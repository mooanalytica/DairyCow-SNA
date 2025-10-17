# Combined ByteTrack + MMPose (ZebraPose) pipeline WITH per‑joint Kalman smoothing.

import os
import sys
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict, deque
import joblib
import math
import torch
torch.backends.cudnn.benchmark = True


# -------------------- EDIT THESE PATHS/SETTINGS --------------------
BASE = Path(__file__).resolve().parent

# Make local repos importable
sys.path.insert(0, str(BASE / "ByteTrack"))
sys.path.insert(0, str(BASE / "ZebraPoseViTPose"))

# Input videos - CHANGE THE DIRECTORY TO THE FOLDER CONTAINING THE INPUT VIDEOS TO BE TRACKED
VIDEO_DIR_OR_LIST   = r"E:\Tracking_Videos\Input_Videos"
VIDEO_FILES         = []

# Model weights - CHANGE THESE PATHS TO YOUR WEIGHTS
DET_MODEL_WEIGHTS   = r"Object_Detection_Trained_Model.pt"
ID_MODEL_WEIGHTS    = r"Identification_Model_Trained.pt"
POSE_CONFIG         = r"ZebraPoseViTPose\ZebraPose\configs\animal\2d_kpt_sview_rgb_img\topdown_heatmap\MAE_pret_syn\s_zebras_old_adam.py"
POSE_CHECKPOINT     = r"Keypoint_Model_Trained.pth"

# Output videos - CHANGE THE DIRECTORY TO THE FOLDER CONTAINING THE INPUT VIDEOS TO BE TRACKED
OUT_PARENT          = r"E:\Tracking_Videos\Pose_Tracker_Outputs"
WRITE_ANNOTATED_MP4 = True

# Devices
DET_DEVICE          = 0   # GPU index or 'cpu'
POSE_DEVICE         = "cuda:0"     # or "cpu"

# Detection params
DET_CLASSES_TO_KEEP = None
DET_CONF_THRESH     = 0.25
DET_IOU_NMS         = 0.5
DET_IMGSZ           = 960

# ByteTrack params (fps is filled per-video)
TRACK_THRESH        = 0.5
MATCH_THRESH        = 0.7
TRACK_BUFFER        = 120

# Visual constants
FONT_SCALE_LABEL = 3
TEXT_THICKNESS   = 2
BBOX_THICKNESS   = 7
POSE_RADIUS      = 14
POSE_THICKNESS   = 2

# Identification window
ID_WINDOW_SIZE_FRAMES   = 20
UNKNOWN_LABEL           = "unknown"
CROP_PAD_FRAC           = 0.05
ID_UNIQUE_ACROSS_TRACKS = True
ID_CONF_MIN             = 0.50

# Pose visualization threshold
KPT_SCORE_THR = 0.2

# ========== Kalman Smoothing (PER-JOINT) ===================
USE_KALMAN_SMOOTHING = True
CONF_USE_THRESHOLD   = 0.05
BASE_R               = 4.0
Q_POS                = 1.0
Q_VEL                = 10.0
STALE_FRAMES_FACTOR  = 2.0

# ================== Interaction Gating + Classification ==================
INTERACT_MIN_SEC        = 4.0          # geometry must hold ≥ this long
INTERACT_DIAG_SIM_RATIO = 0.75         # diag similarity threshold
TIME_WINDOW_SEC         = 30.0         # 1 weight per 30s per pair per class
COOLDOWN_SEC            = 30.0         # block new events for this pair for 30s after any event ends
CONF_MASK_THR           = 0.20         # mask low-conf keypoints same as training
SVM_MARGIN_THR          = 1.0          # margin below this => neutral
SVM_MODEL_PATH          = str(BASE / "svm_model.joblib")  # override via CLI if needed
PROXIMITY_SCALE         = 0.35

# ================= Speed knobs =================
# (A) classify less often
CLASSIFY_EVERY_HZ   = 6          # run SVM ~6 times/sec once gated
CLASSIFY_EVERY      = None       # filled per-video as int(fps / CLASSIFY_EVERY_HZ)

# (B) shrink feature buffer
BUF_SEC_MIN         = 5.0        # keep 5s (≥ 4s gate)

# (C) throttle drawing & CSV
DRAW_EVERY          = 2          # draw pose every N frames (2=every other frame)
CSV_EVERY           = 5          # write keypoints every N frames
DRAW_ONLY_ACTIVE    = False      # set True to draw only when a pair is active

# (D) hysteresis + suspect cap
GAP_TOL_SEC         = 0.5        # allow 0.5s misses without reset
MAX_SUSPECT_PAIRS   = 12         # keep only the closest N pairs per frame


# Helper fns for interaction logic
def _center_and_diag(tlwh):
    x, y, w, h = map(float, tlwh)
    cx, cy = x + w*0.5, y + h*0.5
    diag = math.hypot(w, h)
    return cx, cy, diag

def _is_candidate(t1, t2, proximity=PROXIMITY_SCALE):
    c1x, c1y, d1 = _center_and_diag(t1.tlwh)
    c2x, c2y, d2 = _center_and_diag(t2.tlwh)
    dist = math.hypot(c1x - c2x, c1y - c2y)
    cond1 = dist <= proximity * (d1 + d2)
    sim = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    cond2 = sim >= INTERACT_DIAG_SIM_RATIO
    return cond1 and cond2

def _slot_area_scale(area):
    return math.sqrt(area) if area and area > 0 else float('nan')

import numpy as _np

_AGGR_FUNCS = {
    "mean": _np.nanmean,
    "std":  _np.nanstd,
    "max":  _np.nanmax,
    "p05":  lambda x: _np.nanpercentile(x, 5),
    "p50":  lambda x: _np.nanpercentile(x, 50),
    "p95":  lambda x: _np.nanpercentile(x, 95),
}
def _slope(y):
    idx = _np.arange(len(y), dtype=float)
    mask = _np.isfinite(y)
    if mask.sum() < 2: return _np.nan
    x = idx[mask]; v = y[mask]
    xm, ym = x.mean(), v.mean()
    den = ((x-xm)**2).sum()
    return float(((x-xm)*(v-ym)).sum()/den) if den>0 else 0.0

def _aggr_series(series):
    out = {k: float(fn(series)) for k,fn in _AGGR_FUNCS.items()}
    out["slope"] = _slope(series)
    return out

_PAIR_CACHE = {}  # cache intra-pair indices & names by (K, tuple(kpt_names))

def _get_intra_pairs(K, kpt_names):
    key = (K, tuple(kpt_names) if kpt_names else None)
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    # build upper-triangular (i<j) index pairs and names
    ii, jj = np.triu_indices(K, k=1)
    if kpt_names:
        nm = lambda i: _colsafe(kpt_names[i])
        names = [f"{nm(i)}__{nm(j)}" for i, j in zip(ii, jj)]
    else:
        names = [f"{i}__{j}" for i, j in zip(ii, jj)]
    _PAIR_CACHE[key] = (ii, jj, names)
    return _PAIR_CACHE[key]

def _pair_features(st, kpt_names=None):
    import numpy as np, math
    kA_list = [k for k in st["kptsA"]]
    kB_list = [k for k in st["kptsB"]]
    if not kA_list or not kB_list: return {}
    KA = next((k for k in kA_list if k is not None), None)
    KB = next((k for k in kB_list if k is not None), None)
    if KA is None or KB is None: return {}
    K = KA.shape[0]
    T = len(st["frames"])
    scaleA = np.array(st["scaleA"], float)            # [T]
    scaleB = np.array(st["scaleB"], float)            # [T]
    scaleAB = np.sqrt(scaleA*scaleB)                  # [T]

    def _stack(lst):
        arr = np.full((T, K, 3), np.nan, float)
        for t, k in enumerate(lst):
            if k is not None and k.shape[0] == K:
                arr[t] = k
        return arr
    KA3 = _stack(kA_list)  # [T,K,3]
    KB3 = _stack(kB_list)

    if CONF_MASK_THR > 0:
        KA3[KA3[...,2] < CONF_MASK_THR, :2] = np.nan
        KB3[KB3[...,2] < CONF_MASK_THR, :2] = np.nan

    feats = {}
    ii, jj, nm_pairs = _get_intra_pairs(K, kpt_names)

    # ----- Intra-slot, vectorized over all i<j pairs -----
    Axy = KA3[...,:2]                     # [T,K,2]
    Bxy = KB3[...,:2]
    dA = np.sqrt(((Axy[:, ii, :] - Axy[:, jj, :])**2).sum(-1)) / scaleA[:, None]   # [T,M]
    dB = np.sqrt(((Bxy[:, ii, :] - Bxy[:, jj, :])**2).sum(-1)) / scaleB[:, None]   # [T,M]

    # Aggregations for all pairs at once → vectors of length M
    def _agg_all(mat):  # mat: [T,M]
        out = {
            "mean": np.nanmean(mat, axis=0),
            "std":  np.nanstd(mat, axis=0),
            "max":  np.nanmax(mat, axis=0),
            "p05":  np.nanpercentile(mat, 5, axis=0),
            "p50":  np.nanpercentile(mat, 50, axis=0),
            "p95":  np.nanpercentile(mat, 95, axis=0),
        }
        # slope per pair: vectorized least-squares over T
        idx = np.arange(mat.shape[0], dtype=float)
        mask = np.isfinite(mat)
        # fallback: compute slope per column quickly
        slopes = np.full(mat.shape[1], np.nan, float)
        for m in range(mat.shape[1]):
            msk = mask[:, m]
            if msk.sum() >= 2:
                x = idx[msk]; y = mat[msk, m]
                xm, ym = x.mean(), y.mean()
                den = ((x - xm)**2).sum()
                slopes[m] = ((x - xm) * (y - ym)).sum() / den if den > 0 else 0.0
        out["slope"] = slopes
        return out

    Ag = _agg_all(dA)
    Bg = _agg_all(dB)
    for n, vec in Ag.items():
        for name, val in zip(nm_pairs, vec):
            feats[f"A_{name}__{n}"] = float(val)
    for n, vec in Bg.items():
        for name, val in zip(nm_pairs, vec):
            feats[f"B_{name}__{n}"] = float(val)

    # ----- Cross-slot centroid distance -----
    cAx = np.nanmean(Axy[...,0], axis=1); cAy = np.nanmean(Axy[...,1], axis=1)
    cBx = np.nanmean(Bxy[...,0], axis=1); cBy = np.nanmean(Bxy[...,1], axis=1)
    cent = np.sqrt((cAx-cBx)**2 + (cAy-cBy)**2) / scaleAB
    for n,v in _aggr_series(cent).items():
        feats[f"AB_centroid__{n}"] = v

    # ----- Cross-slot min kpt-to-kpt (broadcast) -----
    diff = Axy[:, :, None, :] - Bxy[:, None, :, :]      # [T,K,K,2]
    dist = np.sqrt((diff**2).sum(-1))                   # [T,K,K]
    maskA = np.isfinite(Axy[...,0]) & np.isfinite(Axy[...,1])
    maskB = np.isfinite(Bxy[...,0]) & np.isfinite(Bxy[...,1])
    valid = maskA[:, :, None] & maskB[:, None, :]
    dist[~valid] = np.nan
    dmins = np.nanmin(dist / scaleAB[:, None, None], axis=(1,2))  # [T]
    for n,v in _aggr_series(dmins).items():
        feats[f"AB_min_kpt2kpt__{n}"] = v

    return feats


def predict_with_margin(pipe, Xrow: dict, cols_cache, medians=None):
    import pandas as pd, numpy as np
    cols = cols_cache.get("cols")
    if cols is None:
        # best: use the scaler's training feature names
        cols = None
        try:
            cols = list(pipe.named_steps.get("scaler").feature_names_in_)
        except Exception:
            pass
        if cols is None:
            # fallback to pipeline-level attribute, or sorted keys
            cols = list(getattr(pipe, "feature_names_in_", []) or sorted(Xrow.keys()))
        cols_cache["cols"] = cols

    x = pd.DataFrame([[Xrow.get(c, np.nan) for c in cols]], columns=cols)
    # fill NaNs similarly to training
    if medians is not None:
        x = x.fillna(medians)
    else:
        x = x.fillna(x.median(numeric_only=True))

    try:
        scores = pipe.decision_function(x)
        if scores.ndim == 1: scores = scores.reshape(1, -1)
        if scores.shape[1] == 1:
            margin = float(abs(scores[0,0])); pred_idx = int(scores[0,0] > 0)
        else:
            order = np.argsort(scores[0])[::-1]
            pred_idx = int(order[0]); margin = float(scores[0, order[0]] - scores[0, order[1]])
        label = pipe.classes_[pred_idx]
        return label, margin
    except Exception:
        # last resort
        label = pipe.predict(x)[0]
        return label, 0.0

# ---------------------- Imports after sys.path ----------------------
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO
from ZebraPoseViTPose.mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    vis_pose_result
)
try:
    from ZebraPoseViTPose.mmpose.datasets import DatasetInfo
except Exception:
    DatasetInfo = None
# from ZebraPoseViTPose.mmpose.core.evaluation import get_similarity
# from ZebraPoseViTPose.mmpose.utils import get_config, collect_env
from ZebraPoseViTPose.mmpose.utils import collect_env

# ------------------------------ Utils ------------------------------
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".ts", ".webm"}

def enumerate_videos():
    files = []
    if VIDEO_FILES:
        for p in VIDEO_FILES:
            if Path(p).suffix.lower() in VIDEO_EXTS:
                files.append(Path(p))
    if VIDEO_DIR_OR_LIST and Path(VIDEO_DIR_OR_LIST).exists():
        for p in Path(VIDEO_DIR_OR_LIST).iterdir():
            if p.suffix.lower() in VIDEO_EXTS:
                files.append(p)
    files = sorted(set(files))
    if not files:
        raise RuntimeError("No video files found. Set VIDEO_DIR_OR_LIST or VIDEO_FILES.")
    return files

def make_writer(out_path, fps, width, height):
    out_path = str(out_path)
    tried = []
    for cc in ('mp4v', 'avc1', 'H264', 'X264'):
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps, (width, height))
        tried.append(cc)
        if w.isOpened():
            return w
        w.release()
    avi_path = out_path if out_path.lower().endswith(".avi") else out_path.rsplit(".", 1)[0] + ".avi"
    for cc in ('MJPG', 'XVID'):
        w = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*cc), fps, (width, height))
        tried.append("AVI:" + cc)
        if w.isOpened():
            return w
        w.release()
    print(f"[warn] writer open failed for {out_path}; tried {tried}")
    return None

def detections_from_ultralytics(det_results, keep_class_ids=None):
    """Return Nx5 [x1,y1,x2,y2,score] float32."""
    if not det_results:
        return np.empty((0, 5), float)
    r = det_results[0]
    if r.boxes is None or r.boxes.shape[0] == 0:
        return np.empty((0, 5), float)
    xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
    conf = r.boxes.conf.cpu().numpy().astype(float)
    if keep_class_ids is not None and r.boxes.cls is not None:
        cls = r.boxes.cls.cpu().numpy().astype(np.int32)
        mask = np.isin(cls, np.asarray(keep_class_ids, dtype=np.int32))
        xyxy = xyxy[mask]
        conf = conf[mask]
    if xyxy.size == 0:
        return np.empty((0, 5), float)
    return np.hstack([xyxy[:, :4], conf[:, None]])

def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    return x, y, x + w, y + h

def crop_with_pad(img, box_xyxy, pad_frac=0.0):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    px = w * pad_frac
    py = h * pad_frac
    x1 = int(max(0, np.floor(x1 - px)))
    y1 = int(max(0, np.floor(y1 - py)))
    x2 = int(min(W, np.ceil(x2 + px)))
    y2 = int(min(H, np.ceil(y2 + py)))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def identity_color(identity: str):
    if not identity or str(identity).lower() == "unknown":
        return (160, 160, 160)
    h = int(hashlib.sha1(str(identity).encode("utf-8")).hexdigest(), 16) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_label_with_bg(img, x1, y1, text, color, font_scale=FONT_SCALE_LABEL):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, 2)
    x2 = x1 + tw + 8
    y2 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y2), (x2, y1), color, -1)
    cv2.putText(img, text, (x1 + 4, y1 - 4), FONT, font_scale, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)

# ----- Interaction drawing helpers -----
POSITIVE_INTERACTIONS = {"licking", "grooming", "allogrooming"}  # treat allogrooming as grooming
NEGATIVE_INTERACTIONS = {"displacement", "headbutting"}
COL_POS = (0, 255, 0)   # BGR: green
COL_NEG = (0, 0, 255)   # BGR: red

def normalize_inter_label(lbl: str):
    """Normalize to title case for display and map synonyms."""
    lab = str(lbl).strip().lower()
    if lab == "allogrooming": lab = "grooming"
    title = lab.capitalize()
    return lab, title

def inter_color(lbl_norm: str):
    if lbl_norm in POSITIVE_INTERACTIONS:
        return COL_POS
    if lbl_norm in NEGATIVE_INTERACTIONS:
        return COL_NEG
    return (255, 255, 255)  # fallback white

def draw_text_outline(img, x, y, text, color, font_scale=FONT_SCALE_LABEL, thickness=TEXT_THICKNESS):
    """Draw colored text with a black outline for readability."""
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), FONT, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)

def _colsafe(name: str) -> str:
    return name.strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_')

def extract_kpt_names(dataset_info, K_fallback=None):
    names = None
    try:
        if hasattr(dataset_info, 'keypoint_info') and isinstance(dataset_info.keypoint_info, dict):
            items = sorted(dataset_info.keypoint_info.items(), key=lambda kv: int(kv[0]))
            names = [_colsafe(info.get('name') or f'kpt_{int(idx)}') for idx, info in items]
    except Exception:
        names = None
    if not names and K_fallback is not None:
        names = [f'kpt_{i}' for i in range(K_fallback)]
    return names

def feats_to_numpy(feat_dict, train_cols, col_index, med_vec):
    x = np.array([feat_dict.get(c, np.nan) for c in train_cols], dtype=np.float32)
    # simple median impute
    nan_mask = ~np.isfinite(x)
    if nan_mask.any():
        x[nan_mask] = med_vec[nan_mask]
    return x.reshape(1, -1)

def predict_with_margin_np(pipe, x_np):
    scores = pipe.decision_function(x_np)
    if scores.ndim == 1: scores = scores.reshape(1, -1)
    if scores.shape[1] == 1:
        margin = float(abs(scores[0,0]))
        pred_idx = int(scores[0,0] > 0)
    else:
        order = np.argsort(scores[0])[::-1]
        pred_idx = int(order[0])
        margin = float(scores[0, order[0]] - scores[0, order[1]])
    label = pipe.classes_[pred_idx]
    return label, margin

def predict_with_margin_df(pipe, x_df):
    scores = pipe.decision_function(x_df)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if scores.shape[1] == 1:
        margin = float(abs(scores[0, 0]))
        pred_idx = int(scores[0, 0] > 0)
    else:
        order = np.argsort(scores[0])[::-1]
        pred_idx = int(order[0])
        margin = float(scores[0, order[0]] - scores[0, order[1]])
    label = pipe.classes_[pred_idx]
    return label, margin


# ---------------- Kalman smoother (per‑joint constant‑velocity) ---------------
class Kalman2D:
    def __init__(self, dt=1/30.0, q_pos=Q_POS, q_vel=Q_VEL, r=BASE_R):
        import numpy as np
        self.np = np
        self.dt = float(dt)
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1, 0],
                           [0,0,0, 1]], float)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.R_base = float(r)
        self.x = np.zeros((4,1), float)
        self.P = np.eye(4, dtype=float) * 1e3
        self.initialized = False

    def init_state(self, x, y):
        self.x[:] = [[x],[y],[0.0],[0.0]]
        self.P[:] = self.np.eye(4, dtype=float) * 10.0
        self.initialized = True

    def step(self, zx, zy, conf=1.0):
        """One predict+update cycle; returns smoothed (x,y)."""
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Build R (optionally inflate when conf is low)
        conf = float(conf)
        scale = 1.0 if conf >= CONF_USE_THRESHOLD else (1.0 + (CONF_USE_THRESHOLD - conf)*5.0)
        R = self.np.eye(2, dtype=float) * (self.R_base * scale)

        # Update
        z = self.np.array([[zx],[zy]], float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ self.np.linalg.inv(S)
        self.x = self.x + K @ y
        I = self.np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0,0]), float(self.x[1,0])

# --------------------------------------------------------------------

def main():
    # Prepare OUT_DIR (auto-increment)
    base_dir = Path(OUT_PARENT)
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base_dir.iterdir() if d.is_dir()]
    idx = len(existing) + 1
    OUT_DIR = base_dir / f"{idx}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing outputs to {OUT_DIR}")

    # Load detector + identification models (Ultralytics)
    det_model = YOLO(DET_MODEL_WEIGHTS)
    id_model  = YOLO(ID_MODEL_WEIGHTS)

    # >>> Force CUDA <<<
    if torch.cuda.is_available():
        # send the underlying torch.nn.Module to GPU
        det_model.model.to('cuda')
        id_model.model.to('cuda')

        # optional: half-precision for an extra boost
        det_model.model.half()
        id_model.model.half()

    if hasattr(id_model, "names"):
        ID_CLASS_NAMES = list(id_model.names.values())
    else:
        ID_CLASS_NAMES = [str(i) for i in range(16)]  # fallback
    ID_CLASS_SET = set(ID_CLASS_NAMES)

    # Load pose model
    assert Path(POSE_CONFIG).is_file(), f"Missing config: {POSE_CONFIG}"
    assert Path(POSE_CHECKPOINT).is_file(), f"Missing checkpoint: {POSE_CHECKPOINT}"
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=POSE_DEVICE)
    pose_dataset = pose_model.cfg.data['test']['type']
    di_cfg = pose_model.cfg.data['test'].get('dataset_info', None)
    if DatasetInfo is not None:
        if di_cfg is None:
            di_cfg = dict(dataset_name='cow_27', flip_pairs=[])
        dataset_info = DatasetInfo(di_cfg)
    else:
        dataset_info = di_cfg  # best effort
    KPT_NAMES = extract_kpt_names(dataset_info)

    # Tracking CSV rows
    track_rows = []  # (video, frame, track_id, x, y, w, h, score, identity, id_conf)

    # Keypoints CSV writer
    kp_csv_path = OUT_DIR / "keypoints.csv"
    kp_file = open(kp_csv_path, "w", newline="", encoding="utf-8")
    kp_writer = csv.writer(kp_file)
    kp_header_written = False

    all_inter_rows = []

    # Iterate videos
    for vpath in enumerate_videos():
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[warn] cannot open {vpath}")
            continue

        # Per interaction class count / id
        inter_seq = defaultdict(int)

        W, H = int(cap.get(3)), int(cap.get(4))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Video writer
        writer = None
        if WRITE_ANNOTATED_MP4:
            out_mp4 = OUT_DIR / f"{vpath.stem}_tracked_pose.mp4"
            writer = make_writer(out_mp4, fps, W, H)

        # Init tracker
        import argparse
        args = argparse.Namespace(
            track_thresh=TRACK_THRESH,
            track_buffer=TRACK_BUFFER,
            match_thresh=MATCH_THRESH,
            frame_rate=fps,
            mot20=False
        )
        tracker = BYTETracker(args)

        # --- SVM model (joblib) ---
        try:
            svm_pipe = joblib.load(SVM_MODEL_PATH)
        except Exception as e:
            print(f"[warn] could not load SVM model at {SVM_MODEL_PATH}: {e}")
            svm_pipe = None

        # --- Feature schema / medians (one-time, guarded) ---
        CLASSIFY_ENABLED = False
        train_cols = None
        med_vec = None

        # Try to get feature names from the fitted scaler (best source)
        if svm_pipe is not None:
            try:
                train_cols = list(svm_pipe.named_steps['scaler'].feature_names_in_)
            except Exception:
                train_cols = None

        # Fallback: load feature schema + medians from your training CSV (if present)
        if train_cols is None:
            TRAIN_FEATURES_CSV = r"F:\Interaction_Cropped_Exports\Individual_Wise_Interaction\ML_Results\svm_features.csv"
            try:
                import pandas as pd
                _df_train_feats = pd.read_csv(TRAIN_FEATURES_CSV)
                train_cols = [c for c in _df_train_feats.columns if c not in ("video_id", "label")]
                med_vec = _df_train_feats[train_cols].median(numeric_only=True).to_numpy(np.float32)
                print(f"[info] loaded {len(train_cols)} training feature columns + medians")
            except Exception as e:
                print(f"[warn] feature schema not found: {e}")

        # If we got cols but no medians yet, at least create zeros (you can replace with better stats later)
        if (train_cols is not None) and (med_vec is None):
            med_vec = np.zeros((len(train_cols),), dtype=np.float32)

        # Build the single-row DataFrame template ONLY if everything is available
        if (svm_pipe is not None) and (train_cols is not None) and (med_vec is not None):
            CLASSIFY_ENABLED = True
            import pandas as pd
            x_df_template = pd.DataFrame([np.zeros(len(train_cols), dtype=np.float32)], columns=train_cols)
            med_series = pd.Series(med_vec, index=train_cols, dtype=np.float32)

            def feats_to_df_inplace(feat_dict, df_row: pd.DataFrame):
                """Fill the 1xD DataFrame in-place with feat_dict values, median-impute NaNs."""
                row = df_row.iloc[0]
                row[:] = med_series.values
                # fill available features
                for k, v in feat_dict.items():
                    if k in df_row.columns:
                        row[k] = v
                # impute invalids
                vals = row.values
                bad = ~np.isfinite(vals)
                if bad.any():
                    vals[bad] = med_series.values[bad]
                    row[:] = vals
                return df_row

            def predict_with_margin_df(pipe, x_df):
                scores = pipe.decision_function(x_df)
                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)
                if scores.shape[1] == 1:
                    margin = float(abs(scores[0, 0]));
                    pred_idx = int(scores[0, 0] > 0)
                else:
                    order = np.argsort(scores[0])[::-1]
                    pred_idx = int(order[0])
                    margin = float(scores[0, order[0]] - scores[0, order[1]])
                label = pipe.classes_[pred_idx]
                return label, margin
        else:
            # Cleanly disable classification — rest of the pipeline still runs
            svm_pipe = None
            print("[warn] Interaction classification disabled: missing model or feature schema")

        col_index = {c: i for i, c in enumerate(train_cols)} if train_cols else {}
        _cols_cache = {}

        # --- Interaction per-video state ---
        BUF_SEC = max(INTERACT_MIN_SEC, BUF_SEC_MIN)
        buf_len = int(math.ceil(BUF_SEC * fps))

        def _make_pair_buf():
            return {
                "frames": deque(maxlen=buf_len),
                "kptsA":  deque(maxlen=buf_len),
                "kptsB":  deque(maxlen=buf_len),
                "scaleA": deque(maxlen=buf_len),
                "scaleB": deque(maxlen=buf_len),
            }

        # tracks geometry history per unordered pair (tidA, tidB)
        pair_geom = defaultdict(lambda: {"suspect_frames": 0, "miss": 0, "prox": None, "last_ok_frame": -1})

        min_frames = int(math.ceil(INTERACT_MIN_SEC * fps))
        gap_tol_fr = int(round(GAP_TOL_SEC * fps))
        CLASSIFY_EVERY = max(1, int(round(fps / CLASSIFY_EVERY_HZ)))

        cooldown_frames = int(round(COOLDOWN_SEC * fps))
        pair_cooldown_until = {}  # (tidA, tidB) -> frame index until which new events are blocked

        id_cooldown_until = {}  # (labA, labB) sorted tuple -> frame index

        def _id_key_for_pair(a_id, b_id):
            """Return a sorted identity tuple or None if either is unknown/unconfirmed."""
            stA = track_state.get(int(a_id), {})
            stB = track_state.get(int(b_id), {})
            la = stA.get("label", UNKNOWN_LABEL)
            lb = stB.get("label", UNKNOWN_LABEL)
            if la == UNKNOWN_LABEL or lb == UNKNOWN_LABEL or not la or not lb:
                return None
            # stable key regardless of order
            return tuple(sorted((str(la), str(lb))))

        print(f"[info] fps={fps:.2f}  classify_every={CLASSIFY_EVERY}f  buf={BUF_SEC:.1f}s  gap_tol={gap_tol_fr}f  max_suspects={MAX_SUSPECT_PAIRS}")

        pair_buf  = defaultdict(_make_pair_buf)
        pair_evt = defaultdict(lambda: {
            "active": False,
            "start_f": None,
            "cur_label": None,
            "last_weight_window_start_f": None,
            "accum_frames": 0,
            "last_margin": None,
            "last_pred_frame": -1,
            "margin_sum": 0.0,
            "margin_cnt": 0
        })

        inter_rows = []

        # Identification window state
        frames_in_window = 0
        track_state = {}
        for lab in ID_CLASS_NAMES:
            pass

        # ------------ NEW: Kalman banks per track ------------
        kf_bank = {}
        last_seen = {}
        stale_limit = int(max(1.0, fps) * STALE_FRAMES_FACTOR)
        # -----------------------------------------------------------------------

        pbar = tqdm(total=nframes if nframes > 0 else None,
                    desc=f"{vpath.name} ({fps:.2f} fps)", unit="frame", leave=False)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ----- DETECT + TRACK (ByteTrack), update track_rows, etc. -----
            # ------------------------ Detection ------------------------
            det_res = det_model.predict(source=frame, imgsz=DET_IMGSZ,
                                        conf=DET_CONF_THRESH, iou=DET_IOU_NMS,
                                        device=DET_DEVICE, verbose=False)
            dets = detections_from_ultralytics(det_res, DET_CLASSES_TO_KEEP)  # Nx5 [x1,y1,x2,y2,score]

            # ------------------------- Tracking ------------------------
            online_targets = tracker.update(dets, (H, W), (H, W))

            # ----------------------- Identification (20‑frame window) ---
            for t in online_targets:
                tlwh = t.tlwh
                tid = int(t.track_id)
                score = float(getattr(t, "score", 1.0))

                # ensure state
                st = track_state.get(tid)
                if st is None:
                    st = {
                        "confirmed": False,
                        "label": UNKNOWN_LABEL,
                        "conf": 0.0,
                        "win_votes": {lab: 0 for lab in ID_CLASS_NAMES},
                        "win_conf": {lab: 0.0 for lab in ID_CLASS_NAMES},
                    }
                    track_state[tid] = st

                # collect crops/votes into the window
                x1, y1, x2, y2 = tlwh_to_xyxy(tlwh)
                crop, _ = crop_with_pad(frame, (x1, y1, x2, y2), pad_frac=CROP_PAD_FRAC)
                if crop.size == 0:
                    continue
                id_pred = id_model.predict(source=crop, imgsz=224, device=DET_DEVICE, verbose=False)[0]
                if id_pred.probs is not None and hasattr(id_pred.probs, "top1"):
                    top_idx = int(id_pred.probs.top1)
                    lab = id_pred.names[top_idx] if hasattr(id_pred, "names") else str(top_idx)
                    conf = float(id_pred.probs.top1conf)
                    if lab in ID_CLASS_SET:
                        st["win_votes"][lab] += 1
                        st["win_conf"][lab] += conf

            frames_in_window += 1
            if frames_in_window >= ID_WINDOW_SIZE_FRAMES:
                used_labels = set()
                # prefer already confirmed labels
                for t in online_targets:
                    st = track_state.get(int(t.track_id))
                    if st and st["confirmed"] and st["label"] != UNKNOWN_LABEL:
                        used_labels.add(st["label"])

                # decide identities for current tracks
                cands = []
                for t in online_targets:
                    tid = int(t.track_id)
                    st = track_state.get(tid)
                    if not st or st["confirmed"]:
                        continue
                    best_lab, best_votes, best_conf_sum = None, -1, -1.0
                    for lab in ID_CLASS_NAMES:
                        v = st["win_votes"][lab];
                        c = st["win_conf"][lab]
                        if (v > best_votes) or (v == best_votes and c > best_conf_sum):
                            best_lab, best_votes, best_conf_sum = lab, v, c
                    cands.append((tid, best_lab, best_votes, best_conf_sum))
                cands.sort(key=lambda x: (x[2], x[3]), reverse=True)

                for tid, best_lab, votes, conf_sum in cands:
                    st = track_state[tid]
                    if best_lab is None or votes <= 0:
                        st["confirmed"] = False
                        st["label"] = UNKNOWN_LABEL
                        st["conf"] = 0.0
                        continue
                    if ID_UNIQUE_ACROSS_TRACKS and best_lab in used_labels:
                        st["confirmed"] = False
                        st["label"] = UNKNOWN_LABEL
                        st["conf"] = 0.0
                        continue
                    avg_conf = conf_sum / max(1, votes)
                    st["confirmed"] = True
                    st["label"] = best_lab if avg_conf >= ID_CONF_MIN else UNKNOWN_LABEL
                    st["conf"] = avg_conf if avg_conf >= ID_CONF_MIN else 0.0
                    used_labels.add(st["label"])

                # reset window counts
                for st in track_state.values():
                    st["win_votes"] = {lab: 0 for lab in ID_CLASS_NAMES}
                    st["win_conf"] = {lab: 0.0 for lab in ID_CLASS_NAMES}
                frames_in_window = 0

            # ----------------------- Pose inference (SELECTIVE) + Interaction --------------------
            # 1) Build geometry-based candidate pairs
            tracks = list(online_targets)
            suspect_pairs = set()
            need_pose_tids = set()
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    a, b = tracks[i], tracks[j]
                    if _is_candidate(a, b, PROXIMITY_SCALE):
                        key = tuple(sorted((int(a.track_id), int(b.track_id))))
                        st = pair_geom[key]

                        # 1) Hysteresis: accumulate "suspect" time
                        st["suspect_frames"] += 1
                        st["miss"] = 0
                        st["last_ok_frame"] = frame_idx

                        # 2) Proximity score for ranking (smaller = closer)
                        c1x, c1y, d1 = _center_and_diag(a.tlwh)
                        c2x, c2y, d2 = _center_and_diag(b.tlwh)
                        denom = (0.5 * d1 + 0.5 * d2) + 1e-6
                        prox = math.hypot(c1x - c2x, c1y - c2y) / denom

                        # for this frame keep the smallest (best) proximity we’ve seen
                        if st["prox"] is None or prox < st["prox"]:
                            st["prox"] = prox

                        suspect_pairs.add(key)
                        need_pose_tids.update(key)

            # reset for non-suspect this frame
            for key, st in list(pair_geom.items()):
                if key not in suspect_pairs:
                    st["miss"] += 1
                    if st["miss"] > gap_tol_fr:
                        st["suspect_frames"] = 0
                        st["miss"] = 0
                        st["prox"] = None

            # Limit how many pairs we process this frame
            if len(suspect_pairs) > MAX_SUSPECT_PAIRS:
                # rank by 'prox' (smaller is closer); pairs without prox go to the end
                ranked = sorted(
                    suspect_pairs,
                    key=lambda k: (
                        -pair_geom[k]["suspect_frames"],  # more progress first
                        pair_geom[k]["prox"] if pair_geom[k]["prox"] is not None else 1e9  # then closeness
                    )
                )
                keep = set(ranked[:MAX_SUSPECT_PAIRS])

                # 1) Trim suspect set
                suspect_pairs = keep

                # 2) Build pose TID set only from kept pairs
                need_pose_tids = set([tid for k in keep for tid in k])

                # 3) Optional: aggressively drop buffers for pairs we didn’t keep (saves CPU/RAM)
                for k in list(pair_buf.keys()):
                    st = pair_geom.get(k)
                    if (k not in keep) and (st is None or st.get("miss", 0) > gap_tol_fr):
                        pair_buf.pop(k, None)

            # 2) Pose only for tids in suspect pairs
            person_results = []
            tid_order = []
            areas = []
            if len(need_pose_tids) > 0:
                for t in online_targets:
                    tid = int(t.track_id)
                    if tid not in need_pose_tids:
                        continue
                    x, y, w, h = map(float, t.tlwh)
                    x = max(0.0, min(x, W - 1.0))
                    y = max(0.0, min(y, H - 1.0))
                    w = max(1.0, min(w, W - x))
                    h = max(1.0, min(h, H - y))
                    person_results.append({'bbox': [x, y, w, h]})
                    tid_order.append(tid)
                    areas.append(w*h)

            pose_results = []
            if person_results:
                pose_results = inference_top_down_pose_model(
                    pose_model, frame, person_results=person_results,
                    bbox_thr=None, format='xywh', dataset=pose_dataset,
                    dataset_info=dataset_info, return_heatmap=False
                )
                if isinstance(pose_results, tuple): pose_results = pose_results[0]
                if pose_results and isinstance(pose_results[0], (list, tuple)): pose_results = pose_results[0]
                for i, pr in enumerate(pose_results):
                    pr['track_id'] = tid_order[i]
                    pr['area']     = float(areas[i])

                # -------------------- NEW: Kalman smoothing --------------------
                if USE_KALMAN_SMOOTHING and pose_results:
                    for pr in pose_results:
                        tid = int(pr.get('track_id', -1))
                        kpts = np.asarray(pr.get('keypoints', []), dtype=float)  # [K,3] = x,y,conf
                        if kpts.size == 0:
                            continue
                        K = kpts.shape[0]

                        # Create or refresh filter bank for this tid
                        bank = kf_bank.get(tid)
                        if (bank is None) or (len(bank) != K):
                            # dt = 1/fps for this video
                            bank = [Kalman2D(dt=1.0 / max(1.0, fps)) for _ in range(K)]
                            # initialize from current measurement
                            for j in range(K):
                                bank[j].init_state(kpts[j, 0], kpts[j, 1])
                            kf_bank[tid] = bank

                        last_seen[tid] = frame_idx

                        # Step each joint through its filter
                        for j in range(K):
                            sx, sy = bank[j].step(kpts[j, 0], kpts[j, 1], kpts[j, 2])
                            kpts[j, 0], kpts[j, 1] = sx, sy

                        # Write smoothed kpts back (so downstream features & drawing see smoothed coords)
                        pr['keypoints'] = kpts

                    # prune stale filters to bound memory
                    for old_tid in list(kf_bank.keys()):
                        if frame_idx - last_seen.get(old_tid, -9999) > stale_limit:
                            kf_bank.pop(old_tid, None)
                            last_seen.pop(old_tid, None)
                # ----------------------------------------------------------------

            # 3) Draw Pose (if any)
            if pose_results and (frame_idx % DRAW_EVERY == 0):
                do_draw = True
                if DRAW_ONLY_ACTIVE:
                    # draw only if any active pair involves this tid
                    active_tids = set([tid for (a, b), evt in pair_evt.items() if evt["active"] for tid in (a, b)])
                    do_draw = any(int(pr['track_id']) in active_tids for pr in pose_results)
                if do_draw:
                    try:
                        frame = vis_pose_result(
                            pose_model, frame, pose_results,
                            dataset=pose_dataset, dataset_info=dataset_info,
                            kpt_score_thr=KPT_SCORE_THR, show=False,
                            radius=POSE_RADIUS, thickness=POSE_THICKNESS
                        )
                    except TypeError:
                        frame = vis_pose_result(
                            pose_model, frame, pose_results,
                            dataset=pose_dataset, dataset_info=dataset_info,
                            kpt_score_thr=KPT_SCORE_THR, show=False
                        )

            # Write keypoints CSV (throttled)
            if pose_results and (frame_idx % CSV_EVERY == 0):
                for pr in pose_results:
                    kpts = np.asarray(pr.get('keypoints', []))
                    tid = pr.get('track_id', -1)
                    if kpts.size:
                        if not kp_header_written:
                            K = kpts.shape[0]
                            hdr = ["video", "frame", "track_id"] + sum(
                                ([f"kpt_{j}_x", f"kpt_{j}_y", f"kpt_{j}_conf"] for j in range(K)), [])
                            kp_writer.writerow(hdr)
                            kp_header_written = True
                        flat = [float(v) for xyz in kpts for v in xyz]
                        kp_writer.writerow([vpath.name, frame_idx, tid] + flat)

            # 4) Update pair buffers with this frame's pose for suspect pairs
            pr_by_tid = {int(pr['track_id']): pr for pr in pose_results} if pose_results else {}
            for (a_id, b_id) in suspect_pairs:
                prA = pr_by_tid.get(a_id)
                prB = pr_by_tid.get(b_id)
                def _safe_kpts(pr):
                    if not pr: return None, float('nan')
                    k = pr.get('keypoints', None)
                    arr = np.asarray(k, float) if k is not None and np.asarray(k).size else None
                    return (arr, _slot_area_scale(pr.get('area', 0.0)))
                kA, sA = _safe_kpts(prA)
                kB, sB = _safe_kpts(prB)
                st = pair_buf[(a_id, b_id)]
                st["frames"].append(frame_idx)
                st["kptsA"].append(kA); st["kptsB"].append(kB)
                st["scaleA"].append(sA); st["scaleB"].append(sB)

            # 5) Classify & maintain event state
            def _finalize_and_write(evt, key, end_f):
                a_id, b_id = key
                dur_frames = max(evt["accum_frames"], 0)
                if dur_frames < min_frames:
                    return
                if evt["last_weight_window_start_f"] is None:
                    evt["last_weight_window_start_f"] = evt["start_f"]
                span_frames = end_f - evt["last_weight_window_start_f"] + 1
                windows_crossed = int(math.ceil((span_frames / fps) / TIME_WINDOW_SEC))
                weight = max(1, windows_crossed)
                duration_s = dur_frames / fps
                evt["last_weight_window_start_f"] = int(evt["last_weight_window_start_f"] +
                                                        weight * TIME_WINDOW_SEC * fps)
                margin_avg = (evt["margin_sum"] / evt["margin_cnt"]) if evt["margin_cnt"] > 0 else None
                inter_rows.append([vpath.name, evt["cur_label"], a_id, b_id,
                                   evt["start_f"], end_f, duration_s, weight, margin_avg])
                # reset per-event accumulators so a new event starts fresh
                evt["margin_sum"] = 0.0
                evt["margin_cnt"] = 0

            if svm_pipe is not None:
                # For each suspect pair that held for ≥ min_frames, we MAY classify (on schedule)
                for key in suspect_pairs:
                    if pair_geom[key]["suspect_frames"] >= min_frames:
                        stbuf = pair_buf[key]
                        evt = pair_evt[key]

                        # --- classify only at ~N Hz ---
                        if CLASSIFY_ENABLED and (frame_idx % CLASSIFY_EVERY) == 0:
                            feats = _pair_features(stbuf, kpt_names=KPT_NAMES)
                            if feats:
                                x_df = feats_to_df_inplace(feats, x_df_template)  # reuses the same DF row
                                pred, margin = predict_with_margin_df(svm_pipe, x_df)
                                if margin < SVM_MARGIN_THR:
                                    pred = "neutral"

                                # ---- event transitions ONLY inside the scheduled branch ----
                                if pred != "neutral":
                                    if not evt["active"] or evt["cur_label"] != pred:
                                        # if switching away from an active label, close it at previous frame
                                        if evt["active"] and evt["cur_label"] != pred:
                                            end_f = frame_idx - 1
                                            _finalize_and_write(evt, key, end_f)
                                            pair_cooldown_until[key] = end_f + cooldown_frames
                                            ik = _id_key_for_pair(*key)
                                            if ik is not None:
                                                id_cooldown_until[ik] = end_f + cooldown_frames
                                            # mark closed
                                            evt["active"] = False
                                            evt["cur_label"] = None
                                            evt["start_f"] = None
                                            evt["accum_frames"] = 0

                                        # only start a new event if cooldown has expired
                                        ok_tid = frame_idx >= pair_cooldown_until.get(key, -1)
                                        ik = _id_key_for_pair(*key)
                                        ok_id = True if ik is None else (frame_idx >= id_cooldown_until.get(ik, -1))
                                        if ok_tid and ok_id:
                                            evt["active"] = True
                                            evt["cur_label"] = pred
                                            evt["start_f"] = frame_idx - min_frames + 1
                                            evt["accum_frames"] = min_frames
                                            evt["margin_sum"] += margin
                                            evt["margin_cnt"] += 1
                                            if evt["last_weight_window_start_f"] is None:
                                                evt["last_weight_window_start_f"] = evt["start_f"]

                                            # Assign a sequential interaction ID per class (for display)
                                            _, display_title = normalize_inter_label(pred)
                                            inter_seq[display_title] += 1
                                            evt["cur_inter_id"] = inter_seq[display_title]
                                        # else: under cooldown → do nothing (stay inactive)
                                    else:
                                        # continuing same label; accumulate margin on scheduled ticks
                                        evt["margin_sum"] += margin
                                        evt["margin_cnt"] += 1
                                else:
                                    # neutral → close if open (and start cooldown)
                                    if evt["active"]:
                                        end_f = frame_idx - 1
                                        _finalize_and_write(evt, key, end_f)
                                        pair_cooldown_until[key] = end_f + cooldown_frames
                                        ik = _id_key_for_pair(*key)
                                        if ik is not None:
                                            id_cooldown_until[ik] = end_f + cooldown_frames
                                        evt["active"] = False
                                        evt["cur_label"] = None
                                        evt["start_f"] = None
                                        evt["accum_frames"] = 0

                                evt["last_margin"] = margin
                                evt["last_pred_frame"] = frame_idx

                        # --- regardless of schedule: if active, accrue one frame ---
                        if evt["active"]:
                            evt["accum_frames"] += 1

                # Close events only when the pair has missed beyond hysteresis tolerance
                for key, evt in list(pair_evt.items()):
                    st = pair_geom.get(key)
                    if evt["active"] and (st is None or st["miss"] > gap_tol_fr):
                        st = pair_geom.get(key)
                        end_f = st["last_ok_frame"] if (st and st["last_ok_frame"] >= 0) else (frame_idx - 1)
                        _finalize_and_write(evt, key, end_f)
                        pair_cooldown_until[key] = end_f + cooldown_frames
                        ik = _id_key_for_pair(*key)
                        if ik is not None:
                            id_cooldown_until[ik] = end_f + cooldown_frames
                        evt["active"] = False
                        evt["cur_label"] = None
                        evt["start_f"] = None
                        evt["accum_frames"] = 0

            # Build per-track overlay text for any active interaction segments
            active_inter_text = {}  # tid -> (text, color)
            for (a_id, b_id), evt in pair_evt.items():
                if not evt["active"] or not evt.get("cur_label"):
                    continue
                st = pair_geom.get((a_id, b_id))
                # Hide as soon as geometry breaks: only draw when no current miss
                if (st is None) or (st.get("miss", 0) > 0):
                    continue
                lbl_norm, title = normalize_inter_label(evt["cur_label"])
                if lbl_norm not in POSITIVE_INTERACTIONS and lbl_norm not in NEGATIVE_INTERACTIONS:
                    continue
                color = inter_color(lbl_norm)
                iid = evt.get("cur_inter_id")
                inter_txt = f"{title} {iid}" if iid else title
                active_inter_text[int(a_id)] = (inter_txt, color)
                active_inter_text[int(b_id)] = (inter_txt, color)

            # ------------------------- Draw tracks ---------------------
            for t in online_targets:
                tlwh = t.tlwh
                tid  = int(t.track_id)
                xi, yi, wi, hi = map(int, tlwh)
                x2i, y2i = xi + wi, yi + hi
                st = track_state.get(tid, {"label": UNKNOWN_LABEL, "conf": 0.0})
                color = identity_color(st["label"])
                cv2.rectangle(frame, (xi, yi), (x2i, y2i), color, BBOX_THICKNESS)
                label_txt = f"{tid}"
                draw_label_with_bg(frame, xi, max(0, yi - 2), label_txt, color, font_scale=FONT_SCALE_LABEL)

                # If this track participates in an active interaction, draw the interaction text next to the ID
                if tid in active_inter_text:
                    inter_txt, inter_col = active_inter_text[tid]
                    # compute width of the ID label to place text to its right
                    FONT = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), bl = cv2.getTextSize(label_txt, FONT, FONT_SCALE_LABEL, 2)
                    base_x = xi + tw + 16
                    base_y = max(18, yi - 6)
                    draw_text_outline(frame, base_x, base_y, inter_txt, inter_col, font_scale=FONT_SCALE_LABEL)

                # Save one row per track per frame for tracking_boxes.csv
                track_rows.append([
                    vpath.name, frame_idx, tid,
                    float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3]),
                    float(getattr(t, "score", 1.0)),
                    st.get("label", UNKNOWN_LABEL),
                    float(st.get("conf", 0.0)),
                ])

            if WRITE_ANNOTATED_MP4 and writer:
                writer.write(frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        # finalize any open events for this video
        if svm_pipe is not None:
            for key, evt in list(pair_evt.items()):
                if evt["active"]:
                    _finalize_and_write(evt, key, frame_idx-1 if frame_idx>0 else 0)
                    evt["active"] = False
            # append to global list
            try:
                all_inter_rows.extend(inter_rows)
            except NameError:
                pass
        if writer:
            writer.release()

    # Save tracking CSV
    track_csv_path = OUT_DIR / "tracking_boxes.csv"
    pd.DataFrame(track_rows, columns=[
        "video", "frame", "track_id", "x", "y", "w", "h", "score", "identity", "id_conf"
    ]).to_csv(track_csv_path, index=False)

    # Save interactions CSV (if any)
    try:
        if len(all_inter_rows) > 0:
            inter_df = pd.DataFrame(all_inter_rows, columns=[
                "video","class","tidA","tidB","start_frame","end_frame","duration_s","weight","margin_avg"
            ])
            inter_df.to_csv(OUT_DIR / "interactions.csv", index=False)
            # optional: adjacency per class
            for cls in sorted([c for c in inter_df["class"].unique() if c != "neutral"]):
                sub = inter_df[inter_df["class"] == cls]
                adj = (sub.groupby(["tidA","tidB"])["weight"].sum()
                        .reset_index()
                        .sort_values(["weight"], ascending=False))
                adj.to_csv(OUT_DIR / f"adjacency_{cls}.csv", index=False)
    except Exception as e:
        print(f"[warn] could not write interactions csv: {e}")

    kp_file.close()
    print(f"\n✅ Done.\n  Tracking CSV: {track_csv_path}\n  Keypoints CSV: {kp_csv_path}\n  Videos/plots:  {OUT_DIR}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
