# Combined ByteTrack + MMPose (ZebraPose) pipeline WITH per‑joint Kalman smoothing.

import argparse
import os
import sys
import csv
import hashlib
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict, deque, Counter
import joblib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- Self-contained project paths/settings --------------------
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
VENDOR_DIR = BASE / "vendor"
INPUTS_DIR = BASE / "inputs"
OUTPUTS_DIR = BASE / "outputs"
BYTE_TRACK_ROOT = VENDOR_DIR / "ByteTrack"
ZEBRAPOSE_ROOT = VENDOR_DIR / "ZebraPoseViTPose"

# Make vendored runtime code importable.
sys.path.insert(0, str(BYTE_TRACK_ROOT))
sys.path.insert(0, str(ZEBRAPOSE_ROOT))

DEFAULT_POSE_CONFIG = (
    ZEBRAPOSE_ROOT
    / "ZebraPose"
    / "configs"
    / "animal"
    / "2d_kpt_sview_rgb_img"
    / "topdown_heatmap"
    / "MAE_pret_syn"
    / "s_zebras_old_adam.py"
)

VIDEO_DIR_OR_LIST   = str(INPUTS_DIR)
VIDEO_FILES         = []

DET_MODEL_WEIGHTS   = str(MODELS_DIR / "Object_Detection_Trained_Model.pt")
ID_MODEL_WEIGHTS    = str(MODELS_DIR / "Identification_Model_Trained.pt")
POSE_CONFIG         = str(DEFAULT_POSE_CONFIG)
POSE_CHECKPOINT     = str(MODELS_DIR / "Keypoint_Model_Trained.pth")

OUT_PARENT          = str(OUTPUTS_DIR)
WRITE_ANNOTATED_MP4 = True
ANNOTATED_OUTPUT_SCALE = 0.5   # write smaller video for much faster encode/IO at 4K input

# Devices
DET_DEVICE          = "auto"   # GPU index, "cpu", or "auto"
POSE_DEVICE         = "auto"   # "cuda:0", "cpu", or "auto"

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
INTERACT_MIN_SEC        = 0.5          # geometry must hold ≥ this long
INTERACT_DIAG_SIM_RATIO = 0.75         # diag similarity threshold
TIME_WINDOW_SEC         = 30.0         # 1 weight per 30s per pair per class
COOLDOWN_SEC            = 30.0         # block new events for this pair for 30s after any event ends
CONF_MASK_THR           = 0.20         # mask low-conf keypoints same as training
SVM_MARGIN_THR          = 1.0          # margin below this => neutral
SVM_MODEL_PATH          = str(MODELS_DIR / "svm_model.joblib")  # override via CLI if needed


# ================== NEW: Two-stage models (Stage-1 TCN + Stage-2 InceptionTime) ==================
# Put these .pt files in models/ or override via CLI.
STAGE1_CKPT_PATH = str(MODELS_DIR / "stage1_tcn_best.pt")
STAGE2_CKPT_PATH = str(MODELS_DIR / "stage2_inception_best.pt")

# Stage-1 decision threshold (will be read from checkpoint if present)
STAGE1_THRESHOLD_OVERRIDE = None  # e.g., 0.965; set None to use ckpt["threshold"]

# Multi-instance crops (seconds) for inference
MI_CROPS_SEC = [1, 2, 4, "full"]

# Match inference to how the temporal models were trained.
RESAMPLE_TO_MODEL_FPS = True

# Evaluate both pair orderings because the sequence features are slot-sensitive
# while track-id order is arbitrary with respect to "initiator" vs "recipient".
BIDIRECTIONAL_PAIR_INFERENCE = True

# -------- Step D: dual proximity gating --------
PROX_HARD_CAP = 0.55        # absolute normalized center distance cap
Q_PROX = 0.15               # rolling quantile (e.g., 0.15 => 15th percentile)
Q_WINDOWS_SEC = [1, 2, 4]   # multi-scale distance windows for quantile gate

# -------- Step D: stability k-of-m before running models --------
GATE_STABLE_M = 12
GATE_STABLE_K = 8

# -------- Step E: stage-1 stability (k-of-m) and stage-2 vote --------
STAGE1_STABLE_M = 12
STAGE1_STABLE_K = 8
STAGE2_VOTE_N = 9
STAGE2_MIN_CONF = 0.60
STAGE2_MIN_START_VOTES = 2

# -------- Logging proximity distributions (for methods justification) --------
LOG_PROX_DISTS = True
PROX_LOG_EVERY_SEC = 10
# ================= Speed knobs =================
# (A) classify less often
CLASSIFY_EVERY_HZ   = 6          # run SVM ~6 times/sec once gated
CLASSIFY_EVERY      = None       # filled per-video as int(fps / CLASSIFY_EVERY_HZ)

# (B) shrink feature buffer
BUF_SEC_MIN         = 5.0        # keep 5s (≥ 4s gate)

# (C) throttle drawing & CSV
DRAW_POSE           = False     # pose overlay is expensive and not needed for interaction logic
DRAW_EVERY          = 2          # draw pose every N frames (2=every other frame)
CSV_EVERY           = 10         # write keypoints every N frames
SAVE_KEYPOINTS_CSV  = False
DRAW_ONLY_ACTIVE    = False      # set True to draw only when a pair is active
ID_EVERY_N_FRAMES   = 4          # identity changes slowly; running this less often is a cheap speed win
POSE_USE_FP16       = True
PROFILE_RUNTIME     = True
PROFILE_EVERY_N_FRAMES = 120

# (D) hysteresis + suspect cap
GAP_TOL_SEC         = 0.5        # allow 0.5s misses without reset
MAX_SUSPECT_PAIRS   = 12         # keep only the closest N pairs per frame


# Helper fns for interaction logic
def _center_and_diag(tlwh):
    x, y, w, h = map(float, tlwh)
    cx, cy = x + w*0.5, y + h*0.5
    diag = math.hypot(w, h)
    return cx, cy, diag

def _normalized_center_distance(t1, t2):
    c1x, c1y, d1 = _center_and_diag(t1.tlwh)
    c2x, c2y, d2 = _center_and_diag(t2.tlwh)
    mean_diag = max(1e-6, 0.5 * (d1 + d2))
    dist = math.hypot(c1x - c2x, c1y - c2y)
    return float(dist / mean_diag), float(d1), float(d2)

def _is_candidate(t1, t2):
    """Candidate pair if bounding boxes overlap (intersection area > 0) AND diagonal-similarity passes."""
    x1, y1, w1, h1 = map(float, t1.tlwh)
    x2, y2, w2, h2 = map(float, t2.tlwh)

    a_x1, a_y1, a_x2, a_y2 = x1, y1, x1 + w1, y1 + h1
    b_x1, b_y1, b_x2, b_y2 = x2, y2, x2 + w2, y2 + h2

    inter_w = min(a_x2, b_x2) - max(a_x1, b_x1)
    inter_h = min(a_y2, b_y2) - max(a_y1, b_y1)
    overlap = (inter_w > 0.0) and (inter_h > 0.0)

    # Keep your original diagonal similarity ratio check
    _, _, d1 = _center_and_diag(t1.tlwh)
    _, _, d2 = _center_and_diag(t2.tlwh)
    sim = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    cond2 = sim >= INTERACT_DIAG_SIM_RATIO

    return overlap and cond2

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
from yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO
from mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    vis_pose_result
)
try:
    from mmpose.datasets import DatasetInfo
except Exception:
    DatasetInfo = None
# from mmpose.core.evaluation import get_similarity
# from mmpose.utils import get_config, collect_env
from mmpose.utils import collect_env

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

def _swap_pair_sequence(X_np: np.ndarray) -> np.ndarray:
    """Swap A/B slots in a (T,F) pair sequence and flip signed deltas."""
    if X_np is None:
        return None
    if X_np.ndim != 2 or X_np.shape[1] < 2:
        return np.array(X_np, copy=True)
    if (X_np.shape[1] - 2) % 8 != 0:
        return np.array(X_np, copy=True)

    out = np.empty_like(X_np)
    num_kpts = (X_np.shape[1] - 2) // 8
    for j in range(num_kpts):
        c = 8 * j
        out[:, c:c+6] = X_np[:, [c+3, c+4, c+5, c+0, c+1, c+2]]
        out[:, c+6] = -X_np[:, c+6]
        out[:, c+7] = -X_np[:, c+7]

    out[:, -2] = X_np[:, -1]
    out[:, -1] = X_np[:, -2]
    return out

def _resample_feature_sequence(X_np: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample a (T,F) sequence so temporal models see their training FPS."""
    if X_np is None:
        return None
    if X_np.ndim != 2 or X_np.shape[0] <= 1:
        return np.array(X_np, copy=True)
    if (not np.isfinite(src_fps)) or (not np.isfinite(dst_fps)) or src_fps <= 0 or dst_fps <= 0:
        return np.array(X_np, copy=True)
    if abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return np.array(X_np, copy=True)

    t_old = X_np.shape[0]
    t_new = max(2, int(round(t_old * float(dst_fps) / float(src_fps))))
    if t_new == t_old:
        return np.array(X_np, copy=True)

    old_grid = np.linspace(0.0, 1.0, t_old, dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, t_new, dtype=np.float32)
    out = np.empty((t_new, X_np.shape[1]), dtype=np.float32)
    for col in range(X_np.shape[1]):
        out[:, col] = np.interp(new_grid, old_grid, X_np[:, col]).astype(np.float32)
    return out

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(), 1e-9, None)


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

def _parse_det_device(raw):
    if isinstance(raw, int):
        return raw
    raw = str(raw).strip()
    return int(raw) if raw.isdigit() else raw

def _resolve_runtime_devices(det_raw, pose_raw):
    cuda_available = torch.cuda.is_available()

    det_value = str(det_raw).strip().lower()
    if det_value == "auto":
        det_device = 0 if cuda_available else "cpu"
    else:
        det_device = _parse_det_device(det_raw)

    pose_value = str(pose_raw).strip().lower()
    if pose_value == "auto":
        pose_device = "cuda:0" if cuda_available else "cpu"
    else:
        pose_device = pose_raw

    runtime_device = "cuda" if cuda_available and str(pose_device).startswith("cuda") else "cpu"
    return det_device, pose_device, runtime_device

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the full self-contained interaction inference pipeline."
    )
    parser.add_argument(
        "--input-dir",
        default=str(INPUTS_DIR),
        help="Directory containing input videos. Defaults to ./inputs",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Specific video file path. Repeat to process multiple videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR),
        help="Base output directory. Each run gets an auto-incremented subfolder.",
    )
    parser.add_argument(
        "--det-weights",
        default=str(MODELS_DIR / "Object_Detection_Trained_Model.pt"),
        help="Detector weights path.",
    )
    parser.add_argument(
        "--id-weights",
        default=str(MODELS_DIR / "Identification_Model_Trained.pt"),
        help="Identity model weights path.",
    )
    parser.add_argument(
        "--pose-config",
        default=str(DEFAULT_POSE_CONFIG),
        help="Pose config path.",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default=str(MODELS_DIR / "Keypoint_Model_Trained.pth"),
        help="Pose checkpoint path.",
    )
    parser.add_argument(
        "--svm-model",
        default=str(MODELS_DIR / "svm_model.joblib"),
        help="Optional SVM model path.",
    )
    parser.add_argument(
        "--stage1-ckpt",
        default=str(MODELS_DIR / "stage1_tcn_best.pt"),
        help="Stage-1 temporal checkpoint path.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        default=str(MODELS_DIR / "stage2_inception_best.pt"),
        help="Stage-2 temporal checkpoint path.",
    )
    parser.add_argument(
        "--det-device",
        default=str(DET_DEVICE),
        help="Ultralytics detector device, for example auto, 0, or cpu.",
    )
    parser.add_argument(
        "--pose-device",
        default=POSE_DEVICE,
        help="Pose model device, for example auto, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--annotated-output-scale",
        type=float,
        default=ANNOTATED_OUTPUT_SCALE,
        help="Scale factor for annotated MP4 output.",
    )
    parser.add_argument(
        "--disable-annotated-video",
        action="store_true",
        help="Skip annotated video writing to reduce runtime and output size.",
    )
    return parser

def configure_from_args(args):
    global VIDEO_DIR_OR_LIST
    global VIDEO_FILES
    global OUT_PARENT
    global DET_MODEL_WEIGHTS
    global ID_MODEL_WEIGHTS
    global POSE_CONFIG
    global POSE_CHECKPOINT
    global SVM_MODEL_PATH
    global STAGE1_CKPT_PATH
    global STAGE2_CKPT_PATH
    global DET_DEVICE
    global POSE_DEVICE
    global DEVICE
    global ANNOTATED_OUTPUT_SCALE
    global WRITE_ANNOTATED_MP4

    VIDEO_DIR_OR_LIST = args.input_dir
    VIDEO_FILES = args.video
    OUT_PARENT = args.output_dir
    DET_MODEL_WEIGHTS = args.det_weights
    ID_MODEL_WEIGHTS = args.id_weights
    POSE_CONFIG = args.pose_config
    POSE_CHECKPOINT = args.pose_checkpoint
    SVM_MODEL_PATH = args.svm_model
    STAGE1_CKPT_PATH = args.stage1_ckpt
    STAGE2_CKPT_PATH = args.stage2_ckpt
    DET_DEVICE, POSE_DEVICE, DEVICE = _resolve_runtime_devices(args.det_device, args.pose_device)
    ANNOTATED_OUTPUT_SCALE = args.annotated_output_scale
    WRITE_ANNOTATED_MP4 = not args.disable_annotated_video

# --------------------------------------------------------------------

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    configure_from_args(args)

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

    # Move Ultralytics models to CUDA only when the resolved detector device uses CUDA.
    if torch.cuda.is_available() and DET_DEVICE != "cpu":
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
    kp_file = None
    kp_writer = None
    if SAVE_KEYPOINTS_CSV:
        kp_file = open(kp_csv_path, "w", newline="", encoding="utf-8")
        kp_writer = csv.writer(kp_file)
    kp_header_written = False

    all_inter_rows = []
    all_no_inter_rows = []

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
            out_w = max(2, int(round(W * ANNOTATED_OUTPUT_SCALE)))
            out_h = max(2, int(round(H * ANNOTATED_OUTPUT_SCALE)))
            writer = make_writer(out_mp4, fps, out_w, out_h)

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

        # ===========================
        # Load Stage-1 + Stage-2 models
        # ===========================
        stage1_ckpt = None
        stage2_ckpt = None

        def _safe_load_ckpt(path_str: str):
            try:
                return torch.load(path_str, map_location=DEVICE)
            except Exception as e:
                print(f"[error] Failed to load checkpoint: {path_str}\\n  {e}")
                return None

        stage1_ckpt = _safe_load_ckpt(STAGE1_CKPT_PATH)
        stage2_ckpt = _safe_load_ckpt(STAGE2_CKPT_PATH)

        if stage1_ckpt is None or stage2_ckpt is None:
            print("[fatal] Missing Stage-1/Stage-2 checkpoints. Update STAGE1_CKPT_PATH/STAGE2_CKPT_PATH.")
            return

        # --- Verify keypoint feature dimensionality ---
        STAGE1_IN_FEATURES = int(stage1_ckpt.get("in_features", -1))
        STAGE2_IN_FEATURES = int(stage2_ckpt.get("in_features", -1))
        STAGE1_KPS = list(stage1_ckpt.get("keypoints", []))
        STAGE2_KPS = list(stage2_ckpt.get("keypoints", []))

        # Prefer stage2 keypoints ordering if both exist (should be same)
        REQUIRED_KPS = STAGE2_KPS or STAGE1_KPS
        if not REQUIRED_KPS:
            print("[fatal] Checkpoints do not contain 'keypoints' list.")
            return

        # Stage-1 threshold
        stage1_thresh = stage1_ckpt.get("threshold", None)

        # Allow override from constants
        if STAGE1_THRESHOLD_OVERRIDE is not None:
            stage1_thresh = float(STAGE1_THRESHOLD_OVERRIDE)

        if stage1_thresh is None:
            print("[fatal] Stage-1 checkpoint missing 'threshold'. Set STAGE1_THRESHOLD_OVERRIDE.")
            return

        # The training script saved threshold as either:
        #   - a float
        #   - a 0-d tensor
        #   - a dict like {"thr": <float>, ...}
        if isinstance(stage1_thresh, dict):
            stage1_thresh = stage1_thresh.get("thr", None)
        if isinstance(stage1_thresh, torch.Tensor):
            stage1_thresh = float(stage1_thresh.detach().cpu().item())

        stage1_thresh = float(stage1_thresh)
        print(f"[info] Stage-1 threshold: {stage1_thresh:.6f}")

        # Build name->index from pose model keypoint names
        # KPT_NAMES exists after pose model init in this script; we will set mapping later when available.
        name_to_kidx = None

        # ---- Stage-1 TCN model definition (matches training) ----
        class TemporalBlock(nn.Module):
            def __init__(self, in_ch, out_ch, k, dilation, dropout):
                super().__init__()
                padding = (k - 1) * dilation
                self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation))
                self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=dilation))
                self.dropout = nn.Dropout(dropout)
                self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x):
                out = self.conv1(x)
                out = out[..., :x.shape[-1]]  # causal trim
                out = F.relu(out)
                out = self.dropout(out)
                out = self.conv2(out)
                out = out[..., :x.shape[-1]]  # causal trim
                out = F.relu(out)
                out = self.dropout(out)
                res = x if self.downsample is None else self.downsample(x)
                return F.relu(out + res)

        class TemporalConvNet(nn.Module):
            def __init__(self, in_ch, channels, k=3, dropout=0.2):
                super().__init__()
                layers = []
                for i, out_ch in enumerate(channels):
                    dilation = 2 ** i
                    layers.append(TemporalBlock(in_ch, out_ch, k, dilation, dropout))
                    in_ch = out_ch
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        def masked_mean_pool_1d(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            # h: (B,C,T)
            B, C, T = h.shape
            mask = (torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
            mask = mask.unsqueeze(1)  # (B,1,T)
            h = h * mask
            denom = mask.sum(dim=2).clamp(min=1.0)
            return h.sum(dim=2) / denom  # (B,C)

        class TemporalBlock(nn.Module):
            # Matches the Stage-1 training script (weight_norm + causal trim).
            def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
                super().__init__()
                padding = (k - 1) * dilation
                self.conv1 = nn.utils.weight_norm(
                    nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation)
                )
                self.conv2 = nn.utils.weight_norm(
                    nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=dilation)
                )
                self.dropout = nn.Dropout(dropout)
                self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B,C,T)
                out = self.conv1(x)
                out = out[..., :x.shape[-1]]  # causal trim
                out = F.relu(out)
                out = self.dropout(out)

                out = self.conv2(out)
                out = out[..., :x.shape[-1]]  # causal trim
                out = F.relu(out)
                out = self.dropout(out)

                res = x if self.downsample is None else self.downsample(x)
                return F.relu(out + res)

        class Stage1TCN(nn.Module):
            # Binary Stage-1 TCN (Interaction vs No-Interaction) compatible with stage1_tcn_best.pt.
            def __init__(self, in_features: int, channels: int = 128, levels: int = 6, kernel_size: int = 3, dropout: float = 0.2):
                super().__init__()
                layers = []
                in_ch = in_features
                for i in range(levels):
                    dil = 2 ** i
                    layers.append(TemporalBlock(in_ch, channels, kernel_size, dil, dropout))
                    in_ch = channels
                self.tcn = nn.Sequential(*layers)
                self.head = nn.Linear(channels, 1)

            def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
                # x: (B,T,F)
                x = x.transpose(1, 2)  # (B,F,T)
                h = self.tcn(x)        # (B,C,T)
                pooled = masked_mean_pool_1d(h, lengths)
                return self.head(pooled)  # (B,1)

        # ---- Stage-2 InceptionTime model definition (matches training script) ----

        class InceptionModule(nn.Module):
            # InceptionTime module (matches Train_Stage2_InceptionTime_MultiInstance.py).
            def __init__(self, in_ch: int, nf: int, kss=(9, 19, 39), bottleneck: int = 32):
                super().__init__()
                self.bottleneck = None
                ch = in_ch
                if bottleneck and in_ch > 1:
                    self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
                    ch = bottleneck

                self.convs = nn.ModuleList([
                    nn.Conv1d(ch, nf, k, padding=k // 2, bias=False) for k in kss
                ])
                self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
                self.conv_pool = nn.Conv1d(in_ch, nf, 1, bias=False)

                self.bn = nn.BatchNorm1d(nf * 4)
                self.act = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B,C,T)
                x0 = self.bottleneck(x) if self.bottleneck is not None else x
                outs = [conv(x0) for conv in self.convs]
                outs.append(self.conv_pool(self.pool(x)))
                out = torch.cat(outs, dim=1)  # (B,4*nf,T)
                return self.act(self.bn(out))

        class InceptionBlock(nn.Module):
            # Stack of InceptionModules with residual connections every 3 blocks.
            def __init__(self, in_ch: int, nf: int = 128, depth: int = 6, kss=(9, 19, 39), bottleneck: int = 32):
                super().__init__()
                self.mods = nn.ModuleList()
                ch = in_ch
                for i in range(depth):
                    self.mods.append(InceptionModule(ch, nf=nf, kss=kss, bottleneck=bottleneck))
                    ch = nf * 4
                self.out_ch = ch

                # residual convs: one per residual connection (every 3 modules), all on 4*nf channels
                n_res = depth // 3
                self.res_convs = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(self.out_ch, self.out_ch, 1, bias=False),
                        nn.BatchNorm1d(self.out_ch),
                    )
                    for _ in range(n_res)
                ])
                self.act = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B,C,T)
                res = None
                res_i = 0
                for i, mod in enumerate(self.mods):
                    x = mod(x)
                    if i == 0:
                        res = x  # start residuals after first module (matches checkpoint shapes)
                    if (i % 3) == 2 and res is not None:
                        x = self.act(x + self.res_convs[res_i](res))
                        res = x
                        res_i += 1
                return x

        class Stage2InceptionTime(nn.Module):
            # 3-class Stage-2 InceptionTime compatible with stage2_inception_best.pt.
            def __init__(self, in_features: int, num_classes: int, nf: int = 128, depth: int = 6,
                         kss=(9, 19, 39), bottleneck: int = 32, dropout: float = 0.25):
                super().__init__()
                self.in_proj = nn.Sequential(
                    nn.Conv1d(in_features, nf, 1, bias=False),
                    nn.BatchNorm1d(nf),
                )
                self.block = InceptionBlock(in_ch=nf, nf=nf, depth=depth, kss=kss, bottleneck=bottleneck)
                self.head = nn.Sequential(
                    nn.LayerNorm(self.block.out_ch),
                    nn.Dropout(dropout),
                    nn.Linear(self.block.out_ch, num_classes)
                )

            def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
                # x: (B,T,F)
                x = x.transpose(1, 2)      # (B,F,T)
                h = self.in_proj(x)        # (B,nf,T)
                h = self.block(h)          # (B,4*nf,T)
                pooled = masked_mean_pool_1d(h, lengths)
                return self.head(pooled)   # (B,K)

        # Instantiate and load weights

        # ---- Stage-1 (binary TCN) ----
        stage1_in_features = int(stage1_ckpt.get("in_features", STAGE1_IN_FEATURES))
        stage1_model = Stage1TCN(
            in_features=stage1_in_features,
            channels=128,
            levels=6,
            kernel_size=3,
            dropout=0.2,
        ).to(DEVICE)

        stage1_model.load_state_dict(stage1_ckpt["state_dict"], strict=True)
        stage1_model.eval()
        stage1_train_fps = float(stage1_ckpt.get("fps", 0) or 0.0)

        # ---- Stage-2 (3-class InceptionTime) ----
        stage2_in_features = int(stage2_ckpt.get("in_features", STAGE2_IN_FEATURES))
        label_to_id = stage2_ckpt.get("label_to_id", {"Displacement": 0, "Headbutting": 1, "Licking": 2})
        stage2_num_classes = len(label_to_id)

        inc_cfg = stage2_ckpt.get("inception", {})
        stage2_model = Stage2InceptionTime(
            in_features=stage2_in_features,
            num_classes=stage2_num_classes,
            nf=int(inc_cfg.get("nf", 128)),
            depth=int(inc_cfg.get("depth", 6)),
            kss=tuple(inc_cfg.get("kss", (9, 19, 39))),
            bottleneck=int(inc_cfg.get("bottleneck", 32)),
            dropout=float(stage2_ckpt.get("dropout", 0.25)),
        ).to(DEVICE)

        stage2_model.load_state_dict(stage2_ckpt["state_dict"], strict=True)
        stage2_model.eval()
        stage2_train_fps = float(stage2_ckpt.get("fps", 0) or 0.0)

        stage2_id_to_label = stage2_ckpt.get("id_to_label", {i: str(i) for i in range(stage2_num_classes)})

        # Stage-2 will run only if Stage-1 stably predicts interaction.
# Stage-2 will run only if Stage-1 stably predicts interaction.
# ble positive.
        print(f"[info] Stage-1 loaded. threshold={stage1_thresh:.6f}")
        print(f"[info] Stage-2 loaded. classes={stage2_id_to_label}")
        if RESAMPLE_TO_MODEL_FPS and stage1_train_fps > 0 and abs(stage1_train_fps - fps) > 1e-6:
            print(f"[info] Stage-1 temporal resampling enabled: video_fps={fps:.2f} -> model_fps={stage1_train_fps:.2f}")
        if RESAMPLE_TO_MODEL_FPS and stage2_train_fps > 0 and abs(stage2_train_fps - fps) > 1e-6:
            print(f"[info] Stage-2 temporal resampling enabled: video_fps={fps:.2f} -> model_fps={stage2_train_fps:.2f}")
        if BIDIRECTIONAL_PAIR_INFERENCE:
            print("[info] Bidirectional pair-order inference enabled for Stage-1/Stage-2.")

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
        pair_prox_hist = defaultdict(lambda: {w: deque(maxlen=int(w*fps)) for w in Q_WINDOWS_SEC})


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
            # event-level state
            "active": False,
            "start_f": None,
            "cur_label": None,
            "last_pos_frame": -1,
            "conf_stage1_max": 0.0,
            "conf_stage2_max": 0.0,

            # stability + hysteresis
            "gate_hist": deque(maxlen=GATE_STABLE_M),      # 1 if dual-gate passes this frame
            "stage1_hist": deque(maxlen=STAGE1_STABLE_M),  # 1 if stage1 positive (any window) at this frame
            "stage2_votes": deque(maxlen=STAGE2_VOTE_N),   # stage2 class ids (for voting)

            # bookkeeping
            "last_pred_frame": -1
        })

        # No-interaction segments: candidate pairs (overlap + diag_ok) but Stage-1 stays negative.
        pair_noevt = defaultdict(lambda: {
            "active": False,
            "start_f": None,
            "last_frame": -1,
            "p1_max": 0.0,
        })

        def _reset_stage2_event(evt):
            evt["active"] = False
            evt["start_f"] = None
            evt["cur_label"] = None
            evt["conf_stage2_max"] = 0.0
            evt["stage2_votes"].clear()

        def _reset_pair_after_gap(evt):
            _reset_stage2_event(evt)
            evt["conf_stage1_max"] = 0.0
            evt["stage1_hist"].clear()
            evt["last_pos_frame"] = -1
            evt["last_pred_frame"] = -1

        inter_rows = []
        no_inter_rows = []  # candidate but classified as no-interaction segments
        prox_logs = []  # [video, frame, t_sec, p1,p5,p10,p15,p25,p50,p75,p90,p95,p99]

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

        prof_times = defaultdict(float)
        prof_counts = defaultdict(int)

        def _prof_add(name: str, dt: float):
            if not PROFILE_RUNTIME:
                return
            prof_times[name] += float(dt)
            prof_counts[name] += 1

        def _prof_report():
            if not PROFILE_RUNTIME:
                return
            order = ["total", "detect_track", "identity", "pose", "classify", "draw", "write"]
            parts = []
            for name in order:
                cnt = prof_counts.get(name, 0)
                if cnt <= 0:
                    continue
                ms = 1000.0 * prof_times[name] / cnt
                parts.append(f"{name}={ms:.1f}ms")
            if parts:
                print(f"[perf] frame={frame_idx}  " + "  ".join(parts))

        frame_idx = 0
        while True:
            t_frame0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # ----- DETECT + TRACK (ByteTrack), update track_rows, etc. -----
            # ------------------------ Detection ------------------------
            t0 = time.perf_counter()
            det_res = det_model.predict(source=frame, imgsz=DET_IMGSZ,
                                        conf=DET_CONF_THRESH, iou=DET_IOU_NMS,
                                        device=DET_DEVICE, verbose=False)
            dets = detections_from_ultralytics(det_res, DET_CLASSES_TO_KEEP)  # Nx5 [x1,y1,x2,y2,score]

            # ------------------------- Tracking ------------------------
            online_targets = tracker.update(dets, (H, W), (H, W))
            _prof_add("detect_track", time.perf_counter() - t0)

            # ----------------------- Identification (20-frame window) --- (BATCHED)
            # Optional speed knob: run ID less often because identities change slowly.
            t0 = time.perf_counter()
            if (frame_idx % ID_EVERY_N_FRAMES) == 0:

                crops = []
                crop_tids = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = int(t.track_id)

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

                    x1, y1, x2, y2 = tlwh_to_xyxy(tlwh)
                    crop, _ = crop_with_pad(frame, (x1, y1, x2, y2), pad_frac=CROP_PAD_FRAC)
                    if crop.size == 0:
                        continue

                    crops.append(crop)
                    crop_tids.append(tid)

                # Run ONE batched predict for all crops
                if len(crops) > 0:
                    id_preds = id_model.predict(
                        source=crops,
                        imgsz=224,
                        device=DET_DEVICE,
                        verbose=False
                    )

                    # Update votes for each crop result
                    for pred, tid in zip(id_preds, crop_tids):
                        st = track_state.get(tid)
                        if st is None:
                            continue

                        if pred.probs is not None and hasattr(pred.probs, "top1"):
                            top_idx = int(pred.probs.top1)
                            lab = pred.names[top_idx] if hasattr(pred, "names") else str(top_idx)
                            conf = float(pred.probs.top1conf)

                            if lab in ID_CLASS_SET:
                                st["win_votes"][lab] += 1
                                st["win_conf"][lab] += conf

            # window counter should still advance 1 per frame
            frames_in_window += 1

            # decide identities when window full
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
                        v = st["win_votes"][lab]
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
            _prof_add("identity", time.perf_counter() - t0)

            # ----------------------- Pose inference (SELECTIVE) + Interaction --------------------
            # 1) Build geometry-based candidate pairs (Step D: dual gating + multi-scale stability)
            tracks = list(online_targets)
            all_pairs = []
            need_pose_tids = set()

            # Optional per-video logging bucket
            if LOG_PROX_DISTS and (frame_idx % max(1, int(PROX_LOG_EVERY_SEC * fps)) == 0):
                prox_log_bucket = []

            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    a, b = tracks[i], tracks[j]

                    # Candidate if boxes overlap (intersection area > 0) and diagonal similarity passes.
                    x1, y1, w1, h1 = map(float, a.tlwh)
                    x2, y2, w2, h2 = map(float, b.tlwh)
                    a_x1, a_y1, a_x2, a_y2 = x1, y1, x1 + w1, y1 + h1
                    b_x1, b_y1, b_x2, b_y2 = x2, y2, x2 + w2, y2 + h2

                    inter_w = min(a_x2, b_x2) - max(a_x1, b_x1)
                    inter_h = min(a_y2, b_y2) - max(a_y1, b_y1)
                    overlap = (inter_w > 0.0) and (inter_h > 0.0)

                    # Diagonal similarity (keep as-is)
                    center_dist_norm, d1, d2 = _normalized_center_distance(a, b)
                    sim = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
                    diag_ok = sim >= INTERACT_DIAG_SIM_RATIO

                    # IoU (used only for ranking/compute budget; not a gating threshold)
                    if overlap:
                        inter_area = inter_w * inter_h
                        area_a = max(0.0, (a_x2 - a_x1)) * max(0.0, (a_y2 - a_y1))
                        area_b = max(0.0, (b_x2 - b_x1)) * max(0.0, (b_y2 - b_y1))
                        union = area_a + area_b - inter_area + 1e-6
                        iou = inter_area / union
                    else:
                        iou = 0.0

                    key = tuple(sorted((int(a.track_id), int(b.track_id))))
                    g = pair_geom[key]
                    prox_hist = pair_prox_hist[key]
                    e = pair_evt[key]

                    hard_prox_ok = center_dist_norm <= PROX_HARD_CAP
                    quantile_ok = False

                    if overlap and diag_ok:
                        for w in Q_WINDOWS_SEC:
                            hist = prox_hist[w]
                            hist.append(center_dist_norm)
                            if len(hist) >= max(3, int(0.5 * w * fps)):
                                qv = float(np.quantile(np.asarray(hist, dtype=np.float32), Q_PROX))
                                if qv <= PROX_HARD_CAP:
                                    quantile_ok = True
                        if LOG_PROX_DISTS and (frame_idx % max(1, int(PROX_LOG_EVERY_SEC * fps)) == 0):
                            prox_log_bucket.append(center_dist_norm)
                    else:
                        for hist in prox_hist.values():
                            hist.clear()

                    gate_pass = bool(overlap and diag_ok and (hard_prox_ok or quantile_ok))
                    e["gate_hist"].append(1 if gate_pass else 0)

                    # Step D (stability k-of-m gate)
                    stable_gate = (sum(e["gate_hist"]) >= GATE_STABLE_K)

                    if stable_gate:
                        # candidate survives gating
                        g["suspect_frames"] += 1
                        g["miss"] = 0
                        g["last_ok_frame"] = frame_idx
                        g["prox"] = float(center_dist_norm)
                        all_pairs.append((float(center_dist_norm), key, int(a.track_id), int(b.track_id)))

            # If enabled, log per-video proximity distribution snapshots
            if LOG_PROX_DISTS and (frame_idx % max(1, int(PROX_LOG_EVERY_SEC * fps)) == 0) and len(prox_log_bucket) > 0:
                try:
                    pcts = [1, 5, 10, 15, 25, 50, 75, 90, 95, 99]
                    vals = np.percentile(np.asarray(prox_log_bucket, dtype=np.float32), pcts).tolist()
                    prox_logs.append([vpath.name, frame_idx, frame_idx / fps] + vals)
                except Exception:
                    pass

            # Keep only the closest N pairs per frame (compute budget control)
            all_pairs.sort(key=lambda x: x[0])
            keep = set([p[1] for p in all_pairs[:MAX_SUSPECT_PAIRS]])

            suspect_pairs = keep
            for (_, key, a_id, b_id) in all_pairs[:MAX_SUSPECT_PAIRS]:
                need_pose_tids.add(a_id)
                need_pose_tids.add(b_id)

            # Decay / miss accounting for pairs not kept this frame
            for k, st in list(pair_geom.items()):
                if k not in suspect_pairs:
                    st["miss"] += 1
                    if st["miss"] > gap_tol_fr:
                        st["suspect_frames"] = 0
                        st["prox"] = None
                        pair_prox_hist.pop(k, None)

            # Remove stale pair buffers to keep memory bounded
            for k in list(pair_buf.keys()):
                st = pair_geom.get(k, None)
                if (k not in suspect_pairs) and (st is None or st.get("miss", 0) > gap_tol_fr):
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
            t0 = time.perf_counter()
            if person_results:
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(
                        enabled=(POSE_USE_FP16 and torch.cuda.is_available() and str(POSE_DEVICE).startswith("cuda")),
                        dtype=torch.float16
                    ):
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
            _prof_add("pose", time.perf_counter() - t0)

            # 3) Draw Pose (if any)
            if DRAW_POSE and WRITE_ANNOTATED_MP4 and pose_results and (frame_idx % DRAW_EVERY == 0):
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
            if SAVE_KEYPOINTS_CSV and pose_results and (frame_idx % CSV_EVERY == 0):
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
                if evt["start_f"] is None:
                    return
                dur_frames = int(end_f - evt["start_f"] + 1)
                if dur_frames < min_frames:
                    return

                # keep your original "weight per time-window crossed" idea
                span_frames = dur_frames
                windows_crossed = int(math.ceil((span_frames / fps) / TIME_WINDOW_SEC))
                weight = max(1, windows_crossed)
                duration_s = dur_frames / fps

                # For compatibility, we keep the 9th column name "margin_avg" but store Stage-2 confidence there.
                stage2_conf = float(evt.get("conf_stage2_max", 0.0))
                stage1_prob = float(evt.get("conf_stage1_max", 0.0))

                inter_rows.append([
                    vpath.name,
                    evt["cur_label"],
                    a_id, b_id,
                    int(evt["start_f"]),
                    int(end_f),
                    float(duration_s),
                    int(weight),
                    stage2_conf,
                    stage1_prob,
                    float(stage1_thresh),
                ])

            # ===============================
            # NEW: Two-stage inference (Step E)
            # ===============================

            def _ensure_kpt_index():
                nonlocal name_to_kidx
                if name_to_kidx is None:
                    name_to_kidx = {str(n): i for i, n in enumerate(KPT_NAMES)}
                    # sanity
                    missing = [k for k in REQUIRED_KPS if k not in name_to_kidx]
                    if missing:
                        print(f"[fatal] Missing keypoints in pose output: {missing[:8]} (total {len(missing)})")
                        raise RuntimeError("Keypoint name mismatch between pose and classifier checkpoints.")

            def _seq_from_pairbuf(stbuf, kps_order):
                # Builds (T,F) features matching training:
                # For each kp: Ax,Ay,Aconf,Bx,By,Bconf,dx,dy  + log(areaA),log(areaB)
                kA_list = stbuf["kptsA"]
                kB_list = stbuf["kptsB"]
                sA_list = stbuf["scaleA"]
                sB_list = stbuf["scaleB"]
                T = len(kA_list)
                if T == 0:
                    return None

                Fdim = 8 * len(kps_order) + 2
                X = np.zeros((T, Fdim), dtype=np.float32)
                eps = 1e-6

                for t in range(T):
                    kA = kA_list[t]  # (K,3) or None
                    kB = kB_list[t]
                    sA = float(sA_list[t]) if sA_list[t] is not None else float("nan")
                    sB = float(sB_list[t]) if sB_list[t] is not None else float("nan")
                    if not np.isfinite(sA) or sA <= 0:
                        sA = 1.0
                    if not np.isfinite(sB) or sB <= 0:
                        sB = 1.0

                    # approximate area for the log terms
                    a_area = sA * sA
                    b_area = sB * sB
                    mean_scale = 0.5 * (sA + sB)

                    col = 0
                    for kp in kps_order:
                        ki = name_to_kidx[kp]
                        if kA is None or kB is None or ki >= len(kA) or ki >= len(kB):
                            ax = ay = ac = bx = by = bc = 0.0
                        else:
                            ax, ay, ac = float(kA[ki, 0]), float(kA[ki, 1]), float(kA[ki, 2])
                            bx, by, bc = float(kB[ki, 0]), float(kB[ki, 1]), float(kB[ki, 2])

                        axn = ax / sA
                        ayn = ay / sA
                        bxn = bx / sB
                        byn = by / sB

                        # relative deltas (normalized)
                        denom = max(eps, mean_scale / sA)
                        dx = (axn - bxn) / denom
                        dy = (ayn - byn) / denom

                        X[t, col:col+8] = [axn, ayn, ac, bxn, byn, bc, dx, dy]
                        col += 8

                    X[t, col] = np.log(a_area + 1.0) / 10.0
                    X[t, col+1] = np.log(b_area + 1.0) / 10.0

                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return X

            def _prepare_seq_for_model(X_np: np.ndarray, target_fps: float) -> np.ndarray:
                if X_np is None:
                    return None
                if not RESAMPLE_TO_MODEL_FPS or target_fps <= 0:
                    return X_np
                return _resample_feature_sequence(X_np, fps, target_fps)

            def _decode_stage2_logits(logits: np.ndarray):
                probs = _softmax_np(logits)
                cid = int(np.argmax(probs))
                conf = float(probs[cid])
                return cid, conf, probs

            # ---------------- BATCH HELPERS FOR STAGE1/STAGE2 ----------------

            def _pad_batch_np(list_X):
                """
                list_X: list of (T,F) numpy arrays (variable T)
                returns: torch tensor (B,Tmax,F), lengths (B,)
                """
                lengths = np.array([x.shape[0] for x in list_X], dtype=np.int64)
                Tmax = int(lengths.max()) if len(lengths) else 0
                Fdim = int(list_X[0].shape[1]) if len(list_X) else 0

                Xpad = np.zeros((len(list_X), Tmax, Fdim), dtype=np.float32)
                for i, x in enumerate(list_X):
                    t = x.shape[0]
                    Xpad[i, :t, :] = x

                x_t = torch.from_numpy(Xpad).to(DEVICE).float()
                lengths_t = torch.from_numpy(lengths).to(DEVICE).long()
                return x_t, lengths_t

            @torch.no_grad()
            def _stage1_batch_probs(list_X):
                """
                list_X: list of (T,F) numpy arrays
                returns probs: numpy array (B,) in [0,1]
                """
                x_t, lengths_t = _pad_batch_np(list_X)

                # AMP speeds up A100/H100 significantly
                with torch.cuda.amp.autocast(
                    enabled=(torch.cuda.is_available() and str(DEVICE).startswith("cuda")),
                    dtype=torch.float16
                ):
                    logits = stage1_model(x_t, lengths_t).squeeze(-1)  # (B,)
                    probs = torch.sigmoid(logits)

                return probs.detach().float().cpu().numpy()

            @torch.no_grad()
            def _stage2_batch_logits(list_X):
                """
                list_X: list of (T,F) numpy arrays
                returns logits: numpy array (B, C)
                """
                x_t, lengths_t = _pad_batch_np(list_X)

                with torch.cuda.amp.autocast(
                    enabled=(torch.cuda.is_available() and str(DEVICE).startswith("cuda")),
                    dtype=torch.float16
                ):
                    logits = stage2_model(x_t, lengths_t)  # (B,C)

                return logits.detach().float().cpu().numpy()

            @torch.no_grad()
            def _stage1_prob(X_np: np.ndarray) -> float:
                # returns probability in [0,1]
                x = torch.from_numpy(X_np[None, ...]).to(DEVICE).float()
                lengths = torch.tensor([X_np.shape[0]], device=DEVICE, dtype=torch.long)
                logit = stage1_model(x, lengths).squeeze(0).squeeze(-1)
                return float(torch.sigmoid(logit).item())

            @torch.no_grad()
            def _stage2_logits(X_np: np.ndarray) -> np.ndarray:
                x = torch.from_numpy(X_np[None, ...]).to(DEVICE).float()
                lengths = torch.tensor([X_np.shape[0]], device=DEVICE, dtype=torch.long)
                logits = stage2_model(x, lengths).squeeze(0).detach().cpu().numpy()
                return logits

            def _multiinstance_stage1(stbuf) -> Tuple[float, bool]:
                # Evaluate stage1 on multiple crops/windows and take max prob (high recall)
                Xfull = _seq_from_pairbuf(stbuf, REQUIRED_KPS)
                if Xfull is None:
                    return 0.0, False

                probs = []
                for w in MI_CROPS_SEC:
                    if w == "full":
                        X = Xfull
                    else:
                        n = int(w * fps)
                        X = Xfull[-n:] if Xfull.shape[0] > n else Xfull
                    probs.append(_stage1_prob(_prepare_seq_for_model(X, stage1_train_fps)))
                    if BIDIRECTIONAL_PAIR_INFERENCE:
                        probs.append(_stage1_prob(_prepare_seq_for_model(_swap_pair_sequence(X), stage1_train_fps)))
                p = float(np.max(probs)) if probs else 0.0
                return p, (p >= stage1_thresh)

            def _multiinstance_stage2(stbuf) -> Tuple[int, float, np.ndarray]:
                # Average logits across crops/windows
                Xfull = _seq_from_pairbuf(stbuf, REQUIRED_KPS)
                if Xfull is None:
                    return 0, 0.0, np.zeros((stage2_num_classes,), dtype=np.float32)

                logits_list = []
                for w in MI_CROPS_SEC:
                    if w == "full":
                        X = Xfull
                    else:
                        n = int(w * fps)
                        X = Xfull[-n:] if Xfull.shape[0] > n else Xfull
                    logits_ab = _stage2_logits(_prepare_seq_for_model(X, stage2_train_fps))
                    if BIDIRECTIONAL_PAIR_INFERENCE:
                        logits_ba = _stage2_logits(_prepare_seq_for_model(_swap_pair_sequence(X), stage2_train_fps))
                        _, conf_ab, _ = _decode_stage2_logits(logits_ab)
                        _, conf_ba, _ = _decode_stage2_logits(logits_ba)
                        logits_list.append(logits_ab if conf_ab >= conf_ba else logits_ba)
                    else:
                        logits_list.append(logits_ab)
                avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
                cls_id, conf, probs = _decode_stage2_logits(avg_logits)
                return cls_id, conf, avg_logits

            def _vote_mode(votes: deque) -> int:
                if not votes:
                    return 0
                c = Counter(list(votes))
                return int(c.most_common(1)[0][0])

            # For each gated pair, run Stage-1 (multi-instance) at schedule and maintain event-level detection
            _ensure_kpt_index()

            # ------------------- BATCHED Stage-1/Stage-2 inference across suspect pairs -------------------
            # We only run on scheduled frames
            t_classify0 = time.perf_counter()
            if (frame_idx % CLASSIFY_EVERY) == 0:

                eligible_keys = []
                Xfull_map = {}

                # Filter eligible pairs first (cooldown + buffer length)
                for key in suspect_pairs:
                    evt = pair_evt[key]
                    stbuf = pair_buf[key]

                    if key in pair_cooldown_until and frame_idx < pair_cooldown_until[key]:
                        continue

                    if len(stbuf["frames"]) < int(min(Q_WINDOWS_SEC) * fps):
                        continue

                    Xfull = _seq_from_pairbuf(stbuf, REQUIRED_KPS)
                    if Xfull is None or Xfull.shape[0] < 2:
                        continue

                    eligible_keys.append(key)
                    Xfull_map[key] = Xfull

                # ---------------- Stage-1 batched multi-instance ----------------
                if len(eligible_keys) > 0:

                    # For each crop window, run stage1 batch once
                    p1_max = {k: 0.0 for k in eligible_keys}

                    for w in MI_CROPS_SEC:
                        list_X = []
                        keys_w = []

                        for k in eligible_keys:
                            Xfull = Xfull_map[k]
                            if w == "full":
                                X = Xfull
                            else:
                                n = int(w * fps)
                                X = Xfull[-n:] if Xfull.shape[0] > n else Xfull

                            list_X.append(_prepare_seq_for_model(X, stage1_train_fps))
                            keys_w.append(k)
                            if BIDIRECTIONAL_PAIR_INFERENCE:
                                list_X.append(_prepare_seq_for_model(_swap_pair_sequence(X), stage1_train_fps))
                                keys_w.append(k)

                        probs = _stage1_batch_probs(list_X)  # (B,)
                        for k, p in zip(keys_w, probs):
                            if p > p1_max[k]:
                                p1_max[k] = float(p)

                    # Update histories + decide stable stage-1
                    stable_stage1_keys = []
                    for k in eligible_keys:
                        evt = pair_evt[k]
                        stbuf = pair_buf[k]

                        p1 = p1_max[k]
                        hit1 = (p1 >= stage1_thresh)

                        evt["conf_stage1_max"] = max(evt.get("conf_stage1_max", 0.0), p1)
                        evt["stage1_hist"].append(1 if hit1 else 0)

                        stable_stage1 = (sum(evt["stage1_hist"]) >= STAGE1_STABLE_K)

                        # ----- No-interaction logging (candidate but Stage-1 negative) -----
                        noevt = pair_noevt[k]
                        if not stable_stage1:
                            if not noevt["active"]:
                                noevt["active"] = True
                                noevt["start_f"] = frame_idx
                                noevt["p1_max"] = float(p1)
                            else:
                                noevt["p1_max"] = max(float(noevt.get("p1_max", 0.0)), float(p1))
                            noevt["last_frame"] = frame_idx
                        else:
                            # If interaction starts, close any running no-interaction segment
                            if noevt.get("active", False) and (noevt.get("start_f") is not None):
                                end_f0 = frame_idx - 1
                                dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
                                if dur0 >= min_frames:
                                    no_inter_rows.append([
                                        vpath.name,
                                        int(k[0]), int(k[1]),
                                        int(noevt["start_f"]),
                                        int(end_f0),
                                        float(dur0 / fps),
                                        float(noevt.get("p1_max", 0.0)),
                                    ])
                            noevt["active"] = False
                            noevt["start_f"] = None
                            noevt["last_frame"] = -1
                            noevt["p1_max"] = 0.0

                        if stable_stage1:
                            evt["last_pos_frame"] = frame_idx
                            evt["last_pred_frame"] = frame_idx
                            stable_stage1_keys.append(k)

                    # ---------------- Stage-2 batched multi-instance (ONLY stable Stage-1) ----------------
                    if len(stable_stage1_keys) > 0:

                        # accumulate logits across windows
                        logits_sum = {k: None for k in stable_stage1_keys}

                        for w in MI_CROPS_SEC:
                            list_X = []
                            keys_w = []

                            for k in stable_stage1_keys:
                                Xfull = Xfull_map[k]
                                if w == "full":
                                    X = Xfull
                                else:
                                    n = int(w * fps)
                                    X = Xfull[-n:] if Xfull.shape[0] > n else Xfull

                                list_X.append(_prepare_seq_for_model(X, stage2_train_fps))
                                keys_w.append((k, False))
                                if BIDIRECTIONAL_PAIR_INFERENCE:
                                    list_X.append(_prepare_seq_for_model(_swap_pair_sequence(X), stage2_train_fps))
                                    keys_w.append((k, True))

                            logits_batch = _stage2_batch_logits(list_X)  # (B,C)

                            best_logits = {}
                            best_conf = {}
                            for (k, _), logits in zip(keys_w, logits_batch):
                                _, conf, _ = _decode_stage2_logits(logits)
                                if (k not in best_conf) or (conf > best_conf[k]):
                                    best_conf[k] = conf
                                    best_logits[k] = logits.astype(np.float32)

                            for k, logits in best_logits.items():
                                if logits_sum[k] is None:
                                    logits_sum[k] = logits
                                else:
                                    logits_sum[k] += logits

                        # finalize stage-2 decision for each stable pair
                        num_windows = float(len(MI_CROPS_SEC))

                        for k in stable_stage1_keys:
                            evt = pair_evt[k]
                            p1_now = float(p1_max.get(k, 0.0))

                            avg_logits = logits_sum[k] / max(1.0, num_windows)
                            cid, cconf, _ = _decode_stage2_logits(avg_logits)
                            evt["last_pred_frame"] = frame_idx

                            if cconf < STAGE2_MIN_CONF:
                                if not evt["active"] and evt["stage2_votes"]:
                                    evt["stage2_votes"].clear()
                                continue

                            evt["stage2_votes"].append(cid)

                            voted_cid = _vote_mode(evt["stage2_votes"])
                            voted_label = stage2_id_to_label.get(voted_cid, str(voted_cid))

                            if (not evt["active"]) and (len(evt["stage2_votes"]) < STAGE2_MIN_START_VOTES):
                                evt["conf_stage2_max"] = max(evt.get("conf_stage2_max", 0.0), cconf)
                                continue

                            # If the class changes while geometry/Stage-1 stay positive, end the
                            # previous segment and immediately start the new label without cooldown.
                            if evt["active"] and evt["cur_label"] != voted_label:
                                end_f = frame_idx - 1
                                _finalize_and_write(evt, k, end_f)
                                _reset_stage2_event(evt)
                                evt["conf_stage1_max"] = p1_now
                                evt["conf_stage2_max"] = float(cconf)
                                evt["stage2_votes"].append(cid)
                                voted_cid = cid
                                voted_label = stage2_id_to_label.get(voted_cid, str(voted_cid))

                            if not evt["active"]:
                                evt["active"] = True
                                evt["cur_label"] = voted_label
                                evt["start_f"] = frame_idx
                            evt["conf_stage2_max"] = max(evt.get("conf_stage2_max", 0.0), cconf)
                            evt["last_pos_frame"] = frame_idx
                            evt["last_pred_frame"] = frame_idx

                    # ---------------- close stale active events for pairs that were NOT stable Stage-1 ----------------
                    # (same behavior as your original "else close after gap tolerance")
                    for k in eligible_keys:
                        evt = pair_evt[k]
                        if evt["active"] and (evt["last_pos_frame"] >= 0) and (
                                frame_idx - evt["last_pos_frame"] > gap_tol_fr):
                            end_f = int(evt["last_pos_frame"])
                            _finalize_and_write(evt, k, end_f)
                            _reset_pair_after_gap(evt)
                            pair_cooldown_until[k] = end_f + cooldown_frames

            else:
                # If not a classify frame, still close stale events if needed (same as your original behavior)
                for key in suspect_pairs:
                    evt = pair_evt[key]
                    if evt["active"] and (evt["last_pos_frame"] >= 0) and (
                            frame_idx - evt["last_pos_frame"] > gap_tol_fr):
                        end_f = int(evt["last_pos_frame"])
                        _finalize_and_write(evt, key, end_f)
                        _reset_pair_after_gap(evt)
                        pair_cooldown_until[key] = end_f + cooldown_frames
            _prof_add("classify", time.perf_counter() - t_classify0)

            
            # Close stale no-interaction segments for pairs that are no longer candidates
            for k, noevt in list(pair_noevt.items()):
                if not noevt.get("active", False):
                    continue
                if (k not in suspect_pairs) and (noevt.get("last_frame", -1) >= 0) and (frame_idx - noevt["last_frame"] > gap_tol_fr):
                    end_f0 = int(noevt["last_frame"])
                    dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
                    if dur0 >= min_frames:
                        no_inter_rows.append([
                            vpath.name,
                            int(k[0]), int(k[1]),
                            int(noevt["start_f"]),
                            int(end_f0),
                            float(dur0 / fps),
                            float(noevt.get("p1_max", 0.0)),
                        ])
                    noevt["active"] = False
                    noevt["start_f"] = None
                    noevt["last_frame"] = -1
                    noevt["p1_max"] = 0.0

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
            t0 = time.perf_counter()
            for t in online_targets:
                tlwh = t.tlwh
                tid  = int(t.track_id)
                xi, yi, wi, hi = map(int, tlwh)
                x2i, y2i = xi + wi, yi + hi
                st = track_state.get(tid, {"label": UNKNOWN_LABEL, "conf": 0.0})
                if WRITE_ANNOTATED_MP4:
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
            _prof_add("draw", time.perf_counter() - t0)

            t0 = time.perf_counter()
            if WRITE_ANNOTATED_MP4 and writer:
                frame_out = frame
                if abs(ANNOTATED_OUTPUT_SCALE - 1.0) > 1e-6:
                    interp = cv2.INTER_AREA if ANNOTATED_OUTPUT_SCALE < 1.0 else cv2.INTER_LINEAR
                    frame_out = cv2.resize(frame, None, fx=ANNOTATED_OUTPUT_SCALE, fy=ANNOTATED_OUTPUT_SCALE, interpolation=interp)
                writer.write(frame_out)
            _prof_add("write", time.perf_counter() - t0)
            _prof_add("total", time.perf_counter() - t_frame0)
            if PROFILE_RUNTIME and frame_idx > 0 and (frame_idx % PROFILE_EVERY_N_FRAMES) == 0:
                _prof_report()

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        # finalize any open events for this video
        for key, evt in list(pair_evt.items()):
            if evt["active"]:
                _finalize_and_write(evt, key, frame_idx-1 if frame_idx > 0 else 0)
                _reset_pair_after_gap(evt)

        # finalize any open no-interaction segments for this video
        for k, noevt in list(pair_noevt.items()):
            if noevt.get("active", False) and (noevt.get("start_f") is not None):
                end_f0 = int(noevt.get("last_frame", frame_idx-1 if frame_idx > 0 else 0))
                dur0 = int(end_f0 - int(noevt["start_f"]) + 1)
                if dur0 >= min_frames:
                    no_inter_rows.append([
                        vpath.name,
                        int(k[0]), int(k[1]),
                        int(noevt["start_f"]),
                        int(end_f0),
                        float(dur0 / fps),
                        float(noevt.get("p1_max", 0.0)),
                    ])
            noevt["active"] = False
            noevt["start_f"] = None
            noevt["last_frame"] = -1
            noevt["p1_max"] = 0.0

        # append to global list
        try:
            all_inter_rows.extend(inter_rows)
            all_no_inter_rows.extend(no_inter_rows)
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
                "video","class","tidA","tidB","start_frame","end_frame","duration_s","weight",
                "stage2_conf","stage1_prob","stage1_thresh"
            ])
            inter_df.to_csv(OUT_DIR / "interactions.csv", index=False)
            if LOG_PROX_DISTS and len(prox_logs) > 0:
                prox_df = pd.DataFrame(prox_logs, columns=[
                    "video","frame","t_sec","p1","p5","p10","p15","p25","p50","p75","p90","p95","p99"
                ])
                prox_df.to_csv(OUT_DIR / "proximity_percentiles.csv", index=False)

            # optional: adjacency per class
            for cls in sorted([c for c in inter_df["class"].unique() if c != "neutral"]):
                sub = inter_df[inter_df["class"] == cls]
                adj = (sub.groupby(["tidA","tidB"])["weight"].sum()
                        .reset_index()
                        .sort_values(["weight"], ascending=False))
                adj.to_csv(OUT_DIR / f"adjacency_{cls}.csv", index=False)
    except Exception as e:
        print(f"[warn] could not write interactions csv: {e}")


    # Save no-interaction segments CSV (candidate overlaps that remained no-interaction)
    try:
        if len(all_no_inter_rows) > 0:
            no_df = pd.DataFrame(all_no_inter_rows, columns=[
                "video","tidA","tidB","start_frame","end_frame","duration_s","stage1_prob_max"
            ])
            no_df.to_csv(OUT_DIR / "no_interactions.csv", index=False)
        else:
            print("[info] No-interaction segments: none.")
    except Exception as e:
        print(f"[warn] could not write no_interactions csv: {e}")

    if kp_file is not None:
        kp_file.close()
    print(f"\n✅ Done.\n  Tracking CSV: {track_csv_path}\n  Keypoints CSV: {kp_csv_path}\n  Videos/plots:  {OUT_DIR}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
