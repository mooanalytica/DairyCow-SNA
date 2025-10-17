# MATCH THRESH, TRACK THRESH, TRACK BUFFER VALUES ARE VARIED IN RANGE TO FIND THE OPTIMAL BYTETRACK PARAMETERS

import os
import json
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
import cv2
from tqdm import tqdm
import argparse
import pandas as pd
from collections import Counter, defaultdict
from typing import Optional, Tuple, Literal
import optuna


# === CONFIG ===
detections_json = r"bytetrack_detections.json"
image_dir = r"Rotated_Frames_Undistorted/"
output_dir = r"Tracking_Results_1/"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DETECTIONS ===
with open(detections_json, 'r') as f:
    raw_detections = json.load(f)
detections = {int(k): np.array(v) for k, v in raw_detections.items()}
total_frames = sum(len(det_array) for det_array in detections.values())

# === INFER IMAGE SIZE ===
sample_img_path = os.path.join(image_dir, sorted(os.listdir(image_dir))[0])
sample_img = cv2.imread(sample_img_path)
frame_height, frame_width = sample_img.shape[:2]

# === INITIALIZE ARRAY TO STORE RESULTS ===
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
accuracyResults = []

# === INITIALIZE MAX MATCH THRESHOLD TO ENTER ===
MAX_MATCH = 400

# === MINIMUM NO. OF FRAMES BELOW WHICH A TRACK IS TO BE DROPPED
MINIMUM_TRACK_FRAMES = 1


# FUNCTION TO FIND ACCURACY RESULT VALUES
def findAccuracyResult(detections, track_thresh=0.5, track_buffer=60, match_thresh=0.8):
    # === INITIALIZE TRACKER ===
    args = argparse.Namespace()
    args.track_thresh = track_thresh
    args.track_buffer = track_buffer
    # args.low_thresh = 0.2
    args.match_thresh = match_thresh
    args.frame_rate = 1 / 12
    args.mot20 = False
    tracker = BYTETracker(args)

    # === RUN TRACKING ===
    results = []
    for frame_id, img_name in enumerate(image_files):
        # img_path = os.path.join(image_dir, img_name)
        # frame = cv2.imread(img_path)

        detections_in_frame = detections.get(frame_id, np.empty((0, 5)))
        img_info = {"height": frame_height, "width": frame_width}
        # print(img_info)
        # online_targets = tracker.update(detections_in_frame, img_info, frame_id)
        online_targets = tracker.update(detections_in_frame, (frame_height, frame_width), (frame_height, frame_width))

        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            results.append([frame_id, tid, *tlwh, t.score])

        # # Optional: Visualize
        # vis = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=30)
        # cv2.imwrite(os.path.join(output_dir, f"tracked_{img_name}"), vis)

    # === SAVE CSV RESULTS ===
    result_df = pd.DataFrame(results, columns=["frame", "id", "x", "y", "w", "h", "score"])
    result_df.to_csv(os.path.join(output_dir, "cow_tracking_results.csv"), index=False)
    # print("\n✅ Tracking complete. Results saved.")

    # === STORE ACCURACY RESULTS ===
    # INPUT PATHS
    DETECTIONS_JSON = r"D:/Thesis/Project/Object_Tracking/bytetrack_detections.json"
    NAMES_JSON = r"D:/Thesis/Project/Object_Tracking/bytetrack_ids.json"
    TRACKING_CSV = r"D:/Thesis/Project/Object_Tracking/Tracking_Results_1/cow_tracking_results.csv"

    # LOAD BYTE-TRACKER OUTPUTS
    # 1) Raw detector boxes per frame
    with open(DETECTIONS_JSON, 'r') as f:
        raw_det = json.load(f)
    detections = {int(k): np.array(v) for k, v in raw_det.items()}

    # 2) Cow‐name lists per frame
    with open(NAMES_JSON, 'r') as f:
        raw_names = json.load(f)
    detection_names = {int(k): v for k, v in raw_names.items()}

    # 3) Tracked results (one row per detection→track assignment)
    df = pd.read_csv(TRACKING_CSV)

    # DEFINE IoU HELPER FUNCTIONS
    def tlwh_to_xyxy(x, y, w, h):
        return np.array([x, y, x + w, y + h])

    def iou(boxA, boxB):
        xa = max(boxA[0], boxB[0]);
        ya = max(boxA[1], boxB[1])
        xb = min(boxA[2], boxB[2]);
        yb = min(boxA[3], boxB[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    # BUILD PER-TRACK NAME HISTORIES
    track_history = defaultdict(list)

    for _, row in df.iterrows():
        frame_id = int(row.frame)
        tid = row.id
        # convert tracked box to xyxy
        box_t = tlwh_to_xyxy(row.x, row.y, row.w, row.h)


        dets = detections.get(frame_id, np.empty((0, 5)))
        names = detection_names.get(frame_id, [])

        best_i, best_iou = -1, 0.0
        for i, det in enumerate(dets):
            box_d = np.array([det[0], det[1], det[2], det[3]])
            score = iou(box_t, box_d)
            if score > best_iou:
                best_iou, best_i = score, i

        # only trust a name if IoU≥0.3 (tweak this if needed)
        if best_iou > 0.3 and best_i < len(names):
            track_history[tid].append(names[best_i])
        else:
            track_history[tid].append(None)

    # COMPUTE AND CREATE A DATAFRAME OF ACCURACY PER TRACK
    trackResults = []
    for tid, history in track_history.items():
        # SKIP SHORT TRACKS
        if len(history) < MINIMUM_TRACK_FRAMES:
            continue

        # count only valid names
        valid = [n for n in history if n is not None]
        if not valid:
            acc = 0.0
            top_name, top_count = None, 0
        else:
            cnts = Counter(valid)
            top_name, top_count = cnts.most_common(1)[0]
            acc = top_count / len(valid) * 100.0

        trackResults.append({
            "track_id": tid,
            "accuracy_%": round(acc, 1),
            "total_frames": len(history),
            "top_name": top_name,
            "top_count": top_count
        })
    if not trackResults:
        return [0, 0.0, 0]

    acc_df = pd.DataFrame(trackResults).sort_values("track_id", ascending=True)
    acc_df.reset_index(drop=True, inplace=True)

    totalTrackedFrames = acc_df["total_frames"].sum()
    totalCorrectDetections = acc_df["top_count"].sum()
    overall_tracking_accuracy = round((100 * totalCorrectDetections / totalTrackedFrames), 2)

    return [len(acc_df), totalTrackedFrames, overall_tracking_accuracy, top_name]


# FUNCTION TO FIND OBJECTIVE SCORE BASED ON THE BYTETRACK OUTPUT
def objective_score(
    T: float,
    F: float,
    A: float,
    *,
    total_frames: float,
    ideal_tracks: float = 6.0,
    weights: Optional[Tuple[float, float, float]] = None,
    mode: Literal["additive", "multiplicative"] = "additive"
) -> float:
    # 1) sub‐scores in [0,1]
    s_T = min(max(ideal_tracks / T, 0.0), 1.0) if T > 0 else 0.0
    s_F = min(max(F / total_frames, 0.0), 1.0)    if total_frames > 0 else 0.0
    s_A = min(max(A / 100.0, 0.0), 1.0)

    # 2) weights
    if weights is None:
        w_T = w_F = w_A = 1/3
    else:
        w_T, w_F, w_A = weights

    # 3) combine
    if mode == "additive":
        return w_T*s_T + w_F*s_F + w_A*s_A
    elif mode == "multiplicative":
        return (s_T**w_T * s_F**w_F * s_A**w_A)
    else:
        raise ValueError("mode must be 'additive' or 'multiplicative'")


# FUNCTION TO FIND DETECTION SCORE FOR OPTIMIZATION
def objective(trial):
    tt = trial.suggest_float("track_thresh", 0.1, 1.0)
    tb = trial.suggest_int("track_buffer", 30, 200)
    mt = trial.suggest_float("match_thresh", 0.01, 1.0)

    # three raw weights in [0,1], one for each component - "track threshold", "track buffer", "match threshold"
    wT_raw = trial.suggest_float("wT_raw", 0.0, 1.0)
    wF_raw = trial.suggest_float("wF_raw", 0.0, 1.0)
    wA_raw = trial.suggest_float("wA_raw", 0.0, 1.0)
    s = wT_raw + wF_raw + wA_raw or 1e-8  # avoid div-by-zero
    weights = (wT_raw / s, wF_raw / s, wA_raw / s)

    # run your tracker with (tt, tb, mt) → returns (n_tracks, frames_tracked, accuracy)
    n_tracks, frames_tracked, accuracy = findAccuracyResult(detections, tt, tb, mt)

    # composite to maximize:
    return objective_score(
        T=n_tracks,
        F=frames_tracked,
        A=accuracy,
        total_frames=total_frames,
        ideal_tracks=6.0,
        weights=weights,
        mode="additive"  # or "multiplicative"
    )


if __name__ == "__main__":
    weights = (0.2, 0.3, 0.5)
    for i in tqdm(range(1, MAX_MATCH+1), desc="Assessing Tracker Match Threshold"):
        n_tracks, frames_tracked, accuracy = findAccuracyResult(detections, track_thresh=0.5, track_buffer=i, match_thresh=0.7)
        # objScore = objective_score(
        #     T=n_tracks,
        #     F=frames_tracked,
        #     A=accuracy,
        #     total_frames=total_frames,
        #     ideal_tracks=6.0,
        #     weights=weights,
        #     mode="additive"  # or "multiplicative"
        # )
        # accuracyResults.append([n_tracks, frames_tracked, accuracy, objScore])
        accuracyResults.append([n_tracks, frames_tracked, accuracy])

    print(accuracyResults)
    # DISPLAY THE TRACKING ACCURACY OF EACH OF THE MATCH THRESHOLD VALUES
    for i in range(1, MAX_MATCH+1):
        accuracyResult = accuracyResults[i-1]
        print("Track Buffer = {}".format(i))
        print("Total Tracks = {}".format(accuracyResult[0]))
        print("Total Frames Tracked = {}".format(accuracyResult[1]))
        print("Tracking Accuracy = {}\n".format(accuracyResult[2]))
        # print("Objective Score = {}\n".format(accuracyResult[3]))
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=10000)
    # print(study.best_params)


