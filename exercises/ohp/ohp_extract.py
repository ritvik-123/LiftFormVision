"""
OHP — Stage 1 (runs in the OHPBonus venv: mediapipe only, no TensorFlow).
Ported from OHP_bonus_features.ipynb::extract_ohp_dual + analyze_upper_body.
Streams progress JSON lines while extracting.
Usage: python ohp_extract.py <video_path> <intermediate_out_path>
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"

SIDE_LANDMARKS = {
    "LEFT":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27},
    "RIGHT": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28},
}

RULES = {
    "lockout_warning": 152, "lockout_strong": 145,
    "grip_narrow": 1.23, "grip_wide": 1.94,
    "stack_warning": 0.55, "wrist_asymmetry": 0.822,
}


@dataclass
class PoseFrame:
    px: dict = field(default_factory=dict)
    norm: dict = field(default_factory=dict)
    vis: dict = field(default_factory=dict)
    detected: bool = False


def angle_3pt(a, b, c):
    a, b, c = map(lambda x: np.asarray(x, float), (a, b, c))
    ba, bc = a - b, c - b
    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d <= 1e-8:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / d, -1, 1)))


def get_geometry(lm):
    LM = {"LS": 11, "RS": 12, "LE": 13, "RE": 14, "LW": 15, "RW": 16}
    LS, RS = [lm[LM["LS"]].x, lm[LM["LS"]].y], [lm[LM["RS"]].x, lm[LM["RS"]].y]
    LE, RE = [lm[LM["LE"]].x, lm[LM["LE"]].y], [lm[LM["RE"]].x, lm[LM["RE"]].y]
    LW, RW = [lm[LM["LW"]].x, lm[LM["LW"]].y], [lm[LM["RW"]].x, lm[LM["RW"]].y]

    shoulder_width = np.linalg.norm(np.array(RS) - np.array(LS))
    if shoulder_width < 1e-6:
        return None

    left_elbow = angle_3pt(LS, LE, LW)
    right_elbow = angle_3pt(RS, RE, RW)

    return {
        "grip_width_ratio": abs(RW[0] - LW[0]) / shoulder_width,
        "left_stack": abs(LW[0] - LE[0]) / shoulder_width,
        "right_stack": abs(RW[0] - RE[0]) / shoulder_width,
        "left_elbow_angle": left_elbow,
        "right_elbow_angle": right_elbow,
        "elbow_asymmetry": abs(left_elbow - right_elbow),
        "wrist_height_asymmetry": abs(LW[1] - RW[1]) / shoulder_width,
        "wrist_height": (LW[1] + RW[1]) / 2,
        "min_visibility": min(lm[LM[k]].visibility for k in LM),
    }


def extract_ohp_dual(video_path):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    video_frames, bonus_rows = [], []

    video_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
        min_tracking_confidence=.5,
    )
    image_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
    )

    with PoseLandmarker.create_from_options(video_opts) as video_lm, \
         PoseLandmarker.create_from_options(image_opts) as image_lm:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            vr = video_lm.detect_for_video(image, int(i * 1000.0 / fps))
            f = PoseFrame()
            if vr.pose_landmarks:
                f.detected = True
                for j, lm in enumerate(vr.pose_landmarks[0]):
                    f.norm[j] = np.array([lm.x, lm.y], float)
                    f.px[j] = np.array([lm.x * w, lm.y * h], float)
                    f.vis[j] = lm.visibility
            video_frames.append(f)

            ir = image_lm.detect(image)   # IMAGE mode -- do not change
            if ir.pose_landmarks:
                feat = get_geometry(ir.pose_landmarks[0])
                if feat:
                    feat.update({"video_id": video_path.stem, "frame_idx": i, "time_sec": i / fps})
                    bonus_rows.append(feat)

            i += 1
            if i % 15 == 0:
                print(json.dumps({"type": "progress", "stage": "extract", "current": i, "total": total_frames}), flush=True)
    cap.release()

    scores = {}
    for side, idx in SIDE_LANDMARKS.items():
        joints = [idx[x] for x in ["elbow", "knee", "wrist", "ankle"]]
        scores[side] = [np.mean([f.vis.get(j, 0) for j in joints]) for f in video_frames if f.detected]
    side = "LEFT" if np.mean(scores["LEFT"] or [0]) >= np.mean(scores["RIGHT"] or [0]) else "RIGHT"
    idx = SIDE_LANDMARKS[side]

    rows = []
    for n, f in enumerate(video_frames):
        row = {
            "video_id": video_path.stem, "frame_idx": n, "detected": f.detected, "fps": fps,
            "knee_angle": np.nan, "elbow_angle": np.nan,
            "knee_visibility": np.nan, "elbow_visibility": np.nan,
            "shoulder_width_px": np.nan, "wrist_height_px": np.nan,
        }
        if f.detected:
            knee = [idx["hip"], idx["knee"], idx["ankle"]]
            elbow = [idx["shoulder"], idx["elbow"], idx["wrist"]]
            row["knee_visibility"] = min(f.vis.get(j, 0) for j in knee)
            row["elbow_visibility"] = min(f.vis.get(j, 0) for j in elbow)
            if row["knee_visibility"] >= .50:
                row["knee_angle"] = angle_3pt(*(f.px[j] for j in knee))
            if row["elbow_visibility"] >= .50:
                row["elbow_angle"] = angle_3pt(*(f.px[j] for j in elbow))
            if all(j in f.px for j in [11, 12]):
                row["shoulder_width_px"] = np.linalg.norm(f.px[12] - f.px[11])
            wrists = [j for j in [15, 16] if j in f.px]
            if wrists:
                row["wrist_height_px"] = np.mean([f.px[j][1] for j in wrists])
        rows.append(row)

    knee_df = pd.DataFrame(rows)
    sw = knee_df["shoulder_width_px"].median()
    knee_df["wrist_height_norm"] = knee_df["wrist_height_px"] / sw if pd.notna(sw) and sw > 0 else np.nan

    bonus_df = pd.DataFrame(bonus_rows)
    return knee_df, bonus_df


def analyze_upper_body(bonus_df):
    x = bonus_df[bonus_df["min_visibility"] >= 0.30].copy() if not bonus_df.empty else bonus_df
    if x.empty:
        return {"grip": "Not assessable", "lockout": "Not assessable",
                "stacking": "Not assessable", "symmetry": "Not assessable",
                "grip_score": None, "lockout_score": None, "stack_score": None, "symmetry_score": None}

    q05, q95 = x["wrist_height"].quantile([.05, .95])
    travel = q95 - q05
    x["top"] = x["wrist_height"] <= q05 + .15 * travel
    x["early"] = x["wrist_height"] >= q95 - .30 * travel

    reliable = x[x["min_visibility"] >= .50]
    top = reliable[reliable["top"]]
    early = reliable[reliable["early"]]

    grip_score = x["grip_width_ratio"].median()
    grip = ("Grip appears unusually wide" if grip_score > RULES["grip_wide"] else
            "Grip appears unusually narrow" if grip_score < RULES["grip_narrow"] else
            "Grip within reference range")

    if len(top) >= 5:
        lockout_score = min(top["left_elbow_angle"].quantile(.90), top["right_elbow_angle"].quantile(.90))
        lockout = ("Strong incomplete lockout" if lockout_score < RULES["lockout_strong"] else
                   "Possible incomplete lockout" if lockout_score < RULES["lockout_warning"] else
                   "Lockout looks good")
    else:
        lockout_score, lockout = None, "Not assessable"

    if len(early) >= 5:
        stack_score = max(early["left_stack"].median(), early["right_stack"].median())
        stacking = "Wrist–elbow misalignment detected" if stack_score > RULES["stack_warning"] else "Stacking looks good"
    else:
        stack_score, stacking = None, "Not assessable"

    if len(reliable) >= 10:
        symmetry_score = reliable["wrist_height_asymmetry"].median()
        symmetry = "Left/right wrist asymmetry detected" if symmetry_score > RULES["wrist_asymmetry"] else "Symmetry looks good"
    else:
        symmetry_score, symmetry = None, "Not assessable"

    return {
        "grip": grip, "lockout": lockout, "stacking": stacking, "symmetry": symmetry,
        "grip_score": float(grip_score) if grip_score is not None else None,
        "lockout_score": float(lockout_score) if lockout_score is not None else None,
        "stack_score": float(stack_score) if stack_score is not None else None,
        "symmetry_score": float(symmetry_score) if symmetry_score is not None else None,
    }


def main():
    video_path, out_path = sys.argv[1], sys.argv[2]
    knee_df, bonus_df = extract_ohp_dual(video_path)
    bonus_verdict = analyze_upper_body(bonus_df)

    payload = {
        "video_name": Path(video_path).name,
        "knee_rows": knee_df.replace({np.nan: None}).to_dict(orient="records"),
        "bonus": bonus_verdict,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()