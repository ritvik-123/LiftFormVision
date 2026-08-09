"""
Squat — single stage (runs in SquatEnv: mediapipe legacy Solutions API,
no TensorFlow). Ported from squat_final.ipynb verbatim. Whole-clip
aggregation (no rep segmentation exists yet). Streams progress every
10 frames, then a wrapped result line.
Usage: python squat_pipeline.py <video_path>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")  # must precede mediapipe import

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # app root
from contracts import ExerciseResult

CONFIG_PATH = Path(__file__).resolve().parent / "models" / "squat_config.json"
mp_pose = mp.solutions.pose


def get_landmark_xy(landmarks, name, w, h):
    lm = landmarks.landmark[mp_pose.PoseLandmark[name].value]
    return np.array([lm.x * w, lm.y * h])


def get_best_side_landmarks(landmarks):
    def avg_visibility(side):
        names = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
        vis = [landmarks.landmark[mp_pose.PoseLandmark[n].value].visibility for n in names]
        return np.mean(vis)
    left_vis, right_vis = avg_visibility("LEFT"), avg_visibility("RIGHT")
    return "LEFT" if left_vis >= right_vis else "RIGHT"


def torso_lean_angle(shoulder, hip):
    torso_vec = shoulder - hip
    vertical_vec = np.array([0, -1])
    cosine = np.dot(torso_vec, vertical_vec) / (np.linalg.norm(torso_vec) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def check_squat_depth(hip_knee_y_diff_norm, threshold):
    return hip_knee_y_diff_norm < threshold


def check_torso_lean(peak_torso_lean_angle, threshold):
    return peak_torso_lean_angle > threshold


def analyze_video(video_path, cfg):
    depth_cfg = cfg["checks"]["depth"]
    torso_cfg = cfg["checks"]["torso_lean"]

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    pose = mp_pose.Pose(**cfg["pose_config"])

    hip_knee_diffs, torso_angles, heel_lifts = [], [], []
    heel_y_baseline, baseline_frames = None, []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            side = get_best_side_landmarks(result.pose_landmarks)
            hip = get_landmark_xy(result.pose_landmarks, f"{side}_HIP", w, h)
            knee = get_landmark_xy(result.pose_landmarks, f"{side}_KNEE", w, h)
            ankle = get_landmark_xy(result.pose_landmarks, f"{side}_ANKLE", w, h)
            shoulder = get_landmark_xy(result.pose_landmarks, f"{side}_SHOULDER", w, h)
            heel = get_landmark_xy(result.pose_landmarks, f"{side}_HEEL", w, h)

            femur_length = np.linalg.norm(hip - knee)
            shin_length = np.linalg.norm(knee - ankle)
            hip_knee_diffs.append((hip[1] - knee[1]) / (femur_length + 1e-6))
            torso_angles.append(torso_lean_angle(shoulder, hip))

            if len(baseline_frames) < 5:
                baseline_frames.append(heel[1])
            if heel_y_baseline is None and len(baseline_frames) == 5:
                heel_y_baseline = np.mean(baseline_frames)
            if heel_y_baseline is not None:
                heel_lifts.append((heel_y_baseline - heel[1]) / (shin_length + 1e-6))

        if frame_idx % 10 == 0 and hip_knee_diffs:
            print(json.dumps({
                "type": "progress", "stage": "squat", "current": frame_idx, "total": total_frames,
                "data": {
                    "peak_depth_so_far": round(float(max(hip_knee_diffs)), 4),
                    "peak_torso_lean_so_far": round(float(max(torso_angles)), 2),
                },
            }), flush=True)

    cap.release()
    pose.close()

    if not hip_knee_diffs:
        return None

    peak_depth = max(hip_knee_diffs)
    peak_torso = max(torso_angles)
    peak_heel = max(heel_lifts) if heel_lifts else 0.0

    return {
        "depth_shallow": bool(check_squat_depth(peak_depth, depth_cfg["threshold"])),
        "torso_lean_tip": bool(check_torso_lean(peak_torso, torso_cfg["threshold"])),
        "peak_heel_lift": round(float(peak_heel), 2),
        "peak_hip_knee_y_diff_norm": round(float(peak_depth), 4),
        "peak_torso_lean_angle": round(float(peak_torso), 2),
    }


def _emit_result(result: ExerciseResult):
    print(json.dumps({"type": "result", "result": json.loads(result.to_json())}), flush=True)


def main():
    video_path = Path(sys.argv[1])
    cfg = json.loads(CONFIG_PATH.read_text())
    verdict = analyze_video(video_path, cfg)

    if verdict is None:
        _emit_result(ExerciseResult(
            exercise="squat", video=video_path.name, ok=True,
            summary="No pose detected in this video.",
            notes=["Check camera angle (3/4 view expected) and lighting."],
        ))
        return

    notes = [
        "Depth looks shallow." if verdict["depth_shallow"] else "Depth looks good.",
        "Torso lean looks high (tip only, not a hard fault)." if verdict["torso_lean_tip"] else "Torso lean looks fine.",
        f"Heel lift: {verdict['peak_heel_lift']} (informational only, not flagged).",
    ]

    if verdict["depth_shallow"]:
        summary = "Depth looks shallow."
    elif verdict["torso_lean_tip"]:
        summary = "Torso lean is a bit high -- worth a look, not necessarily a fault."
    else:
        summary = "Form looks good."

    _emit_result(ExerciseResult(
        exercise="squat", video=video_path.name, ok=True,
        summary=summary,
        flags={"depth_shallow": verdict["depth_shallow"], "torso_lean_tip": verdict["torso_lean_tip"]},
        scores={
            "peak_hip_knee_y_diff_norm": verdict["peak_hip_knee_y_diff_norm"],
            "peak_torso_lean_angle": verdict["peak_torso_lean_angle"],
            "peak_heel_lift": verdict["peak_heel_lift"],
        },
        notes=notes,
    ))


if __name__ == "__main__":
    main()