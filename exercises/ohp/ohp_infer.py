"""
OHP — Stage 2 (runs in the DeepGPU venv: TensorFlow/keras).
Ported from OHP_final.ipynb::preprocess_knee_frames + make_ohp_report.
Streams a progress line per window, then a wrapped result line.
Usage: python ohp_infer.py <intermediate_in_path>
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # app root
from contracts import ExerciseResult

MODELS_DIR = Path(__file__).resolve().parent / "models"
KNEE_THRESHOLD = 0.6782
WINDOW = 30
STRIDE = 5


def preprocess_knee_frames(df, preprocessor):
    df = df.sort_values(["video_id", "frame_idx"]).copy()

    df["pose_missing"] = (~df["detected"].astype(bool)).astype(np.float32)
    df["knee_angle_missing"] = df["knee_angle"].isna().astype(np.float32)
    df["elbow_angle_missing"] = df["elbow_angle"].isna().astype(np.float32)
    df["wrist_height_missing"] = df["wrist_height_norm"].isna().astype(np.float32)

    for col in ["knee_angle", "elbow_angle", "wrist_height_norm"]:
        smooth = f"{col}_smooth"
        df[smooth] = df.groupby("video_id")[col].transform(lambda x: x.rolling(5, min_periods=1).median())
        df.loc[df[col].isna(), smooth] = np.nan

    for src, dst in {
        "knee_angle_smooth": "knee_angle_velocity",
        "elbow_angle_smooth": "elbow_angle_velocity",
        "wrist_height_norm_smooth": "wrist_height_velocity",
    }.items():
        df[dst] = df.groupby("video_id")[src].diff() * df["fps"]

    for src, dst in {
        "knee_angle_velocity": "knee_angle_acceleration",
        "wrist_height_velocity": "wrist_height_acceleration",
    }.items():
        df[dst] = df.groupby("video_id")[src].diff() * df["fps"]

    for col, bounds in preprocessor["clip_bounds"].items():
        if col in df:
            df[col] = df[col].clip(bounds["lower"], bounds["upper"])

    cont = preprocessor["continuous_columns"]
    ind = preprocessor["indicator_columns"]

    X_cont = preprocessor["imputer"].transform(df[cont])
    X_cont = preprocessor["scaler"].transform(X_cont)

    df[cont] = X_cont.astype(np.float32)
    df[ind] = df[ind].astype(np.float32)

    return df


def score_knee(knee_df: pd.DataFrame):
    preprocessor = joblib.load(MODELS_DIR / "knee_frame_preprocessor.joblib")
    model = keras.models.load_model(MODELS_DIR / "knee_temporal_cnn.keras")

    proc = preprocess_knee_frames(knee_df, preprocessor).reset_index(drop=True)
    cols = preprocessor["model_columns"]

    starts = list(range(0, len(proc) - WINDOW + 1, STRIDE))
    if not starts:
        return 0.0, False

    window_scores = []
    for k, s in enumerate(starts, 1):
        X = proc.loc[s:s + WINDOW - 1, cols].to_numpy(np.float32).reshape(1, WINDOW, -1)
        pred = float(model.predict(X, verbose=0)[0][0])
        window_scores.append(pred)
        running_top3 = float(np.mean(sorted(window_scores, reverse=True)[:3]))
        print(json.dumps({
            "type": "progress", "stage": "infer", "current": k, "total": len(starts),
            "data": {"running_knee_score": round(running_top3, 4)},
        }), flush=True)

    score = float(np.mean(sorted(window_scores, reverse=True)[:3]))
    return score, score >= KNEE_THRESHOLD


def make_ohp_report(knee_label: bool, knee_score: float, upper: dict) -> ExerciseResult:
    issues, notes = [], []

    if knee_label:
        issues.append("Knee movement error detected.")
    if "unusually wide" in upper["grip"]:
        issues.append("Grip appears unusually wide.")
    elif "unusually narrow" in upper["grip"]:
        issues.append("Grip appears unusually narrow.")
    if "incomplete lockout" in upper["lockout"].lower():
        issues.append(upper["lockout"] + ".")
    if "misalignment" in upper["stacking"].lower():
        issues.append("Wrist–elbow misalignment detected.")
    if "asymmetry detected" in upper["symmetry"].lower():
        issues.append("Left/right wrist asymmetry detected.")

    unavailable = [k for k in ["grip", "lockout", "stacking", "symmetry"] if upper[k] == "Not assessable"]
    if unavailable:
        notes.append("Not assessable: " + ", ".join(unavailable))

    summary = issues[0] if issues else "No notable issues detected."
    if len(issues) > 1:
        notes = issues[1:] + notes

    flags = {
        "knee_error": knee_label,
        "grip_wide": upper["grip"] == "Grip appears unusually wide",
        "grip_narrow": upper["grip"] == "Grip appears unusually narrow",
        "lockout_issue": "incomplete lockout" in upper["lockout"].lower(),
        "stacking_issue": "misalignment" in upper["stacking"].lower(),
        "symmetry_issue": "asymmetry detected" in upper["symmetry"].lower(),
    }
    scores = {"knee_score": round(knee_score, 4)}
    for key in ["grip_score", "lockout_score", "stack_score", "symmetry_score"]:
        if upper.get(key) is not None:
            scores[key] = round(upper[key], 4)

    return ExerciseResult(exercise="ohp", video="", ok=True, summary=summary, flags=flags, scores=scores, notes=notes)


def main():
    in_path = sys.argv[1]
    payload = json.loads(Path(in_path).read_text())

    knee_df = pd.DataFrame(payload["knee_rows"])
    knee_score, knee_label = score_knee(knee_df)

    result = make_ohp_report(knee_label, knee_score, payload["bonus"])
    result.video = payload["video_name"]
    print(json.dumps({"type": "result", "result": json.loads(result.to_json())}), flush=True)


if __name__ == "__main__":
    main()