# LiftFormVision

A locally-deployed computer-vision application that analyzes barbell exercise video — squat, overhead press, and barbell row — and reports form errors in real time, without a human coach in the loop.

This repo is the **deployed application**. The models, thresholds, and feature-engineering work it runs were developed in [`298-Major-Project`](https://github.com/ritvik-123/298-Major-Project); this repo is where that work gets wired into an actual runnable web app.

## What It Does

- **Upload mode** — pick an exercise, upload a clip, watch progress stream in, then see the result with a skeleton overlay rendered inline as a real, playable H.264 video.
- **Live-camera mode** — the browser records 5-second chunks; each chunk runs through the same analysis pipeline as upload mode, with results appearing in a feed linked to each clip.
- **Per-exercise diagnostics**:
  - **Squat**: depth (validated threshold), torso lean (advisory tip), heel lift (informational).
  - **Overhead press**: a temporal-CNN knee-error classifier, plus rule-based grip/lockout/stacking/symmetry checks, each gated on landmark visibility.
  - **Barbell row**: not currently shipped as a check — see [Known Limitations](#known-limitations).

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask (Python), server-rendered HTML templates |
| Pose estimation | MediaPipe — legacy **Solutions API** (squat) and Tasks **PoseLandmarker API** (OHP, both `VIDEO` and `IMAGE` running modes) |
| Learned model | TensorFlow / Keras — 1D CNN for OHP knee-error classification |
| Classical models / thresholds | scikit-learn (logistic regression baselines), NumPy-based geometric feature calibration |
| Video I/O & overlay rendering | OpenCV (`cv2`), H.264 encoding for playable overlay output |
| Data handling | pandas, NumPy |
| Process orchestration | Python `subprocess`, isolated per-exercise virtual environments, JSON as the inter-process contract |
| Frontend | HTML/CSS/JS (Flask `templates/`), browser `MediaRecorder` API for live-camera chunk capture |
| Persistence (app-level) | Local filesystem (`uploads/`, `temp/`) — no database |

## Architecture

The core engineering constraint driving this app's structure: **MediaPipe's legacy Solutions API and TensorFlow have conflicting protobuf version requirements** and cannot be reliably imported together in one Python process. Rather than fight that at the dependency-resolution level, each exercise runs in its own isolated virtual environment and its own subprocess.

```
                         ┌────────────────────┐
   Browser (upload /     │   Flask (app.py)    │
   live camera) ───────▶ │   AppEnv venv       │
                         │   no MediaPipe/TF    │
                         └─────────┬───────────┘
                                   │ dispatch via
                                   │ exercise_registry.py (Strategy pattern)
                                   │ runner.py (subprocess pipeline chain)
                                   ▼
        ┌──────────────────────────────────────────────┐
        │                                                │
  ┌─────▼─────┐                              ┌───────────▼──────────┐
  │ SquatEnv  │                              │  OHP pipeline (2-stage)│
  │ legacy    │                              │  ┌─────────────────┐  │
  │ MediaPipe │                              │  │ OHPBonus venv   │  │
  │ Solutions │                              │  │ MediaPipe pose  │  │
  │ API       │                              │  │ + upper-body    │  │
  └───────────┘                              │  │ geometry rules  │  │
                                              │  └────────┬────────┘  │
                                              │           │ features  │
                                              │  ┌────────▼────────┐  │
                                              │  │ DeepGPU venv    │  │
                                              │  │ TensorFlow      │  │
                                              │  │ knee CNN infer. │  │
                                              │  └─────────────────┘  │
                                              └────────────────────────┘
```

- Each subprocess reports its result back to Flask as **plain JSON**, following the schema defined in `contracts.py` — `app.py` never needs to know which exercise or environment produced a given result.
- `exercise_registry.py` implements a **Strategy pattern**: each exercise registers its own entry point, input requirements, and output shape, so adding a new exercise doesn't require touching `app.py`.
- `runner.py` implements the **subprocess pipeline chain** — for OHP specifically, it chains two environments per video: `OHPBonus` extracts pose features first, then `DeepGPU` consumes them for knee-CNN scoring.
- Skeleton-overlay video is rendered in the *same pass* that computes the analysis, not as a separate post-processing step.

## Repository Structure

```
app.py                  Flask entry point — routes, upload handling, live-camera endpoint
config.py               App-wide configuration (paths, per-exercise environment/script locations)
contracts.py            Shared JSON result schema every exercise subprocess must return
exercise_registry.py    Strategy pattern — registers each exercise's pipeline entry point
runner.py               Subprocess orchestration / pipeline chaining (e.g. OHP's two-stage handoff)

Envs/
  AppEnv/                Virtual environment for the Flask app itself (no MediaPipe/TensorFlow)
  (SquatEnv, OHPBonus, DeepGPU are set up locally — see Setup below)

exercises/               Per-exercise pipeline code (feature extraction, inference, thresholds)
templates/                Flask HTML templates (upload UI, results view, live-camera UI)
uploads/                  User-uploaded video storage
temp/                     Scratch space for intermediate pipeline artifacts
```

## Setup

The app requires **four** Python environments because of the MediaPipe/TensorFlow conflict described above. Package versions below match what the pipelines were built and validated against.

```bash
# 1. AppEnv — runs Flask only, no CV/ML libraries
python -m venv Envs/AppEnv
Envs/AppEnv/Scripts/activate     # or source Envs/AppEnv/bin/activate on Linux/Mac
pip install flask

# 2. SquatEnv — legacy MediaPipe Solutions API
python -m venv Envs/SquatEnv
Envs/SquatEnv/Scripts/activate
pip install mediapipe==0.10.14 opencv-python numpy pandas scikit-learn

# 3. OHPBonus — MediaPipe Tasks PoseLandmarker API (pose + upper-body rules), no TensorFlow
python -m venv Envs/OHPBonus
Envs/OHPBonus/Scripts/activate
pip install mediapipe==0.10.14 opencv-python numpy pandas

# 4. DeepGPU — TensorFlow, for knee-CNN inference
python -m venv Envs/DeepGPU
Envs/DeepGPU/Scripts/activate
pip install tensorflow==2.21.0 numpy pandas scikit-learn joblib
```

Point `config.py` at the interpreter paths for each environment (`SquatEnv`, `OHPBonus`, `DeepGPU`), and place the trained artifacts exported from [`298-Major-Project`](https://github.com/ritvik-123/298-Major-Project) — `knee_temporal_cnn.keras`, `knee_frame_preprocessor.joblib`, `squat_config.json`, and the MediaPipe `pose_landmarker*.task` files — under the relevant `exercises/<exercise>/models/` directory.

### Run

```bash
Envs/AppEnv/Scripts/activate
python app.py
```

Then open `http://localhost:5000` (or the port Flask reports) in a browser. Upload a clip, or grant camera access to use live mode.

> **Note:** the app was developed and validated on Windows. MediaPipe's Tasks API has a known Windows path-resolution issue with drive-letter paths — model files are loaded via `model_asset_buffer` (reading bytes directly) rather than `model_asset_path` to work around it.

## Model & Threshold Provenance

No models are trained inside this repo. All thresholds, the knee CNN weights, and feature-engineering logic were developed and validated in [`298-Major-Project`](https://github.com/ritvik-123/298-Major-Project), then exported here as frozen config/model artifacts. See that repo (and the project report) for methodology, validation numbers, and known failure modes per exercise.

## Known Limitations

- **Barbell row has no shipped check.** Pose estimation fails on ~34% of frames due to occlusion; the best appearance-based model (64.0% validation accuracy) showed clear overfitting and was judged not reliable enough for production.
- **No rep segmentation.** Squat and OHP checks currently aggregate over the whole clip, not per individual repetition.
- **The OHP knee model outputs a raw score**, not a calibrated probability — it's compared against a fixed decision threshold (0.6782), not read as a confidence percentage.
- **Single-lifter assumption.** Multi-person frames (e.g. a spotter in view) are not specifically handled.
- **Camera-angle dependent.** Visibility gating means some upper-body checks will correctly report "Not assessable" rather than a forced diagnosis when the camera angle doesn't support it.

## Related

- [`298-Major-Project`](https://github.com/ritvik-123/298-Major-Project) — modeling notebooks, dataset details, and validation methodology behind every pipeline this app runs.
