"""
Central place for every path/constant the app needs.
Edit this file (not the others) when environments move.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXERCISES_DIR = BASE_DIR / "exercises"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
OVERLAY_DIR = BASE_DIR / "static" / "overlays"   # served automatically by Flask's static handler
OVERLAY_URL_PREFIX = "/static/overlays"

UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# --- per-exercise interpreters -------------------------------------------------
OHP_BONUS_PYTHON = r"D:\My Work\Computer Science Courses and Projects\Envs\OHPBonus\Scripts\python.exe"
OHP_DEEPGPU_PYTHON = r"D:\My Work\Computer Science Courses and Projects\Envs\DeepGPU\Scripts\python.exe"
SQUAT_PYTHON = r"D:\My Work\Computer Science Courses and Projects\Envs\SquatEnv\Scripts\python.exe"

STAGE_TIMEOUT = 300  # seconds per stage
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}  # .webm added for live-camera chunk uploads