"""
Registry of exercise pipelines (Strategy pattern).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import config


@dataclass
class Stage:
    python_exe: str
    script: str
    label: str                       # shown in the live progress panel
    extra_env: Optional[Dict[str, str]] = None


@dataclass
class ExerciseSpec:
    name: str
    label: str
    stages: List[Stage]


REGISTRY: Dict[str, ExerciseSpec] = {
    "ohp": ExerciseSpec(
        name="ohp", label="Overhead Press",
        stages=[
            Stage(python_exe=config.OHP_BONUS_PYTHON,
                  script=str(config.EXERCISES_DIR / "ohp" / "ohp_extract.py"),
                  label="Extracting pose & upper-body features"),
            Stage(python_exe=config.OHP_DEEPGPU_PYTHON,
                  script=str(config.EXERCISES_DIR / "ohp" / "ohp_infer.py"),
                  label="Scoring knee movement"),
        ],
    ),
    "squat": ExerciseSpec(
        name="squat", label="Squat",
        stages=[
            Stage(
                python_exe=config.SQUAT_PYTHON,
                script=str(config.EXERCISES_DIR / "squat" / "squat_pipeline.py"),
                label="Analyzing squat form",
                extra_env={"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
            ),
        ],
    ),
}


def get_exercise(name: str) -> ExerciseSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown exercise '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name]