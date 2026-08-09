"""
Common interface every exercise pipeline's final stage must print to
stdout as JSON. app.py and runner.py only ever touch this shape.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class ExerciseResult:
    exercise: str
    video: str
    ok: bool
    summary: str
    flags: Dict[str, bool] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    overlay_url: Optional[str] = None   # /static/overlays/<file>.mp4 -- landmark-annotated playback, when available

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ExerciseResult":
        return ExerciseResult(**json.loads(s))