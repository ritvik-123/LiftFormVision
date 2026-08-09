"""
Runs an ExerciseSpec's stages as subprocesses, one per venv, streaming
progress + the final ExerciseResult back as they happen.

Protocol: each exercise script prints one JSON object per line to
stdout, flushed immediately:
  {"type": "progress", "stage": "...", "current": N, "total": N|null, "data": {...}}
  {"type": "result", "result": {...ExerciseResult fields...}}   # last stage only

Note: per-stage hard timeouts aren't enforced during streaming (reading
a live pipe with a timeout is unreliable on Windows) -- acceptable for
a local single-user tool, worth revisiting if this ever runs unattended.
"""
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Iterator, Optional

import config
from contracts import ExerciseResult
from exercise_registry import ExerciseSpec, get_exercise


def _spawn(python_exe: str, script: str, args: list, extra_env: Optional[dict]) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [python_exe, script, *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )


def stream_exercise(exercise_name: str, video_path: Path) -> Iterator[dict]:
    """Yields progress dicts as they happen, then one {"type": "result", ...}
    dict, then stops. Yields {"type": "error", ...} and stops on failure."""
    spec: ExerciseSpec = get_exercise(exercise_name)
    run_id = uuid.uuid4().hex[:8]
    carry = str(video_path)

    for i, stage in enumerate(spec.stages):
        is_last = i == len(spec.stages) - 1
        intermediate_path = config.TEMP_DIR / f"{run_id}_stage{i}.json"

        args = [carry]
        if not is_last:
            args.append(str(intermediate_path))

        yield {"type": "progress", "stage": stage.label, "current": 0, "total": None,
               "message": f"Starting: {stage.label}..."}

        try:
            proc = _spawn(stage.python_exe, stage.script, args, stage.extra_env)
        except FileNotFoundError as e:
            yield {"type": "error", "message": f"Could not start stage {i} ({stage.script}): {e}"}
            return

        result_line = None
        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue  # stray print from a library -- not our protocol, ignore
            if obj.get("type") == "result":
                result_line = obj
            else:
                yield obj

        stderr_text = proc.stderr.read()
        returncode = proc.wait()

        if returncode != 0:
            yield {"type": "error", "message": f"Stage {i} ({stage.script}) exited {returncode}:\n{stderr_text[-2000:]}"}
            return

        if is_last:
            if result_line is None:
                yield {"type": "error", "message": f"Stage {i} finished with no result line.\nstderr:\n{stderr_text[-2000:]}"}
                return
            result = ExerciseResult(**result_line["result"])
            result.video = video_path.name
            yield {"type": "result", "result": result.__dict__}
            return
        else:
            carry = str(intermediate_path)


def run_exercise(exercise_name: str, video_path: Path) -> ExerciseResult:
    """Non-streaming convenience wrapper -- drains stream_exercise() and
    returns just the final ExerciseResult. Handy for quick manual testing."""
    for event in stream_exercise(exercise_name, video_path):
        if event["type"] == "result":
            return ExerciseResult(**event["result"])
        if event["type"] == "error":
            return ExerciseResult(exercise=exercise_name, video=video_path.name, ok=False,
                                   summary="Analysis failed.", error=event["message"])
    return ExerciseResult(exercise=exercise_name, video=video_path.name, ok=False,
                           summary="No result produced.", error="stream ended without a result")