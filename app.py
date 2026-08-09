"""
Local deployment entry point. Streams NDJSON progress to the browser as
each exercise script works through the video.
"""
import json
from pathlib import Path
from flask import Flask, Response, render_template, request

import config
from exercise_registry import REGISTRY
from runner import stream_exercise

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", exercises=REGISTRY)


@app.route("/analyze", methods=["POST"])
def analyze():
    exercise_name = request.form.get("exercise")
    video_file = request.files.get("video")

    if exercise_name not in REGISTRY:
        return {"type": "error", "message": "Pick a valid exercise."}, 400
    if not video_file or video_file.filename == "":
        return {"type": "error", "message": "Pick a video file."}, 400

    ext = Path(video_file.filename).suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        return {"type": "error", "message": f"Unsupported file type {ext}."}, 400

    save_path = config.UPLOAD_DIR / video_file.filename
    video_file.save(save_path)

    def generate():
        for event in stream_exercise(exercise_name, save_path):
            yield json.dumps(event) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)