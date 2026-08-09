"""
Shared helper for writing browser-playable (H.264) annotated videos.
cv2.VideoWriter's codec support is unreliable for browser playback, so
frames are piped as raw bytes into ffmpeg (via imageio-ffmpeg's bundled
static binary) with libx264 instead.
"""
import subprocess
import imageio_ffmpeg


def open_overlay_writer(out_path, width, height, fps):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_overlay_frame(proc, frame_bgr):
    try:
        proc.stdin.write(frame_bgr.tobytes())
    except (BrokenPipeError, OSError):
        pass  # ffmpeg died -- let close_overlay_writer surface the real error


def close_overlay_writer(proc):
    if proc.stdin:
        proc.stdin.close()
    proc.wait()