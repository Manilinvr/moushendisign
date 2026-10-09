"""Обёртки над ffmpeg/ffprobe."""
import json
import os
import subprocess
from pathlib import Path


class ToolError(RuntimeError):
    pass


def run(cmd, quiet=True):
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, stdout=subprocess.PIPE if quiet else None,
                       stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-15:])
        raise ToolError(f"Команда завершилась с ошибкой ({r.returncode}): {' '.join(cmd[:6])} …\n{tail}")
    return r.stdout


def ffmpeg(*args):
    return run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", *args])


def probe(path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(out)


def duration(path) -> float:
    return float(probe(path)["format"]["duration"])


def has_audio(path) -> bool:
    return any(s["codec_type"] == "audio" for s in probe(path)["streams"])


def frame(path, t, out, width=None):
    """Один кадр из видео в момент t (сек)."""
    vf = ["-vf", f"scale={width}:-2"] if width else []
    ffmpeg("-ss", f"{max(0, t):.3f}", "-i", path, "-frames:v", "1", *vf, "-q:v", "3", out)
    return Path(out)


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)
