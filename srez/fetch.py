"""Загрузка стримов по ссылкам через yt-dlp.

Чтобы было быстро, сначала качается только звук (это в 20–50 раз меньше
видео), по нему ищутся моменты, а потом видео скачивается только кусками
вокруг выбранных моментов.
"""
import json
from pathlib import Path

from .media import ToolError, run

PAD = 6  # запас по краям куска, сек


def info(url: str) -> dict:
    out = run(["yt-dlp", "-J", "--no-warnings", url])
    return json.loads(out)


def audio(url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    done = list(workdir.glob("audio.*"))
    done = [p for p in done if p.suffix not in (".part", ".ytdl")]
    if done:
        return done[0]
    run(["yt-dlp", "--no-warnings", "-N", "8", "-f", "audio_only/Audio_Only/bestaudio/worst",
         "-o", str(workdir / "audio.%(ext)s"), url], quiet=False)
    return next(p for p in workdir.glob("audio.*") if p.suffix not in (".part", ".ytdl"))


def section(url: str, start: float, end: float, out: Path, height: int = 1080) -> tuple[Path, float]:
    """Кусок видео [start−PAD, end+PAD]. Возвращает файл и смещение начала момента в нём."""
    if out.exists():
        return out, min(PAD, start)
    a, b = max(0.0, start - PAD), end + PAD
    fmt = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"
    try:
        run(["yt-dlp", "--no-warnings", "-N", "8", "-f", fmt, "--download-sections", f"*{a:.2f}-{b:.2f}",
             "--merge-output-format", "mp4", "-o", str(out), url])
    except ToolError:
        out.unlink(missing_ok=True)
        raise
    return out, start - a
