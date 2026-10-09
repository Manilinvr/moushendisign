"""Загрузка стримов по ссылкам через yt-dlp.

Чтобы было быстро, сначала качается только звук (это в 20–50 раз меньше
видео), по нему ищутся моменты, а потом видео скачивается только кусками
вокруг выбранных моментов.
"""
import functools
import http.client
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from .media import ToolError, ffmpeg, probe, run

PAD = 6  # запас по краям куска, сек


def _proxy():
    """Куски видео качает ffmpeg, а он прокси из окружения сам не берёт — передаём явно."""
    p = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return ["--proxy", p] if p else []


def _get(url: str, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (OSError, http.client.HTTPException):  # обрыв соединения — пробуем ещё раз
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def info(url: str) -> dict:
    out = run(["yt-dlp", "-J", "--no-warnings", url])
    return json.loads(out)


def audio(url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    done = list(workdir.glob("audio.*"))
    done = [p for p in done if p.suffix not in (".part", ".ytdl", ".raw")]
    if done:
        return done[0]
    try:
        run(["yt-dlp", "--no-warnings", "-N", "8", "-f", "audio_only/Audio_Only/bestaudio/worst",
             "-o", str(workdir / "audio.%(ext)s"), url], quiet=False)
    except ToolError:
        # стрим ещё идёт: yt-dlp качает его через ffmpeg и не может — берём уже записанное сегментами
        return _hls_audio(url, workdir / "audio.m4a")
    return next(p for p in workdir.glob("audio.*") if p.suffix not in (".part", ".ytdl", ".raw"))


def _hls_audio(url: str, out: Path) -> Path:
    """Звук записи целиком по сегментам HLS (работает и для записи стрима, который ещё в эфире)."""
    from concurrent.futures import ThreadPoolExecutor
    m3u8 = run(["yt-dlp", "--no-warnings", "-g", "-f", "audio_only/Audio_Only/worst", url]).split()[0]
    text = _get(m3u8).decode("utf-8", "replace")
    init = re.search(r'#EXT-X-MAP:URI="([^"]+)"', text)
    segs = [urljoin(m3u8, ln.strip()) for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    raw = out.with_suffix(".raw")
    with raw.open("wb") as f:
        if init:
            f.write(_get(urljoin(m3u8, init.group(1))))
        with ThreadPoolExecutor(8) as ex:
            for chunk in ex.map(_get, segs):      # map сохраняет порядок сегментов
                f.write(chunk)
    try:
        ffmpeg("-i", raw, "-map", "0:a:0", "-c", "copy", out)
    finally:
        raw.unlink(missing_ok=True)
    return out


@functools.lru_cache(maxsize=None)
def _playlist(url: str, height: int):
    """Сегменты HLS-плейлиста записи: (init-сегмент или None, ((начало, длительность, ссылка), …))."""
    m3u8 = run(["yt-dlp", "--no-warnings", "-g", "-f", f"best[height<={height}]/best", url]).split()[0]
    text = _get(m3u8).decode("utf-8", "replace")
    if not text.startswith("#EXTM3U"):
        return None, ()
    init, segs, t, dur = None, [], 0.0, None
    for line in map(str.strip, text.splitlines()):
        if line.startswith("#EXT-X-MAP:"):
            init = urljoin(m3u8, re.search(r'URI="([^"]+)"', line).group(1))
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            segs.append((t, dur, urljoin(m3u8, line)))
            t, dur = t + dur, None
    return init, tuple(segs)


def _hls_section(url, a, b, out: Path, height) -> float | None:
    """Скачивает только сегменты, покрывающие [a, b]. Возвращает время начала куска в стриме.

    ffmpeg (через него yt-dlp режет куски) на записях Twitch в формате fMP4 не
    перематывает, а читает всё подряд с начала, поэтому сегменты берём сами.
    """
    init, segs = _playlist(url, height)
    pick = [s for s in segs if s[0] + s[1] > a and s[0] < b]
    if not pick:
        return None
    raw = out.with_name(out.stem + ".raw")
    with raw.open("wb") as f:
        if init:
            f.write(_get(init))
        for _, _, u in pick:
            f.write(_get(u))
    try:
        ffmpeg("-i", raw, "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", out)
    finally:
        raw.unlink(missing_ok=True)
    return pick[0][0]


def section(url: str, start: float, end: float, out: Path, height: int = 1080) -> tuple[Path, float]:
    """Кусок видео [start−PAD, end+PAD]. Возвращает файл и смещение начала момента в нём."""
    meta = out.with_suffix(".json")
    if out.exists():
        off = json.loads(meta.read_text())["offset"] if meta.exists() else min(PAD, start)
        return out, off
    a, b = max(0.0, start - PAD), end + PAD
    try:
        t0 = _hls_section(url, a, b, out, height)
        if t0 is None:
            fmt = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"
            run(["yt-dlp", "--no-warnings", *_proxy(), "-N", "8", "-f", fmt, "--download-sections",
                 f"*{a:.2f}-{b:.2f}", "--merge-output-format", "mp4", "-o", str(out), url])
            t0 = a
        if not any(s["codec_type"] == "video" for s in probe(out)["streams"]):
            raise ToolError(f"В скачанном куске нет видео: {out.name}")
    except Exception:
        out.unlink(missing_ok=True)
        raise
    meta.write_text(json.dumps({"offset": start - t0}))
    return out, start - t0
