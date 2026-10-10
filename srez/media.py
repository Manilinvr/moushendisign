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


def snap_tb(stream: dict) -> str:
    """Фильтр: кадры источника ровно на его сетку (settb=1/частота кадров).

    В стримах 60 к/с время кадров округлено до миллисекунд. Если не выровнять, переход
    на 30 к/с иногда берёт один и тот же кадр дважды — картинка дёргается и «подвисает»."""
    try:
        num, den = (int(x) for x in stream.get("r_frame_rate", "0/1").split("/"))
    except ValueError:
        return "null"
    if num <= 0 or den <= 0 or num / den > 240:
        return "null"
    return f"settb={den}/{num}"


def judder(path, width=160, height=96, grid=4) -> tuple[float, list[tuple[float, float]]]:
    """Подвисания в готовом видео: кадр целиком повторяется посреди движения (а не статичная сцена).

    Кадр делится на grid×grid частей: повтор — когда не изменилась ни одна часть, движение — когда
    сильно изменилась хоть одна. Возвращает (долю повторов, отрезки в секундах, где их 3+ за секунду)."""
    fps = 30.0
    for s in probe(path)["streams"]:
        if s["codec_type"] == "video":
            n, d = (int(x) for x in s.get("avg_frame_rate", "30/1").split("/"))
            fps = n / d if n and d else 30.0
    import numpy as np
    size = width * height
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vf",
                          f"scale={width}:{height},format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    diffs, prev = [], None
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        cur = np.frombuffer(buf, np.uint8).astype(np.int16).reshape(height, width)
        if prev is not None:
            tiles = np.abs(cur - prev).reshape(grid, height // grid, grid, width // grid).mean(axis=(1, 3))
            diffs.append(float(tiles.max()))
        prev = cur
    p.wait()
    d = np.array(diffs)
    if len(d) < 3:
        return 0.0, []
    same, moving = d < 0.3, d > 1.0
    near = np.zeros_like(same)
    near[1:] |= moving[:-1]
    near[:-1] |= moving[1:]
    bad = same & near                                  # повтор кадра рядом с движением
    per_sec = np.add.reduceat(bad, np.arange(0, len(bad), int(round(fps))))
    spans = []
    for i, n in enumerate(per_sec):
        if n >= 3:                                     # 3+ повтора за секунду уже видно глазом
            if spans and spans[-1][1] == i:
                spans[-1] = (spans[-1][0], i + 1)
            else:
                spans.append((i, i + 1))
    return float(bad.mean()), [(float(a), float(b)) for a, b in spans]
