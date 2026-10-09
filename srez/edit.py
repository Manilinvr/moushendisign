"""Монтажные решения внутри клипа: зум на реакцию, подпись-реакция, вырезка пауз.

Всё считается по звуку клипа и словам из распознавания (их уже сделал censor),
лицо для зума ищет детектор YuNet. Решения кэшируются в work/<проект>/edit/,
а в candidates.json их можно поправить вручную (zoom_at, caption, sfx у момента).
"""
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from .censor import is_bad
from .media import frame

FPS_GRID = 30
YUNET = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
         "face_detection_yunet/face_detection_yunet_2023mar.onnx")
ZOOM_LEN = 1.9              # сколько держится зум, сек
ZOOM_MIN, ZOOM_MAX = 1.3, 2.2


def _snap(t):
    return round(t * FPS_GRID) / FPS_GRID


def _rms_db(path, offset, dur, hop=0.05):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}",
                          "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    n = int(16000 * hop)
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    return 10 * np.log10((x[: k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-10)


# ---------------------------------------------------------------- лицо

_det = None


def _detector(cache_dir: Path):
    global _det
    if _det is None:
        import cv2
        model = cache_dir / "yunet.onnx"
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(YUNET, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                model.write_bytes(r.read())
        _det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6)
    return _det


def zoom_box(img_path, cache_dir: Path):
    """Кадр 16:9 вокруг самого крупного лица (в пикселях исходника) или None."""
    import cv2
    im = cv2.imread(str(img_path))
    if im is None:
        return None
    h0, w0 = im.shape[:2]
    s = 960 / w0
    small = cv2.resize(im, (960, int(h0 * s)))
    det = _detector(cache_dir)
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None or not len(faces):
        return None
    fx, fy, fw, fh = (v / s for v in max(faces, key=lambda f: f[2] * f[3] * f[14])[:4])
    bh = min(max(fh * 4.2, h0 / ZOOM_MAX), h0 / ZOOM_MIN)
    bw = bh * 16 / 9
    cx, cy = fx + fw / 2, fy + fh * 0.8
    margin = bw * 0.015                    # запас под тряску
    x = min(max(cx - bw / 2, margin), w0 - bw - margin)
    y = min(max(cy - bh / 2, margin), h0 - bh - margin)
    even = lambda v: int(v) // 2 * 2       # noqa: E731
    return [even(x), even(y), even(bw), even(bh)]


# ---------------------------------------------------------------- подпись

def _clean(word):
    return re.sub(r"[^\wёЁ?!-]", "", word.strip()).strip("-").upper()


def bleep_text(word):
    """Мат в подписи звёздочками: БЛЯТЬ → Б**ТЬ."""
    w = _clean(word)
    letters = re.sub(r"[?!-]", "", w)
    if len(letters) < 3:
        return letters[:1] + "*" * (len(letters) - 1) + w[len(letters):]
    return letters[0] + "*" * (len(letters) - 3) + letters[-2:] + w[len(letters):]


def caption_words(words, tp, before=1.2, after=0.9, n=3):
    """До n слов, сказанных вокруг пика реакции: подпись-панчлайн."""
    near = [w for w in words if w[1] > tp - before and w[0] < tp + after and _clean(w[2])]
    if not near:
        return ""
    near = near[-n:]
    return " ".join(bleep_text(w[2]) if is_bad(w[2]) else _clean(w[2]) for w in near)


# ---------------------------------------------------------------- паузы

def keep_intervals(rms, words, dur, min_gap, edge=0.2, protect=(), hop=0.05):
    """Куски клипа, которые оставляем: вырезаем тишину без слов длиннее min_gap."""
    if min_gap <= 0 or len(rms) == 0:
        return [(0.0, _snap(dur))]
    quiet = rms < np.median(rms) - 8
    t = np.arange(len(rms)) * hop
    for a, b, _ in words:                  # слова — не тишина, даже если тихие
        quiet[(t >= a - 0.1) & (t <= b + 0.1)] = False
    for a, b in protect:
        quiet[(t >= a) & (t <= b)] = False
    cuts, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            a, b = t[i], min(dur, t[j - 1] + hop)
            if b - a > min_gap and a > 0.3 and b < dur - 0.3:
                cuts.append((_snap(a + edge), _snap(b - edge)))
            i = j
        else:
            i += 1
    keep, cur = [], 0.0
    for a, b in cuts:
        if a > cur:
            keep.append((cur, a))
        cur = b
    keep.append((cur, _snap(dur)))
    return [(a, b) for a, b in keep if b - a >= 1 / FPS_GRID]


def remap(t, keep):
    """Время в исходном клипе → время после вырезки пауз."""
    out = 0.0
    for a, b in keep:
        if t < a:
            return out
        if t <= b:
            return out + t - a
        out += b - a
    return out


# ---------------------------------------------------------------- всё вместе

def decide(m, path, offset, words, settings, cache: Path, assets: Path) -> dict:
    """Решения для клипа m (время — от начала клипа)."""
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
    else:
        dur = m.length
        cache.parent.mkdir(parents=True, exist_ok=True)
        rms = _rms_db(path, offset, dur)
        tp = None
        if settings.zoom or settings.captions:
            # пик реакции: самый громкий полсекундный отрезок рядом с найденным пиком
            guess = min(max(m.peak - m.start, 1.0), dur - 2.0)
            lo, hi = int(max(0.5, guess - 2.5) / 0.05), int(min(dur - ZOOM_LEN - 0.5, guess + 2.5) / 0.05)
            if hi > lo + 10 and len(rms) > hi + 10:
                win = np.convolve(rms, np.ones(10) / 10, mode="same")
                tp = _snap((lo + int(np.argmax(win[lo:hi]))) * 0.05 - 0.25)
        d = {"tp": tp, "boxes": {}, "auto_caption": caption_words(words, tp) if tp is not None else "",
             "rms": [round(float(v), 1) for v in rms]}
    # ручные правки из candidates.json важнее
    tp = d["tp"]
    if m.zoom_at is not None:
        tp = None if m.zoom_at < 0 else _snap(m.zoom_at)
    zoom = None
    if tp is not None and settings.zoom and m.length - 0.4 - tp >= 0.8:
        key = f"{tp:.2f}"
        if key not in d["boxes"]:        # лицо ищем в тот момент, где будет зум
            img = cache.with_suffix(".jpg")
            frame(path, offset + tp + 0.3, img)
            d["boxes"][key] = zoom_box(img, assets)
        if d["boxes"][key]:
            zoom = (tp, _snap(min(tp + ZOOM_LEN, m.length - 0.4)), d["boxes"][key])
    cache.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    text = "" if m.caption == "-" else (m.caption or d["auto_caption"])
    end = zoom[1] if zoom else tp + ZOOM_LEN if tp is not None else 0
    caption = (max(0.0, tp - 0.1), min(end, m.length - 0.3), text) if (text and tp is not None
                                                                       and settings.captions) else None
    protect = [(zoom[0] - 0.3, zoom[1] + 0.3)] if zoom else []
    if caption:
        protect.append((caption[0] - 0.3, caption[1] + 0.3))
    keep = keep_intervals(np.array(d["rms"]), words, m.length, settings.jumpcut, protect=protect)
    sfx = [("boom", zoom[0])] if zoom else []
    sfx += [(name, float(t)) for name, t in (m.sfx or [])]
    return {"zoom": zoom, "caption": caption, "keep": keep, "sfx": sfx,
            "dur": round(sum(b - a for a, b in keep), 3)}
