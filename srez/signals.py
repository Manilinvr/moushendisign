"""Сигналы «здесь что-то происходит»: громкость звука и активность чата.

Всё считается на сетке с шагом HOP секунд и переводится в z-оценки
относительно локального фона, поэтому тихий и громкий стример
сравниваются честно.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HOP = 0.5          # шаг сетки, сек
SR = 8000          # частота, до которой ужимаем звук для анализа


# ---------------------------------------------------------------- звук

def loudness(path) -> np.ndarray:
    """Громкость (дБ) каждые HOP секунд. Декодируется только звук — это быстро."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(SR), "-f", "s16le", "-"]
    n = int(SR * HOP)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, rest = [], np.zeros(0, dtype=np.int16)
    chunk = n * 2 * 2000
    while True:
        buf = proc.stdout.read(chunk)
        if not buf:
            break
        x = np.concatenate([rest, np.frombuffer(buf, dtype=np.int16)])
        k = len(x) // n
        frames = x[: k * n].reshape(k, n).astype(np.float32) / 32768.0
        out.append(10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10))
        rest = x[k * n:]
    proc.wait()
    if not out:
        raise RuntimeError(f"Не удалось прочитать звук из {path}")
    return np.concatenate(out)


# ---------------------------------------------------------------- чат

# Маркеры реакций зрителей: эмоуты Twitch/7TV и русские слова
LEXICON = {
    "funny": ["kekw", "lul", "lulw", "omegalul", "icant", "xdd", "kekl", "pepelaugh", "lmao",
              "ахах", "хаха", "хахах", "ору", "орнул", "ржу", "ржака", "смешно", "угар",
              "😂", "🤣", "хд", "xd"],
    "scary": ["monkas", "monkaw", "monkaomega", "d:", "waytoodank", "скример", "страшно",
              "жуть", "жутко", "крип", "испуг", "напугал", "😱", "😨", "😰"],
    "hype": ["pog", "pogchamp", "poggers", "pogu", "pag", "ez", "clap", "gg", "wow", "goat",
             "вау", "ого", "имба", "жесть", "легенда", "красава", "🔥", "💪"],
    "shock": ["wtf", "что", "чё", "шо", "???", "aware", "stare", "modcheck", "weirdchamp",
              "cringe", "кринж", "бан", "ban", "sadge", "o7", "😳", "🤨"],
}
LABELS = {"funny": "угар", "scary": "жуть", "hype": "эпик", "shock": "шок", "talk": "момент"}

_token = re.compile(r"[\w:?]+|[^\w\s]", re.UNICODE)


def classify(text: str) -> str | None:
    t = text.lower()
    words = set(_token.findall(t))
    best, hits = None, 0
    for cat, keys in LEXICON.items():
        h = sum(1 for k in keys if (k in words) or (len(k) > 3 and k in t) or (not k.isalnum() and k in t))
        if h > hits:
            best, hits = cat, h
    return best


def load_chat(path) -> list[tuple[float, str]]:
    """Чат в нашем JSONL или в JSON от TwitchDownloader."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    msgs = []
    if raw.lstrip().startswith("{") and '"comments"' in raw[:2000]:
        for c in json.loads(raw)["comments"]:
            msgs.append((float(c["content_offset_seconds"]), c["message"]["body"]))
    else:
        for line in raw.splitlines():
            if line.strip():
                m = json.loads(line)
                msgs.append((float(m["t"]), m["text"]))
    msgs.sort()
    return msgs


def chat_curves(msgs, length: int):
    """Сообщений в секунду и доли реакций по категориям на сетке HOP."""
    cats = list(LEXICON)
    rate = np.zeros(length)
    share = np.zeros((len(cats), length))
    for t, text in msgs:
        i = int(t / HOP)
        if 0 <= i < length:
            rate[i] += 1
            c = classify(text)
            if c:
                share[cats.index(c), i] += 1
    rate_s = smooth(rate, 10) / HOP
    share_s = np.array([smooth(s, 10) / HOP for s in share])
    return rate_s, share_s, cats


# ---------------------------------------------------------------- нормировка

def _rolling_baseline(x, block_sec=60, span_blocks=5):
    """Медианный фон за ~5 минут вокруг каждой точки (быстро, через блоки)."""
    b = max(1, int(block_sec / HOP))
    nb = max(1, int(np.ceil(len(x) / b)))
    pad = np.pad(x, (0, nb * b - len(x)), mode="edge")
    med = np.median(pad.reshape(nb, b), axis=1)
    h = span_blocks // 2
    smooth = np.array([np.median(med[max(0, i - h): i + h + 1]) for i in range(nb)])
    centers = (np.arange(nb) + 0.5) * b
    return np.interp(np.arange(len(x)), centers, smooth)


def zscore(x):
    base = _rolling_baseline(x)
    d = x - base
    mad = np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-6
    return d / mad


def smooth(x, sec):
    """Скользящее среднее; края дополняются крайним значением, а не нулём."""
    w = max(1, int(sec / HOP))
    if w == 1:
        return np.asarray(x, dtype=float)
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def lag(x, sec):
    """Значение sec секунд назад."""
    k = int(sec / HOP)
    return np.concatenate([np.full(k, x[0]), x[:-k]]) if 0 < k < len(x) else x


def squash(x, top=10):
    """Мягкий потолок: сильные моменты остаются различимы, но один выброс не задавит всё."""
    return np.where(x > 0, top * np.tanh(x / top), np.maximum(x, -3))


def audio_score(db):
    level = zscore(smooth(db, 2))
    # резкий скачок громкости: крик, смех, взрыв
    jump = np.maximum(0, smooth(db, 1) - lag(smooth(db, 4), 3))
    jump = jump / (np.median(jump[jump > 0]) + 1e-6) if np.any(jump > 0) else jump
    return squash(0.75 * level + 0.25 * np.clip(jump, 0, 8))


def chat_score(rate, share):
    react = share.sum(axis=0) / (rate + 1e-6)          # доля сообщений-реакций
    z = zscore(np.log1p(rate))
    return squash(z + 1.5 * np.clip(react, 0, 1))
