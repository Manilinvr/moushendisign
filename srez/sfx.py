"""Звуки СРЕЗа и музыка.

Эффекты синтезируются кодом (numpy), поэтому никаких лицензий и Content ID.
Музыка — треки Кевина Маклауда (incompetech.com) по лицензии CC BY 4.0:
можно монетизировать на YouTube, автор указывается в описании (см. MUSIC).
"""
import math
import urllib.request
import wave
from pathlib import Path

import numpy as np

SR = 48000

MUSIC = {
    "funkorama": ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Funkorama.mp3",
                  '"Funkorama" Kevin MacLeod (incompetech.com)\n'
                  "Licensed under Creative Commons: By Attribution 4.0 License\n"
                  "http://creativecommons.org/licenses/by/4.0/"),
}


def _t(sec):
    return np.arange(int(SR * sec)) / SR


def _env(n, attack, release):
    """Огибающая: быстрый подъём и плавный спад (секунды)."""
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(attack, 1e-4))
    tail = t > (n / SR - release)
    e[tail] *= np.clip((n / SR - t[tail]) / release, 0, 1)
    return e


def _bandpass_sweep(x, f0, f1, q=1.4):
    """Шум через полосовой фильтр, частота которого плывёт от f0 к f1."""
    n = len(x)
    out = np.zeros(n)
    z1 = z2 = 0.0
    for i in range(n):
        f = f0 * (f1 / f0) ** (i / n)
        w = 2 * math.pi * f / SR
        alpha = math.sin(w) / (2 * q)
        b0, a0, a1, a2 = alpha, 1 + alpha, -2 * math.cos(w), 1 - alpha
        y = (b0 * x[i] - b0 * (x[i - 2] if i > 1 else 0) - a1 * z1 - a2 * z2) / a0
        out[i] = y
        z2, z1 = z1, y
    return out


def whoosh(sec=0.42):
    rng = np.random.default_rng(7)
    x = _bandpass_sweep(rng.standard_normal(int(SR * sec)), 350, 4200)
    x *= _env(len(x), sec * 0.45, sec * 0.5)
    pan = np.linspace(0, 1, len(x))          # пролетает слева направо
    return np.stack([x * np.cos(pan * math.pi / 2), x * np.sin(pan * math.pi / 2)], 1)


def boom(sec=0.75):
    t = _t(sec)
    f = 38 + 34 * np.exp(-t * 9)
    x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
    click = np.random.default_rng(3).standard_normal(len(t)) * np.exp(-t / 0.006) * 0.3
    click = np.convolve(click, np.ones(12) / 12, mode="same")    # глуше: без верхов
    x = np.tanh(1.8 * (x + click))
    return np.stack([x, x], 1)


def ding(sec=1.0):
    t = _t(sec)
    x = sum(a * np.sin(2 * math.pi * f * t) * np.exp(-t / d)
            for f, a, d in ((1760, 1.0, 0.35), (3527, 0.35, 0.2), (5290, 0.12, 0.1)))
    x *= np.minimum(1, t / 0.003)
    return np.stack([x, x], 1)


def boing(sec=0.6):
    t = _t(sec)
    f = 170 * (1 + 0.45 * np.exp(-t * 5) * np.sin(2 * math.pi * 13 * t)) * (1 - 0.25 * t)
    x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25) * np.minimum(1, t / 0.005)
    return np.stack([x, x], 1)


def logo(sec=1.4):
    """Звуковой логотип: «вжух», удар и яркий аккорд."""
    out = np.zeros((int(SR * sec), 2))
    w = whoosh(0.32)
    out[:len(w)] += w * 0.8
    at = int(SR * 0.3)
    b = boom(0.8)[: len(out) - at]
    out[at:at + len(b)] += b * 0.7
    t = _t(sec - 0.3)
    chord = sum(sum(np.sin(2 * math.pi * f * h * t) / h ** 1.6 for h in range(1, 6))
                for f in (523.25, 659.25, 783.99, 1174.66)) / 4
    chord *= np.exp(-t / 0.45) * np.minimum(1, t / 0.004)
    out[at:at + len(chord)] += np.stack([chord, chord], 1) * 0.55
    return out


SOUNDS = {"whoosh": whoosh, "boom": boom, "ding": ding, "boing": boing, "logo": logo}


def path(name: str, folder: Path, gain_db: float = -6.0) -> Path:
    """WAV со звуком (генерируется один раз), пик нормирован к gain_db."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.wav"
    if out.exists():
        return out
    x = SOUNDS[name]()
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (gain_db / 20)
    with wave.open(str(out), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((x * 32767).astype("<i2").tobytes())
    return out


def music(name: str, folder: Path) -> tuple[Path, str]:
    """Трек из каталога MUSIC (скачивается один раз) и строка с указанием автора для описания."""
    url, credit = MUSIC[name]
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.mp3"
    if not out.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out.with_suffix(".part").write_bytes(r.read())
        out.with_suffix(".part").replace(out)
    return out, credit
