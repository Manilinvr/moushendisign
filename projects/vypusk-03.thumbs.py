"""Превью выпуска №3: python projects/vypusk-03.thumbs.py

Кадры — из исходников клипов с ChatGPT (work/vypusk-03/render/src_*.mp4 — появляются после `srez picks`).
Стрим очень тёмный (чёрная рубашка на тёмном фоне), поэтому кадр сначала осветляется, а вырезка —
моделью для портретов birefnet-portrait: isnet-general-use здесь теряет тело и прихватывает кресло.
A — «эмоция + предмет»: стример хохочет, рядом светится телефон с голосовым ChatGPT. «ТЫ — ПЫЛЕСОС!»
B — «до → после»: серьёзно говорит в телефон (синий фон) → лежит от смеха (оранжевый). «ИИ СОШЁЛ С УМА».
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from srez import plates, poster  # noqa: E402

SRC = ROOT / "projects/work/vypusk-03/render"
OUT = ROOT / "projects/output"
TMP = ROOT / "projects/work/vypusk-03/thumbs"
TMP.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720


def frame(clip, t):
    """Кадр исходника клипа на секунде t клипа."""
    p = TMP / f"full_{clip}_{t}.jpg"
    if not p.exists():
        src = next(SRC.glob(f"src_vod2887268104-{clip}_*.mp4"))
        off = json.loads(src.with_suffix(".json").read_text())["offset"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(off + t), "-i", str(src), "-frames:v", "1",
                        "-q:v", "2", str(p)], check=True)
    return Image.open(p)


def bright(im):
    """Тени вытянуты гаммой — иначе чёрная рубашка сливается с тёмной комнатой."""
    a = (np.asarray(im.convert("RGB")).astype(np.float32) / 255) ** 0.55
    return ImageEnhance.Contrast(Image.fromarray((a * 255).astype(np.uint8))).enhance(1.15)


def cut(name, img):
    p = TMP / f"cutp_{name}.png"
    if not p.exists():
        poster.cutout(bright(img), model="birefnet-portrait").save(p)
    return Image.open(p)


def phone(w=250, h=470):
    """Телефон с голосовым режимом ChatGPT: тёмный экран и светящийся шар (без логотипов)."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=42, fill=(22, 22, 28, 255))
    d.rounded_rectangle((12, 12, w - 13, h - 13), radius=32, fill=(6, 8, 18, 255))
    d.rounded_rectangle((w // 2 - 36, 24, w // 2 + 36, 44), radius=10, fill=(22, 22, 28, 255))   # вырез камеры
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w / 2, h * 0.46, w * 0.27
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = np.clip(1 - dist / r, 0, 1) ** 0.6
    halo = np.clip(1 - dist / (r * 2.3), 0, 1) ** 2 * 0.8
    rgb = np.zeros((h, w, 3), np.float32)
    rgb += core[..., None] * np.array([235, 250, 255]) + halo[..., None] * np.array([40, 170, 255])
    orb = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    mask = Image.fromarray((np.clip(core + halo, 0, 1) * 255).astype(np.uint8))
    screen = Image.new("L", (w, h), 0)
    ImageDraw.Draw(screen).rounded_rectangle((12, 12, w - 13, h - 13), radius=32, fill=255)
    im.paste(orb, (0, 0), Image.fromarray(np.minimum(np.asarray(mask), np.asarray(screen))))
    for i, bh in enumerate((26, 54, 80, 54, 26)):            # «волна» голоса под шаром
        x = int(cx - 70 + i * 35)
        d.rounded_rectangle((x - 9, int(h * 0.78) - bh // 2, x + 9, int(h * 0.78) + bh // 2), radius=9,
                            fill=(120, 210, 255, 255))
    return im


laugh = cut("laugh", frame("k18", 87).crop((480, 120, 1500, 1080)))       # хохочет над «подкупом»
serious = cut("serious", frame("k18", 51).crop((520, 120, 1500, 1080)))   # серьёзно говорит в телефон
serious = poster.erase(serious, [(0, 560), (330, 560), (330, 960), (0, 960)])   # стойка микрофона слева

# ---------------------------------------------------------------- A: «ТЫ — ПЫЛЕСОС!»
im = poster.burst((W, H), center=(860, 300), inner=(255, 150, 20), outer=(48, 10, 0))
L = poster.pop(laugh, color=1.15, contrast=1.1)
s = 1.18
L = L.resize((int(L.width * s), int(L.height * s)), Image.LANCZOS)
poster.place(im, L, (880 - int(640 * s), 330 - int(400 * s)), outline=9)
poster.place(im, phone(), (150, 120), height=400, outline=8, glow=(40, 190, 255), glow_size=36, angle=12)
poster.finish(im, nick="frametamer666", words=["ТЫ —", "ПЫЛЕСОС!"]).save(
    OUT / "vypusk-03_превью_A_ты_пылесос.jpg", quality=93)

# ---------------------------------------------------------------- B: «ИИ СОШЁЛ С УМА» (до → после)
im = poster.burst((W, H), center=(300, 260), inner=(40, 120, 210), outer=(4, 10, 26))
right = poster.burst((W, H), center=(980, 260), inner=(255, 140, 20), outer=(48, 10, 0))
x0, x1 = 700, 600
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).polygon([(x0, 0), (W, 0), (W, H), (x1, H)], fill=255)
im.paste(right, (0, 0), mask)
S = poster.pop(serious)
left = Image.new("RGBA", (W, H), (0, 0, 0, 0))
poster.place(left, S, (330 - 564, 250 - 502), outline=9)       # лицо — в (330, 250)
lmask = Image.new("L", (W, H), 0)
ImageDraw.Draw(lmask).polygon([(0, 0), (x0, 0), (x1, H), (0, H)], fill=255)
im.paste(left, (0, 0), Image.fromarray(np.minimum(np.asarray(lmask), np.asarray(left.getchannel("A")))))
band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(band).polygon([(x0 - 34, 0), (x0 + 34, 0), (x1 + 34, H), (x1 - 34, H)], fill=plates.LIME + (255,))
im.alpha_composite(band)
L = poster.pop(laugh, color=1.15, contrast=1.1)
s = 1.02
L = L.resize((int(L.width * s), int(L.height * s)), Image.LANCZOS)
poster.place(im, L, (975 - int(682 * s), 270 - int(400 * s)), outline=9)   # лицо — в (975, 270)
poster.arrow(im, (560, 450), (760, 450), width=30)
poster.finish(im, nick="frametamer666", words=["ИИ", "СОШЁЛ С УМА"]).save(
    OUT / "vypusk-03_превью_B_ии_сошёл_с_ума.jpg", quality=93)
print("Готово:", OUT)
