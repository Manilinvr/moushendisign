"""Превью выпуска №4: python projects/vypusk-04.thumbs.py

Кадры — из исходников клипов Бустера (work/vypusk-04/render/src_*.mp4 — появляются после `srez picks`).
Тема выпуска — медиабои Бустера: «Влад А4 подерётся только за 800 миллионов».
A — «эмоция + предмет»: Бустер схватился за голову, рядом светится боксёрская перчатка. «800 МЛН ЗА БОЙ?!»
B — «до → после»: серьёзный Бустер (синий фон) → хохочет (оранжевый). «БОЙ ЗА 800 МЛН».
За спиной у Бустера коробка «WIN» и знак Tornado (спонсоры) — вырезка убирает фон вместе с ними.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from srez import plates, poster  # noqa: E402

SRC = ROOT / "projects/work/vypusk-04/render"
OUT = ROOT / "projects/output"
TMP = ROOT / "projects/work/vypusk-04/thumbs"
TMP.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720


def frame(clip, t):
    """Кадр исходника клипа Бустера на секунде t клипа."""
    p = TMP / f"full_{clip}_{t}.jpg"
    if not p.exists():
        src = next(SRC.glob(f"src_vod2896940076-{clip}_*.mp4"))
        off = json.loads(src.with_suffix(".json").read_text())["offset"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(off + t), "-i", str(src), "-frames:v", "1",
                        "-q:v", "2", str(p)], check=True)
    return Image.open(p)


def cut(name, img, model="birefnet-portrait"):
    p = TMP / f"cut_{name}.png"
    if not p.exists():
        poster.cutout(img, model=model).save(p)
    return Image.open(p)


def glove(w=420, h=520, color=(225, 28, 36)):
    """Боксёрская перчатка: кулак, большой палец, белая полоса на манжете, объём — светом сверху слева."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    fist = (int(w * 0.10), int(h * 0.02), int(w * 0.97), int(h * 0.70))
    thumb = (int(w * 0.00), int(h * 0.30), int(w * 0.40), int(h * 0.66))
    cuff = (int(w * 0.20), int(h * 0.58), int(w * 0.88), int(h * 0.99))
    d.ellipse(fist, fill=255)
    d.ellipse(thumb, fill=255)
    d.rounded_rectangle(cuff, radius=int(w * 0.08), fill=255)
    # свет: ярче сверху слева, темнее к краям — как у шара
    cx, cy = w * 0.42, h * 0.22
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    shade = np.clip(1.25 - dist * 1.35, 0.35, 1.25)[..., None]
    rgb = np.clip(np.array(color, np.float32) * shade, 0, 255)
    im = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    im.putalpha(mask)
    d = ImageDraw.Draw(im)
    lw = max(4, w // 70)
    dark = (90, 6, 10, 255)
    d.arc(thumb, start=200, end=40, fill=dark, width=lw)                       # шов большого пальца
    band = (cuff[0], int(h * 0.68), cuff[2], int(h * 0.80))
    d.rectangle(band, fill=(245, 245, 245, 255))                                 # белая полоса манжеты
    d.line([(cuff[0], band[1]), (cuff[2], band[1])], fill=dark, width=lw)
    d.line([(cuff[0], band[3]), (cuff[2], band[3])], fill=dark, width=lw)
    d.line([(cuff[0] + lw, cuff[1] + lw), (cuff[2] - lw, cuff[1] + lw)], fill=dark, width=lw)
    # блик
    hl = Image.new("L", (w, h), 0)
    ImageDraw.Draw(hl).ellipse((int(w * 0.30), int(h * 0.07), int(w * 0.62), int(h * 0.24)), fill=150)
    hl = hl.filter(ImageFilter.GaussianBlur(w // 22))
    white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    white.putalpha(Image.fromarray(np.minimum(np.asarray(hl), np.asarray(mask))))
    im.alpha_composite(white)
    return im


facepalm = cut("facepalm", frame("k04", 1.5).crop((300, 60, 1500, 1080)))     # схватился за голову
laugh = cut("laugh", frame("k04", 55.3).crop((350, 60, 1550, 1080)))          # хохочет
serious = cut("serious", frame("k03", 0.5).crop((420, 60, 1500, 1080)))        # серьёзно смотрит в камеру


def face_box(rgba):
    """Примерный центр головы вырезки: верхняя часть непрозрачной области."""
    a = np.asarray(rgba.getchannel("A")) > 128
    ys, xs = np.nonzero(a)
    top = ys.min()
    band = a[top: top + max(1, (ys.max() - top) // 4)]
    bx = np.nonzero(band.any(axis=0))[0]
    return (bx.min() + bx.max()) // 2, top + (ys.max() - top) // 6


# ---------------------------------------------------------------- A: «800 МЛН ЗА БОЙ?!»
im = poster.burst((W, H), center=(880, 300), inner=(255, 120, 30), outer=(52, 6, 0))
P = poster.pop(facepalm, color=1.15, contrast=1.1)
s = 1.12
P = P.resize((int(P.width * s), int(P.height * s)), Image.LANCZOS)
fx, fy = face_box(P)
poster.place(im, P, (900 - fx, 250 - fy), outline=9)
poster.place(im, glove(), (110, 105), height=360, outline=8, glow=(255, 70, 40), glow_size=40, angle=-18)
poster.finish(im, nick="buster", words=["800 МЛН", "ЗА БОЙ?!"]).save(
    OUT / "vypusk-04_превью_A_800_млн_за_бой.jpg", quality=93)

# ---------------------------------------------------------------- B: «до → после»
im = poster.burst((W, H), center=(300, 260), inner=(40, 120, 210), outer=(4, 10, 26))
right = poster.burst((W, H), center=(980, 260), inner=(255, 140, 20), outer=(48, 10, 0))
x0, x1 = 700, 600
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).polygon([(x0, 0), (W, 0), (W, H), (x1, H)], fill=255)
im.paste(right, (0, 0), mask)
S = poster.pop(serious)
fx, fy = face_box(S)
left = Image.new("RGBA", (W, H), (0, 0, 0, 0))
poster.place(left, S, (320 - fx, 240 - fy), outline=9)
lmask = Image.new("L", (W, H), 0)
ImageDraw.Draw(lmask).polygon([(0, 0), (x0, 0), (x1, H), (0, H)], fill=255)
im.paste(left, (0, 0), Image.fromarray(np.minimum(np.asarray(lmask), np.asarray(left.getchannel("A")))))
band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(band).polygon([(x0 - 34, 0), (x0 + 34, 0), (x1 + 34, H), (x1 - 34, H)], fill=plates.LIME + (255,))
im.alpha_composite(band)
L = poster.pop(laugh, color=1.15, contrast=1.1)
s = 1.0
L = L.resize((int(L.width * s), int(L.height * s)), Image.LANCZOS)
fx, fy = face_box(L)
poster.place(im, L, (985 - fx, 250 - fy), outline=9)
poster.arrow(im, (560, 450), (760, 450), width=30)
poster.finish(im, nick="buster", words=["БОЙ ЗА", "800 МЛН"]).save(
    OUT / "vypusk-04_превью_B_бой_за_800_млн.jpg", quality=93)
print("Готово:", OUT)
