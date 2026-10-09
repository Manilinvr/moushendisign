"""Превью выпуска №2 (образец для следующих): python projects/vypusk-02.thumbs.py

Кадры берутся из исходников клипов RavshanN (work/vypusk-02/render/src_*.mp4 — появляются после `srez picks`).
A — «эмоция + предмет»: кричащий Равшан и светящийся шлем Железного человека.
B — «до → после»: в капюшоне (синий фон) → в шлеме (красный фон), между ними лаймовый срез и стрелка.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from srez import plates, poster  # noqa: E402

SRC = ROOT / "projects/work/vypusk-02/render"
OUT = ROOT / "projects/output"
TMP = ROOT / "projects/work/vypusk-02/thumbs"
TMP.mkdir(parents=True, exist_ok=True)


def frame(clip, t):
    p = TMP / f"{clip[:20]}_{t}.jpg"
    if not p.exists():
        src = next(SRC.glob(f"src_{clip}_*.mp4"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1", "-q:v", "2",
                        str(p)], check=True)
    return Image.open(p)


def cut(name, img, **kw):
    p = TMP / f"cut_{name}.png"
    if not p.exists():
        poster.cutout(img, **kw).save(p)
    return Image.open(p)


# ---------------------------------------------------------------- A: «ЖЕЛЕЗНЫЙ РАВШАН?!»
shout = cut("shout", frame("vod2895984355-06", 30.25))
shout = poster.erase(shout, [(650, 150), (800, 145), (940, 145), (940, 236), (805, 236), (760, 252), (705, 290),
                             (650, 335)])                       # мяч и футболка на полке за кулаком
helmet = cut("helmet", frame("vod2895984355-06", 2.0).crop((560, 180, 1160, 700)))
helmet = poster.erase(helmet, [(185, 0), (262, 0), (262, 46), (185, 46)])   # вешалка над шлемом
s = 1.25
P = poster.pop(shout)
P = P.resize((int(P.width * s), int(P.height * s)), Image.LANCZOS)
im = poster.burst((1280, 720), center=(880, 300), inner=(235, 40, 22), outer=(22, 4, 8))
poster.place(im, P, (880 - int(1190 * s), 720 - int(560 * s)), outline=9)
poster.place(im, poster.pop(helmet, color=1.25, contrast=1.15), (175, 95), height=420, outline=8,
             glow=(255, 190, 40), glow_size=34, angle=8)
poster.finish(im, nick="RavshanN", words=["ЖЕЛЕЗНЫЙ", "РАВШАН?!"]).save(
    OUT / "vypusk-02_превью_A_железный_равшан.jpg", quality=93)

# ---------------------------------------------------------------- B: «НИКТО НЕ УЗНАЕТ»
hood = cut("hood", frame("vod2895984355-05", 36.5).crop((720, 0, 1460, 760)))
hood = poster.erase(hood, [(522, 248), (618, 248), (618, 348), (592, 342), (552, 302), (522, 288)])  # голова с постера
a = np.asarray(hood).copy()                # тёмный хвостик от неё у плеча (куртка светлая — тёмное здесь лишнее)
reg = a[250:370, 490:630]
reg[..., 3][reg[..., :3].mean(axis=2) < 90] = 0
hood = Image.fromarray(a)
iron = cut("iron", frame("vod2895984355-06", 0.25).crop((380, 0, 1420, 760)))
W, H = 1280, 720
im = poster.burst((W, H), center=(330, 260), inner=(40, 120, 190), outer=(6, 12, 24))
right = poster.burst((W, H), center=(960, 260), inner=(235, 40, 22), outer=(22, 4, 8))
x0, x1 = 700, 600                          # косой стык сверху и снизу
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).polygon([(x0, 0), (W, 0), (W, H), (x1, H)], fill=255)
im.paste(right, (0, 0), mask)
Hd = poster.pop(hood)
poster.place(im, Hd, (-20, 60), height=int(Hd.height * 1.12), outline=9)
band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(band).polygon([(x0 - 34, 0), (x0 + 34, 0), (x1 + 34, H), (x1 - 34, H)], fill=plates.LIME + (255,))
im.alpha_composite(band)
Ir = poster.pop(iron, color=1.25)
poster.place(im, Ir, (650, 115), height=int(Ir.height * 0.86), outline=9)
poster.arrow(im, (545, 455), (775, 455), width=30)
poster.finish(im, nick="RavshanN", words=["НИКТО НЕ", "УЗНАЕТ"]).save(
    OUT / "vypusk-02_превью_B_никто_не_узнает.jpg", quality=93)
print("Готово:", OUT)
