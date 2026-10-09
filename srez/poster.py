"""Кликабельные превью «как у больших нарезок»: стример вырезан из кадра, белая обводка и тень,
яркий фон с лучами, предмет-«крючок» со свечением, крупные слова. Ник и логотип — по правилам канала.

    from srez import poster
    face = poster.cutout(Image.open("кадр.jpg").crop(box))          # вырезка из фона (rembg)
    im = poster.burst((1280, 720), center=(900, 330), inner=(235, 40, 25), outer=(18, 4, 6))
    poster.place(im, face, (560, 0), height=820, outline=10)       # стикер с обводкой и тенью
    poster.finish(im, nick="RavshanN", words=["ЖЕЛЕЗНЫЙ", "РАВШАН?!"]).save("превью.jpg")

Вырезка (rembg, модель скачивается при первом запуске ~180 МБ) берётся из исходника клипа:
маски рекламы в нём не нужны, если в рамку не попадает угол с баннером.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import plates

_sessions = {}


def cutout(img: Image.Image, model="isnet-general-use", keep_largest=True) -> Image.Image:
    """Человек или предмет без фона (RGBA). Чистит полупрозрачный «туман» и мелкие лишние куски."""
    from rembg import new_session, remove
    if model not in _sessions:
        _sessions[model] = new_session(model)
    out = remove(img.convert("RGB"), session=_sessions[model])
    a = np.asarray(out.getchannel("A")).astype(np.float32)
    a[a < 90] = 0
    if keep_largest:                       # остаётся только самый большой кусок (без голов с постеров и т.п.)
        from scipy import ndimage
        lab, n = ndimage.label(a > 0)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
            a[lab != 1 + int(np.argmax(sizes))] = 0
    out.putalpha(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)))
    return out


def erase(rgba: Image.Image, polygon) -> Image.Image:
    """Стирает часть вырезки (многоугольник в пикселях вырезки) — убрать лишний предмет за спиной."""
    a = rgba.getchannel("A")
    ImageDraw.Draw(a).polygon(polygon, fill=0)
    out = rgba.copy()
    out.putalpha(a)
    return out


def pop(rgba: Image.Image, color=1.18, contrast=1.12, sharp=1.6) -> Image.Image:
    """Сочнее и резче — чтобы лицо читалось на маленькой картинке."""
    a = rgba.getchannel("A")
    rgb = ImageEnhance.Color(rgba.convert("RGB")).enhance(color)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Sharpness(rgb).enhance(sharp)
    rgb.putalpha(a)
    return rgb


def burst(size, center, inner, outer, rays=18, ray_alpha=38, vignette=True) -> Image.Image:
    """Фон: радиальный градиент от центра и расходящиеся лучи (как в превью больших нарезок)."""
    W, H = size
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(W, H) * 1.6
    t = np.clip(d, 0, 1)[..., None] ** 0.9
    rgb = (np.array(inner, np.float32) * (1 - t) + np.array(outer, np.float32) * t).astype(np.uint8)
    im = Image.fromarray(rgb, "RGB").convert("RGBA")
    if rays:
        lay = Image.new("L", size, 0)
        dr = ImageDraw.Draw(lay)
        R = math.hypot(W, H)
        for i in range(rays):
            a0 = 2 * math.pi * i / rays
            a1 = a0 + math.pi / rays
            dr.polygon([(cx, cy), (cx + R * math.cos(a0), cy + R * math.sin(a0)),
                        (cx + R * math.cos(a1), cy + R * math.sin(a1))], fill=ray_alpha)
        lay = lay.filter(ImageFilter.GaussianBlur(2))
        white = Image.new("RGBA", size, (255, 255, 255, 0))
        white.putalpha(lay)
        im.alpha_composite(white)
    if vignette:
        v = np.clip((d - 0.55) * 1.4, 0, 0.75)
        dark = Image.new("RGBA", size, (0, 0, 0, 0))
        dark.putalpha(Image.fromarray((v * 255).astype(np.uint8)))
        im.alpha_composite(dark)
    return im


def place(im: Image.Image, rgba: Image.Image, xy, *, height=None, width=None, outline=10, outline_color=(255, 255, 255),
          shadow=18, glow=None, glow_size=40, angle=0):
    """Кладёт вырезку «стикером»: тень, свечение (цвет glow), обводка, сама картинка. xy — левый верхний угол."""
    W, H = im.size
    if height or width:
        s = (height / rgba.height) if height else (width / rgba.width)
        rgba = rgba.resize((max(1, int(rgba.width * s)), max(1, int(rgba.height * s))), Image.LANCZOS)
    if angle:
        rgba = rgba.rotate(angle, resample=Image.BICUBIC, expand=True)
    pad = max(outline, shadow, glow_size if glow else 0) * 3
    a = Image.new("L", (rgba.width + 2 * pad, rgba.height + 2 * pad), 0)
    a.paste(rgba.getchannel("A"), (pad, pad))
    x, y = xy[0] - pad, xy[1] - pad
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))

    def put(mask, color, dx=0, dy=0):
        c = Image.new("RGBA", mask.size, color + (0,))
        c.putalpha(mask)
        tmp = Image.new("RGBA", im.size, (0, 0, 0, 0))
        tmp.paste(c, (x + dx, y + dy))
        layer.alpha_composite(tmp)

    if glow:
        put(a.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(glow_size)).point(
            lambda v: min(255, v * 2)), glow)
    if shadow:
        put(a.filter(ImageFilter.GaussianBlur(shadow)).point(lambda v: int(v * 0.8)), (0, 0, 0),
            shadow // 2, shadow)
    if outline:
        put(a.filter(ImageFilter.MaxFilter(2 * outline + 1)).filter(ImageFilter.GaussianBlur(1)), outline_color)
    tmp = Image.new("RGBA", im.size, (0, 0, 0, 0))
    tmp.paste(rgba, (xy[0], xy[1]))
    layer.alpha_composite(tmp)
    im.alpha_composite(layer)
    return im


def arrow(im: Image.Image, start, end, *, width=26, color=(255, 255, 255), outline=(12, 12, 14), head=1.0):
    """Толстая стрелка с тёмной обводкой — показать, куда смотреть."""
    (x0, y0), (x1, y1) = start, end
    ang = math.atan2(y1 - y0, x1 - x0)
    L = math.hypot(x1 - x0, y1 - y0)
    hl, hw = width * 2.2 * head, width * 1.6 * head
    pts = [(0, -width / 2), (L - hl, -width / 2), (L - hl, -hw), (L, 0), (L - hl, hw), (L - hl, width / 2),
           (0, width / 2)]
    ca, sa = math.cos(ang), math.sin(ang)
    poly = [(x0 + px * ca - py * sa, y0 + px * sa + py * ca) for px, py in pts]
    d = ImageDraw.Draw(im)
    d.polygon(poly, fill=color, outline=outline, width=max(4, width // 4))
    return im


def finish(im: Image.Image, *, nick, words, text_pos="bottom", max_w=0.46, size=104) -> Image.Image:
    """Ник, логотип, слова — по правилам канала (как в plates.thumbnail)."""
    plates.brand(im, nick)
    plates.words_block(im, words, text_pos=text_pos, max_w=max_w, size=size)
    return im.convert("RGB")
