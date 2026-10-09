"""Графика ролика в фирменном стиле: плашка стримера, заставка, концовка."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "fonts" / "Unbounded-VariableFont_wght.ttf"
MARK = ROOT / "logo" / "srez-mark-lime.png"
LOGO = ROOT / "logo" / "srez-logo-transparent-white.png"

INK = (12, 12, 14)
LIME = (198, 255, 51)
WHITE = (245, 245, 240)
GRAY = (140, 140, 146)
TRACK = (38, 38, 43)

_fonts = {}


def font(size, wght=400):
    key = (size, wght)
    if key not in _fonts:
        f = ImageFont.truetype(str(FONT), size)
        f.set_variation_by_axes([wght])
        _fonts[key] = f
    return _fonts[key]


def fmt_count(n: int) -> str:
    """1 234 567 → «1,2 млн», 845 300 → «845 тыс.», 5 400 → «5,4 тыс.»."""
    def dec(v):
        return f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",")
    if n >= 1_000_000:
        return f"{dec(n / 1e6)} млн"
    if n >= 10_000:
        return f"{n // 1000} тыс."
    if n >= 1000:
        return f"{dec(n / 1e3)} тыс."
    return str(n)


def followers_label(n: int, label: str) -> str:
    if n >= 1000 or not label.startswith("фолловеров"):
        return label
    forms = ("фолловер", "фолловера", "фолловеров")
    k = 0 if n % 10 == 1 and n % 100 != 11 else 1 if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 2
    return forms[k] + label[len("фолловеров"):]


def _cover(img: Image.Image, w, h) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), (w, h), Image.LANCZOS)


def _avatar(path, size, name):
    if path and Path(path).exists():
        im = _cover(Image.open(path), size, size)
    else:
        im = Image.new("RGB", (size, size), LIME)
        d = ImageDraw.Draw(im)
        letter = (name or "?")[0].upper()
        d.text((size / 2, size / 2), letter, font=font(int(size * 0.46), 800), fill=INK, anchor="mm")
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * 4, size * 4), fill=255)
    out = Image.new("RGBA", (size, size))
    out.paste(im, (0, 0), mask.resize((size, size), Image.LANCZOS))
    return out


def _brand_corner(canvas, W, H):
    logo = Image.open(LOGO).convert("RGBA")
    h = int(H * 0.05)
    logo = logo.resize((int(logo.width * h / logo.height), h), Image.LANCZOS)
    canvas.alpha_composite(logo, (W - logo.width - int(W * 0.035), int(H * 0.05)))


def background(frame_path, W, H) -> Image.Image:
    """Размытый и затемнённый кадр следующего клипа."""
    if frame_path and Path(frame_path).exists():
        bg = _cover(Image.open(frame_path), W, H).filter(ImageFilter.GaussianBlur(W / 60))
        bg = Image.blend(bg, Image.new("RGB", (W, H), INK), 0.62)
    else:
        bg = Image.new("RGB", (W, H), INK)
    return bg


def streamer_card(W, H, *, name, login, followers, label_text, avatar=None, tag=None, title=None,
                  platform="Twitch") -> Image.Image:
    """Прозрачный слой с карточкой стримера (отдельно от фона — для анимации)."""
    k = W / 1920
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    av = int(220 * k)
    pad = int(64 * k)
    f_name, f_link = font(int(76 * k), 800), font(int(32 * k), 400)
    f_num, f_lab, f_tag = font(int(46 * k), 800), font(int(32 * k), 400), font(int(24 * k), 700)
    link = f"{platform.lower()}.tv/{login}" if login else platform
    count = fmt_count(followers) if followers is not None else None
    lab = " " + followers_label(followers, label_text) if followers is not None else ""

    text_w = max(
        d.textlength(name, font=f_name),
        d.textlength(link, font=f_link),
        (d.textlength(count, font=f_num) + d.textlength(lab, font=f_lab)) if count else 0,
    )
    cw = int(pad + av + 56 * k + text_w + pad * 1.3)
    ch = int(av + pad * 2)
    x0, y0 = (W - cw) // 2, (H - ch) // 2 - int(20 * k)

    d.rounded_rectangle((x0, y0, x0 + cw, y0 + ch), radius=int(44 * k), fill=INK + (238,),
                        outline=TRACK, width=max(1, int(2 * k)))
    # аватар с лаймовым кольцом
    ax, ay = x0 + pad, y0 + pad
    ring = int(7 * k)
    d.ellipse((ax - ring, ay - ring, ax + av + ring, ay + av + ring), fill=LIME)
    d.ellipse((ax - ring + int(4 * k), ay - ring + int(4 * k), ax + av + ring - int(4 * k),
               ay + av + ring - int(4 * k)), fill=INK)
    layer.alpha_composite(_avatar(avatar, av, name), (ax, ay))

    tx = ax + av + int(56 * k)
    # строки текста по вертикали от центра аватара
    ty = ay + int(av * 0.5)
    d.text((tx, ty - int(18 * k)), name, font=f_name, fill=WHITE, anchor="ls")
    d.text((tx, ty + int(34 * k)), link, font=f_link, fill=GRAY, anchor="ls")
    if count:
        d.text((tx, ty + int(100 * k)), count, font=f_num, fill=LIME, anchor="ls")
        d.text((tx + d.textlength(count, font=f_num), ty + int(100 * k)), lab, font=f_lab,
               fill=WHITE, anchor="ls")
    if tag:
        tag = tag.upper()
        tw = d.textlength(tag, font=f_tag)
        px, py = int(20 * k), int(12 * k)
        tx0, ty1 = x0 + cw - pad - int(tw) - px * 2, y0 - int(22 * k)
        d.rounded_rectangle((tx0, ty1, tx0 + tw + px * 2, ty1 + int(24 * k) + py * 2),
                            radius=int(30 * k), fill=LIME)
        d.text((tx0 + px, ty1 + py + int(24 * k)), tag, font=f_tag, fill=INK, anchor="ls")
    if title:
        d.text((W / 2, y0 + ch + int(80 * k)), title, font=font(int(40 * k), 500), fill=WHITE, anchor="ms")
    _brand_corner(layer, W, H)
    return layer


def intro(W, H, title) -> Image.Image:
    im = Image.new("RGBA", (W, H), INK + (255,))
    logo = Image.open(LOGO).convert("RGBA")
    h = int(H * 0.17)
    logo = logo.resize((int(logo.width * h / logo.height), h), Image.LANCZOS)
    im.alpha_composite(logo, ((W - logo.width) // 2, int(H * 0.36)))
    if title:
        ImageDraw.Draw(im).text((W / 2, H * 0.66), title, font=font(int(H * 0.04), 400), fill=GRAY, anchor="ms")
    return im


def outro(W, H) -> Image.Image:
    """Фон для конечной заставки YouTube: слева место под «Подписаться», справа — под видео."""
    k = W / 1920
    im = Image.new("RGBA", (W, H), INK + (255,))
    d = ImageDraw.Draw(im)
    logo = Image.open(LOGO).convert("RGBA")
    h = int(110 * k)
    logo = logo.resize((int(logo.width * h / logo.height), h), Image.LANCZOS)
    im.alpha_composite(logo, ((W - logo.width) // 2, int(120 * k)))
    d.text((W / 2, 330 * k), "Подпишись, чтобы не пропустить новый срез", font=font(int(40 * k), 400),
           fill=GRAY, anchor="ms")
    # контуры под элементы конечной заставки
    r = int(170 * k)
    cx, cy = int(560 * k), int(700 * k)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=TRACK, width=int(4 * k))
    vx, vy, vw = int(880 * k), int(510 * k), int(680 * k)
    d.rounded_rectangle((vx, vy, vx + vw, vy + vw * 9 // 16), radius=int(20 * k), outline=TRACK, width=int(4 * k))
    # таймлайн из баннера
    y = int(1000 * k)
    d.rectangle((0, y - 2, W, y + 2), fill=TRACK)
    d.rectangle((W // 2 - int(330 * k), y - 3, W // 2 + int(330 * k), y + 3), fill=LIME)
    return im


# ---------------------------------------------------------------- стиль 2: поверх клипа

SLANT = 0.36  # наклон «среза» у плашек: сдвиг по x на единицу высоты, как косой разрез в логотипе


def _slanted(w, h, fill):
    im = Image.new("RGBA", (w + int(h * SLANT), h), (0, 0, 0, 0))
    ImageDraw.Draw(im).polygon([(0, 0), (w + int(h * SLANT), 0), (w, h), (0, h)], fill=fill)
    return im


def lower_third(W, *, name, login, followers, title=None, avatar=None, platform="Twitch") -> Image.Image:
    """Плашка стримера поверх клипа: название момента на лайме, под ним аватар, ник, ссылка и фолловеры."""
    k = W / 1920
    fn, fl, fb = font(int(40 * k), 800), font(int(23 * k), 500), font(int(23 * k), 800)
    link = f"{platform.lower()}.tv/{login}" if login else platform
    line2 = f"{link}  ·  " if followers is not None else link
    count = fmt_count(followers) if followers is not None else ""
    text_w = max(fn.getlength(name), fl.getlength(line2) + fb.getlength(count))
    av, pad = int(80 * k), int(16 * k)
    cw, ch = int(av + pad * 2 + 22 * k + text_w + 34 * k), int(112 * k)
    card = _slanted(cw, ch, INK + (232,))
    d = ImageDraw.Draw(card)
    ring = int(4 * k)
    d.ellipse((pad - ring, (ch - av) // 2 - ring, pad + av + ring, (ch + av) // 2 + ring), fill=LIME)
    card.alpha_composite(_avatar(avatar, av, name), (pad, (ch - av) // 2))
    tx = pad * 2 + av + int(22 * k) - pad
    d.text((tx, int(18 * k)), name, font=fn, fill=WHITE)
    d.text((tx, int(70 * k)), line2, font=fl, fill=GRAY)
    if count:
        d.text((tx + fl.getlength(line2), int(70 * k)), count, font=fb, fill=LIME)
    if not title:
        return card
    ft = font(int(28 * k), 800)
    th = int(52 * k)
    tag = _slanted(int(ft.getlength(title.upper()) + 40 * k), th, LIME + (255,))
    ImageDraw.Draw(tag).text((int(20 * k), th // 2), title.upper(), font=ft, fill=INK, anchor="lm")
    gap = int(10 * k)
    out = Image.new("RGBA", (max(card.width, tag.width), th + gap + ch), (0, 0, 0, 0))
    out.alpha_composite(tag, (0, 0))
    out.alpha_composite(card, (0, th + gap))
    return out


def caption(W, text) -> Image.Image:
    """Подпись-реакция: крупно белым с обводкой, последнее слово — на лаймовой плашке.
    Длинная подпись сначала уменьшается, слова отбрасываются только если не влезает и мелким шрифтом."""
    k = W / 1920
    words = text.split()
    gap, padx = int(24 * k), int(18 * k)

    def layout(size, ws):
        f = font(size, 900)
        ph = int(size * 1.32)
        widths = [int(f.getlength(w)) for w in ws]
        return f, ph, widths, sum(widths) + gap * (len(ws) - 1) + padx * 2 + int(ph * SLANT)

    size = int(88 * k)
    f, ph, widths, total = layout(size, words)
    while total > W * 0.9 and size > int(60 * k):
        size -= int(4 * k) or 1
        f, ph, widths, total = layout(size, words)
    while total > W * 0.9 and len(words) > 1:     # всё равно не влезает — оставляем последние слова
        words = words[1:]
        f, ph, widths, total = layout(size, words)
    im = Image.new("RGBA", (total + int(20 * k), ph + int(20 * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = int(10 * k), im.height // 2
    for i, (w, ww) in enumerate(zip(words, widths)):
        if i == len(words) - 1:
            im.alpha_composite(_slanted(ww + padx * 2, ph, LIME + (255,)), (x, y - ph // 2 + int(4 * k)))
            d.text((x + padx, y), w, font=f, fill=INK, anchor="lm")
        else:
            d.text((x, y), w, font=f, fill=WHITE, anchor="lm", stroke_width=max(2, size // 10), stroke_fill=INK)
            x += ww + gap
    return im


def disclaimer(W, text) -> Image.Image:
    """Строка-предупреждение в начале ролика: тёмная плашка, лаймовая точка, белый текст."""
    k = W / 1920
    f = font(int(30 * k), 600)
    h, pad, dot = int(64 * k), int(26 * k), int(14 * k)
    w = int(pad * 2 + dot + 16 * k + f.getlength(text))
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=INK + (225,))
    d.ellipse((pad, (h - dot) // 2, pad + dot, (h + dot) // 2), fill=LIME)
    d.text((pad + dot + int(16 * k), h // 2), text, font=f, fill=WHITE, anchor="lm")
    return im


def sting_frame(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    """Кадр перехода (t от 0 до 1): лаймовый косой срез проходит по кадру, в середине — знак СРЕЗа."""
    W, H = a.size
    out = (a if t < 0.5 else b).convert("RGBA").copy()
    span = W + H * SLANT * 2
    lead = -H * SLANT + span * min(1, t * 1.6)
    tail = -H * SLANT + span * max(0, (t - 0.38) * 1.6)
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(band).polygon([(tail, H), (tail + H * SLANT, 0), (lead + H * SLANT, 0), (lead, H)],
                                 fill=LIME + (255,))
    out.alpha_composite(band)
    if 0.3 <= t <= 0.7:
        import math
        mark = Image.open(ROOT / "logo" / "srez-mark-black.png").convert("RGBA")
        s = int(H * (0.30 + 0.06 * math.sin((t - 0.3) / 0.4 * math.pi)))
        mark = mark.resize((s, s), Image.LANCZOS)
        out.alpha_composite(mark, ((W - s) // 2, (H - s) // 2))
    return out.convert("RGB")


def thumbnail(img: Image.Image, box, *, nick, words, text_pos="bottom", W=1280, H=720) -> Image.Image:
    """Превью по правилам канала: лицо справа, 2–3 слова слева снизу (последнее на лайме),
    ник на тёмной плашке слева сверху, логотип СРЕЗа справа сверху, правый нижний угол пустой.
    text_pos="top" — слова слева сверху под ником, если главное в кадре внизу."""
    from PIL import ImageEnhance
    k = W / 1280
    x, y, w, h = box
    base = img.convert("RGB").crop((x, y, x + w, y + h)).resize((W, H), Image.LANCZOS)
    base = ImageEnhance.Contrast(ImageEnhance.Color(base).enhance(1.25)).enhance(1.12)
    im = base.convert("RGBA")
    # затемнение слева снизу под текст
    grad = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(grad)
    for i in range(0, W, 4):
        for j in range(0, H, 4):
            dy = (j / H) if text_pos == "top" else (1 - j / H)
            v = max(0.0, 1 - ((i / W) / 0.62) ** 2 - (dy / 0.75) ** 2)
            gd.rectangle((i, j, i + 3, j + 3), fill=int(215 * v))
    shade = Image.new("RGBA", (W, H), INK + (255,))
    shade.putalpha(grad.filter(ImageFilter.GaussianBlur(20 * k)))
    im.alpha_composite(shade)
    d = ImageDraw.Draw(im)
    # ник слева сверху
    fn = font(int(30 * k), 800)
    nw = int(fn.getlength(nick))
    d.rounded_rectangle((int(28 * k), int(28 * k), int(28 * k + 64 * k + nw), int(84 * k)), radius=int(28 * k),
                        fill=INK + (230,))
    d.ellipse((int(46 * k), int(48 * k), int(62 * k), int(64 * k)), fill=LIME)
    d.text((int(76 * k), int(56 * k)), nick, font=fn, fill=WHITE, anchor="lm")
    # логотип канала справа сверху, с тенью — как фирменный знак серии
    logo = Image.open(LOGO).convert("RGBA")
    lh = int(64 * k)
    logo = logo.resize((int(logo.width * lh / logo.height), lh), Image.LANCZOS)
    pos = (W - logo.width - int(30 * k), int(30 * k))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow.paste(logo, pos, logo)
    halo = glow.getchannel("A").filter(ImageFilter.GaussianBlur(10 * k)).point(lambda v: min(255, v * 2))
    dark = Image.new("RGBA", (W, H), INK + (255,))
    dark.putalpha(halo)
    im.alpha_composite(dark)
    im.alpha_composite(glow)
    # 2–3 слова слева снизу: не шире левой половины кадра
    lines = words if isinstance(words, list) else words.split()
    size = int(96 * k)
    while size > 40 and max(font(size, 900).getlength(x) for x in lines) > W * 0.46:
        size -= 4
    f = font(size, 900)
    step, ph = int(size * 1.1), int(size * 1.14)
    y0 = (int(112 * k) + ph // 2 if text_pos == "top"
          else H - int(36 * k) - ph // 2 - (len(lines) - 1) * step)
    for i, wd in enumerate(lines):
        yy = y0 + i * step
        if i == len(lines) - 1:
            ww = int(f.getlength(wd))
            im.alpha_composite(_slanted(ww + int(40 * k), ph, LIME + (255,)), (int(30 * k), yy - ph // 2))
            d.text((int(50 * k), yy), wd, font=f, fill=INK, anchor="lm")
        else:
            d.text((int(40 * k), yy), wd, font=f, fill=WHITE, anchor="lm", stroke_width=int(8 * k),
                   stroke_fill=INK)
    return im.convert("RGB")
