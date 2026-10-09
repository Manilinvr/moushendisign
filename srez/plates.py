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
