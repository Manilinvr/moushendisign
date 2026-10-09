#!/usr/bin/env python3
"""Генератор фирменного стиля канала СРЕЗ.

Собирает все SVG (текст сразу переводится в кривые, поэтому шрифт
для просмотра не нужен) и пишет build/manifest.json, по которому
src/render.cjs делает PNG нужных для YouTube размеров.

    python3 src/build.py && node src/render.cjs
"""
import json
import math
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "fonts" / "Unbounded-VariableFont_wght.ttf"

# Палитра
INK = "#0C0C0E"    # почти чёрный фон
LIME = "#C6FF33"   # фирменный лайм
WHITE = "#F5F5F0"  # тёплый белый
GRAY = "#8C8C92"   # второстепенный текст
TRACK = "#26262B"  # линии, подложки

NAME = "СРЕЗ"
WORD_TRACK = 0.035  # воздух между буквами, чтобы фирменная засечка «Е» читалась


# ---------------------------------------------------------------- текст

class Face:
    """Шрифт Unbounded нужной насыщенности: шейпинг и контуры через HarfBuzz."""

    def __init__(self, wght):
        face = hb.Face(hb.Blob.from_file_path(str(FONT)))
        self.font = hb.Font(face)
        self.font.set_variations({"wght": wght})
        self.upem = face.upem
        self.cap = TTFont(FONT)["OS/2"].sCapHeight / self.upem

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {"kern": True})
        return list(zip(buf.glyph_infos, buf.glyph_positions))

    def width(self, text, size, track=0.0):
        g = self._shape(text)
        adv = sum(p.x_advance for _, p in g) + track * self.upem * (len(g) - 1)
        return adv * size / self.upem

    def path(self, text, size, x, y, track=0.0, anchor="start"):
        """Контур строки; y — базовая линия, track — трекинг в долях кегля."""
        s = size / self.upem
        if anchor == "middle":
            x -= self.width(text, size, track) / 2
        elif anchor == "end":
            x -= self.width(text, size, track)
        pen = SVGPathPen(None, ntos=num)
        cur = 0
        for info, pos in self._shape(text):
            t = TransformPen(pen, (s, 0, 0, -s, x + (cur + pos.x_offset) * s, y - pos.y_offset * s))
            self.font.draw_glyph_with_pen(info.codepoint, t)
            cur += pos.x_advance + track * self.upem
        return pen.getCommands()


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


BLACK = Face(800)
BOLD = Face(700)
MEDIUM = Face(500)
REGULAR = Face(400)


# ---------------------------------------------------------------- знак

def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _add(a, b): return (a[0] + b[0], a[1] + b[1])
def _mul(a, k): return (a[0] * k, a[1] * k)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1]
def _unit(a):
    d = math.hypot(*a)
    return (a[0] / d, a[1] / d)


def rounded_polygon(pts, r, steps=32):
    """Многоугольник со скруглёнными углами радиуса r, разбитый на точки."""
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = _unit(_sub(p0, p1)), _unit(_sub(p2, p1))
        ang = math.acos(max(-1.0, min(1.0, _dot(v1, v2))))
        a = _add(p1, _mul(v1, r / math.tan(ang / 2)))
        b = _add(p1, _mul(v2, r / math.tan(ang / 2)))
        c = _add(p1, _mul(_unit(_add(v1, v2)), r / math.sin(ang / 2)))
        a0 = math.atan2(a[1] - c[1], a[0] - c[0])
        a1 = math.atan2(b[1] - c[1], b[0] - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        for k in range(steps + 1):
            t = a0 + da * k / steps
            out.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t)))
    return out


def clip_halfplane(poly, n, p, off):
    """Оставляет часть многоугольника, где n·(x − p) ≥ off (Сазерленд — Ходжман)."""
    f = lambda q: _dot(n, _sub(q, p)) - off
    out = []
    for i in range(len(poly)):
        a, b = poly[i - 1], poly[i]
        fa, fb = f(a), f(b)
        if fb >= 0:
            if fa < 0:
                out.append(_add(a, _mul(_sub(b, a), fa / (fa - fb))))
            out.append(b)
        elif fa >= 0:
            out.append(_add(a, _mul(_sub(b, a), fa / (fa - fb))))
    return out


# Знак в квадрате 100×100: кнопка «play», рассечённая косым срезом «/».
TRIANGLE = [(25, 17), (25, 83), (83, 50)]
CORNER = 7.5
CUT_POINT = (55, 50)
CUT_DIR = _unit((0.42, -1))           # направление разреза «/»
CUT_NORMAL = (-CUT_DIR[1], CUT_DIR[0])  # смотрит вправо
CUT_GAP = 7
CUT_SLIDE = 5                          # насколько правый кусок съехал вдоль разреза


def mark_paths(x, y, size):
    """Две части знака, вписанного в квадрат size×size с левым верхним углом (x, y)."""
    shape = rounded_polygon(TRIANGLE, CORNER)
    left = clip_halfplane(shape, _mul(CUT_NORMAL, -1), CUT_POINT, CUT_GAP / 2)
    right = clip_halfplane(shape, CUT_NORMAL, CUT_POINT, CUT_GAP / 2)
    right = [_add(q, _mul(CUT_DIR, CUT_SLIDE)) for q in right]
    k = size / 100
    res = []
    for piece in (left, right):
        pts = [(x + px * k, y + py * k) for px, py in piece]
        res.append("M" + " L".join(f"{num(px)} {num(py)}" for px, py in pts) + " Z")
    return res


def mark(x, y, size, color):
    return "".join(f'<path d="{d}" fill="{color}"/>' for d in mark_paths(x, y, size))


# ---------------------------------------------------------------- документы

def svg(w, h, body, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" style="display:block">{rect}{body}</svg>\n')


def lockup(x, y, cap, fg, accent):
    """Горизонтальный логотип: знак + слово. (x, y) — левый край и верх заглавных."""
    size = cap / BLACK.cap  # кегль, при котором высота заглавных = cap
    m = cap * 1.7           # знак чуть выше строки
    lead = m * TRIANGLE[0][0] / 100  # пустое поле слева от треугольника внутри квадрата знака
    body = mark(x - lead, y + cap / 2 - m / 2, m, accent)
    body += f'<path d="{BLACK.path(NAME, size, x - lead + m * 0.9, y + cap, track=WORD_TRACK)}" fill="{fg}"/>'
    return body, lockup_width(cap)


def lockup_width(cap):
    m = cap * 1.7
    return m * (0.9 - TRIANGLE[0][0] / 100) + BLACK.width(NAME, cap / BLACK.cap, track=WORD_TRACK)


ASSETS = []


def save(rel, w, h, content, png=None, scale=1, transparent=False):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if png:
        ASSETS.append({"svg": rel, "png": png, "w": w, "h": h, "scale": scale, "transparent": transparent})


def build_mark():
    # Сам знак на прозрачном фоне — для монтажа и для сторонних редакторов
    for name, color in (("lime", LIME), ("black", INK), ("white", WHITE)):
        save(f"logo/srez-mark-{name}.svg", 100, 100, svg(100, 100, mark(0, 0, 100, color)),
             png=f"logo/srez-mark-{name}.png", scale=10.24, transparent=True)


def build_logo():
    cap, pad = 120, 90
    w = round(lockup_width(cap) + pad * 2)
    h = round(cap * 1.7 + pad * 2)
    y = (h - cap) / 2
    variants = (
        ("dark", INK, WHITE, LIME, False),
        ("light", WHITE, INK, INK, False),
        ("transparent-white", None, WHITE, LIME, True),
        ("transparent-black", None, INK, INK, True),
    )
    for name, bg, fg, acc, tr in variants:
        body, _ = lockup(pad, y, cap, fg, acc)
        save(f"logo/srez-logo-{name}.svg", w, h, svg(w, h, body, bg),
             png=f"logo/srez-logo-{name}.png", scale=2, transparent=tr)


def build_avatar():
    # 800×800 — рекомендованный YouTube размер. Знак занимает центр,
    # чтобы круглая обрезка его не задевала.
    s = 800
    m = 500
    off = (s - m) / 2
    save("avatar/srez-avatar-lime.svg", s, s, svg(s, s, mark(off, off, m, INK), LIME),
         png="avatar/srez-avatar-lime-800.png")
    save("avatar/srez-avatar-dark.svg", s, s, svg(s, s, mark(off, off, m, LIME), INK),
         png="avatar/srez-avatar-dark-800.png")
    # Превью — как аватар выглядит в круге
    circ = (f'<defs><clipPath id="c"><circle cx="400" cy="400" r="400"/></clipPath></defs>'
            f'<g clip-path="url(#c)"><rect width="800" height="800" fill="{LIME}"/>'
            f'{mark(off, off, m, INK)}</g>')
    save("avatar/preview-circle.svg", s, s, svg(s, s, circ),
         png="avatar/preview-circle.png", scale=0.5, transparent=True)


def build_watermark():
    # Значок «подписаться» в углу видео: 150×150, прозрачный фон
    s = 150
    body = f'<circle cx="75" cy="75" r="75" fill="{LIME}"/>' + mark(26, 27, 98, INK)
    save("watermark/srez-watermark.svg", s, s, svg(s, s, body),
         png="watermark/srez-watermark-150.png", transparent=True)


def timeline(w, y, seg0, seg1, tc0, tc1, tc_y):
    """Полоса таймлайна с выделенным фрагментом — «срезом» из стрима."""
    tick_h = 34
    b = f'<rect x="0" y="{y - 2}" width="{w}" height="4" fill="{TRACK}"/>'
    b += f'<rect x="{seg0}" y="{y - 3}" width="{seg1 - seg0}" height="6" fill="{LIME}"/>'
    for x in (seg0, seg1):
        b += f'<rect x="{x - 3}" y="{y - tick_h / 2}" width="6" height="{tick_h}" rx="3" fill="{LIME}"/>'
    b += f'<path d="{MEDIUM.path(tc0, 22, seg0, tc_y, track=0.06)}" fill="{GRAY}"/>'
    b += f'<path d="{MEDIUM.path(tc1, 22, seg1, tc_y, track=0.06, anchor="end")}" fill="{GRAY}"/>'
    return b


def build_banner():
    W, H = 2560, 1440
    # Безопасная зона, видимая на всех устройствах: 1546×423 по центру
    sx, sy, sw, sh = (W - 1546) / 2, (H - 423) / 2, 1546, 423
    cx = W / 2

    cap = 150
    lw = lockup_width(cap)
    top = sy + 46
    logo, _ = lockup(cx - lw / 2, top, cap, WHITE, LIME)

    tag = "лучшие моменты стримов — без воды"
    tag_y = top + cap + 88
    tagline = f'<path d="{REGULAR.path(tag, 40, cx, tag_y, track=0.01, anchor="middle")}" fill="{GRAY}"/>'

    tl_y = sy + sh - 72
    seg0, seg1 = cx - 330, cx + 330
    tl = timeline(W, tl_y, seg0, seg1, "02:14:07", "02:14:59", tl_y + 52)

    body = logo + tagline + tl
    save("banner/srez-banner.svg", W, H, svg(W, H, body, INK), png="banner/srez-banner-2560x1440.png")

    # Проверочная версия с разметкой зон YouTube
    guides = (f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="none" stroke="#FF3B5C" '
              f'stroke-width="4" stroke-dasharray="16 10"/>'
              f'<rect x="0" y="{sy}" width="{W}" height="{sh}" fill="none" stroke="#3BA0FF" '
              f'stroke-width="4" stroke-dasharray="16 10"/>')
    legend = (f'<path d="{MEDIUM.path("красная рамка — видно везде (телефон)", 30, sx, sy - 24)}" fill="#FF3B5C"/>'
              f'<path d="{MEDIUM.path("синяя рамка — компьютер", 30, 40, sy + sh + 50)}" fill="#3BA0FF"/>'
              f'<path d="{MEDIUM.path("весь холст — телевизор", 30, 40, 70)}" fill="{GRAY}"/>')
    save("banner/preview-safe-zones.svg", W, H, svg(W, H, body + guides + legend, INK),
         png="banner/preview-safe-zones.png", scale=0.5)


def build_thumbnail():
    W, H = 1280, 720

    # Подложка-оверлей: затемнение снизу слева + фирменный значок
    grad = ('<defs><linearGradient id="fade" x1="0" y1="1" x2="0.7" y2="0.1">'
            f'<stop offset="0" stop-color="{INK}" stop-opacity="0.92"/>'
            f'<stop offset="0.55" stop-color="{INK}" stop-opacity="0.35"/>'
            f'<stop offset="1" stop-color="{INK}" stop-opacity="0"/></linearGradient></defs>'
            f'<rect width="{W}" height="{H}" fill="url(#fade)"/>')
    badge = (f'<rect x="1150" y="40" width="90" height="90" rx="22" fill="{LIME}"/>'
             + mark(1162, 52, 66, INK))
    save("thumbnail/srez-thumbnail-overlay.svg", W, H, svg(W, H, grad + badge),
         png="thumbnail/srez-thumbnail-overlay-1280x720.png", transparent=True)

    # Пример готового превью (вместо силуэта — кадр со стрима)
    bg = ('<defs><radialGradient id="glow" cx="0.72" cy="0.38" r="0.7">'
          '<stop offset="0" stop-color="#3A3A44"/><stop offset="1" stop-color="#121216"/></radialGradient></defs>'
          f'<rect width="{W}" height="{H}" fill="url(#glow)"/>'
          '<circle cx="890" cy="300" r="128" fill="#55555F"/>'
          '<path d="M650 720 C650 540 760 470 890 470 C1020 470 1130 540 1130 720 Z" fill="#55555F"/>')
    pill_txt = "ИМЯ СТРИМЕРА"
    pill_w = MEDIUM.width(pill_txt, 26, track=0.04) + 84
    pill = (f'<rect x="60" y="56" width="{pill_w}" height="56" rx="28" fill="{INK}"/>'
            f'<circle cx="94" cy="84" r="9" fill="{LIME}"/>'
            f'<path d="{MEDIUM.path(pill_txt, 26, 118, 84 + 26 * MEDIUM.cap / 2, track=0.04)}" fill="{WHITE}"/>')
    l1, l2 = "ЧАТ", "В ШОКЕ"
    size = 118
    capp = size * BLACK.cap
    y2 = H - 70
    y1 = y2 - capp - 52
    w2 = BLACK.width(l2, size, -0.02)
    text = (f'<path d="{BLACK.path(l1, size, 72, y1, -0.02)}" fill="{WHITE}"/>'
            f'<rect x="56" y="{y2 - capp - 22}" width="{w2 + 36}" height="{capp + 44}" rx="10" fill="{LIME}"/>'
            f'<path d="{BLACK.path(l2, size, 74, y2, -0.02)}" fill="{INK}"/>')
    save("thumbnail/srez-thumbnail-example.svg", W, H, svg(W, H, bg + grad + badge + pill + text),
         png="thumbnail/srez-thumbnail-example-1280x720.png")


def main():
    build_mark()
    build_logo()
    build_avatar()
    build_watermark()
    build_banner()
    build_thumbnail()
    out = ROOT / "build" / "manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ASSETS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SVG: {len(ASSETS)} шт., манифест: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
