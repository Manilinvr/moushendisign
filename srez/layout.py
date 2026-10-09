"""Поиск рекламы на экране стрима и листы для проверки масок.

Баннеры, логотипы спонсоров и QR-коды висят на одном месте, пока игра и
вебка меняются. Поэтому по кадрам из разных моментов стрима ищутся
статичные участки с чёткими контурами. Это кандидаты: интерфейс игры и
рамка вебки тоже статичны, решение принимается по листу проверки.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .media import frame
from .plates import LIME, font

GW, GH = 192, 108   # сетка анализа


def _grow(static, x1, x2, y1, y2, share=0.8):
    """Расширяет рамку, пока соседняя полоса тоже статична: однотонный фон баннера без контуров."""
    changed = True
    while changed:
        changed = False
        if x2 < GW and static[y1:y2, x2].mean() >= share:
            x2, changed = x2 + 1, True
        if x1 > 0 and static[y1:y2, x1 - 1].mean() >= share:
            x1, changed = x1 - 1, True
        if y2 < GH and static[y2, x1:x2].mean() >= share:
            y2, changed = y2 + 1, True
        if y1 > 0 and static[y1 - 1, x1:x2].mean() >= share:
            y1, changed = y1 - 1, True
    return x1, x2, y1, y2


def static_boxes(images, min_area=0.002, max_area=0.35):
    if len(images) < 3:
        return []
    g = np.stack([np.asarray(im.convert("L").resize((GW, GH)), dtype=np.float32) for im in images])
    std, mean = g.std(axis=0), g.mean(axis=0)
    gx = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
    gy = np.abs(np.diff(mean, axis=0, prepend=mean[:1]))
    cand = (std < 8) & ((gx + gy) > 18)
    # слегка расширяем, чтобы буквы баннера слились в один блок
    pad = np.pad(cand, 2)
    grown = np.zeros_like(cand)
    for dy in range(5):
        for dx in range(5):
            grown |= pad[dy:dy + GH, dx:dx + GW]
    seen = np.zeros_like(grown)
    boxes = []
    for y0, x0 in zip(*np.nonzero(grown)):
        if seen[y0, x0]:
            continue
        q, xs, ys = deque([(y0, x0)]), [], []
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            xs.append(x)
            ys.append(y)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < GH and 0 <= nx < GW and grown[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        x1, x2, y1, y2 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
        x1, x2, y1, y2 = _grow(std < 8, x1, x2, y1, y2)
        x1, y1, x2, y2 = max(0, x1 - 1), max(0, y1 - 1), min(GW, x2 + 1), min(GH, y2 + 1)
        area = (x2 - x1) * (y2 - y1) / (GW * GH)
        if min_area <= area <= max_area:
            boxes.append((x1 / GW, y1 / GH, (x2 - x1) / GW, (y2 - y1) / GH))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _grid(d, x0, y0, w, h, labels=True):
    f = font(max(10, w // 40), 500)
    for i in range(1, 10):
        x, y = x0 + w * i / 10, y0 + h * i / 10
        d.line((x, y0, x, y0 + h), fill=(255, 255, 255, 70), width=1)
        d.line((x0, y, x0 + w, y), fill=(255, 255, 255, 70), width=1)
        if labels:
            d.text((x + 2, y0 + 2), f".{i}", font=f, fill=(255, 255, 255, 200))
            d.text((x0 + 2, y + 2), f".{i}", font=f, fill=(255, 255, 255, 200))


def _boxes(d, x0, y0, w, h, boxes, color, prefix, fill=None):
    f = font(max(12, w // 35), 700)
    for i, (bx, by, bw, bh) in enumerate(boxes, 1):
        r = (x0 + bx * w, y0 + by * h, x0 + (bx + bw) * w, y0 + (by + bh) * h)
        d.rectangle(r, outline=color, width=3, fill=fill)
        d.text((r[0] + 4, r[1] + 2), f"{prefix}{i}", font=f, fill=color)


def sheet(items, masks, out: Path, tile=640):
    """Лист проверки: кадры стрима с сеткой 10%, масками (красные) и кандидатами (лаймовые)."""
    tmp = out.parent / f".{out.stem}_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    allframes = []
    for i, (path, t) in enumerate(items):
        p = tmp / f"{i}.jpg"
        frame(path, t, p, width=1280)
        allframes.append(Image.open(p).convert("RGB"))
    if not allframes:
        return None, []
    sugg = static_boxes(allframes)
    images = allframes[:: max(1, len(allframes) // 6)][:6]
    th = tile * 9 // 16
    cols = 3 if len(images) > 4 else 2
    rows = (len(images) + cols - 1) // cols
    board = Image.new("RGBA", (tile * cols, th * rows), (12, 12, 14, 255))
    over = Image.new("RGBA", board.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    mboxes = [m.box for m in masks]
    for i, im in enumerate(images):
        x0, y0 = (i % cols) * tile, (i // cols) * th
        board.paste(im.resize((tile, th)), (x0, y0))
        _grid(d, x0, y0, tile, th, labels=(i == 0))
        _boxes(d, x0, y0, tile, th, mboxes, (255, 59, 92, 255), "M", fill=(255, 59, 92, 60))
        _boxes(d, x0, y0, tile, th, sugg, (*LIME, 255), "S")
    Image.alpha_composite(board, over).convert("RGB").save(out, quality=88)

    # крупный кадр для точных координат
    big = images[len(images) // 2].resize((1280, 720)).convert("RGBA")
    ov = Image.new("RGBA", big.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    _grid(d, 0, 0, 1280, 720)
    _boxes(d, 0, 0, 1280, 720, mboxes, (255, 59, 92, 255), "M", fill=(255, 59, 92, 60))
    _boxes(d, 0, 0, 1280, 720, sugg, (*LIME, 255), "S")
    Image.alpha_composite(big, ov).convert("RGB").save(out.with_name(out.stem + "_big.jpg"), quality=88)
    out.with_suffix(".json").write_text(json.dumps({"suggested": sugg}, indent=1), encoding="utf-8")
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return out, sugg
