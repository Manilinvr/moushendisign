"""YouTube Shorts из готовых выпусков: вертикально 1080×1920, 15–30 с, крупные субтитры по словам.

    python -m srez shorts projects/shorts-01.toml        все шортсы из файла → output/shorts/
    python -m srez shorts projects/shorts-01.toml 03     только третий (по номеру или части имени)

Файл шортсов:

    [[short]]
    name = "01_ravshan_maska"                         # имя файла
    title = "Равшан спрятался от [фанатов]"           # заголовок сверху; слово в [скобках] — на лаймовой плашке
    layout = "face"       # face — кадр 9:16 по лицу (живая камера); split — вебка сверху, игра снизу; fit — кадр целиком
    parts = [["vypusk-02", "vod2895984355-05", 18.0, 50.0]]   # куски: проект, клип, с какой по какую секунду клипа
    # face_x = 0.3        # если в кадре несколько лиц — какое брать (доля ширины кадра)
    # crop = [0.3, 0.5, 1.0]   # face: кадр вручную — центр x, центр y, высота (доли кадра), если лицо не ловится
    # game_x = 0.5        # split: центр игры по ширине кадра
    # cam = [0, 0.55, 0.16, 0.27]   # split: вебка вручную (доли кадра), если лицо не находится
    # sub_y = 0.45         # где строка субтитров (доля высоты экрана), если по умолчанию она на лице
    # drop = ["returned"]          # слова, которые распознались мусором, — убрать из субтитров
    # fix = { "лиску" = "Алиску" }   # исправить слово в субтитрах

Берётся всё из выпуска: исходники клипов, маски рекламы, запикивание мата и распознанные слова.
Паузы короче, чем в выпуске (шортс должен лететь), мат в субтитрах — звёздочками.
"""
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import censor, edit, plates
from .media import cpu_count, ffmpeg, probe
from .moments import Moment
from .project import load
from .render import _source_media, load_masks, mask_filters

W, H, FPS = 1080, 1920, 30
BAND = 210                      # split: полоса заголовка сверху; ниже вебка, под ней игра
GAP = 0.45                      # паузы длиннее — вырезаются
LIME_ASS, WHITE_ASS, INK_ASS = "&H0033FFC6&", "&H00F0F5F5&", "&H000E0C0C&"


# ---------------------------------------------------------------- шрифт для субтитров

def _black_font(folder: Path) -> Path:
    """Unbounded самой жирной начертки отдельным файлом — libass не умеет выбирать вес у переменного шрифта."""
    out = folder / "Unbounded-Black.ttf"
    if not out.exists():
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        folder.mkdir(parents=True, exist_ok=True)
        f = instancer.instantiateVariableFont(TTFont(str(plates.FONT)), {"wght": 900})
        for rec in f["name"].names:
            if rec.nameID in (1, 4, 16):
                rec.string = "Unbounded Black"
            elif rec.nameID == 6:
                rec.string = "UnboundedBlack"
            elif rec.nameID in (2, 17):
                rec.string = "Regular"
        f.save(str(out))
    return out


# ---------------------------------------------------------------- куски и вырезка пауз

def _part(spec_part, cache):
    ep, mid, a, b = spec_part
    if ep not in cache:
        p = load(Path("projects") / f"{ep}.toml")
        cache[ep] = (p, json.loads((p.work / "candidates.json").read_text(encoding="utf-8")), load_masks(p))
    p, data, masks = cache[ep]
    m = Moment(**next(x for x in data["moments"] if x["id"] == mid))
    src, off = _source_media(p, m, p.work / "render")
    cc = json.loads((p.work / "censor" / f"{m.id}_{m.start:.1f}-{m.end:.1f}.json").read_text(encoding="utf-8"))
    info = data["sources"][m.source]
    words = censor.merge_hyphens([tuple(x) for x in cc["words"]])
    bleeps = censor.bad_spans(words, m.length)
    return {"p": p, "m": m, "src": src, "off": off, "a": float(a), "b": float(b), "masks": masks[m.source],
            "bleeps": sorted(bleeps), "words": words,
            "name": info["name"], "login": info.get("login")}


def _intervals(pt):
    """Отрезки куска без пауз (секунды клипа)."""
    a, b = pt["a"], pt["b"]
    rms = edit._rms_db(pt["src"], pt["off"] + a, b - a)
    words = [(x - a, y - a, t) for x, y, t in pt["words"] if y > a and x < b]
    keep = edit.keep_intervals(rms, words, b - a, GAP, edge=0.12)
    return [(a + x, a + y) for x, y in keep]


def _segment(pt, a, b, out: Path):
    """Кусок клипа 1920×1080: реклама закрыта, мат запикан (как в выпуске)."""
    info = probe(pt["src"])
    v = next(x for x in info["streams"] if x["codec_type"] == "video")
    has_audio = any(x["codec_type"] == "audio" for x in info["streams"])
    dur = b - a
    m = pt["m"]
    act = [x for x in pt["masks"] if x.active(m.start + a, m.start + b)]
    graph, cur = mask_filters(act, m.start + a, dur, v["width"], v["height"])
    graph.append(f"[{cur}]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                 f"setsar=1,fps={FPS},format=yuv420p[v]")
    inp = ["-ss", f"{pt['off'] + a:.3f}", "-t", f"{dur:.3f}", "-i", str(pt["src"])]
    if has_audio:
        graph.append("[0:a:0]aresample=48000,aformat=channel_layouts=stereo[sp0]")
    else:
        inp += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph.append("[1:a]anull[sp0]")
    bl = [(max(0.0, x - a), min(dur, y - a)) for x, y in pt["bleeps"] if y > a and x < b]
    graph += censor.audio_filter(bl, dur, "beep")
    ffmpeg(*inp, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[sp]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
           "-t", f"{dur:.3f}", out)


# ---------------------------------------------------------------- лицо

def _faces(path, dur, cache_dir, step=1 / 3):
    """Самое крупное лицо в кадрах через каждые step секунд: [(t, cx, cy, fh)] в пикселях 1920×1080."""
    import cv2
    det = edit._detector(cache_dir)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={1 / step},scale=960:540",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 540, 960, 3)
    det.setInputSize((960, 540))
    out = []
    for i, f in enumerate(frames):
        _, faces = det.detect(np.ascontiguousarray(f))
        if faces is None or not len(faces):
            continue
        out.append([(i * step, (x + w / 2) * 2, (y + h / 2) * 2, h * 2) for x, y, w, h in
                    (fc[:4] for fc in faces if fc[14] > 0.6)])
    return [o for o in out if o]


def _pick(faces, hint_x=None):
    """В каждом кадре — нужное лицо: ближайшее к подсказке или самое крупное."""
    res = []
    for row in faces:
        if hint_x is not None:
            f = min(row, key=lambda f: abs(f[1] - hint_x * 1920))
        else:
            f = max(row, key=lambda f: f[3])
        res.append(f)
    return res


def _path_expr(points, dur, lo, hi):
    """Плавная траектория x(t) кадра: сглаживание и выражение ffmpeg из отрезков по 0,5 с."""
    if not points:
        return f"{(lo + hi) / 2:.0f}"
    ts = np.arange(0, dur + 0.5, 0.5)
    t0 = np.array([p[0] for p in points])
    x0 = np.array([p[1] for p in points])
    xs = np.interp(ts, t0, x0)
    k = 3                                               # сглаживание ±1,5 с
    xs = np.array([xs[max(0, i - k):i + k + 1].mean() for i in range(len(xs))])
    xs = np.clip(xs, lo, hi)
    if xs.max() - xs.min() < 60:                        # почти не двигается — стоим на месте
        return f"{np.median(xs):.0f}"
    expr = f"{xs[-1]:.0f}"
    for i in range(len(ts) - 2, -1, -1):
        expr = (f"if(lt(t,{ts[i + 1]:.2f}),{xs[i]:.0f}+({xs[i + 1] - xs[i]:.0f})*(t-{ts[i]:.2f})/0.5,{expr})")
    return expr


# ---------------------------------------------------------------- заголовок и субтитры

def _title_png(text, credit, out: Path, boxed=True):
    """Заголовок сверху: до двух строк, слово в [скобках] — на лаймовой плашке; ниже — ник со ссылкой."""
    k = 1.0
    f = plates.font(int(58 * k), 900)
    fc = plates.font(int(30 * k), 600)
    words, hot = [], False                              # слова как куски (текст, на плашке)
    for w in text.split():
        segs, cur_t = [], ""
        for c in w:
            if c in "[]":
                if cur_t:
                    segs.append((cur_t, hot))
                cur_t, hot = "", c == "["
            else:
                cur_t += c
        if cur_t:
            segs.append((cur_t, hot))
        words.append(segs)
    lines, cur = [], []
    for wd in words:                                    # перенос по ширине
        trial = " ".join("".join(t for t, _ in x) for x in cur + [wd])
        if cur and f.getlength(trial) > W - 160:
            lines.append(cur)
            cur = [wd]
        else:
            cur.append(wd)
    lines.append(cur)
    lh = int(58 * 1.25)
    h = 40 + lh * len(lines) + 54
    im = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if boxed:
        d.rounded_rectangle((40, 0, W - 40, h), radius=34, fill=plates.INK + (225,))
    y = 20 + lh // 2
    for ln in lines:
        parts = []                                      # (текст, на плашке, пробел перед)
        for wi, segs in enumerate(ln):
            for si, (t, hot) in enumerate(segs):
                sp = wi > 0 and si == 0
                if hot and sp and parts and parts[-1][1]:   # несколько слов в [скобках] — одна плашка
                    parts[-1] = (parts[-1][0] + " " + t, True, parts[-1][2])
                else:
                    parts.append((t, hot, sp))
        pad = 16                                        # отступ текста от края плашки
        total = sum(f.getlength(t) + (2 * pad + (lh - 6) * plates.SLANT * 0.7 if hot else 0)
                    for t, hot, _ in parts) + \
            f.getlength(" ") * sum(1 for *_, sp in parts if sp)
        x = (W - total) / 2
        for t, hot, sp in parts:
            if sp:
                x += f.getlength(" ")
            tw = f.getlength(t)
            if hot:
                plate = plates._slanted(int(tw + 2 * pad), lh - 6, plates.LIME + (255,))
                im.alpha_composite(plate, (int(x), int(y - (lh - 6) / 2)))
                d.text((x + pad, y), t, font=f, fill=plates.INK, anchor="lm")
                x += tw + 2 * pad + (lh - 6) * plates.SLANT * 0.7   # косой край плашки
            else:
                d.text((x, y), t, font=f, fill=plates.WHITE, anchor="lm")
                x += tw
        y += lh
    d.ellipse((W / 2 - fc.getlength(credit) / 2 - 26, y + 4, W / 2 - fc.getlength(credit) / 2 - 12, y + 18),
              fill=plates.LIME)
    d.text((W / 2 + 8, y + 11), credit, font=fc, fill=(230, 230, 225), anchor="mm")
    im.save(out)
    return im.height


def _ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _stars(w):
    """Мат звёздочками: первая буква и одна-две последние остаются (БЛЯТЬ → Б**ТЬ, ЕБУ → Е*У)."""
    import re
    m = re.match(r"^(\W*)([\w-]+?)(\W*)$", w)
    if not m:
        return w
    pre, core, post = m.groups()
    core = core.replace("-", "")                       # «уй-ё-бищ» → одно слово под звёздочками
    n = len(core)
    keep = 2 if n >= 5 else 1
    if n <= 2:
        return pre + core[0] + "*" * (n - 1) + post
    return pre + core[0] + "*" * (n - 1 - keep) + core[-keep:] + post


def _word(w):
    w = w.strip()
    if "*" not in w and censor.is_bad(w):
        w = _stars(w)
    return w.upper()


def _ass(words, dur, y, out: Path, font_name="Unbounded Black"):
    """Субтитры по словам: 2–3 слова на экране, произносимое — лаймовым, кусок «выпрыгивает»."""
    words = [(a, b, _word(t)) for a, b, t in words if _word(t).strip(" .,!?-")]
    chunks, cur = [], []
    for wd in words:
        text = " ".join(x[2] for x in cur + [wd])
        gap = cur and wd[0] - cur[-1][1] > 0.6
        if cur and (len(cur) >= 3 or len(text) > 16 or gap or cur[-1][2][-1:] in ".?!"):
            chunks.append(cur)
            cur = []
        cur.append(wd)
    if cur:
        chunks.append(cur)
    ev = []
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else dur
        end_chunk = min(nxt, ch[-1][1] + 0.6, dur)
        for wi, (a, b, _) in enumerate(ch):
            s = a if wi else min(a, ch[0][0])
            e = ch[wi + 1][0] if wi + 1 < len(ch) else end_chunk
            if e <= s:
                continue
            txt = " ".join((f"{{\\c{LIME_ASS}}}{t}{{\\c{WHITE_ASS}}}" if j == wi else t)
                           for j, (_, _, t) in enumerate(ch))
            pop = r"\fscx118\fscy118\t(0,110,\fscx100\fscy100)" if wi == 0 else ""
            ev.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Sub,,0,0,0,,{{\\pos(540,{y}){pop}}}{txt}")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font_name},90,{WHITE_ASS},{WHITE_ASS},{INK_ASS},&H7F000000&,0,0,0,0,100,100,0,0,1,8,4,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- сборка одного шортса

def build(sh: dict, out_dir: Path, log=print):
    name = sh["name"]
    work = Path("projects/work/shorts") / name
    work.mkdir(parents=True, exist_ok=True)
    cache = {}
    parts = [_part(p, cache) for p in sh["parts"]]
    assets = Path("projects/work/_assets")
    # 1. куски без пауз → один горизонтальный исходник
    segs, words, t = [], [], 0.0
    for pi, pt in enumerate(parts):
        for si, (a, b) in enumerate(_intervals(pt)):
            sig = hashlib.md5(repr(([x for x in pt["bleeps"] if x[1] > a and x[0] < b],
                                    [x for x in pt["masks"] if x.active(pt["m"].start + a, pt["m"].start + b)]))
                              .encode()).hexdigest()[:8]           # другие маски или мат — другой файл
            seg = work / f"seg_{pi}_{si}_{a:.2f}-{b:.2f}_{sig}.mkv"
            if not seg.exists():
                _segment(pt, a, b, seg)
            segs.append(seg)
            words += [(t + x - a, t + min(y, b) - a, w) for x, y, w in pt["words"] if a <= x < b]
            t += b - a
    dur = t
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs), encoding="utf-8")
    src = work / "src.mkv"
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", src)
    # 2. вертикальная раскладка
    lay = sh.get("layout", "face")
    login = parts[0]["login"]
    credit = f"{parts[0]['name']} · twitch.tv/{login}" if login else parts[0]["name"]
    title = work / "title.png"
    th = _title_png(sh["title"], credit, title, boxed=lay == "fit")
    band = max(BAND, th + 20)
    hint = sh.get("face_x")
    if hint is None and sh.get("cam"):              # в split с рамкой вебки ищем лицо в ней
        hint = sh["cam"][0] + sh["cam"][2] / 2
    faces = _pick(_faces(src, dur, assets), hint) if lay in ("face", "split") else []
    g = []
    if lay == "face":
        # сверху полоса с заголовком, под ней кадр по лицу: лицо примерно на трети высоты, не под заголовком
        vh = H - band
        if faces:
            fh = float(np.median([f[3] for f in faces]))
            fy = float(np.median([f[2] for f in faces]))
        else:
            fh, fy = 300.0, 540.0
        ch = min(max(fh * 7.5, 760), 1080)
        cw = min(1920, ch * W / vh)
        cy0 = int(min(max(fy - ch * 0.36, 0), 1080 - ch)) // 2 * 2
        xe = _path_expr([(tt, cx - cw / 2) for tt, cx, cy, fh_ in faces], dur, 0, 1920 - cw)
        if sh.get("crop"):                      # кадр вручную: центр x, центр y и высота (доли кадра)
            mx, my, mh = sh["crop"]
            ch = min(1080, mh * 1080)
            cw = min(1920, ch * W / vh)
            cy0 = int(min(max(my * 1080 - ch / 2, 0), 1080 - ch)) // 2 * 2
            xe = f"{min(max(mx * 1920 - cw / 2, 0), 1920 - cw):.0f}"
        g += [f"[0:v]crop={int(cw) // 2 * 2}:{int(ch) // 2 * 2}:'{xe}':{cy0},scale={W}:{vh}:flags=lanczos,setsar=1[fv]",
              f"color=c=0x0C0C0E:s={W}x{H}:r={FPS}:d={dur:.3f}[bg]",
              f"[bg][fv]overlay=0:{band},drawbox=x=0:y={band - 3}:w={W}:h=6:color=0xC6FF33:t=fill[base]"]
        title_y, sub_y = 0, band + int(vh * 0.66)
    elif lay == "split":
        fx = float(np.median([f[1] for f in faces])) if faces else None
        fy = float(np.median([f[2] for f in faces])) if faces else None
        if sh.get("cam"):                       # рамка вебки (доли кадра)
            x, y, w, h = (v * k for v, k in zip(sh["cam"], (1920, 1080, 1920, 1080)))
            inside = [f for f in faces if x <= f[1] <= x + w and y <= f[2] <= y + h]   # лицо именно в вебке
            fx = float(np.median([f[1] for f in inside])) if inside else None
            fy = float(np.median([f[2] for f in inside])) if inside else None
        elif faces:                             # вокруг лица
            fh = float(np.median([f[3] for f in faces]))
            h = min(1080, fh * 2.9)
            w = h * 1.5
            x, y = min(max(fx - w / 2, 0), 1920 - w), min(max(fy - h / 2, 0), 1080 - h)
        else:
            x, y, w, h = 0, 0, 1920, 1080
        cam_h = int(min(max(W * h / w, 560), 860)) // 2 * 2      # панель под форму вебки
        aspect = W / cam_h
        if w / h < aspect:                      # вебка уже панели — режем по высоте вокруг лица
            cw, ch = w, w / aspect
            cx0, cy0 = x, min(max((fy if fy else y + h / 2) - ch * 0.45, y), y + h - ch)
        else:                                   # шире — режем по ширине вокруг лица
            cw, ch = h * aspect, h
            cx0, cy0 = min(max((fx if fx else x + w / 2) - cw / 2, x), x + w - cw), y
        game_h = H - band - cam_h
        gw = min(1920, int(1080 * W / game_h)) // 2 * 2
        if "game_x" in sh:
            gcx = sh["game_x"] * 1920
        elif x + w < 0.45 * 1920:               # вебка слева — игру берём правее неё
            gcx = (x + w + 1920) / 2
        elif x > 0.55 * 1920:                   # справа — левее
            gcx = x / 2
        else:
            gcx = 960
        gx = int(min(max(gcx - gw / 2, 0), 1920 - gw)) // 2 * 2
        e = lambda v: int(v) // 2 * 2           # noqa: E731
        g += [f"[0:v]split=2[c][gm]",
              f"[c]crop={e(cw)}:{e(ch)}:{e(cx0)}:{e(cy0)},scale={W}:{cam_h}:flags=lanczos,setsar=1[cv]",
              f"[gm]crop={gw}:1080:{gx}:0,scale={W}:{game_h}:flags=lanczos,setsar=1[gv]",
              f"color=c=0x0C0C0E:s={W}x{H}:r={FPS}:d={dur:.3f}[bg]",
              f"[bg][cv]overlay=0:{band}[b1]",
              f"[b1][gv]overlay=0:{band + cam_h},drawbox=x=0:y={band + cam_h - 3}:w={W}:h=6:"
              f"color=0xC6FF33:t=fill[base]"]
        title_y, sub_y = 0, band + cam_h
    else:   # fit
        g += ["[0:v]split=2[b][f]",
              f"[b]scale=-2:{H},crop={W}:{H},boxblur=24:2,eq=brightness=-0.12,setsar=1[bb]",
              f"[f]scale={W}:-2,setsar=1[ff]",
              "[bb][ff]overlay=0:(H-h)/2-80[base]"]
        title_y, sub_y = 230, 1330
    # 3. заголовок, субтитры, звук
    if lay in ("split", "face"):
        title_y = max(0, (band - th) // 2)
    if "sub_y" in sh:   # субтитры закрывают лицо — переставить (доля высоты экрана)
        sub_y = int(H * sh["sub_y"])
    font = _black_font(assets / "fonts")
    drop = {w.lower() for w in sh.get("drop", [])}
    fix = {k.lower(): v for k, v in sh.get("fix", {}).items()}
    clean = []
    for a, b, wd in words:
        key = wd.strip().strip(".,!?").lower()
        if key in drop or any(ord(c) > 0x2000 and c not in "«»—–…" for c in wd):   # иероглифы и т.п. — мусор
            continue
        clean.append((a, b, fix.get(key, wd)))
    ass = work / "subs.ass"
    _ass(clean, dur, sub_y, ass)
    (work / "subs.txt").write_text(" ".join(_word(w) for _, _, w in clean), encoding="utf-8")
    g += [f"[base][1:v]overlay=0:{title_y}[tv]",
          f"[tv]subtitles={ass}:fontsdir={font.parent},format=yuv420p[v]",
          "[0:a]loudnorm=I=-14:TP=-1.0:LRA=11,aresample=48000[a]"]
    out = out_dir / f"{name}.mp4"
    ffmpeg("-i", src, "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", title,
           "-filter_complex", ";".join(g), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-threads", str(cpu_count()), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)
    note = "" if 14.5 <= dur <= 30.5 else f"  ⚠ длина {dur:.1f} с — вне 15–30 с"
    log(f"  {name}: {dur:.1f} с, {lay}{note}")
    return out, dur


def run(path, only=None, log=print):
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    out_dir = Path("projects/output/shorts")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = cfg.get("short", [])
    if only:
        items = [s for i, s in enumerate(items, 1) if only == f"{i:02d}" or only == str(i) or only in s["name"]]
    log(f"Шортсов: {len(items)}")
    res = []
    for sh in items:
        res.append(build(sh, out_dir, log))
    return res
