"""Сборка ролика: клипы, плашки стримеров, концовка, склейка, таймкоды."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import fetch, plates
from .media import cpu_count, duration, ffmpeg, frame, probe

AENC_PCM = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]


def _venc(s, threads):
    return ["-c:v", "libx264", "-preset", s.preset, "-crf", str(s.crf), "-pix_fmt", "yuv420p",
            "-r", str(s.fps), "-g", str(s.fps * 2), "-threads", str(threads)]


def _even(v):
    return max(2, int(v) // 2 * 2)


def mask_filters(masks, clip_start, dur, src_w, src_h, style="blur"):
    """Цепочка filter_complex, закрывающая рекламу. Возвращает (фильтры, метка выхода)."""
    act = [m for m in masks if m.active(clip_start, clip_start + dur)]
    if not act:
        return [], "0:v"
    out, cur = [], "0:v"
    if style != "fill":
        out.append(f"[0:v]split={len(act) + 1}[mb]" + "".join(f"[mc{i}]" for i in range(len(act))))
        cur = "mb"
    for i, m in enumerate(act):
        x, y, w, h = m.box
        px, py = _even(x * src_w) if x > 0 else 0, _even(y * src_h) if y > 0 else 0
        pw, ph = _even(min(w, 1 - x) * src_w), _even(min(h, 1 - y) * src_h)
        pw, ph = min(pw, src_w - px), min(ph, src_h - py)
        en = ""
        if m.start is not None or m.end is not None:
            a = m.start - clip_start if m.start is not None else -1
            b = m.end - clip_start if m.end is not None else dur + 1
            en = f":enable='between(t,{a:.2f},{b:.2f})'"
        if style == "fill":
            out.append(f"[{cur}]drawbox=x={px}:y={py}:w={pw}:h={ph}:color=0x0C0C0E:t=fill{en}[mv{i}]")
        else:
            sigma = max(12, min(pw, ph) // 4)
            out.append(f"[mc{i}]crop={pw}:{ph}:{px}:{py},gblur=sigma={sigma}:steps=3,"
                       f"eq=brightness=-0.06[mk{i}]")
            out.append(f"[{cur}][mk{i}]overlay={px}:{py}{en}[mv{i}]")
        cur = f"mv{i}"
    return out, cur


def encode_clip(src, offset, dur, out: Path, s, threads, masks=(), clip_start=0.0):
    W, H = s.width, s.height
    info = probe(src)
    v = next(x for x in info["streams"] if x["codec_type"] == "video")
    graph, cur = mask_filters(masks, clip_start, dur, v["width"], v["height"], s.mask_style)
    graph.append(f"[{cur}]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                 f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C0E,setsar=1,fps={s.fps},"
                 f"fade=t=in:st=0:d=0.12[v]")
    audio = any(x["codec_type"] == "audio" for x in info["streams"])
    if audio:
        graph.append("[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
                     f"afade=t=in:d=0.1,afade=t=out:st={max(0, dur - 0.25):.2f}:d=0.25[a]")
    tmp = out.with_name(out.stem + ".tmp.mkv")
    inp = ["-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", src]
    if not audio:
        inp += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    ffmpeg(*inp, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]" if audio else "1:a",
           *_venc(s, threads), *AENC_PCM, "-shortest", tmp)
    tmp.replace(out)


def plate_video(bg, card, out: Path, s, threads):
    """Фон стоит, карточка выезжает снизу с проявлением, звучит короткий «вжух»."""
    d, k = s.plate_seconds, s.width / 1920
    bg_png, card_png = out.with_suffix(".bg.png"), out.with_suffix(".card.png")
    bg.save(bg_png)
    card.save(card_png)
    fc = (f"[0:v]fps={s.fps},format=yuv420p[bg];"
          "[1:v]format=rgba,fade=t=in:st=0:d=0.12:alpha=1[c];"
          f"[bg][c]overlay=x=0:y='{int(60 * k)}*pow(1-min(t/0.22,1),3)',format=yuv420p[v];"
          "[2:a]highpass=f=500,lowpass=f=6000,volume=0.5,"
          "afade=t=in:d=0.06,afade=t=out:st=0.08:d=0.3,"
          f"apad,atrim=0:{d},aformat=sample_rates=48000:channel_layouts=stereo[a]")
    ffmpeg("-loop", "1", "-framerate", s.fps, "-t", d, "-i", bg_png,
           "-loop", "1", "-framerate", s.fps, "-t", d, "-i", card_png,
           "-f", "lavfi", "-t", d, "-i", "anoisesrc=c=pink:r=48000:a=0.5",
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
           *_venc(s, threads), *AENC_PCM, out)
    bg_png.unlink()
    card_png.unlink()


def still_video(img, dur, out: Path, s, threads, fade_in=0.3, fade_out=0.3):
    png = out.with_suffix(".png")
    img.convert("RGB").save(png)
    vf = f"format=yuv420p,fade=t=in:st=0:d={fade_in},fade=t=out:st={dur - fade_out}:d={fade_out}"
    ffmpeg("-loop", "1", "-framerate", s.fps, "-t", dur, "-i", png,
           "-f", "lavfi", "-t", dur, "-i", "anullsrc=r=48000:cl=stereo",
           "-vf", vf, "-map", "0:v", "-map", "1:a", *_venc(s, threads), *AENC_PCM, "-shortest", out)
    png.unlink()


def contact_sheet(src, times, out: Path, width=480):
    """Сетка 2×2 из кадров — чтобы быстро понять, что происходит в моменте."""
    from PIL import Image
    tiles = []
    for i, t in enumerate(times):
        p = out.with_name(f"{out.stem}_{i}.jpg")
        frame(src, t, p, width=width)
        tiles.append(Image.open(p).convert("RGB"))
        p.unlink()
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * 2, h * 2))
    for i, im in enumerate(tiles[:4]):
        sheet.paste(im.resize((w, h)), ((i % 2) * w, (i // 2) * h))
    sheet.save(out, quality=85)
    return out


def _tc(sec):
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


MASKS = "masks.json"


def load_masks(project):
    """Маски из проекта плюс work/<проект>/masks.json (его удобно править после проверки)."""
    import json
    from .project import parse_masks
    extra = {}
    f = project.work / MASKS
    if f.exists():
        extra = json.loads(f.read_text(encoding="utf-8"))
    return {src.id: src.masks + parse_masks(extra.get(src.id)) for src in project.sources}


def _source_media(project, m, w):
    """Файл с моментом и смещение начала момента в нём (для ссылок — скачанный кусок)."""
    src = next(x for x in project.sources if x.id == m.source)
    if src.file:
        return src.file, m.start
    tag = f"{m.start:.1f}-{m.end:.1f}"
    return fetch.section(src.url, m.start, m.end, w / f"src_{m.id}_{tag}.mp4", project.settings.height)


def prepare(project, moments, log=print):
    """Скачивает куски видео и собирает листы проверки рекламы по каждому стримеру."""
    from . import layout
    w = project.work / "render"
    w.mkdir(parents=True, exist_ok=True)
    log(f"Готовлю видео для {len(moments)} моментов…")
    with ThreadPoolExecutor(4) as ex:
        media = dict(zip([m.id for m in moments], ex.map(lambda m: _source_media(project, m, w), moments)))
    masks = load_masks(project)
    sheets = project.work / "layouts"
    sheets.mkdir(exist_ok=True)
    for src in project.sources:
        ms = [m for m in moments if m.source == src.id]
        if not ms:
            continue
        if src.file:  # кадры равномерно по всему стриму
            total = duration(src.file)
            items = [(src.file, total * (i + 0.5) / 9) for i in range(9)]
        else:         # кадры из скачанных кусков
            items = [(media[m.id][0], media[m.id][1] + m.length * k) for m in ms for k in (0.2, 0.5, 0.8)]
            items = items[:: max(1, len(items) // 9)][:9]
        out, sugg = layout.sheet(items, masks[src.id], sheets / f"{src.id}.jpg")
        log(f"  {src.id}: лист проверки {out.name}, кандидатов в рекламу: {len(sugg)}")
    return media


def render(project, moments, meta, log=print):
    s = project.settings
    W, H = s.width, s.height
    w = project.work / "render"
    w.mkdir(parents=True, exist_ok=True)
    project.out.mkdir(parents=True, exist_ok=True)
    jobs = s.jobs or max(1, cpu_count() // 2)
    threads = max(1, cpu_count() // jobs)
    masks = load_masks(project)

    def build(item):
        i, m = item
        info = meta[m.source]
        mk = [x for x in masks[m.source] if x.active(m.start, m.end)]
        key = f"{m.start:.1f}-{m.end:.1f}"
        if mk:  # другие маски — другой файл, чтобы не взять старый клип из кэша
            key += "_m" + hashlib.md5(repr((s.mask_style, mk)).encode()).hexdigest()[:8]
        clip = w / f"clip_{m.id}_{key}.mkv"
        if not clip.exists():
            path, off = _source_media(project, m, w)
            encode_clip(path, off, m.length, clip, s, threads, masks=mk, clip_start=m.start)
        fr = frame(clip, 0.5, w / f"frame_{m.id}.jpg", width=W // 2)
        card = plates.streamer_card(
            W, H, name=info["name"], login=info.get("login"), followers=info.get("followers"),
            label_text=s.followers_label, avatar=info.get("avatar"), tag=m.label,
            title=m.title or None, platform=info.get("platform", "Twitch"))
        plate = w / f"plate_{i:03d}.mkv"
        plate_video(plates.background(fr, W, H), card, plate, s, threads)
        log(f"  готово {i + 1}/{len(moments)}: {info['name']} [{m.label}] {_tc(m.start)}–{_tc(m.end)}")
        return plate, clip

    log(f"Монтаж {len(moments)} моментов, параллельно {jobs}…")
    with ThreadPoolExecutor(jobs) as ex:
        pairs = list(ex.map(build, enumerate(moments)))

    parts = []
    if s.intro:
        p = w / "intro.mkv"
        still_video(plates.intro(W, H, project.title), 2.5, p, s, cpu_count())
        parts.append(p)
    for plate, clip in pairs:
        parts += [plate, clip]
    if s.outro_seconds > 0:
        p = w / "outro.mkv"
        still_video(plates.outro(W, H), s.outro_seconds, p, s, cpu_count(), fade_out=0.6)
        parts.append(p)

    lst = w / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    out = project.out / f"{project.name}.mp4"
    log("Склейка…")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)

    # таймкоды: глава начинается с плашки стримера
    t, chapters = (2.5 if s.intro else 0.0), []
    for m, (plate, clip) in zip(moments, pairs):
        info = meta[m.source]
        chapters.append((t, f"{info['name']} — {m.title or m.label}"))
        t += duration(plate) + duration(clip)
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    desc = description(project, moments, meta, chapters)
    desc_path = project.out / f"{project.name}_описание.txt"
    desc_path.write_text(desc, encoding="utf-8")
    log(f"Готово: {out} ({_tc(duration(out))})")
    log(f"Описание с таймкодами: {desc_path}")
    return out, desc_path


def description(project, moments, meta, chapters):
    used = []
    for m in moments:
        if m.source not in used:
            used.append(m.source)
    lines = [project.title, "", "Стримеры в выпуске:"]
    for sid in used:
        i = meta[sid]
        link = f"https://twitch.tv/{i['login']}" if i.get("login") else ""
        fol = f" ({plates.fmt_count(i['followers'])} фолловеров)" if i.get("followers") is not None else ""
        lines.append(f"▸ {i['name']} — {link}{fol}".rstrip(" —"))
    lines += ["", "Таймкоды:"]
    lines += [f"{_tc(t)} — {name}" for t, name in chapters]
    vods = [(meta[sid]["name"], meta[sid].get("vod_url")) for sid in used if meta[sid].get("vod_url")]
    if vods:
        lines += ["", "Оригинальные трансляции:"] + [f"▸ {n}: {u}" for n, u in vods]
    tags = " ".join(f"#{meta[sid]['login']}" for sid in used if meta[sid].get("login"))
    lines += ["", "Все права на трансляции принадлежат их авторам. Поддержите стримеров подпиской на Twitch.",
              "", "СРЕЗ — лучшие моменты стримов без воды.", "", f"#нарезка #стримы {tags}".strip()]
    return "\n".join(lines) + "\n"
