"""Сборка ролика: клипы, плашки стримеров, концовка, склейка, таймкоды."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import fetch, plates
from .media import cpu_count, duration, ffmpeg, frame, has_audio

AENC_PCM = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]


def _venc(s, threads):
    return ["-c:v", "libx264", "-preset", s.preset, "-crf", str(s.crf), "-pix_fmt", "yuv420p",
            "-r", str(s.fps), "-g", str(s.fps * 2), "-threads", str(threads)]


def encode_clip(src, offset, dur, out: Path, s, threads):
    W, H = s.width, s.height
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C0E,setsar=1,fps={s.fps},"
          f"fade=t=in:st=0:d=0.15")
    af = ("loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
          f"afade=t=in:d=0.12,afade=t=out:st={max(0, dur - 0.3):.2f}:d=0.3")
    tmp = out.with_name(out.stem + ".tmp.mkv")
    inp = ["-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", src]
    if has_audio(src):
        ffmpeg(*inp, "-map", "0:v:0", "-map", "0:a:0", "-vf", vf, "-af", af,
               *_venc(s, threads), *AENC_PCM, tmp)
    else:
        ffmpeg(*inp, "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
               "-map", "0:v:0", "-map", "1:a", "-vf", vf, *_venc(s, threads), *AENC_PCM, "-shortest", tmp)
    tmp.replace(out)


def plate_video(bg, card, out: Path, s, threads):
    """Фон стоит, карточка выезжает снизу с проявлением, звучит короткий «вжух»."""
    d, k = s.plate_seconds, s.width / 1920
    bg_png, card_png = out.with_suffix(".bg.png"), out.with_suffix(".card.png")
    bg.save(bg_png)
    card.save(card_png)
    fc = (f"[0:v]fps={s.fps},format=yuv420p[bg];"
          "[1:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1[c];"
          f"[bg][c]overlay=x=0:y='{int(70 * k)}*pow(1-min(t/0.4,1),3)',format=yuv420p[v];"
          "[2:a]highpass=f=500,lowpass=f=6000,volume=0.5,"
          "afade=t=in:d=0.1,afade=t=out:st=0.12:d=0.5,"
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


def render(project, moments, meta, log=print):
    s = project.settings
    W, H = s.width, s.height
    w = project.work / "render"
    w.mkdir(parents=True, exist_ok=True)
    project.out.mkdir(parents=True, exist_ok=True)
    jobs = s.jobs or max(1, cpu_count() // 2)
    threads = max(1, cpu_count() // jobs)
    srcs = {x.id: x for x in project.sources}

    def build(item):
        i, m = item
        src, info = srcs[m.source], meta[m.source]
        tag = f"{m.start:.1f}-{m.end:.1f}"
        clip = w / f"clip_{m.id}_{tag}.mkv"
        if not clip.exists():
            if src.file:
                path, off = src.file, m.start
            else:
                path, off = fetch.section(src.url, m.start, m.end, w / f"src_{m.id}_{tag}.mp4", s.height)
            encode_clip(path, off, m.length, clip, s, threads)
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
