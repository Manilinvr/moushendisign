"""Сборка ролика: клипы, переходы (или плашки стримеров), хук, концовка, склейка, таймкоды."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import censor, edit, fetch, plates, sfx
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


def encode_clip(src, offset, dur, out: Path, s, threads, masks=(), clip_start=0.0, bleeps=(),
                edit=None, card=None, side="left", assets=None):
    """Клип 1920×1080: реклама закрыта, мат запикан. В стиле sting ещё зум на реакцию с «бум»,
    подпись-реакция, вырезанные паузы и плашка стримера поверх первых секунд."""
    W, H, fps = s.width, s.height, s.fps
    k = W / 1920
    info = probe(src)
    v = next(x for x in info["streams"] if x["codec_type"] == "video")
    audio = any(x["codec_type"] == "audio" for x in info["streams"])
    edit = edit or {}
    inp = ["-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", src]
    n_in = 1

    def add_input(*args):
        nonlocal n_in
        inp.extend(args)
        n_in += 1
        return n_in - 1

    graph, cur = mask_filters(masks, clip_start, dur, v["width"], v["height"], s.mask_style)
    fit = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C0E,setsar=1,fps={fps}")
    zoom = edit.get("zoom")
    if zoom:  # резкий зум на лицо с затухающей тряской
        t0, t1, (bx, by, bw, bh) = zoom
        a = bw * 0.012
        x = f"{bx}+{a:.1f}*sin(2*PI*9*t)*exp(-t/0.3)"
        y = f"{by}+{a:.1f}*cos(2*PI*7*t)*exp(-t/0.3)"
        graph += [f"[{cur}]split=3[z0][z1][z2]",
                  f"[z0]trim=end={t0:.4f},setpts=PTS-STARTPTS,{fit}[p0]",
                  f"[z1]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
                  f"crop={bw}:{bh}:'{x}':'{y}',{fit}[p1]",
                  f"[z2]trim=start={t1:.4f},setpts=PTS-STARTPTS,{fit}[p2]",
                  "[p0][p1][p2]concat=n=3:v=1:a=0[zc]"]
    else:
        graph.append(f"[{cur}]{fit}[zc]")
    cur = "zc"
    cap = edit.get("caption")
    if cap:   # подпись снизу по центру; если в это время ещё висит плашка — сверху
        c0, c1, png = cap
        ci = add_input("-loop", "1", "-framerate", fps, "-t", f"{dur:.3f}", "-i", png)
        top = card and c0 < s.lower_third + 0.4
        y = f"{H * 0.13:.0f}-h/2" if top else f"{H * 0.80:.0f}-h/2"
        graph.append(f"[{cur}][{ci}:v]overlay=x=(W-w)/2:y={y}:enable='between(t,{c0:.2f},{c1:.2f})'[cv]")
        cur = "cv"
    keep = edit.get("keep") or [(0.0, dur)]
    new_dur = sum(b - a for a, b in keep)
    cut = len(keep) > 1 or keep[0][0] > 0 or keep[-1][1] < dur - 1e-3
    if cut:   # вырезаем паузы: кадры и звук по одним и тем же отрезкам
        expr = "+".join(f"between(t,{a:.4f},{b - 0.5 / fps:.4f})" for a, b in keep)
        graph.append(f"[{cur}]select='{expr}',setpts=N/({fps}*TB)[sv]")
        cur = "sv"
    if card and s.lower_third > 0:   # плашка стримера выезжает сбоку, висит и уезжает
        li = add_input("-loop", "1", "-framerate", fps, "-t", f"{new_dur:.3f}", "-i", card)
        L, m = s.lower_third, int(56 * k)
        ease_in, ease_out = "(1-pow(1-t/0.25,3))", f"pow((t-{L})/0.25,2)"
        if side == "right":
            x = f"if(lt(t,0.25),W-(w+{m})*{ease_in},if(lt(t,{L}),W-w-{m},W-w-{m}+(w+{m})*{ease_out}))"
        else:
            x = f"if(lt(t,0.25),-w+(w+{m})*{ease_in},if(lt(t,{L}),{m},{m}-(w+{m})*{ease_out}))"
        graph.append(f"[{cur}][{li}:v]overlay=x='{x}':y=H-h-{m}:enable='lt(t,{L + 0.3:.2f})'[lv]")
        cur = "lv"
    graph.append(f"[{cur}]format=yuv420p[v]")

    if audio:
        graph.append("[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[sp0]")
    else:
        ai = add_input("-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo")
        graph.append(f"[{ai}:a]anull[sp0]")
    graph += censor.audio_filter(bleeps, dur, s.censor)
    acur = "sp"
    cues = edit.get("sfx") or []
    if cues and assets:
        labels = []
        for j, (name, t) in enumerate(cues):
            si = add_input("-i", sfx.path(name, assets / "sfx"))
            ms = int(max(0.0, t) * 1000)
            graph.append(f"[{si}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                         f"adelay={ms}|{ms},volume={SFX_GAIN.get(name, 0.6)}[fx{j}]")
            labels.append(f"[fx{j}]")
        graph.append(f"[{acur}]{''.join(labels)}amix=inputs={len(labels) + 1}:normalize=0:duration=first[mx]")
        acur = "mx"
    if cut:
        expr = "+".join(f"between(t,{a:.4f},{b - 1e-4:.4f})" for a, b in keep)
        graph.append(f"[{acur}]asetnsamples=n=160:p=0,aselect='{expr}',asetpts=N/SR/TB[sa]")
        acur = "sa"
    fade = 0.25 if s.transition == "plate" else 0.12
    graph.append(f"[{acur}]afade=t=in:d=0.05,afade=t=out:st={max(0, new_dur - fade):.3f}:d={fade}[a]")

    tmp = out.with_name(out.stem + ".tmp.mkv")
    ffmpeg(*inp, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
           *_venc(s, threads), *AENC_PCM, "-t", f"{new_dur:.3f}", tmp)
    tmp.replace(out)
    return new_dur


SFX_GAIN = {"boom": 0.75, "ding": 0.5, "boing": 0.55, "whoosh": 0.6, "logo": 0.8}


def sting_video(a_img, b_img, out: Path, s, threads, sound):
    """Переход: лаймовый срез с логотипом между последним кадром клипа и первым кадром следующего."""
    from PIL import Image
    a = Image.open(a_img).convert("RGB").resize((s.width, s.height))
    b = Image.open(b_img).convert("RGB").resize((s.width, s.height))
    n = max(2, round(s.sting_seconds * s.fps))
    tmp = out.with_name(out.stem + "_f")
    tmp.mkdir(exist_ok=True)
    for i in range(n):
        plates.sting_frame(a, b, i / (n - 1)).save(tmp / f"{i:03d}.png")
    d = n / s.fps
    ffmpeg("-framerate", s.fps, "-i", tmp / "%03d.png", "-i", sound,
           "-filter_complex", f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,apad,atrim=0:{d:.3f}[a]",
           "-map", "0:v", "-map", "[a]", *_venc(s, threads), *AENC_PCM, "-t", f"{d:.3f}", out)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def hook_video(pieces, out: Path, s, threads, music=None):
    """Хук в начале ролика: несколько панчлайнов подряд под тихую музыку."""
    inp, graph, lab = [], [], ""
    for i, (clip, t, d) in enumerate(pieces):
        inp += ["-ss", f"{t:.3f}", "-t", f"{d:.3f}", "-i", clip]
        graph.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        lab += f"[v{i}][a{i}]"
    total = sum(d for _, _, d in pieces)
    graph.append(f"{lab}concat=n={len(pieces)}:v=1:a=1[v][hc]")
    if music:
        inp += ["-ss", "8", "-t", f"{total:.3f}", "-i", music]
        graph.append(f"[{len(pieces)}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.22,"
                     f"afade=t=in:d=0.2,afade=t=out:st={max(0, total - 0.3):.2f}:d=0.3[mb];"
                     "[hc][mb]amix=inputs=2:normalize=0:duration=first,"
                     "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]")
    else:
        graph.append("[hc]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]")
    ffmpeg(*inp, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
           *_venc(s, threads), *AENC_PCM, out)


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


def still_video(img, dur, out: Path, s, threads, fade_in=0.3, fade_out=0.3, music=None):
    png = out.with_suffix(".png")
    img.convert("RGB").save(png)
    vf = f"format=yuv420p,fade=t=in:st=0:d={fade_in},fade=t=out:st={dur - fade_out}:d={fade_out}"
    if music:   # концовка под музыку: тише и с затуханием
        aud = ["-ss", "40", "-t", dur, "-i", music]
        af = ["-af", "loudnorm=I=-18:TP=-2:LRA=11,aresample=48000,aformat=sample_rates=48000:"
                     f"channel_layouts=stereo,afade=t=in:d=0.6,afade=t=out:st={dur - 2}:d=2"]
    else:
        aud, af = ["-f", "lavfi", "-t", dur, "-i", "anullsrc=r=48000:cl=stereo"], []
    ffmpeg("-loop", "1", "-framerate", s.fps, "-t", dur, "-i", png, *aud,
           "-vf", vf, *af, "-map", "0:v", "-map", "1:a", *_venc(s, threads), *AENC_PCM, "-t", dur, out)
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


def censor_spans(project, moments, log=print):
    """Где в каждом клипе мат: распознаём речь клипов (результат кэшируется)."""
    w = project.work / "render"
    log(f"Ищу мат в {len(moments)} клипах…")
    out = {}
    for i, m in enumerate(moments, 1):
        path, off = _source_media(project, m, w)
        cache = project.work / "censor" / f"{m.id}_{m.start:.1f}-{m.end:.1f}.json"
        out[m.id] = censor.spans(path, off, m.length, cache)
        if i % 10 == 0 or i == len(moments):
            log(f"  проверено {i}/{len(moments)}, запикать: {sum(len(v) for v in out.values())} слов")
    return out


def _speech(project, m, path, off):
    """Слова клипа с временем (распознавание общее с запикиванием мата, кэшируется)."""
    cache = project.work / "censor" / f"{m.id}_{m.start:.1f}-{m.end:.1f}.json"
    try:
        censor.spans(path, off, m.length, cache)
    except ImportError:          # нет faster-whisper: без подписей, паузы — только по звуку
        return []
    return json.loads(cache.read_text(encoding="utf-8"))["words"]


def render(project, moments, meta, log=print):
    s = project.settings
    W, H = s.width, s.height
    w = project.work / "render"
    w.mkdir(parents=True, exist_ok=True)
    project.out.mkdir(parents=True, exist_ok=True)
    jobs = s.jobs or max(1, cpu_count() // 2)
    threads = max(1, cpu_count() // jobs)
    masks = load_masks(project)
    bleeps = censor_spans(project, moments, log) if s.censor != "off" else {}
    styled = s.transition != "plate"
    assets = project.work.parent / "_assets"
    srcs = {x.id: x for x in project.sources}
    music, credit = (sfx.music(s.music, assets / "music") if s.music else (None, ""))

    def build(item):
        i, m = item
        info = meta[m.source]
        mk = [x for x in masks[m.source] if x.active(m.start, m.end)]
        key = f"{m.start:.1f}-{m.end:.1f}"
        if mk:  # другие маски — другой файл, чтобы не взять старый клип из кэша
            key += "_m" + hashlib.md5(repr((s.mask_style, mk)).encode()).hexdigest()[:8]
        bl = bleeps.get(m.id, [])
        if bl:  # то же для запикивания
            key += "_b" + hashlib.md5(repr((s.censor, bl)).encode()).hexdigest()[:8]
        path, off = _source_media(project, m, w)
        ed, card, side = None, None, srcs[m.source].plate_side
        if styled:
            ed = edit.decide(m, path, off, _speech(project, m, path, off), s,
                             project.work / "edit" / f"{m.id}_{m.start:.1f}-{m.end:.1f}.json", assets)
            if ed["caption"]:
                c0, c1, text = ed["caption"]
                png = w / f"cap_{m.id}_{hashlib.md5(text.encode()).hexdigest()[:8]}.png"
                plates.caption(W, text).save(png)
                ed["caption"] = (c0, c1, png)
            if s.lower_third > 0:
                card = w / f"card_{m.id}.png"
                plates.lower_third(W, name=info["name"], login=info.get("login"), followers=info.get("followers"),
                                   title=m.title or None, avatar=info.get("avatar"),
                                   platform=info.get("platform", "Twitch")).save(card)
            sig = (ed, s.lower_third, side, m.title, info["name"], info.get("followers"), s.fps)
            key += "_s" + hashlib.md5(repr(sig).encode()).hexdigest()[:8]
        clip = w / f"clip_{m.id}_{key}.mkv"
        if not clip.exists():
            encode_clip(path, off, m.length, clip, s, threads, masks=mk, clip_start=m.start, bleeps=bl,
                        edit=ed, card=card, side=side, assets=assets)
        log(f"  готово {i + 1}/{len(moments)}: {info['name']} — {m.title or m.label}")
        if styled:
            z = ed["zoom"]
            return {"clip": clip, "zoom": edit.remap(z[0], ed["keep"]) if z else None}
        fr = frame(clip, 0.5, w / f"frame_{m.id}.jpg", width=W // 2)
        plate_card = plates.streamer_card(
            W, H, name=info["name"], login=info.get("login"), followers=info.get("followers"),
            label_text=s.followers_label, avatar=info.get("avatar"), tag=m.label,
            title=m.title or None, platform=info.get("platform", "Twitch"))
        plate = w / f"plate_{i:03d}.mkv"
        plate_video(plates.background(fr, W, H), plate_card, plate, s, threads)
        return {"clip": clip, "plate": plate}

    log(f"Монтаж {len(moments)} моментов, параллельно {jobs}…")
    with ThreadPoolExecutor(jobs) as ex:
        built = list(ex.map(build, enumerate(moments)))

    parts, starts = [], []        # starts — где в ролике начинается каждый клип (для таймкодов)
    if s.intro:
        p = w / "intro.mkv"
        still_video(plates.intro(W, H, project.title), 2.5, p, s, cpu_count())
        parts.append(p)
    if styled:
        def edge(clip, at_end):
            p = w / f"edge_{clip.stem}_{'b' if at_end else 'a'}.jpg"
            frame(clip, max(0.0, duration(clip) - 0.05) if at_end else 0.0, p)
            return p

        hook_n = min(s.hook, len(built))
        if hook_n:   # хук: самые сильные моменты с зумом (кроме первого и последнего клипа), по 1,8 с
            cand = sorted((-moments[i].score, i) for i in range(1, len(built) - 1) if built[i]["zoom"] is not None)
            best = [built[i] for _, i in cand[:hook_n]]
            if best:
                hook = w / "hook.mkv"
                hook_video([(b["clip"], max(0.0, b["zoom"] - 0.35), 1.8) for b in best], hook, s, cpu_count(),
                           music)
                parts.append(hook)
                st = w / "sting_hook.mkv"
                sting_video(edge(hook, True), edge(built[0]["clip"], False), st, s, cpu_count(),
                            sfx.path("logo", assets / "sfx"))
                parts.append(st)
        log("Переходы…")
        for i, b in enumerate(built):
            starts.append(len(parts))
            parts.append(b["clip"])
            if i + 1 < len(built):
                st = w / f"sting_{i:03d}.mkv"
                sting_video(edge(b["clip"], True), edge(built[i + 1]["clip"], False), st, s, cpu_count(),
                            sfx.path("whoosh", assets / "sfx"))
                parts.append(st)
    else:
        for b in built:
            starts.append(len(parts))
            parts += [b["plate"], b["clip"]]
    if s.outro_seconds > 0:
        p = w / "outro.mkv"
        still_video(plates.outro(W, H), s.outro_seconds, p, s, cpu_count(), fade_out=0.6, music=music)
        parts.append(p)

    lst = w / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    out = project.out / f"{project.name}.mp4"
    log("Склейка…")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)

    # таймкоды: глава начинается с перехода (или плашки) перед клипом
    durs = [duration(p) for p in parts]
    chapters = []
    for m, k in zip(moments, starts):
        t = sum(durs[:k]) - (durs[k - 1] if styled and k > 0 and parts[k - 1].name.startswith("sting") else 0)
        chapters.append((t, f"{meta[m.source]['name']} — {m.title or m.label}"))
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    desc = description(project, moments, meta, chapters, credit)
    desc_path = project.out / f"{project.name}_описание.txt"
    desc_path.write_text(desc, encoding="utf-8")
    log(f"Готово: {out} ({_tc(duration(out))})")
    log(f"Описание с таймкодами: {desc_path}")
    return out, desc_path


def description(project, moments, meta, chapters, credit=""):
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
              "", "СРЕЗ — лучшие моменты стримов без воды."]
    if credit:
        lines += ["", "Музыка:", credit]
    lines += ["", f"#нарезка #стримы {tags}".strip()]
    return "\n".join(lines) + "\n"
