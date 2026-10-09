"""Ручной отбор моментов (редактура): расшифровки кандидатов, файл отбора, листы кадров.

Порядок работы (подробно — в CLAUDE.md):
    python -m srez analyze     проект.toml          кандидаты по звуку
    python -m srez transcripts проект.toml --top 30 расшифровка лучших кандидатов → work/<проект>/transcripts.md
    (Claude читает расшифровки и пишет projects/<проект>.picks.toml)
    python -m srez picks       проект.toml          применяет отбор, качает куски, ставит зумы на фразы
    python -m srez sheets      проект.toml          листы кадров клипов → work/<проект>/sheets/
    python -m srez prepare     проект.toml          листы рекламы → маски в проект
    python -m srez render      проект.toml

Файл отбора projects/<проект>.picks.toml:

    opener = "vod123-45"      # первый клип (самый сильный и понятный без контекста)
    closer = "vod123-67"      # последний клип (финал)
    pad = 8                   # секунд контекста с каждой стороны клипа

    [[pick]]
    id = "vod123-45"
    title = "Спор, у кого пузо больше"   # название момента на плашке и в таймкодах
    caption = "5 КГ РАЗНИЦЫ!"            # подпись-реакция на зуме (2–4 слова)
    at = "5 кг"                          # с какой фразы начинается зум (ищется в распознанной речи)
    # zoom_at = 12.3                     # или точная секунда клипа; -1 — без зума
    # sfx = [["ding", 8.2]]              # звуки: whoosh, boom, ding, boing, logo
    # hook = 1                           # панчлайн этого клипа — первый в хуке (2, 3 — следующие)
    # from = "0:56:05"                   # точное начало/конец клипа во времени стрима
    # to = "0:57:40"                     # (склеить соседние куски одной истории, обрезать лишнее)
"""
import json
import re
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from .moments import Moment, _quiet_point, select

HOP = 0.5


def _tc(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _sec(tc):
    """'1:02:03' или '62:03' или число → секунды."""
    if isinstance(tc, (int, float)):
        return float(tc)
    sec = 0.0
    for part in str(tc).split(":"):
        sec = sec * 60 + float(part)
    return sec


# ---------------------------------------------------------------- расшифровки

def transcripts(project, top=30, log=print):
    """Расшифровка речи в top лучших кандидатах каждого стрима → transcripts.json и transcripts.md."""
    from .censor import _get_model
    data = json.loads((project.work / "candidates.json").read_text(encoding="utf-8"))
    out_path = project.work / "transcripts.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    per = {}
    for m in data["moments"]:
        per.setdefault(m["source"], []).append(m)
    todo = [m for ms in per.values() for m in sorted(ms, key=lambda m: -m["score"])[:top] if m["id"] not in done]
    log(f"Расшифровываю {len(todo)} кандидатов…")
    for i, m in enumerate(todo, 1):
        audio = next(p for p in (project.work / m["source"]).glob("audio.*") if p.suffix not in (".part", ".ytdl"))
        a = max(0.0, m["start"] - 2)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{a:.2f}", "-t", f"{m['end'] + 2 - a:.2f}",
                              "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                             capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        segs, _ = _get_model().transcribe(x, language="ru", beam_size=1, vad_filter=True,
                                          condition_on_previous_text=False)
        lines = [(round(a + s.start - m["start"], 1), s.text.strip()) for s in segs]
        done[m["id"]] = {"text": lines,
                         "speech": round(sum(len(t) for _, t in lines) / max(1, m["end"] - m["start"]), 1)}
        if i % 10 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            log(f"  {i}/{len(todo)}")
    # обзор для чтения: по стримам, сильные сверху
    md = [f"# Расшифровки кандидатов: {project.title}", ""]
    for sid, ms in per.items():
        md += [f"## {data['sources'][sid]['name']} ({sid})", ""]
        for m in sorted((m for m in ms if m["id"] in done), key=lambda m: -m["score"]):
            t = done[m["id"]]
            text = " / ".join(x for _, x in t["text"])
            md.append(f"- **{m['id']}** {_tc(m['start'])} {m['end'] - m['start']:.0f}с сила {m['score']:.1f} "
                      f"речь {t['speech']}: {text}")
        md.append("")
    (project.work / "transcripts.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Обзор: {project.work / 'transcripts.md'}")


# ---------------------------------------------------------------- отбор

def picks_path(project) -> Path:
    return project.path.with_name(project.path.stem + ".picks.toml")


def _norm(x):
    return re.sub(r"[^\wё ]", "", x.lower().replace("ё", "е"))


def find_phrase(words, phrase):
    """Время первого слова фразы в распознанной речи клипа или None."""
    toks = [_norm(w[2]).strip() for w in words]
    target = _norm(phrase).split()
    for i in range(len(toks)):
        if all(i + j < len(toks) and target[j] in toks[i + j] for j in range(len(target))):
            return words[i][0]
    return None


def apply_picks(project, log=print):
    """Отбор из picks.toml → candidates.json: keep, названия, контекст по краям, порядок, зумы и подписи."""
    from .render import _source_media, _speech
    cfg = tomllib.loads(picks_path(project).read_text(encoding="utf-8"))
    picks = {p["id"]: p for p in cfg.get("pick", [])}
    pad = float(cfg.get("pad", 8))
    cpath = project.work / "candidates.json"
    orig = project.work / "candidates.auto.json"
    if not orig.exists():
        orig.write_text(cpath.read_text(encoding="utf-8"), encoding="utf-8")
    data = json.loads(orig.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in data["moments"]}
    unknown = [k for k in picks if k not in by_id]
    if unknown:
        raise KeyError(f"В candidates.json нет моментов: {', '.join(unknown)}")
    for m in data["moments"]:
        p = picks.get(m["id"])
        m["keep"] = p is not None
        m["title"] = p.get("title", "") if p else ""
        m["caption"] = (p.get("caption") or "-") if p else ""
        m["sfx"] = p.get("sfx", []) if p else []
        m["zoom_at"] = p.get("zoom_at") if p else None
        m["hook"] = int(p.get("hook", 0)) if p else 0
    # больше контекста по краям: режем по паузам, не залезая в соседний выбранный момент
    for sid in {m["source"] for m in data["moments"]}:
        db = np.load(project.work / sid / "loudness.npy")
        ms = sorted((m for m in data["moments"] if m["source"] == sid and m["keep"]), key=lambda m: m["start"])
        total = len(db) * HOP
        for i, m in enumerate(ms):
            lo = ms[i - 1]["end"] if i else 0
            hi = ms[i + 1]["start"] if i + 1 < len(ms) else total
            p = picks[m["id"]]
            if "from" in p:          # границы, заданные вручную, важнее автоматических
                s = _sec(p["from"])
            else:
                s = _quiet_point(db, m["start"] - pad, -3, 2) if pad else m["start"]
            e = _sec(p["to"]) if "to" in p else _quiet_point(db, m["end"] + pad, -2, 3) if pad else m["end"]
            m["start"], m["end"] = round(max(lo, s), 2), round(min(hi, e), 2)
    for key, score in (("opener", 100.0), ("closer", 99.0)):
        if cfg.get(key):
            by_id[cfg[key]]["score"] = score
    cpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # зум на фразу: качаем кусок, распознаём речь клипа и ищем, где сказана фраза
    w = project.work / "render"
    w.mkdir(parents=True, exist_ok=True)
    miss = []
    for m in data["moments"]:
        p = picks.get(m["id"])
        if not p or not p.get("at") or p.get("zoom_at") is not None:
            continue
        mo = Moment(**m)
        path, off = _source_media(project, mo, w)
        t = find_phrase(_speech(project, mo, path, off), p["at"])
        if t is None:
            miss.append(m["id"])
        else:
            m["zoom_at"] = round(max(0.5, t - 0.15), 2)
    cpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    per = {}
    for d in data["moments"]:
        per.setdefault(d["source"], []).append(Moment(**d))
    chosen = select(per, project.settings)
    kept = [m for ms in per.values() for m in ms if m.keep]
    s = project.settings
    est = sum(m.length * (0.9 if s.jumpcut > 0 else 1) + s.sting_seconds for m in chosen) + s.outro_seconds + 6
    log(f"Отобрано: {len(kept)} моментов, {sum(m.length for m in kept) / 60:.1f} мин клипов; "
        f"в ролик {len(chosen)} ≈ {est / 60:.1f} мин после вырезки пауз (цель {s.minutes})")
    dropped = [m.id for m in kept if m not in chosen]
    if dropped:
        log("Не влезли по длине: " + ", ".join(dropped))
    if miss:
        log("Фраза для зума не найдена (поправьте at или задайте zoom_at): " + ", ".join(miss))
    return chosen


# ---------------------------------------------------------------- листы кадров

def sheets(project, moments, log=print):
    """По 4 кадра на клип, 6 клипов на лист → work/<проект>/sheets/clips_NN.jpg."""
    from PIL import Image, ImageDraw
    from .media import frame
    from .plates import LIME, font
    from .render import _source_media
    out = project.work / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    w = project.work / "render"
    TW, TH, ROWS = 400, 225, 6
    rows = []
    for i, m in enumerate(moments):
        src, off = _source_media(project, m, w)
        row = Image.new("RGB", (TW * 4, TH + 28), (12, 12, 14))
        ImageDraw.Draw(row).text((6, 4), f"{i + 1:02d} {m.id} · {m.title} · {m.length:.0f}с", font=font(18, 700),
                                 fill=LIME)
        for k, frac in enumerate((0.1, 0.37, 0.63, 0.9)):
            tmp = out / f".f{k}.jpg"
            frame(src, off + m.length * frac, tmp, width=TW)
            row.paste(Image.open(tmp).convert("RGB").resize((TW, TH)), (k * TW, 28))
            tmp.unlink()
        rows.append(row)
    for n in range(0, len(rows), ROWS):
        part = rows[n:n + ROWS]
        sheet = Image.new("RGB", (TW * 4, (TH + 28) * len(part)))
        for j, r in enumerate(part):
            sheet.paste(r, (0, j * (TH + 28)))
        sheet.save(out / f"clips_{n // ROWS + 1:02d}.jpg", quality=80)
    log(f"Листы кадров: {out} ({(len(rows) + ROWS - 1) // ROWS} шт.)")
