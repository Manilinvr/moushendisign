"""Анализ стримов: данные о стримере, звук, чат → кандидаты в нарезку."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import fetch, twitch
from .media import duration
from .moments import Moment, find
from .render import _tc, contact_sheet
from .signals import classify, load_chat

CANDIDATES = "candidates.json"


def _json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def streamer_meta(src, sw, log):
    """Имя, логин, фолловеры и аватар. Ручные значения из проекта важнее."""
    meta = {"name": src.name or src.twitch or src.id, "login": src.twitch, "followers": src.followers,
            "avatar": str(src.avatar) if src.avatar else None, "platform": src.platform,
            "vod_url": src.vod_url, "title": src.title}
    if not src.twitch or src.platform.lower() != "twitch":
        return meta
    cache = sw / "twitch_user.json"
    info = _json(cache)
    if info is None:
        try:
            info = twitch.user_info(src.twitch)
            _save(cache, info)
            log(f"  {info['name']}: {info['followers']:,} фолловеров".replace(",", " "))
        except Exception as e:  # сеть, Twitch недоступен — не повод останавливаться
            log(f"  ! Не удалось получить данные канала {src.twitch} с Twitch: {e}")
            return meta
    meta["name"] = src.name or info["name"]
    if meta["followers"] is None:
        meta["followers"] = info["followers"]
    if not meta["avatar"] and info.get("avatar_url"):
        p = sw / "avatar.jpg"
        try:
            if not p.exists():
                twitch.download(info["avatar_url"], p)
            meta["avatar"] = str(p)
        except Exception:
            pass
    return meta


def analyze_source(project, src, log):
    s = project.settings
    sw = project.work / src.id
    sw.mkdir(parents=True, exist_ok=True)
    tag = src.twitch or src.id
    log(f"[{tag}] начинаю")

    if src.url:
        info = _json(sw / "info.json")
        if info is None:
            info = fetch.info(src.url)
            info = {k: info.get(k) for k in ("id", "title", "duration", "uploader", "uploader_id",
                                             "channel_id", "timestamp", "extractor_key")}
            _save(sw / "info.json", info)
        src.title = src.title or info.get("title")
        if not src.twitch and "twitch" in (info.get("extractor_key") or "").lower():
            src.twitch = (info.get("uploader_id") or "").lower() or None
            src.name = src.name or info.get("uploader")
        log(f"[{tag}] качаю только звук…")
        audio = fetch.audio(src.url, sw)
    else:
        audio = src.file

    curve = sw / "loudness.npy"
    if curve.exists():
        db = np.load(curve)
    else:
        log(f"[{tag}] слушаю звук…")
        from .signals import loudness
        db = loudness(audio)
        np.save(curve, db)
    total = len(db) * 0.5
    log(f"[{tag}] длительность {_tc(total)}")

    msgs = []
    chat = src.chat
    if not chat and src.url:
        m = re.search(r"twitch\.tv/videos/(\d+)", src.url)
        if m:
            chat = sw / "chat.jsonl"
            if not chat.exists():
                log(f"[{tag}] качаю чат…")
                try:
                    twitch.download_chat(m.group(1), chat, length=info.get("duration"))
                except Exception as e:
                    log(f"[{tag}] ! чат не скачался ({e}), ищу моменты только по звуку")
                    chat = None
    if chat and chat.exists():
        msgs = load_chat(chat)
        log(f"[{tag}] чат: {len(msgs):,} сообщений".replace(",", " "))

    limit = int(s.minutes * 60 / max(20, (s.pre_roll + s.post_roll))) * 2 + 10
    found, score = find(src.id, db, msgs, s, limit=limit)
    np.save(sw / "score.npy", score)
    meta = streamer_meta(src, sw, log)
    log(f"[{tag}] найдено моментов: {len(found)}")
    return src.id, found, meta, msgs


def chat_excerpt(msgs, m, s, n=8):
    a, b = m.peak - 4 + s.chat_delay, m.peak + 14 + s.chat_delay
    window = [txt for t, txt in msgs if a <= t <= b]
    react = [x for x in window if classify(x)]
    other = [x for x in window if not classify(x) and len(x) > 12]
    seen, out = set(), []
    for x in react[: n // 2] + sorted(other, key=len, reverse=True)[: n] + react[n // 2:]:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= n:
            break
    return out, len(window)


def analyze(project, log=print):
    project.work.mkdir(parents=True, exist_ok=True)
    jobs = min(len(project.sources), 4)
    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(lambda src: analyze_source(project, src, log), project.sources))

    data = {"title": project.title, "sources": {}, "moments": []}
    review = project.work / "review"
    review.mkdir(exist_ok=True)
    lines = [f"# Кандидаты: {project.title}", "",
             "Поставьте `\"keep\": false` в candidates.json у лишних моментов и запустите `render`.", ""]
    for (sid, found, meta, msgs), src in zip(results, project.sources):
        data["sources"][sid] = meta
        data["moments"] += [m.to_dict() for m in found]
        lines += [f"## {meta['name']} ({sid})", ""]
        for m in sorted(found, key=lambda m: -m.score):
            ex, n = chat_excerpt(msgs, m, project.settings) if msgs else ([], 0)
            lines.append(f"### {m.id} · {_tc(m.start)}–{_tc(m.end)} · {m.label} · сила {m.score:.1f}")
            if src.file:
                sheet = review / f"{m.id}.jpg"
                if not sheet.exists():
                    ts = [m.start + 1, m.peak - 2, m.peak + 2, m.end - 1]
                    contact_sheet(src.file, ts, sheet)
                lines.append(f"![{m.id}]({sheet.relative_to(project.work).as_posix()})")
            if ex:
                lines.append(f"Чат ({n} сообщ.): " + " | ".join(x.replace("\n", " ")[:80] for x in ex))
            lines.append("")
    _save(project.work / CANDIDATES, data)
    (project.work / "review.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"Кандидаты: {project.work / CANDIDATES}")
    log(f"Обзор для проверки: {project.work / 'review.md'}")
    return data


def load_candidates(project):
    data = _json(project.work / CANDIDATES)
    if data is None:
        raise FileNotFoundError("Сначала запустите analyze")
    per_source = {}
    for d in data["moments"]:
        per_source.setdefault(d["source"], []).append(Moment(**d))
    return data, per_source
