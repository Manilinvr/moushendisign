"""Мат в клипах: распознаём речь с точным временем слов и запикиваем ругательства.

Нужен faster-whisper (pip install faster-whisper). Модель скачивается один раз,
распознаётся только звук выбранных клипов, результат кэшируется.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

MODEL = "small"
CHECK_MODEL = "medium"              # проверочный проход только для мата: small часто пропускает короткий мат
CHECKS = (("words_novad", MODEL), ("words_check", CHECK_MODEL))   # проверочные проходы — без фильтра тишины
PAD_BEFORE, PAD_AFTER = 0.12, 0.2   # запас писка вокруг слова, сек: время слов распознаватель даёт с ошибкой
MERGE_GAP = 0.3                     # писки ближе этого сливаются в один — в зазоры проскакивают куски слов
BLEEP_VERSION = 2                   # поменялся звук писка — поднять, чтобы клипы и куски шортсов пересобрались

# Корни мата и грубых слов. Слово проверяется целиком (в нижнем регистре, ё → е).
_ROOTS = [
    r"^(на|по|ни|не|о|а|ва|во|рас|за|при|от|до)?ху[йяеюи]",       # хуй, нахуя, похуй, охуеть, вахуе, отхуя
    r"пизд",                                                    # пизда, спиздил
    r"пизн",                                                    # «припизднутый», распознанное как «припизнутый»
    r"^(за|у|вы|на|по|от|отъ|про|при|раз|разъ|въ|съ|подъ|объ|изъ|до|пере|недо|долбо|долба|бо)?еб(?!ол)(а|у|л|н|и|ы|о|е|ш|т|к|с|ь|$)",
    r"еб(ан|ну|уч|ло|лан|ись|ал|ищ)",                              # ...ебаный, уебище внутри слова
    r"^бля",                                                    # блять, бляха
    r"^хун(я|и|ю|е)$",                                          # «хуйня», распознанная как «хуня»
    r"^(на|по)хуа",                                             # «нахуя», распознанное как «нахуай» (шортсы №2)
    r"^съевал",                                                 # «съебался», распознанное как «съевался»
    r"^сук(а|и|е|у|ой|ин|ины|ами|ам)?$",
    r"^муд(ак|ил|о|ач)",
    r"^пид[оа]?р",
    r"^г[ао]ндон",
    r"шлюх",
    r"шелюх",                                                   # «шлюха», распознанная как «шелюха»
    r"^бать$",                                                  # «ебать», распознанное как «бать»
    r"^долбо[её]",                                              # «долбоёб», распознанный как «долбоёк»
    r"залуп",
    r"^(на|по|ни)?хер(а|ня|ни|ов|ом|ь)?$",
    r"^(f+u+c+k|shit|bitch)",
    r"^\*+$",                                                   # whisper иногда сам пишет мат звёздочками
]
_BAD = [re.compile(p) for p in _ROOTS]


def is_bad(word: str) -> bool:
    w = re.sub(r"[^a-zа-я*]", "", word.lower().replace("ё", "е"))
    return bool(w) and any(p.search(w) for p in _BAD)


_models = {}


def _get_model(name=MODEL):
    if name not in _models:
        from faster_whisper import WhisperModel
        _models[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    return _models[name]


def _pcm(path, start, dur) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                          "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


def merge_hyphens(words):
    """«Ха -ха -ха», «теперь -то», «уй -ё -бищ» распознаются кусками — склеиваем в одно слово."""
    out = []
    for a, b, wd in words:
        if out and wd.strip().startswith("-") and a - out[-1][1] < 0.6:
            out[-1] = (out[-1][0], b, out[-1][2].rstrip() + wd.strip())
        else:
            out.append((a, b, wd))
    return out


def bad_spans(words, dur: float) -> list[tuple[float, float]]:
    """Отрезки писка по словам (a, b, слово): мат с запасом, соседние отрезки сливаются."""
    words = [tuple(w) for w in words]
    out = []
    for a, b, word in sorted(set(words) | set(merge_hyphens(words))):   # и куски, и склеенное слово
        if is_bad(word):
            a, b = max(0.0, a - PAD_BEFORE), min(dur, b + PAD_AFTER)
            if out and a <= out[-1][1] + MERGE_GAP:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
    return [(round(float(a), 2), round(float(b), 2)) for a, b in out]


def union(*lists) -> list[tuple[float, float]]:
    """Несколько списков отрезков → один: пересекающиеся сливаются."""
    out = []
    for a, b in sorted(x for xs in lists for x in xs):
        if out and a <= out[-1][1] + MERGE_GAP:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [(round(float(a), 2), round(float(b), 2)) for a, b in out]


def cached_spans(d: dict, dur: float) -> list[tuple[float, float]]:
    """Мат по всем спискам слов из кэша (основное распознавание и проверочные)."""
    return union(*(bad_spans(d.get(key, []), dur) for key in ("words", *dict(CHECKS))))


def _recognize(path, offset, dur, vad, model=MODEL):
    segs, _ = _get_model(model).transcribe(_pcm(path, offset, dur), language="ru", beam_size=1, vad_filter=vad,
                                           word_timestamps=True, condition_on_previous_text=False)
    return [[round(float(w.start), 2), round(float(w.end), 2), w.word] for sg in segs for w in (sg.words or [])]


def spans(path, offset: float, dur: float, cache: Path) -> list[tuple[float, float]]:
    """Отрезки мата (секунды от начала клипа) в куске [offset, offset + dur] файла path.

    Основной проход (small, с фильтром тишины) — его слова идут в подписи, зумы и вырезку пауз.
    Проверочные (small и medium без фильтра) — только для мата: фильтр тишины выбрасывает целые фразы
    тихой речи (голос из телефона, озвучка в игре), а каждая модель где-то пропускает короткий мат
    («блять» между словами) — пищит то, что услышала хоть одна. В кэше хранятся все списки слов;
    мат по ним считается заново каждый раз, чтобы новые корни доходили и до уже распознанных клипов."""
    d = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    fresh = any(key not in d for key in ("words", *dict(CHECKS)))
    if "words" not in d:
        d["words"] = _recognize(path, offset, dur, True)
    for key, model in CHECKS:
        if key not in d:
            d[key] = _recognize(path, offset, dur, False, model)
    out = cached_spans(d, dur)
    if fresh or d.get("spans") != [list(x) for x in out]:
        d["spans"] = out
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return out


def audio_filter(bleeps, dur, style="beep"):
    """Фильтры: глушим речь на отрезках и подкладываем писк. Вход [sp0], выход [sp]."""
    if not bleeps:
        return ["[sp0]anull[sp]"]
    cond = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in bleeps)
    # enable у звука срабатывает на границах кадров, а loudnorm отдаёт кадры по 100 мс — края мата
    # проскакивали мимо заглушения. Режем звук на кадры по 5 мс, чтобы заглушение совпало с писком.
    out = [f"[sp0]asetnsamples=n=240:p=0,volume=0:enable='{cond}'[muted]"]
    if style != "beep":
        return out + ["[muted]anull[sp]"]
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    out.append(f"sine=f=1000:r=48000:d={dur:.3f}:samples_per_frame=240,volume=1.8,{fmt},"
               f"volume=0:enable='not({cond})'[bp]")
    out.append(f"[muted]{fmt}[mf];[mf][bp]amix=inputs=2:normalize=0:duration=first[sp]")
    return out
