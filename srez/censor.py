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
PAD_BEFORE, PAD_AFTER = 0.05, 0.1   # запас писка вокруг слова, сек

# Корни мата и грубых слов. Слово проверяется целиком (в нижнем регистре, ё → е).
_ROOTS = [
    r"^(на|по|ни|о|а|рас|за|при)?ху[йяеюи]",                    # хуй, нахуя, похуй, охуеть, хуево
    r"пизд",                                                    # пизда, спиздил
    r"^(за|у|вы|на|по|от|отъ|про|при|раз|разъ|въ|съ|подъ|объ|изъ|до|пере|недо|долбо|долба|бо)?еб(?!ол)(а|у|л|н|и|ы|о|е|ш|т|к|с|ь|$)",
    r"еб(ан|ну|уч|ло|лан|ись|ал|ищ)",                              # ...ебаный, уебище внутри слова
    r"^бля",                                                    # блять, бляха
    r"^хун(я|и|ю|е)$",                                          # «хуйня», распознанная как «хуня»
    r"^сук(а|и|е|у|ой|ин|ины|ами|ам)?$",
    r"^муд(ак|ил|о|ач)",
    r"^пид[оа]?р",
    r"^г[ао]ндон",
    r"шлюх",
    r"залуп",
    r"^(на|по|ни)?хер(а|ня|ни|ов|ом|ь)?$",
    r"^(f+u+c+k|shit|bitch)",
    r"^\*+$",                                                   # whisper иногда сам пишет мат звёздочками
]
_BAD = [re.compile(p) for p in _ROOTS]


def is_bad(word: str) -> bool:
    w = re.sub(r"[^a-zа-я*]", "", word.lower().replace("ё", "е"))
    return bool(w) and any(p.search(w) for p in _BAD)


_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    return _model


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
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
    return [(round(float(a), 2), round(float(b), 2)) for a, b in out]


def spans(path, offset: float, dur: float, cache: Path) -> list[tuple[float, float]]:
    """Отрезки мата (секунды от начала клипа) в куске [offset, offset + dur] файла path.

    В кэше хранятся распознанные слова; мат по ним считается заново каждый раз,
    чтобы новые слова в списке мата доходили и до уже распознанных клипов."""
    if cache.exists():
        return bad_spans([tuple(w) for w in json.loads(cache.read_text(encoding="utf-8"))["words"]], dur)
    segs, _ = _get_model().transcribe(_pcm(path, offset, dur), language="ru", beam_size=1, vad_filter=True,
                                      word_timestamps=True, condition_on_previous_text=False)
    words = [(float(w.start), float(w.end), w.word) for sg in segs for w in (sg.words or [])]
    out = bad_spans(words, dur)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"spans": out, "words": [[round(a, 2), round(b, 2), w] for a, b, w in words]},
                                ensure_ascii=False), encoding="utf-8")
    return out


def audio_filter(bleeps, dur, style="beep"):
    """Фильтры: глушим речь на отрезках и подкладываем писк. Вход [sp0], выход [sp]."""
    if not bleeps:
        return ["[sp0]anull[sp]"]
    cond = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in bleeps)
    out = [f"[sp0]volume=0:enable='{cond}'[muted]"]
    if style != "beep":
        return out + ["[muted]anull[sp]"]
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    out.append(f"sine=f=1000:r=48000:d={dur:.3f},volume=1.8,{fmt},volume=0:enable='not({cond})'[bp]")
    out.append(f"[muted]{fmt}[mf];[mf][bp]amix=inputs=2:normalize=0:duration=first[sp]")
    return out
