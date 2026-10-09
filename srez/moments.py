"""Поиск пиковых моментов, их границ и сборка списка для ролика."""
from dataclasses import asdict, dataclass, field

import numpy as np

from .signals import HOP, LABELS, audio_score, chat_curves, chat_score, smooth


@dataclass
class Moment:
    id: str
    source: str
    start: float
    end: float
    peak: float
    score: float
    label: str = "момент"
    keep: bool = True
    title: str = ""
    zoom_at: float | None = None   # секунда клипа для зума на реакцию; None — сам найдёт, -1 — без зума
    caption: str = ""              # подпись-реакция; "" — из распознанной речи, "-" — без подписи
    sfx: list = field(default_factory=list)   # [["ding", 12.3], …] — звуки в секундах клипа

    @property
    def length(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d.update(start=round(self.start, 2), end=round(self.end, 2), peak=round(self.peak, 2),
                 score=round(self.score, 2))
        return d


def combined_score(db, msgs, s):
    """Итоговая «интересность» на каждую точку сетки + категории чата."""
    a = audio_score(db)
    if not msgs:
        return smooth(a, 4), None, None
    rate, share, cats = chat_curves(msgs, len(db))
    c = chat_score(rate, share)
    shift = int(s.chat_delay / HOP)
    c = np.concatenate([c[shift:], np.full(shift, c[-1])])          # чат → время события
    share = np.concatenate([share[:, shift:], np.repeat(share[:, -1:], shift, axis=1)], axis=1)
    return smooth(0.6 * c + 0.4 * a, 4), share, cats


def _quiet_point(db, t, lo, hi):
    """Самое тихое место (пауза между фразами) в окне [t+lo, t+hi]."""
    i0, i1 = int(max(0, t + lo) / HOP), int(min(len(db) * HOP, t + hi) / HOP)
    if i1 <= i0 + 1:
        return t
    seg = smooth(db, 1)[i0:i1]
    return (i0 + int(np.argmin(seg))) * HOP


def find(source_id, db, msgs, s, limit=None):
    score, share, cats = combined_score(db, msgs, s)
    total = len(db) * HOP
    # локальные максимумы
    peaks = np.where((score[1:-1] > score[:-2]) & (score[1:-1] >= score[2:]) & (score[1:-1] > 1.0))[0] + 1
    lo, hi = s.skip_start / HOP, (total - s.skip_end) / HOP
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    order = peaks[np.argsort(-score[peaks])]
    taken = []
    gap = s.min_gap / HOP
    for p in order:
        if all(abs(p - q) >= gap for q in taken):
            taken.append(p)
        if limit and len(taken) >= limit:
            break

    found = []
    for p in sorted(taken):
        t = p * HOP
        start = _quiet_point(db, t - s.pre_roll, -4, 3)
        end = _quiet_point(db, t + s.post_roll, -2, 5)
        # растягиваем, пока активность держится выше половины пика
        j = p
        while j + 1 < len(score) and score[j + 1] > score[p] * 0.5 and (j - p) * HOP < s.clip_max:
            j += 1
        end = max(end, min(j * HOP + 3, t + s.clip_max))
        start, end = max(0, start), min(total, end)
        if end - start < s.clip_min:
            end = min(total, start + s.clip_min)
        if end - start > s.clip_max:
            start = max(start, t - s.clip_max * 0.6)
            end = min(end, start + s.clip_max)
        label = "момент"
        if share is not None:
            w = share[:, max(0, p - int(8 / HOP)): p + int(8 / HOP)].sum(axis=1)
            if w.max() > 0:
                label = LABELS[cats[int(np.argmax(w))]]
        found.append(Moment(id="", source=source_id, start=start, end=end, peak=t,
                            score=float(score[p]), label=label))

    # сливаем пересекающиеся
    merged = []
    for m in found:
        if merged and m.start < merged[-1].end and m.end - merged[-1].start <= s.clip_max:
            prev = merged[-1]
            prev.end = max(prev.end, m.end)
            if m.score > prev.score:
                prev.score, prev.peak, prev.label = m.score, m.peak, m.label
        elif merged and m.start < merged[-1].end:
            m.start = merged[-1].end
            if m.end - m.start >= s.clip_min:
                merged.append(m)
        else:
            merged.append(m)
    for i, m in enumerate(merged, 1):
        m.id = f"{source_id}-{i:02d}"
    return merged, score


def select(per_source: dict[str, list[Moment]], s) -> list[Moment]:
    """Берём лучшие моменты, пока не наберём нужную длину ролика, и расставляем по порядку."""
    budget = s.minutes * 60 - s.outro_seconds - (2.5 if s.intro else 0)
    pool = [m for ms in per_source.values() for m in ms if m.keep]
    if not pool:
        return []
    n_src = len(per_source)
    cap = budget / n_src * 2.2 if n_src > 1 else budget        # не даём одному стримеру занять всё
    used = {k: 0.0 for k in per_source}
    chosen, total = [], 0.0
    # сначала лучший момент каждого стрима, затем все остальные по силе
    firsts = [max(ms, key=lambda m: m.score) for k in per_source
              if (ms := [m for m in pool if m.source == k])]
    rest = sorted((m for m in pool if m not in firsts), key=lambda m: -m.score)
    for m in sorted(firsts, key=lambda m: -m.score) + rest:
        if total >= budget:
            break
        if used[m.source] + m.length > cap and m not in firsts:
            continue
        chosen.append(m)
        used[m.source] += m.length
        # в стиле sting между клипами переход 0,4 с, а паузы внутри клипов вырезаются (~10%)
        total += (m.length + s.plate_seconds if s.transition == "plate"
                  else m.length * (0.9 if s.jumpcut > 0 else 1) + s.sting_seconds)
    return arrange(chosen, s)


def arrange(chosen: list[Moment], s) -> list[Moment]:
    if len(chosen) < 3:
        return sorted(chosen, key=lambda m: -m.score)
    best = sorted(chosen, key=lambda m: -m.score)
    opener, closer = best[0], best[1]
    middle = [m for m in chosen if m is not opener and m is not closer]
    if s.order == "stream":
        middle.sort(key=lambda m: (m.source, m.start))
    else:
        # по кругу между стримерами, внутри стрима — в хронологии
        queues = {}
        for m in sorted(middle, key=lambda m: m.start):
            queues.setdefault(m.source, []).append(m)
        keys = sorted(queues, key=lambda k: -max(m.score for m in queues[k]))
        middle, last = [], None
        while any(queues.values()):
            for k in keys:
                if queues[k] and (k != last or sum(1 for q in queues.values() if q) == 1):
                    middle.append(queues[k].pop(0))
                    last = k
    return [opener, *middle, closer]
