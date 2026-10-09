"""Командная строка нарезчика.

    python -m srez analyze проект.toml   найти моменты во всех стримах
    python -m srez prepare проект.toml   скачать нужные куски видео и найти рекламу на экране
    python -m srez render  проект.toml   смонтировать ролик из найденного
    python -m srez auto    проект.toml   всё сразу
    python -m srez check                 проверить, что всё установлено
"""
import argparse
import shutil
import sys
import time
import urllib.request

from . import project as proj
from .moments import select
from .pipeline import analyze, load_candidates
from .render import _tc, prepare, render


def log(msg):
    print(msg, flush=True)


def cmd_analyze(p):
    analyze(p, log)


def chosen_moments(p):
    data, per_source = load_candidates(p)
    chosen = select(per_source, p.settings)
    if not chosen:
        sys.exit("Нет моментов для монтажа: проверьте candidates.json")
    return data, chosen


def cmd_prepare(p):
    _, chosen = chosen_moments(p)
    prepare(p, chosen, log)
    log(f"Листы проверки: {p.work / 'layouts'}. Маски рекламы — в {p.work / 'masks.json'}")


def cmd_render(p):
    data, chosen = chosen_moments(p)
    s = p.settings
    gap = s.plate_seconds if s.transition == "plate" else s.sting_seconds
    total = sum(m.length + gap for m in chosen)
    log(f"Отобрано {len(chosen)} моментов, до вырезки пауз ≈ {_tc(total + s.outro_seconds)}")
    render(p, chosen, data["sources"], log)


def cmd_check(_):
    ok = True
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        found = shutil.which(tool)
        ok &= bool(found) or tool == "yt-dlp"
        log(f"{'✓' if found else '✗'} {tool}" + ("" if found else " — не найден"))
    for name, url in (("Twitch", "https://gql.twitch.tv/gql"), ("YouTube", "https://www.youtube.com")):
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=8)
            log(f"✓ {name} доступен")
        except urllib.error.HTTPError:
            log(f"✓ {name} доступен")
        except Exception as e:
            log(f"✗ {name} недоступен ({e.__class__.__name__}): ссылки работать не будут, только файлы")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="srez", description="Нарезка лучших моментов стримов в один ролик")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "prepare", "render", "auto"):
        sp = sub.add_parser(name)
        sp.add_argument("project", help="файл проекта .toml")
        sp.add_argument("--minutes", type=float, help="переопределить длину ролика")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    p = proj.load(a.project)
    if a.minutes:
        p.settings.minutes = a.minutes
    t0 = time.time()
    if a.cmd in ("analyze", "auto"):
        cmd_analyze(p)
    if a.cmd in ("prepare", "auto"):
        cmd_prepare(p)
    if a.cmd in ("render", "auto"):
        cmd_render(p)
    log(f"Заняло {_tc(time.time() - t0)}")


if __name__ == "__main__":
    main()
