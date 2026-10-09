"""Описание проекта (какие стримы, сколько минут) из TOML-файла."""
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    minutes: float = 30          # целевая длина ролика
    clip_min: float = 12         # самый короткий момент, сек
    clip_max: float = 75         # самый длинный момент, сек
    pre_roll: float = 18         # сколько брать до пика реакции
    post_roll: float = 10        # и после него
    chat_delay: float = 8        # чат реагирует позже, чем происходит событие
    min_gap: float = 45          # минимальное расстояние между моментами одного стрима
    skip_start: float = 60       # пропустить начало стрима («скоро начнём», приветствия)
    skip_end: float = 30         # и прощание в конце
    order: str = "mix"           # mix — чередовать стримеров, stream — по стримам подряд
    width: int = 1920
    height: int = 1080
    fps: int = 30
    preset: str = "veryfast"     # пресет x264: быстрее — ultrafast, качественнее — medium
    crf: int = 20
    plate_seconds: float = 1.4   # длительность плашки перед клипом (transition = "plate")
    transition: str = "sting"    # sting — переход 0,4 с с логотипом; plate — плашка стримера перед клипом
    sting_seconds: float = 0.4
    lower_third: float = 4.0     # сколько секунд держится плашка стримера поверх клипа; 0 — без неё
    zoom: bool = True            # зум на лицо в момент реакции (со звуком «бум»)
    captions: bool = True        # подпись-реакция из 2–3 слов на зуме
    jumpcut: float = 0.8         # вырезать паузы без слов длиннее этого (сек); 0 — не вырезать
    hook: int = 3                # сколько панчлайнов показать в самом начале; 0 — без хука
    music: str = "funkorama"     # музыка под хук и концовку (каталог в srez/sfx.py); "" — без музыки
    disclaimer: str = "Развлекательный контент. Возможна ненормативная лексика"   # в начале ролика и в описании
    mask_style: str = "blur"     # как закрывать рекламу: blur — размыть, fill — закрасить
    censor: str = "off"          # мат: beep — запикать, mute — заглушить, off — оставить
    intro: bool = False          # короткая заставка с логотипом в начале
    outro_seconds: float = 12    # финальная заставка под конечные элементы YouTube
    followers_label: str = "фолловеров на Twitch"
    jobs: int = 0                # параллельных ffmpeg; 0 — по числу ядер


@dataclass
class Mask:
    """Прямоугольник, который нужно закрыть (реклама, баннер, QR-код).

    box — x, y, ширина, высота в долях кадра (0…1); start/end — секунды стрима,
    в которые баннер висит на экране (None — весь стрим).
    """
    box: tuple[float, float, float, float]
    start: float | None = None
    end: float | None = None

    def active(self, a: float, b: float) -> bool:
        return (self.start is None or self.start < b) and (self.end is None or self.end > a)


def parse_time(v) -> float | None:
    """«1:20:05», «20:05» или число секунд."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    sec = 0.0
    for part in str(v).split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_masks(items) -> list[Mask]:
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append(Mask(tuple(it["box"]), parse_time(it.get("from")), parse_time(it.get("to"))))
        else:
            out.append(Mask(tuple(it)))
    for m in out:
        x, y, w, h = m.box
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Маска {m.box}: координаты должны быть долями кадра от 0 до 1")
    return out


@dataclass
class Source:
    id: str
    url: str | None = None
    file: Path | None = None
    chat: Path | None = None
    twitch: str | None = None         # логин канала на Twitch
    name: str | None = None           # отображаемое имя, если хочется переопределить
    followers: int | None = None      # ручное значение, если Twitch недоступен
    avatar: Path | None = None
    platform: str = "Twitch"
    title: str | None = None          # название трансляции (для описания)
    vod_url: str | None = None
    masks: list[Mask] = field(default_factory=list)
    plate_side: str = "left"          # где плашка стримера поверх клипа: left или right (если там вебка)


@dataclass
class Project:
    path: Path
    title: str
    settings: Settings
    sources: list[Source] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def work(self) -> Path:
        return self.root / "work" / self.name

    @property
    def out(self) -> Path:
        return self.root / "output"


def _slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "_", text, flags=re.UNICODE).strip("_").lower()
    return s[:40] or "stream"


def load(path: str | Path) -> Project:
    path = Path(path).resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    video = data.get("video", {})
    settings = Settings(**{k: v for k, v in video.items() if k != "title"})

    def rel(p):
        if not p:
            return None
        p = Path(p).expanduser()
        return p if p.is_absolute() else (path.parent / p).resolve()

    sources, seen = [], set()
    for i, s in enumerate(data.get("stream", []), 1):
        url, file = s.get("url"), rel(s.get("file"))
        if not url and not file:
            raise ValueError(f"[[stream]] №{i}: нужен url или file")
        if file and not file.exists():
            raise FileNotFoundError(f"[[stream]] №{i}: файл не найден: {file}")
        m = re.search(r"twitch\.tv/videos/(\d+)", url or "")
        base = f"vod{m.group(1)}" if m else _slug(file.stem if file else url.rsplit("/", 1)[-1])
        sid, n = base, 2
        while sid in seen:
            sid, n = f"{base}_{n}", n + 1
        seen.add(sid)
        sources.append(Source(
            id=sid, url=url, file=file, chat=rel(s.get("chat")),
            twitch=(s.get("twitch") or "").lower() or None, name=s.get("name"),
            followers=s.get("followers"), avatar=rel(s.get("avatar")),
            platform=s.get("platform", "Twitch"), vod_url=url, masks=parse_masks(s.get("masks")),
            plate_side=s.get("plate_side", "left"),
        ))
    if not sources:
        raise ValueError("В проекте нет ни одного [[stream]]")
    return Project(path=path, title=video.get("title", "Лучшие моменты стримов"),
                   settings=settings, sources=sources)
