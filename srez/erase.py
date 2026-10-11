"""Стирание рекламы с неподвижной камеры: вместо размытого прямоугольника — дорисованный фон.

Для маски со `style = "erase"` (предмет в комнате у неподвижной камеры) или `"paint"` (наклейка-оверлей поверх
неподвижной картинки — подложка кладётся всегда, без проверки на руки):
1) из всех клипов, где маска активна, берётся по кадру в секунду и считается медиана — «эталон»: как выглядит
   это место с рекламой, без случайно зашедших рук и плеч;
2) нейросеть LaMa (ONNX) дорисовывает на эталоне фон вместо рекламы — «чистая подложка»;
3) при сборке подложка кладётся поверх кадра только там, где кадр похож на эталон: если стример заслонил рекламу
   рукой или плечом, там остаётся живой кадр (`plate_filters`).
Модель LaMa (~200 МБ) скачивается один раз в ~/.cache/lama.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ERASE_VERSION = 2                   # поменял подложку или ключ — поднять, чтобы клипы пересобрались
SIZE = 512                          # LaMa работает с квадратом 512×512
FEATHER = 6                         # мягкий край подложки, пиксели
DIFF = 40                           # отличие от эталона (0–255), с которого пиксель считается заслонённым
MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL = Path.home() / ".cache/lama/lama_fp32.onnx"


def _model():
    import onnxruntime as ort
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(MODEL), MODEL_URL], check=True)
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


def window(box_px, src_w, src_h):
    """Квадрат SIZE×SIZE вокруг рамки рекламы (в пикселях исходника), не вылезая за кадр."""
    x, y, w, h = box_px
    if w > SIZE - 2 * FEATHER or h > SIZE - 2 * FEATHER:
        raise ValueError(f"Маска erase {w}×{h} больше {SIZE - 2 * FEATHER} пикселей — разбейте на части")
    x0 = int(min(max(0, x + w / 2 - SIZE / 2), src_w - SIZE))
    y0 = int(min(max(0, y + h / 2 - SIZE / 2), src_h - SIZE))
    return x0, y0


def plate(box_px, sources, out_dir: Path, log=print) -> dict:
    """Эталон, чистая подложка и мягкая маска для рамки box_px (x, y, w, h в пикселях исходника).

    sources — [(файл, начало, длительность)] кусков, где реклама на экране. Результат кэшируется."""
    from PIL import Image, ImageDraw, ImageFilter
    sig = hashlib.md5(repr((ERASE_VERSION, box_px, sources)).encode()).hexdigest()[:10]
    meta = out_dir / f"erase_{sig}.json"
    if meta.exists():
        return json.loads(meta.read_text())
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(sources[0][0])], capture_output=True, text=True, check=True)
    src_w, src_h = map(int, probe.stdout.strip().split(","))
    x0, y0 = window(box_px, src_w, src_h)
    frames = []
    for path, start, dur in sources:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                              "-vf", f"fps=1,crop={SIZE}:{SIZE}:{x0}:{y0}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                             capture_output=True, check=True).stdout
        frames.append(np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3))
    ref = np.median(np.concatenate(frames), axis=0).astype(np.uint8)
    x, y, w, h = box_px
    hole = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(hole).rectangle((x - x0, y - y0, x - x0 + w, y - y0 + h), fill=255)
    hole = hole.filter(ImageFilter.MaxFilter(2 * (FEATHER // 2) + 1))          # с запасом на край рекламы
    img = ref.astype(np.float32).transpose(2, 0, 1)[None] / 255
    mk = (np.asarray(hole, np.float32)[None, None] > 127).astype(np.float32)
    clean = _model().run(None, {"image": img, "mask": mk})[0][0].transpose(1, 2, 0)
    if clean.max() <= 1.5:
        clean = clean * 255
    alpha = hole.filter(ImageFilter.MaxFilter(2 * FEATHER + 1)).filter(ImageFilter.GaussianBlur(FEATHER / 2))
    files = {k: out_dir.resolve() / f"erase_{sig}_{k}.png" for k in ("ref", "clean", "alpha")}
    Image.fromarray(ref).save(files["ref"])
    Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8)).save(files["clean"])
    alpha.save(files["alpha"])
    info = {"x": x0, "y": y0, **{k: str(v) for k, v in files.items()}}
    meta.write_text(json.dumps(info))
    log(f"  подложка для стирания рекламы: {box_px} → {meta.name}")
    return info


def plate_filters(info, cur, i, en="", keyed=True):
    """Цепочка filter_complex: подложка поверх кадра там, где реклама не заслонена (keyed=False — везде,
    для наклеек-оверлеев, которые меняются и никем не заслоняются). Возвращает (фильтры, выход)."""
    x, y = info["x"], info["y"]
    if not keyed:
        return [
            f"movie={info['clean']},format=rgb24[ecl{i}]",
            f"movie={info['alpha']},format=gray[eal{i}]",
            f"[ecl{i}][eal{i}]alphamerge[ep{i}]",
            f"[{cur}][ep{i}]overlay={x}:{y}{en}[mv{i}]",
        ], f"mv{i}"
    return [
        f"[{cur}]split[ea{i}][eb{i}]",
        f"[eb{i}]crop={SIZE}:{SIZE}:{x}:{y},format=gbrp[ec{i}]",
        f"movie={info['ref']},format=gbrp[er{i}]",
        f"movie={info['clean']},format=rgb24[ecl{i}]",
        f"movie={info['alpha']},format=gray[eal{i}]",
        # где кадр заметно отличается от эталона — там рука или плечо: подложку туда не кладём
        # тонкие контуры (камера сдвинулась на пиксель, сжатие) отбрасываем: эрозия, потом расширение назад с запасом
        f"[ec{i}][er{i}]blend=all_mode=difference,format=gray,lut=y='if(gt(val,{DIFF}),255,0)',"
        f"{','.join(['erosion'] * 3 + ['dilation'] * 6)},gblur=sigma=3[eo{i}]",
        f"[eal{i}][eo{i}]blend=all_mode=subtract[ek{i}]",
        f"[ecl{i}][ek{i}]alphamerge[ep{i}]",
        f"[ea{i}][ep{i}]overlay={x}:{y}{en}[mv{i}]",
    ], f"mv{i}"
