"""Visualización del mapa de sospecha sobre la foto."""

import numpy as np
from PIL import Image, ImageDraw


def overlay(img: Image.Image, heat: Image.Image, alpha: float = 0.55) -> Image.Image:
    """Superpone el mapa (0-255) en rojo sobre la foto; más rojo = más sospechoso."""
    base = np.asarray(img.convert("RGB"), np.float32)
    h = np.asarray(heat.resize(img.size), np.float32)[..., None] / 255
    red = np.array([255, 40, 40], np.float32)
    out = base * (1 - alpha * h) + red * (alpha * h)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def suspicious_box(heat: Image.Image, frac: float = 0.5):
    """Caja alrededor de la zona más sospechosa (píxeles con >= frac del máximo)."""
    h = np.asarray(heat, np.float32)
    if h.max() <= 0:
        return None
    ys, xs = np.where(h >= frac * h.max())
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_box(img: Image.Image, box, color=(255, 0, 0), width: int = 4) -> Image.Image:
    out = img.convert("RGB").copy()
    if box:
        ImageDraw.Draw(out).rectangle(box, outline=color, width=width)
    return out


def zoom(img: Image.Image, box, pad: int = 60) -> Image.Image | None:
    if not box:
        return None
    x0, y0, x1, y1 = box
    return img.crop((max(0, x0 - pad * 2), max(0, y0 - pad), min(img.width, x1 + pad * 2), min(img.height, y1 + pad)))
