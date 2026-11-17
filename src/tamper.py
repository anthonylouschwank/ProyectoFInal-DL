"""Alteración de fotos de facturas, como lo haría alguien con una app de edición del celular.

Las alteraciones se aplican sobre la foto final (no sobre el documento), igual que en un
fraude real: se toma la foto y después se edita. Hay dos técnicas:

- reescribir:     se tapa el valor con el color del papel y se escribe uno nuevo encima
- copiar_digito:  se copia un dígito de otra parte de la misma factura y se pega sobre otro
                  (por ejemplo, un 1 se vuelve 7)

Cada alteración tiene un nivel de dificultad, que se guarda para reportar el recall por nivel:
- fácil (50 %):   lo típico de una app del celular: otra fuente, texto horizontal aunque la
                  foto esté inclinada, tinta negra pura, parche liso
- medio (30 %):   misma fuente, rotación y tono de tinta, pero parche liso y sin desenfoque
- difícil (20 %): falsificación cuidadosa: además imita el ruido y el desenfoque de la foto,
                  o copia un dígito de la misma factura

Cada alteración devuelve la máscara exacta de lo editado y el "monto en riesgo" (cuánto
dinero gana el que la alteró), que usa el modelo de valor.
"""

import io
import math
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .normalize import fecha as parse_fecha
from .synth_fel import FONTS, apply_h, make_nit

ALL_FONTS = sorted(p.name for p in FONTS.iterdir() if p.suffix in (".ttf", ".otf"))
# Probabilidad de alterar cada campo y técnica disponible
CAMPOS = [("total", 0.55), ("fecha", 0.12), ("nit_receptor", 0.15), ("iva", 0.06), ("nit_emisor", 0.06),
          ("item", 0.06)]


# ------------------------------------------------------------------------ utilidades

def seg_quad(seg, H, a=0, b=None, pad=0.18):
    """Cuadrilátero (4x2) en la foto de los caracteres [a, b) del segmento."""
    b = len(seg["value"]) if b is None else b
    x0, x1 = seg["x"][a], seg["x"][b]
    h = seg["y1"] - seg["y0"]
    y0, y1 = seg["y0"] - pad * h, seg["y1"] + pad * h
    return apply_h(H, [(x0 - 1, y0), (x1 + 1, y0), (x1 + 1, y1), (x0 - 1, y1)])


def local_scale(H, seg):
    """Cuántos píxeles de foto mide un píxel del documento alrededor del segmento."""
    p = apply_h(H, [(seg["x"][0], seg["y0"]), (seg["x"][0], seg["y1"])])
    return float(np.linalg.norm(p[1] - p[0])) / max(1.0, seg["y1"] - seg["y0"])


def quad_angle(q):
    return math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))


def ring_stats(arr, q, grow=8):
    """Color del papel y de la tinta alrededor del cuadrilátero."""
    h, w = arr.shape[:2]
    x0, y0 = np.floor(q.min(0)).astype(int) - grow
    x1, y1 = np.ceil(q.max(0)).astype(int) + grow
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(patch) < 10:
        return np.array([240, 240, 235], np.float32), np.array([30, 30, 30], np.float32)
    lum = patch.mean(1)
    paper = patch[lum >= np.percentile(lum, 60)].mean(0)
    ink = patch[lum <= np.percentile(lum, 4)].mean(0)
    return paper, ink


def parse_amount(text):
    m = re.search(r"\d[\d,]*\.\d{2}", text)
    return float(m.group(0).replace(",", "")) if m else None


def format_like(orig: str, x: float) -> str:
    """Formatea x con el mismo estilo que el texto original (prefijo Q, comas de miles)."""
    m = re.match(r"^\D*", orig)
    prefix = m.group(0) if m else ""
    body = f"{x:,.2f}" if ("," in orig or (x >= 1000 and prefix.strip())) else f"{x:.2f}"
    return prefix + body


def change_digit(text, rng, positions=None):
    pos = [i for i, c in enumerate(text) if c.isdigit()] if positions is None else positions
    if not pos:
        return text, None
    i = rng.choice(pos)
    new = rng.choice([d for d in "0123456789" if d != text[i]])
    return text[:i] + new + text[i + 1:], i


# ------------------------------------------------------------------- nuevos valores

def new_value(seg, inv, rng):
    """(texto nuevo, monto en riesgo) para el campo del segmento."""
    v, tag, total = seg["value"], seg["tag"], inv["total"]
    if tag in ("total", "item", "iva"):
        old = seg.get("amount") or parse_amount(v) or 0.0
        if rng.random() < 0.35:  # cambiar el primer dígito por uno mayor (149.36 -> 749.36)
            i = next((k for k, c in enumerate(v) if c.isdigit()), None)
            if i is not None and v[i] != "9":
                cand = v[:i] + rng.choice([d for d in "23456789" if d > v[i]]) + v[i + 1:]
                new_amt = parse_amount(cand)
                if new_amt:
                    return cand, round(new_amt - old, 2) if tag != "item" else 0.0
        new_amt = round(old * rng.uniform(1.25, 3.0) if rng.random() < 0.6 else old + rng.choice([50, 100, 200, 500]), 2)
        return format_like(v, new_amt), round(new_amt - old, 2) if tag != "item" else 0.0
    if tag == "fecha":  # reutilizar una factura vieja o fuera del periodo: todo el monto en riesgo
        year = re.search(r"\d{4}", v)
        digits = [i for i, c in enumerate(v[:10]) if c.isdigit() and not (year and year.start() <= i < year.end())]
        for _ in range(30):  # solo día o mes, y que la fecha nueva exista
            new, _ = change_digit(v, rng, digits)
            if new != v and parse_fecha(new):
                return new, total
        return v, 0.0
    if tag == "nit_receptor":  # poner el NIT de la empresa en una compra personal
        return make_nit(rng), total
    if tag == "nit_emisor":
        body = [i for i, c in enumerate(v) if c.isdigit()][:-1]
        new, _ = change_digit(v, rng, body)
        return new, total
    return v, 0.0


# ----------------------------------------------------------------------- técnicas

def _font(name, size):
    return ImageFont.truetype(str(FONTS / name), max(6, int(round(size))))


def reescribir(img, seg, H, new_text, rng, nivel="facil"):
    arr = np.asarray(img).copy()
    q = seg_quad(seg, H)
    paper, ink = ring_stats(arr, q)
    scale = local_scale(H, seg)
    hard = nivel == "dificil"  # imita además el ruido y el desenfoque de la foto
    cuidadoso = nivel in ("medio", "dificil")  # misma fuente, rotación y tinta
    otras = [f for f in ALL_FONTS if f != seg["font"]]
    font_name = seg["font"] if cuidadoso else rng.choice(otras)
    font = _font(font_name, seg["size"] * scale * (1.0 if cuidadoso else rng.uniform(0.85, 1.2)))
    angle = quad_angle(q) if (cuidadoso or rng.random() < 0.3) else 0.0
    color = tuple(int(c) for c in (ink if cuidadoso else [rng.randint(0, 40)] * 3))

    # Texto nuevo en una capa propia, alineado como el original (montos a la derecha)
    tw = font.getlength(new_text)
    l, t, r, b = font.getbbox(new_text)
    layer = Image.new("RGBA", (int(tw + 20), int(b + 20)), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((10, 10), new_text, font=font, fill=color + (255,))
    layer = layer.rotate(-angle, resample=Image.BICUBIC, expand=True)
    if hard:
        layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))

    right_aligned = seg["tag"] in ("total", "item", "iva")
    old_w = float(np.linalg.norm(q[1] - q[0]))
    extra = max(0.0, tw - old_w) + 4
    # Zona a tapar: el valor original, extendida si el texto nuevo es más largo
    u = (q[1] - q[0]) / max(old_w, 1e-6)
    cover = q.copy()
    if right_aligned:
        cover[0] -= u * extra
        cover[3] -= u * extra
    else:
        cover[1] += u * extra
        cover[2] += u * extra
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in cover], fill=255)

    fill = np.zeros_like(arr, dtype=np.float32) + paper
    if hard:  # imitar el ruido de la foto para que el parche no se vea liso
        fill += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 4), fill.shape[:2])[..., None]
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.8 if hard else 0.3)), np.float32)[..., None] / 255
    out = Image.fromarray((arr * (1 - m) + fill * m).clip(0, 255).astype(np.uint8))

    # Pegar el texto: el borde derecho (o izquierdo) coincide con el del valor original
    anchor = (q[1] + q[2]) / 2 if right_aligned else (q[0] + q[3]) / 2
    cx, cy = anchor - u * tw / 2 if right_aligned else anchor + u * tw / 2
    out.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)
    return out, mask


def copiar_digito(img, seg, H, segs, rng):
    """Copia un dígito de otro lugar de la factura sobre un dígito del valor."""
    v = seg["value"]
    targets = [i for i, c in enumerate(v) if c.isdigit()]
    sources = [(s, i) for s in segs for i, c in enumerate(s["value"]) if c.isdigit()]
    if not targets or not sources:
        return None
    if seg["tag"] in ("total", "iva", "item"):
        # Para montos: un dígito de la parte entera, reemplazado por uno mayor (inflar el monto)
        entera = v.split(".")[0]
        targets = [i for i in targets if i < len(entera) and v[i] != "9"]
        if not targets:
            return None
        ti = rng.choice(targets)
        cands = [(s, i) for s, i in sources if s["value"][i] > v[ti]]
    else:
        ti = rng.choice(targets)
        cands = [(s, i) for s, i in sources if s["value"][i] != v[ti]]
        if seg["tag"] == "fecha":
            year = re.search(r"\d{4}", v)
            if ti >= 10 or (year and year.start() <= ti < year.end()):
                return None
    if not cands:
        return None
    src_seg, si = rng.choice(cands)
    sq, tq = seg_quad(src_seg, H, si, si + 1, pad=0.1), seg_quad(seg, H, ti, ti + 1, pad=0.1)
    sb = (*np.floor(sq.min(0)).astype(int), *np.ceil(sq.max(0)).astype(int))
    tb = (*np.floor(tq.min(0)).astype(int), *np.ceil(tq.max(0)).astype(int))
    if sb[2] - sb[0] < 2 or sb[3] - sb[1] < 2 or tb[2] - tb[0] < 2 or tb[3] - tb[1] < 2:
        return None
    patch = img.crop(sb).resize((tb[2] - tb[0], tb[3] - tb[1]), Image.BICUBIC)
    feather = Image.new("L", patch.size, 255).filter(ImageFilter.GaussianBlur(0.6))
    out = img.copy()
    out.paste(patch, tb[:2], feather)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle(tb, fill=255)
    new_text = v[:ti] + src_seg["value"][si] + v[ti + 1:]
    if seg["tag"] == "fecha" and not parse_fecha(new_text):
        return None
    return out, mask, new_text


# ------------------------------------------------------------------ interfaz pública

NIVELES = (0.5, 0.3, 0.2)  # probabilidad de fácil, medio y difícil


def tamper(img, segs, H, inv, rng, pesos=NIVELES, p_copiar=0.5):
    """Altera un campo de la foto. Devuelve (imagen, máscara, info) o None si no se pudo."""
    disponibles = {s["tag"] for s in segs}
    campos = [(c, p) for c, p in CAMPOS if c in disponibles]
    if not campos:
        return None
    tag = rng.choices([c for c, _ in campos], [p for _, p in campos])[0]
    seg = rng.choice([s for s in segs if s["tag"] == tag])
    nivel = rng.choices(["facil", "medio", "dificil"], list(pesos))[0]
    if nivel == "dificil" and tag != "nit_receptor" and rng.random() < p_copiar:
        r = copiar_digito(img, seg, H, segs, rng)
        if r:
            out, mask, new_text = r
            old_amt, new_amt = parse_amount(seg["value"]), parse_amount(new_text)
            if tag in ("total", "iva") and old_amt is not None and new_amt is not None:
                riesgo = round(new_amt - old_amt, 2)
            else:
                riesgo = 0.0 if tag == "item" else inv["total"]
            return out, mask, {"metodo": "copiar_digito", "campo": tag, "original": seg["value"],
                               "nuevo": new_text, "monto_en_riesgo": riesgo, "dificultad": "dificil"}
    new_text, riesgo = new_value(seg, inv, rng)
    if new_text == seg["value"]:
        return None
    out, mask = reescribir(img, seg, H, new_text, rng, nivel)
    return out, mask, {"metodo": "reescribir", "campo": tag, "original": seg["value"], "nuevo": new_text,
                       "monto_en_riesgo": riesgo, "dificultad": nivel}


def draw_geometry(rng):
    """Post-proceso que comparten la foto auténtica y su gemela alterada (misma geometría)."""
    return {"crop": (rng.uniform(0, 0.05), rng.uniform(0, 0.05), rng.uniform(0.95, 1), rng.uniform(0.95, 1))
            if rng.random() < 0.3 else None,
            "brillo": (rng.uniform(0.85, 1.2), rng.uniform(-15, 15)) if rng.random() < 0.3 else None,
            "escala": rng.uniform(0.6, 0.95) if rng.random() < 0.5 else None}


def finalize(img, rng, mask=None, geo=None):
    """Lo que le pasa a cualquier foto antes de llegar al sistema (alterada o no): recorte,
    ajuste de brillo, reescalado y recompresión tipo WhatsApp. Así el modelo no puede usar la
    doble compresión ni el tamaño como atajo. `geo` (de draw_geometry) fija la geometría para
    que la foto auténtica y la alterada queden alineadas píxel a píxel; la compresión sí es
    independiente para cada una."""
    geo = geo or draw_geometry(rng)
    if geo["crop"]:
        w, h = img.size
        fx0, fy0, fx1, fy1 = geo["crop"]
        box = (int(w * fx0), int(h * fy0), int(w * fx1), int(h * fy1))
        img = img.crop(box)
        mask = mask.crop(box) if mask is not None else None
    if geo["brillo"]:
        c, b = geo["brillo"]
        arr = (np.asarray(img, np.float32) - 128) * c + 128 + b
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if geo["escala"]:
        size = (int(img.width * geo["escala"]), int(img.height * geo["escala"]))
        img = img.resize(size, Image.BICUBIC)
        mask = mask.resize(size, Image.NEAREST) if mask is not None else None
    for _ in range(rng.choice([0, 1, 1, 2])):  # compresiones intermedias
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(60, 95))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img, mask
