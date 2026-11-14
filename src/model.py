"""Modelos para detectar facturas alteradas.

- "gap":  clasificador estándar: EfficientNet + promedio global + capa lineal.
- "mapa": la misma red produce un mapa de sospecha por zona (logits a resolución 1/16) y la
          factura se clasifica con el promedio de las k zonas más sospechosas. Una alteración
          ocupa una parte chica de la foto; el promedio global la diluye, el top-k no. El mapa
          se puede supervisar con la máscara de la alteración y sirve como mapa de calor.
"""

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Detector(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", head: str = "mapa", topk: int = 8,
                 pretrained: bool = True):
        super().__init__()
        self.head, self.topk = head, topk
        self.enc = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(3, 4))
        c16, c32 = self.enc.feature_info.channels()
        if head == "gap":
            self.fc = nn.Linear(c32, 1)
        else:
            self.fuse = nn.Sequential(nn.Conv2d(c16 + c32, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
                                      nn.Conv2d(128, 1, 1))

    def forward(self, x):
        f16, f32 = self.enc(x)
        if self.head == "gap":
            return self.fc(f32.mean((2, 3))).squeeze(1), None
        up = F.interpolate(f32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        mapa = self.fuse(torch.cat([f16, up], 1)).squeeze(1)  # (B, H/16, W/16) logits
        k = min(self.topk, mapa[0].numel())
        logit = mapa.flatten(1).topk(k, dim=1).values.mean(1)
        return logit, mapa


def letterbox(img: Image.Image, size: tuple[int, int], fill=(0, 0, 0)):
    """Redimensiona conservando proporción y rellena hasta size=(alto, ancho).
    Devuelve (imagen, escala, desplazamiento x, desplazamiento y)."""
    H, W = size
    s = min(H / img.height, W / img.width)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new(img.mode, (W, H), fill if img.mode == "RGB" else 0)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR if img.mode == "RGB" else Image.NEAREST), (ox, oy))
    return canvas, s, ox, oy


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def prepare(img: Image.Image, cfg):
    """Imagen lista para el modelo y cómo volver al tamaño original.

    Con cfg["native"] (modelos entrenados con recortes a resolución completa) la foto entra
    sin reducir (como máximo cfg["max_side"] px) y se rellena hasta múltiplo de 128; si no, se
    redimensiona a cfg["size"] conservando la proporción. Devuelve (imagen, (s, ox, oy))."""
    img = img.convert("RGB")
    if cfg.get("native"):
        ms = cfg.get("max_side", 1600)
        s = min(1.0, ms / max(img.size))
        im = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR) if s < 1 else img
        # Múltiplos de 128: pocos tamaños distintos, así cuDNN reutiliza sus planes de ejecución
        # (con un tamaño distinto por foto, armar esos planes es más lento que la red misma)
        W, H = math.ceil(im.width / 128) * 128, math.ceil(im.height / 128) * 128
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(im, (0, 0))
        return canvas, (s, 0, 0)
    lb, s, ox, oy = letterbox(img, tuple(cfg["size"]))
    return lb, (s, ox, oy)


def _back(m: np.ndarray, lb_size, geom, orig_size):
    """Mapa (a cualquier resolución) -> imagen PIL del tamaño de la foto original."""
    s, ox, oy = geom
    m = np.asarray(Image.fromarray(m).resize(lb_size, Image.BILINEAR))
    nw, nh = round(orig_size[0] * s), round(orig_size[1] * s)
    return Image.fromarray(m[oy:oy + nh, ox:ox + nw]).resize(orig_size, Image.BILINEAR)


def load_detector(path, device=None):
    """Carga un checkpoint guardado por train/train.py. Devuelve (modelo, config)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = Detector(cfg["backbone"], cfg["head"], cfg["topk"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def predict(model, cfg, img: Image.Image, device=None):
    """Probabilidad de alteración y mapa de sospecha (en coordenadas de la imagen original)."""
    device = device or next(model.parameters()).device
    lb, geom = prepare(img, cfg)
    x = to_tensor(lb)[None].to(device)
    with torch.autocast(device.type if hasattr(device, "type") else str(device), dtype=torch.bfloat16,
                        enabled=str(device).startswith("cuda")):
        logit, mapa = model(x)
    prob = torch.sigmoid(logit.float())[0].item()
    heat = None
    if mapa is not None:
        m = (torch.sigmoid(mapa.float())[0].cpu().numpy() * 255).astype(np.uint8)
        heat = _back(m, lb.size, geom, img.size)
    return prob, heat


def gradcam(model, cfg, img: Image.Image, device=None):
    """Grad-CAM sobre la última etapa del encoder (stride 32) respecto al logit de "alterada".
    Funciona con las dos cabezas; devuelve un mapa en [0, 255] del tamaño de la imagen."""
    device = device or next(model.parameters()).device
    lb, geom = prepare(img, cfg)
    x = to_tensor(lb)[None].to(device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        f16, f32 = model.enc(x)
        f32.retain_grad()
        if model.head == "gap":
            logit = model.fc(f32.mean((2, 3))).squeeze(1)
        else:
            up = F.interpolate(f32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
            mapa = model.fuse(torch.cat([f16, up], 1)).squeeze(1)
            logit = mapa.flatten(1).topk(min(model.topk, mapa[0].numel()), dim=1).values.mean(1)
        logit.sum().backward()
    w = f32.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * f32).sum(1))[0].detach().float().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return _back((cam * 255).astype(np.uint8), lb.size, geom, img.size)
