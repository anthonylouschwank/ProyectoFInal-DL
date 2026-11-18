"""Entrena el detector de facturas alteradas.

    python -m train.train --head gap  --size 768 576 --out models/it1_gap          # foto reducida, clasificador estándar
    python -m train.train --head mapa --size 768 576 --out models/it2_mapa         # foto reducida, mapa de sospecha
    python -m train.train --head mapa --crop 512 --out models/it3_recortes         # recortes a resolución completa
    python -m train.train --head mapa --crop 512 --epochs 15 --extra-train data/tamper_dificil/train \
        --out models/it4_mas_dificiles                                              # + lote con más casos difíciles
    python -m train.train --head mapa --crop 512 --extra-train data/tamper_dificil/train --p-mask 0.5 \
        --select recall@fpr1% --out models/it5_menos_falsas                         # it4 corrigiendo falsas alarmas
    python -m train.train --smoke                                                   # prueba rápida

Con --crop N se entrena con recortes de N×N a resolución completa: por cada par se toma la
misma ventana de la foto auténtica y de su gemela alterada (están alineadas píxel a píxel),
casi siempre sobre la zona editada. La validación y el uso son con la foto completa.

Guarda en --out: best.pt (mejor métrica de validación según --select), logs/epochs.csv, logs/curvas.png y
logs/train_summary.json (tiempo, GPU y memoria: costo de entrenamiento para el informe).
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import Detector, letterbox, prepare, to_tensor  # noqa: E402


def load_split(d: Path, limit: int = 0):
    recs = [json.loads(line) for line in (d / "metadata.jsonl").open(encoding="utf-8")]
    for r in recs:
        r["dir"] = str(d)
    if limit:  # conservar pares completos
        pares = sorted({r["pair"] for r in recs})[: limit // 2]
        recs = [r for r in recs if r["pair"] in set(pares)]
    return recs


class TamperDataset(Dataset):
    def __init__(self, recs, size, train: bool):
        self.recs, self.size, self.train = recs, tuple(size), train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(Path(r["dir"]) / r["file"]).convert("RGB")
        mask = Image.open(Path(r["dir"]) / r["mask"]) if r["label"] else Image.new("L", img.size, 0)
        lb, *_ = letterbox(img, self.size)
        lm, *_ = letterbox(mask, self.size)
        H, W = self.size
        # Máscara a la resolución del mapa (1/16): fracción alterada de cada celda
        m = torch.from_numpy(np.asarray(lm, np.float32) / 255)[None, None]
        m = F.adaptive_avg_pool2d(m, (H // 16, W // 16))[0, 0]
        m = (m * 4).clamp(0, 1)  # una celda con >= 25 % alterado cuenta como alterada
        return to_tensor(lb), torch.tensor(float(r["label"])), m


def seed_worker(_):
    """Semilla distinta por worker y por época para los recortes aleatorios (funciona en Windows)."""
    random.seed(torch.initial_seed() % 2**32)


def mask_grid(m: Image.Image, h: int, w: int) -> torch.Tensor:
    """Máscara a la resolución del mapa (1/16): una celda con >= 25 % alterado cuenta como alterada."""
    t = torch.from_numpy(np.asarray(m, np.float32) / 255)[None, None]
    return (F.adaptive_avg_pool2d(t, (h, w))[0, 0] * 4).clamp(0, 1)


def _pad(im: Image.Image, W: int, H: int) -> Image.Image:
    canvas = Image.new(im.mode, (W, H), 0)
    canvas.paste(im, (0, 0))
    return canvas


class CropPairs(Dataset):
    """Por cada par, la misma ventana de la foto auténtica y de la alterada.

    Con peso_monto, el recorte alterado pesa en la pérdida según el monto en riesgo de su
    alteración (log, normalizado a promedio 1, entre 0.3 y 3): un fraude de Q1,000 importa
    más que uno de Q5. Los recortes auténticos, o los alterados cuya ventana no incluye la
    edición, pesan 1."""

    def __init__(self, recs, crop: int, p_mask: float = 0.75, peso_monto: bool = False):
        by = {}
        for r in recs:
            by.setdefault(r["pair"], {})[r["label"]] = r
        self.pairs = [v for v in by.values() if 0 in v and 1 in v]
        self.crop, self.p_mask = crop, p_mask
        logm = np.array([np.log1p(max(0.0, v[1]["tamper"]["monto_en_riesgo"])) for v in self.pairs])
        self.pesos = np.clip(logm / logm.mean(), 0.3, 3.0) if peso_monto else np.ones(len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, t = self.pairs[i][0], self.pairs[i][1]
        A = Image.open(Path(a["dir"]) / a["file"]).convert("RGB")
        T = Image.open(Path(t["dir"]) / t["file"]).convert("RGB")
        M = Image.open(Path(t["dir"]) / t["mask"])
        c = self.crop
        if min(T.size) < c:  # tickets angostos (tipo escáner): rellenar hasta el tamaño del recorte
            W, H = max(T.width, c), max(T.height, c)
            A, T, M = (_pad(im, W, H) for im in (A, T, M))
        W, H = T.size
        bb = M.getbbox()
        if bb and random.random() < self.p_mask:  # ventana que contiene la zona editada
            x0 = random.randint(max(0, min(bb[2] - c, W - c)), max(0, min(bb[0], W - c)))
            y0 = random.randint(max(0, min(bb[3] - c, H - c)), max(0, min(bb[1], H - c)))
        else:
            x0, y0 = random.randint(0, W - c), random.randint(0, H - c)
        box = (x0, y0, x0 + c, y0 + c)
        mc = M.crop(box)
        label = float((np.asarray(mc) > 0).sum() >= 60)
        g = c // 16
        x = torch.stack([to_tensor(A.crop(box)), to_tensor(T.crop(box))])
        m = torch.stack([torch.zeros(g, g), mask_grid(mc, g, g)])
        w = torch.tensor([1.0, float(self.pesos[i]) if label else 1.0])
        return x, torch.tensor([0.0, label]), m, w


class FullImages(Dataset):
    """Fotos completas tal como se usan en producción (validación)."""

    def __init__(self, recs, cfg):
        self.recs, self.cfg = recs, cfg

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        x, _ = prepare(Image.open(Path(r["dir"]) / r["file"]), self.cfg)
        monto = r["tamper"]["monto_en_riesgo"] if r["label"] else 0.0
        return to_tensor(x), torch.tensor(float(r["label"])), torch.tensor(float(monto))


def best_value(ys, ps, montos, prev=0.03, c_rev=10.0, c_fric=20.0):
    """Ahorro por cada 1,000 facturas en el mejor umbral (mismo modelo de valor que eval/evaluate.py)."""
    alt, aut = ys == 1, ys == 0
    best = -np.inf
    for t in np.linspace(0, 1, 101):
        a = ps >= t
        tpr, fpr = a[alt].mean(), a[aut].mean()
        v = (1000 * prev * (montos[alt] * a[alt]).mean() - 1000 * (prev * tpr + (1 - prev) * fpr) * c_rev
             - 1000 * (1 - prev) * fpr * c_fric)
        best = max(best, v)
    return float(best)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, ms, losses = [], [], [], []
    for x, y, m in loader:
        ms += m.flatten().tolist()
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logit.float(), y).item())
        ys += y.tolist()
        ps += torch.sigmoid(logit.float()).tolist()
    model.train()
    ys, ps = np.array(ys), np.array(ps)
    fpr, tpr, _ = roc_curve(ys, ps)
    out = {"val_loss": float(np.mean(losses)), "auc": float(roc_auc_score(ys, ps)),
           "recall@fpr1%": float(tpr[fpr <= 0.01].max()),
            "ap": float(average_precision_score(ys, ps)), "acc": float(((ps >= 0.5) == ys).mean()),
            "recall@0.5": float((ps[ys == 1] >= 0.5).mean()), "fpr@0.5": float((ps[ys == 0] >= 0.5).mean())}
    if len(ms) == len(ys) and np.any(np.array(ms) > 0):  # validación con montos: ahorro en el mejor umbral
        out["ahorro"] = best_value(ys, ps, np.array(ms))
    return out


def plot(log_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ep = pd.read_csv(log_dir / "epochs.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(ep["epoca"], ep["train_loss"], "o-", label="train (total)")
    ax[0].plot(ep["epoca"], ep["train_loss_cls"], "o--", label="train (clasificación)")
    ax[0].plot(ep["epoca"], ep["val_loss"], "s-", label="validación (clasificación)")
    ax[0].set(title="Loss", xlabel="época", yscale="log")
    ax[0].legend()
    for c in ("auc", "ap", "acc"):
        ax[1].plot(ep["epoca"], ep[c], "o-", label=c.upper())
    ax[1].set(title="Validación", xlabel="época", ylim=(0.4, 1.01))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(log_dir / "curvas.png", dpi=130)
    plt.close(fig)


def main(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    lim = 64 if a.smoke else 0
    tr, va = load_split(data / "train", lim), load_split(data / "val", 32 if a.smoke else 0)
    for i, d in enumerate(a.extra_train):  # lotes extra: se renumeran los pares para que no choquen
        extra = load_split(Path(d), lim)
        for r in extra:
            r["pair"] = f"extra{i}_{r['pair']}"
        tr += extra
    print(f"train={len(tr)} val={len(va)} tamaño={a.size} cabeza={a.head} peso_mascara={a.seg_weight}", flush=True)
    config = {k: v for k, v in vars(a).items() if k in ("backbone", "head", "topk", "size", "seg_weight", "lr", "crop")}
    config["size"] = list(a.size)
    if a.crop:
        config |= {"native": True, "max_side": 1600}
        tds = CropPairs(tr, a.crop, a.p_mask, a.peso_monto)
        vl = DataLoader(FullImages(va, config), batch_size=1, num_workers=a.workers, pin_memory=True)
    else:
        tds = TamperDataset(tr, a.size, True)
        vl = DataLoader(TamperDataset(va, a.size, False), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    tl = DataLoader(tds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0, worker_init_fn=seed_worker)

    model = Detector(a.backbone, a.head, a.topk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    epochs = 1 if a.smoke else a.epochs
    total = epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    rows, best, t0 = [], (-1.0, -1.0), time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        te, tot, tot_cls, n = time.time(), 0.0, 0.0, 0
        for batch in tl:
            x, y, m = batch[:3]
            w = batch[3] if len(batch) > 3 else torch.ones_like(y)
            if x.dim() == 5:  # pares de recortes: (B, 2, ...) -> (2B, ...)
                x, y, m, w = x.flatten(0, 1), y.flatten(0, 1), m.flatten(0, 1), w.flatten(0, 1)
            w = w.to(device)
            x = x.to(device, non_blocking=True)
            y, m = y.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, mapa = model(x)
            loss_cls = F.binary_cross_entropy_with_logits(logit.float(), y, weight=w)
            loss = loss_cls
            if mapa is not None and a.seg_weight > 0:
                loss = loss + a.seg_weight * F.binary_cross_entropy_with_logits(mapa.float(), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot, tot_cls, n = tot + loss.item(), tot_cls + loss_cls.item(), n + 1
        mv = evaluate(model, vl, device)
        row = {"epoca": epoch, "train_loss": round(tot / n, 5), "train_loss_cls": round(tot_cls / n, 5),
               **{k: round(v, 4) for k, v in mv.items()}, "min": round((time.time() - te) / 60, 2)}
        rows.append(row)
        with (out / "logs" / "epochs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
        plot(out / "logs")
        flag = ""
        score = (mv[a.select], mv["auc"])  # métrica elegida; el AUC desempata
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": config, "epoca": epoch, "val": mv}, out / "best.pt")
            flag = "  <- guardado"
        print(f"época {epoch}/{epochs}  loss {row['train_loss']:.4f}  val AUC {mv['auc']:.4f}  "
              f"recall@fpr1% {mv['recall@fpr1%']:.3f}  "
              + (f"ahorro Q{mv['ahorro']:,.0f}  " if "ahorro" in mv else "") + f"AP {mv['ap']:.4f}  "
              f"acc {mv['acc']:.3f}  ({row['min']} min, VRAM máx {torch.cuda.max_memory_allocated() / 2**30:.1f} GB){flag}",
              flush=True)

    summary = {"config": config, "gpu": torch.cuda.get_device_name(), "minutos": round((time.time() - t0) / 60, 1),
               "vram_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "epocas": epochs,
               "n_train": len(tr), "n_val": len(va), "seleccion": a.select, "mejor_val": list(best),
               "parametros_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    (out / "logs" / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "tamper"))
    p.add_argument("--out", default=str(ROOT / "models" / "detector"))
    p.add_argument("--extra-train", nargs="*", default=[], help="carpetas extra con metadata.jsonl para entrenar")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--head", choices=["gap", "mapa"], default="mapa")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seg-weight", type=float, default=1.0, help="peso de la pérdida del mapa con la máscara")
    p.add_argument("--size", type=int, nargs=2, default=[768, 576], metavar=("ALTO", "ANCHO"))
    p.add_argument("--crop", type=int, default=0, help="entrenar con recortes NxN a resolución completa")
    p.add_argument("--p-mask", type=float, default=0.75,
                   help="con --crop, fracción de recortes centrados en la zona editada (el resto, al azar)")
    p.add_argument("--peso-monto", action="store_true",
                   help="ponderar la pérdida de clasificación por el monto en riesgo de cada alteración")
    p.add_argument("--select", choices=["auc", "recall@fpr1%", "ahorro"], default="auc",
                   help="métrica de validación para guardar best.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
