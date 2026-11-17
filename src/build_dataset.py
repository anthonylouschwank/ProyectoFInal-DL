"""Construye el dataset de facturas auténticas y alteradas.

Por cada factura sintética se toma una "foto" y se guardan dos versiones de esa misma foto:
la auténtica y una alterada (src/tamper.py). Las dos pasan por el mismo post-proceso
(recorte, brillo y reescalado idénticos; recompresión JPEG independiente), así que quedan
alineadas píxel a píxel y la única diferencia sistemática es la alteración. Las dos versiones
van siempre al mismo split.

    python -m src.build_dataset --out data/tamper --n-train 3000 --n-val 300 --n-test 300
    # Lote extra con más casos difíciles (iteración 4)
    python -m src.build_dataset --out data/tamper_dificil --n-train 3000 --n-val 0 --n-test 0 \
        --seed 4040 --pesos 0.3 0.35 0.35 --p-copiar 0.8

Crea data/tamper/<split>/images/*.jpg, masks/*.png (zona alterada) y metadata.jsonl con una
línea por imagen: {"file", "label" (1 = alterada), "mask", "pair", "layout", "total", "tamper"}.
"""

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

from .synth_fel import make_invoice, photograph, render_carta, render_ticket
from .tamper import NIVELES, draw_geometry, finalize, tamper

ROOT = Path(__file__).resolve().parent.parent


def generate(seed: int, carta_frac: float, pesos=NIVELES, p_copiar: float = 0.5):
    """Factura, foto auténtica y foto alterada a partir de una semilla (determinista)."""
    rng = random.Random(seed)
    for _ in range(20):  # reintenta si la factura no tiene campos alterables
        inv = make_invoice(rng)
        layout = "carta" if rng.random() < carta_frac else "ticket"
        doc, segs = render_carta(inv, rng) if layout == "carta" else render_ticket(inv, rng)
        photo, H = photograph(doc, rng)
        t = tamper(photo, segs, H, inv, rng, pesos, p_copiar)
        if t:
            return rng, inv, layout, photo, t
    raise RuntimeError(f"no se pudo alterar la factura con semilla {seed}")


def make_pair(args):
    idx, seed, out_dir, carta_frac, pesos, p_copiar = args
    rng, inv, layout, photo, (fake, mask, info) = generate(seed, carta_frac, pesos, p_copiar)
    base = {"pair": idx, "layout": layout, "categoria": inv["cat"], "total": inv["total"], "seed": seed,
            "carta_frac": carta_frac, "pesos": list(pesos), "p_copiar": p_copiar, "gt": inv["gt"]}
    recs = []
    geo = draw_geometry(rng)  # misma geometría para las dos: quedan alineadas píxel a píxel
    for label, img, m in [(0, photo, None), (1, fake, mask)]:
        img, m = finalize(img, rng, m, geo)
        name = f"{idx:05d}_{'alterada' if label else 'autentica'}"
        img.save(out_dir / "images" / f"{name}.jpg", quality=rng.randint(55, 92))
        rec = {"file": f"images/{name}.jpg", "label": label, **base}
        if label:
            m.save(out_dir / "masks" / f"{name}.png")
            rec |= {"mask": f"masks/{name}.png", "tamper": info}
        recs.append(rec)
    return recs


def build(out: Path, split: str, n: int, base_seed: int, workers: int, carta_frac: float,
          pesos=NIVELES, p_copiar: float = 0.5):
    d = out / split
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    jobs = [(i, base_seed * 1_000_003 + i, d, carta_frac, pesos, p_copiar) for i in range(n)]
    with Pool(workers) as pool, (d / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for k, recs in enumerate(pool.imap(make_pair, jobs, chunksize=4), 1):
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if k % 250 == 0 or k == n:
                print(f"  {split}: {k}/{n} pares", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "tamper"))
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--carta-frac", type=float, default=0.25)
    p.add_argument("--pesos", type=float, nargs=3, default=list(NIVELES), metavar=("FACIL", "MEDIO", "DIFICIL"),
                   help="mezcla de dificultad de las alteraciones")
    p.add_argument("--p-copiar", type=float, default=0.5, help="en las difíciles, probabilidad de copiar un dígito")
    a = p.parse_args()
    out = Path(a.out)
    for k, (split, n) in enumerate([("train", a.n_train), ("val", a.n_val), ("test", a.n_test)]):
        if n:
            build(out, split, n, a.seed + k, a.workers, a.carta_frac, a.pesos, a.p_copiar)  # semilla distinta por split
    print(f"Listo: {out}")
