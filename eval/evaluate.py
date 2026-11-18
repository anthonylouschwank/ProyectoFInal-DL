"""Evalúa un detector en el conjunto de prueba: métricas, localización, líneas base y valor.

    python -m eval.evaluate --model models/it2_mapa/best.pt                 # prueba sintética
    python -m eval.evaluate --model models/it2_mapa/best.pt --real eval/real/real.csv

Modelo de valor (por cada 1,000 facturas de reembolso):
    valor(t) = fraude detenido - costo de auditar las marcadas - fricción con empleados honestos
    fraude detenido   = 1000 · prevalencia · E[monto en riesgo · 1(p >= t) | alterada]
    costo de auditar  = 1000 · P(p >= t) · costo_revision
    fricción          = 1000 · (1 - prevalencia) · FPR(t) · costo_friccion
Se compara contra "no revisar nada" (valor 0) y "auditar todo". Los supuestos están en
--prevalencia, --costo-revision y --costo-friccion, y se mueven ±50 % en la sensibilidad.

El umbral se elige en el conjunto de VALIDACIÓN (el que maximiza el valor ahí) y el valor
se reporta en prueba con ese umbral. El óptimo calculado directamente en prueba se guarda
solo como referencia ("oraculo_prueba"): elegirlo así sería ajustar a la prueba.

Líneas base sin deep learning:
- reglas FEL: las validaciones del Proyecto 3 sobre los valores *visibles* en la foto,
  suponiendo una lectura perfecta (cota superior: en la realidad habría que leerlos con OCR)
- ELA (Error Level Analysis): diferencia contra una recompresión JPEG; puntaje = percentil
  alto de la diferencia por bloques

Salida en results/<modelo>_<conjunto>/: predicciones.csv, metricas.json, curva_valor.csv,
sensibilidad.csv, curva_valor.png y roc.png.
"""

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import normalize as N  # noqa: E402
from src.build_dataset import generate  # noqa: E402
from src.model import gradcam, load_detector, predict  # noqa: E402
from src.validate import validate  # noqa: E402

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


# ---------------------------------------------------------------- líneas base

def visible_fields(r):
    """Campos que se ven en la foto: los originales, con el campo alterado reemplazado."""
    gt = dict(r["gt"])
    if r["label"]:
        t = r["tamper"]
        if t["campo"] in ("total", "iva"):
            gt[t["campo"]] = N.money(t["nuevo"])
        elif t["campo"] in ("fecha", "nit_emisor", "nit_receptor"):
            gt[t["campo"]] = t["nuevo"]
    return gt


def rules_flag(r) -> int:
    issues = validate(visible_fields(r), today=date(2026, 12, 31))
    return int(any(i["regla"] not in ("fecha_antigua",) for i in issues))


def ela_score(img: Image.Image, quality=90) -> float:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    re = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    d = np.abs(np.asarray(img, np.float32) - np.asarray(re, np.float32)).mean(2)
    h, w = (d.shape[0] // 16) * 16, (d.shape[1] // 16) * 16
    blocks = d[:h, :w].reshape(h // 16, 16, w // 16, 16).mean((1, 3))
    return float(np.percentile(blocks, 99.5) / (np.median(blocks) + 1e-3))


# ---------------------------------------------------------------- valor

def value_curve(df, prev, c_rev, c_fric):
    alt, aut = df[df.label == 1], df[df.label == 0]
    rows = []
    for t in THRESHOLDS:
        tpr = (alt.prob >= t).mean()
        fpr = (aut.prob >= t).mean()
        detenido = 1000 * prev * (alt.monto * (alt.prob >= t)).mean()
        marcadas = 1000 * (prev * tpr + (1 - prev) * fpr)
        rows.append({"umbral": t, "recall": tpr, "fpr": fpr, "facturas_auditadas_1000": marcadas,
                     "fraude_detenido_Q": detenido, "fraude_perdido_Q": 1000 * prev * alt.monto.mean() - detenido,
                     "costo_auditoria_Q": marcadas * c_rev, "costo_friccion_Q": 1000 * (1 - prev) * fpr * c_fric,
                     "valor_neto_Q_1000": detenido - marcadas * c_rev - 1000 * (1 - prev) * fpr * c_fric})
    return pd.DataFrame(rows)


def point_value(tpr, fpr, monto_detectado_medio, prev, c_rev, c_fric):
    """Valor de una política binaria (reglas, auditar todo)."""
    marcadas = 1000 * (prev * tpr + (1 - prev) * fpr)
    return 1000 * prev * monto_detectado_medio - marcadas * c_rev - 1000 * (1 - prev) * fpr * c_fric


def val_predictions(model, cfg, data_dir: Path) -> pd.DataFrame:
    """Probabilidades en validación (sin Grad-CAM: es rápido) para elegir el umbral."""
    rows = []
    for r in (json.loads(line) for line in (data_dir / "val" / "metadata.jsonl").open(encoding="utf-8")):
        prob, _ = predict(model, cfg, Image.open(data_dir / "val" / r["file"]))
        rows.append({"label": r["label"], "prob": prob,
                     "monto": r["tamper"]["monto_en_riesgo"] if r["label"] else 0.0})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve.valor_neto_Q_1000.idxmax()].umbral)


# ---------------------------------------------------------------- principal

def load_records(a):
    if a.real:
        df = pd.read_csv(a.real, dtype={"archivo_imagen": str})
        base = Path(a.real).parent
        recs = [{"path": str(base / r.archivo_imagen), "label": int(r.label), "file": r.archivo_imagen,
                 "monto": float(r.monto_en_riesgo) if r.label else 0.0, "layout": "real",
                 "campo": getattr(r, "campo", "") if r.label else "", "metodo": "real" if r.label else "",
                 "mask": None} for r in df.itertuples()]
        return recs, "real"
    d = Path(a.data) / a.split
    recs = [json.loads(line) for line in (d / "metadata.jsonl").open(encoding="utf-8")]
    for r in recs:
        if "gt" not in r:  # datasets generados antes de guardar los campos: se reconstruyen
            r["gt"] = generate(r["seed"], r.get("carta_frac", 0.25))[1]["gt"]
        r["path"] = str(d / r["file"])
        r["mask"] = str(d / r["mask"]) if r.get("mask") else None
        r["monto"] = r["tamper"]["monto_en_riesgo"] if r["label"] else 0.0
        r["campo"] = r["tamper"]["campo"] if r["label"] else ""
        r["metodo"] = r["tamper"]["metodo"] if r["label"] else ""
        r["dificultad"] = r["tamper"].get("dificultad", "") if r["label"] else ""
    return recs, a.split


def main(a):
    recs, name = load_records(a)
    model, cfg = load_detector(a.model)
    tag = Path(a.model).parent.name
    out = ROOT / "results" / f"{tag}_{name}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in enumerate(recs):
        img = Image.open(r["path"]).convert("RGB")
        prob, heat = predict(model, cfg, img)
        row = {"file": r["file"], "label": r["label"], "prob": prob, "layout": r["layout"], "campo": r["campo"],
               "metodo": r["metodo"], "dificultad": r.get("dificultad", ""), "monto": r["monto"],
               "ela": ela_score(img)}
        if "gt" in r:
            row["reglas"] = rules_flag(r)
        if r["label"] and r["mask"]:  # localización: ¿el punto más sospechoso cae en la zona alterada?
            mask = np.asarray(Image.open(r["mask"]).resize(img.size)) > 0
            cam = np.asarray(gradcam(model, cfg, img))
            row["acierto_gradcam"] = bool(mask.flat[cam.argmax()])
            if heat is not None:
                row["acierto_mapa"] = bool(mask.flat[np.asarray(heat).argmax()])
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(recs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "predicciones.csv", index=False)

    y, p = df.label.values, df.prob.values
    alt = df[df.label == 1]
    m = {"modelo": tag, "conjunto": name, "n": len(df), "auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
         "acc@0.5": float(((p >= 0.5) == y).mean()), "recall@0.5": float((alt.prob >= 0.5).mean()),
         "fpr@0.5": float((df[df.label == 0].prob >= 0.5).mean()),
         "auc_ela": roc_auc_score(y, df.ela.values)}
    fpr_c, tpr_c, thr_c = roc_curve(y, p)
    for target in (0.01, 0.05):
        m[f"recall@fpr{int(target * 100)}%"] = float(tpr_c[fpr_c <= target].max())
    m["recall_por_campo@0.5"] = alt.groupby("campo").prob.apply(lambda s: round(float((s >= 0.5).mean()), 3)).to_dict()
    m["recall_por_metodo@0.5"] = alt.groupby("metodo").prob.apply(lambda s: round(float((s >= 0.5).mean()), 3)).to_dict()
    if "dificultad" in alt and alt.dificultad.notna().any():
        m["recall_por_dificultad@0.5"] = alt.groupby("dificultad").prob.apply(
            lambda s: round(float((s >= 0.5).mean()), 3)).to_dict()
        aut = df[df.label == 0]
        m["auc_por_dificultad"] = {d: round(float(roc_auc_score(np.r_[np.zeros(len(aut)), np.ones(len(g))],
                                                                np.r_[aut.prob, g.prob])), 4)
                                   for d, g in alt.groupby("dificultad")}
    m["auc_por_formato"] = {lay: float(roc_auc_score(g.label, g.prob)) for lay, g in df.groupby("layout")
                            if g.label.nunique() == 2}
    if "acierto_gradcam" in df:
        m["localizacion_gradcam"] = float(alt.acierto_gradcam.mean())
    if "acierto_mapa" in df:
        m["localizacion_mapa"] = float(alt.acierto_mapa.mean())
    if "reglas" in df:
        m["reglas_recall"] = float(alt.reglas.mean())
        m["reglas_fpr"] = float(df[df.label == 0].reglas.mean())

    # Valor: el umbral se elige en validación y se evalúa en prueba
    dv = val_predictions(model, cfg, Path(a.data))
    dv.to_csv(out / "predicciones_val.csv", index=False)
    vkw = {"prev": a.prevalencia, "c_rev": a.costo_revision, "c_fric": a.costo_friccion}
    t_val = best_threshold(value_curve(dv, **vkw))
    curve = value_curve(df, **vkw)
    curve.to_csv(out / "curva_valor.csv", index=False)
    best = curve[curve.umbral == t_val].iloc[0]
    oraculo = curve.loc[curve.valor_neto_Q_1000.idxmax()]
    m["valor"] = {"supuestos": {"prevalencia": a.prevalencia, "costo_revision_Q": a.costo_revision,
                                "costo_friccion_Q": a.costo_friccion},
                  "fraude_total_sin_control_Q_1000": 1000 * a.prevalencia * alt.monto.mean(),
                  "umbral_optimo": t_val, "umbral_elegido_en": "validacion",
                  "en_umbral_optimo": best.round(4).to_dict(),
                  "oraculo_prueba": {"umbral": float(oraculo.umbral),
                                     "valor_neto_Q_1000": round(float(oraculo.valor_neto_Q_1000), 1)},
                  "auditar_todo_Q_1000": point_value(1, 1, alt.monto.mean(), a.prevalencia, a.costo_revision,
                                                     a.costo_friccion)}
    if "reglas" in df:
        m["valor"]["reglas_Q_1000"] = point_value(m["reglas_recall"], m["reglas_fpr"],
                                                  (alt.monto * alt.reglas).mean(), a.prevalencia,
                                                  a.costo_revision, a.costo_friccion)
    sens = []
    for label, kw in [("base", {}), ("prevalencia 1%", {"prev": 0.01}), ("prevalencia 5%", {"prev": 0.05}),
                      ("prevalencia 10%", {"prev": 0.10}),
                      ("costo revisión -50%", {"c_rev": a.costo_revision * 0.5}),
                      ("costo revisión +50%", {"c_rev": a.costo_revision * 1.5}),
                      ("fricción -50%", {"c_fric": a.costo_friccion * 0.5}),
                      ("fricción +50%", {"c_fric": a.costo_friccion * 1.5})]:
        args = {"prev": a.prevalencia, "c_rev": a.costo_revision, "c_fric": a.costo_friccion, **kw}
        c = value_curve(df, **args)
        t_s = best_threshold(value_curve(dv, **args))  # umbral re-elegido en validación para el escenario
        b = c[c.umbral == t_s].iloc[0]
        base_t = c[c.umbral == best.umbral].iloc[0]
        sens.append({"escenario": label, "umbral_optimo": t_s, "valor_optimo_Q_1000": round(b.valor_neto_Q_1000, 1),
                     "valor_con_umbral_base_Q_1000": round(base_t.valor_neto_Q_1000, 1),
                     "auditar_todo_Q_1000": round(point_value(1, 1, alt.monto.mean(), args["prev"], args["c_rev"],
                                                              args["c_fric"]), 1)})
    pd.DataFrame(sens).to_csv(out / "sensibilidad.csv", index=False)
    m["sensibilidad"] = sens
    (out / "metricas.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(curve.umbral, curve.valor_neto_Q_1000, "-", lw=2, label="detector (valor neto)")
    ax[0].plot(curve.umbral, curve.fraude_detenido_Q, "--", label="fraude detenido")
    ax[0].plot(curve.umbral, -(curve.costo_auditoria_Q + curve.costo_friccion_Q), ":", label="− costo auditoría y fricción")
    ax[0].axhline(m["valor"]["auditar_todo_Q_1000"], color="gray", ls="-.", label="auditar todo")
    if "reglas_Q_1000" in m["valor"]:
        ax[0].axhline(m["valor"]["reglas_Q_1000"], color="purple", ls="-.", lw=1, label="reglas FEL (lectura perfecta)")
    ax[0].axvline(best.umbral, color="k", ls=":", lw=1)
    ax[0].set(title=f"Valor por cada 1,000 facturas (umbral {best.umbral:.2f} elegido en validación: "
                    f"Q{best.valor_neto_Q_1000:,.0f})",
              xlabel="umbral de probabilidad", ylabel="Q")
    ax[0].legend(fontsize=8)
    ax[1].plot(fpr_c, tpr_c, label=f"detector (AUC {m['auc']:.3f})")
    fe, te, _ = roc_curve(y, df.ela.values)
    ax[1].plot(fe, te, label=f"ELA (AUC {m['auc_ela']:.3f})")
    if "reglas" in df:
        ax[1].plot(m["reglas_fpr"], m["reglas_recall"], "s", label="reglas FEL")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].set(title="Curva ROC", xlabel="tasa de falsos positivos", ylabel="recall")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "curva_valor.png", dpi=130)

    if a.guardar_umbral:  # el umbral óptimo queda junto al modelo para la app
        (Path(a.model).parent / "umbral.json").write_text(json.dumps(
            {"umbral": float(best.umbral), "fuente": str(out / "curva_valor.csv"), **m["valor"]["supuestos"]},
            indent=1), encoding="utf-8")
    resumen = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
               if k not in ("sensibilidad", "valor")}
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print(f"Valor: umbral {best.umbral:.2f} (elegido en validación) -> Q{best.valor_neto_Q_1000:,.0f} por 1,000 facturas "
          f"[óptimo mirando la prueba: {oraculo.umbral:.2f} -> Q{oraculo.valor_neto_Q_1000:,.0f}] "
          f"(auditar todo: Q{m['valor']['auditar_todo_Q_1000']:,.0f}; "
          f"reglas: Q{m['valor'].get('reglas_Q_1000', float('nan')):,.0f}; "
          f"fraude total sin control: Q{m['valor']['fraude_total_sin_control_Q_1000']:,.0f})")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=str(ROOT / "data" / "tamper"), help="carpeta con val/ y test/")
    p.add_argument("--split", default="test")
    p.add_argument("--real", default="", help="CSV con archivo_imagen,label,monto_en_riesgo[,campo]")
    p.add_argument("--prevalencia", type=float, default=0.03, help="fracción de facturas alteradas")
    p.add_argument("--costo-revision", type=float, default=10.0, help="Q por auditar una factura (~10 min)")
    p.add_argument("--costo-friccion", type=float, default=20.0, help="Q por acusar a un empleado honesto")
    p.add_argument("--guardar-umbral", action="store_true", help="guardar el umbral óptimo junto al modelo")
    main(p.parse_args())
