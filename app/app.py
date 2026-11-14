"""FacturaSegura: revisa si la foto de una factura fue alterada antes de aprobar un reembolso.

    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import gradcam, load_detector, predict  # noqa: E402
from src.viz import draw_box, overlay, suspicious_box, zoom  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "final" / "best.pt"

st.set_page_config(page_title="FacturaSegura", page_icon="🧾", layout="wide")


@st.cache_resource
def get_model(path: str):
    return load_detector(path)


def default_threshold(model_path: Path) -> float:
    """Umbral óptimo según el modelo de valor (eval/evaluate.py), si existe."""
    f = model_path.parent / "umbral.json"
    if f.exists():
        return float(json.loads(f.read_text())["umbral"])
    return 0.5


with st.sidebar:
    st.header("Configuración")
    model_path = Path(st.text_input("Modelo", str(DEFAULT_MODEL)))
    if not model_path.exists():
        st.error(f"No se encontró el modelo en {model_path}. Ver la guía de uso.")
        st.stop()
    umbral = st.slider("Umbral para auditar", 0.0, 1.0, default_threshold(model_path), 0.01,
                       help="Probabilidad de alteración a partir de la cual la factura se manda a auditoría. "
                            "El valor por defecto es el que maximiza el ahorro según el modelo de valor.")
    vista = st.radio("Mapa de calor", ["Mapa de sospecha del modelo", "Grad-CAM"])
    st.caption("Las fotos se procesan en esta computadora; no se envían a ningún servicio externo.")

model, cfg = get_model(str(model_path))

tab_app, tab_how = st.tabs(["🧾 Revisar facturas", "📊 Cómo funciona y valor"])

with tab_app:
    st.title("FacturaSegura")
    st.write("Sube la foto de una factura de reembolso. El modelo estima si fue **editada** (total, fecha o NIT "
             "cambiados) y marca la zona sospechosa.")
    files = st.file_uploader("Fotos de facturas", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    for up in files or []:
        st.divider()
        try:
            img = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        except Exception:
            st.error(f"{up.name}: no se pudo abrir como imagen.")
            continue
        with st.spinner("Analizando..."):
            prob, heat = predict(model, cfg, img)
            if vista == "Grad-CAM" or heat is None:
                heat = gradcam(model, cfg, img)
        box = suspicious_box(heat)
        left, right = st.columns([1, 1])
        left.image(draw_box(overlay(img, heat), box if prob >= umbral else None), caption=up.name, width="stretch")
        with right:
            if prob >= umbral:
                st.error(f"🔎 **Auditar.** Probabilidad de alteración: **{prob:.0%}** (umbral {umbral:.0%}).")
                z = zoom(img, box)
                if z is not None:
                    st.image(z, caption="Zona más sospechosa", width="stretch")
            else:
                st.success(f"✅ **Aprobar.** Probabilidad de alteración: **{prob:.0%}** (umbral {umbral:.0%}).")
            if abs(prob - umbral) < 0.1:
                st.warning("Zona gris: la probabilidad está cerca del umbral. Si el monto es alto, conviene revisarla.")
            st.progress(min(1.0, prob), text="Probabilidad de alteración")
            st.caption("El modelo detecta ediciones visuales (texto reescrito, dígitos pegados). No sustituye a la "
                       "auditoría: una factura marcada no es prueba de fraude, solo una alerta para revisarla.")

with tab_how:
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
  foto [label="Foto de la factura\\n(reembolso)"];
  cnn [label="EfficientNet-B0\\nmapa de sospecha 1/16"];
  pool [label="Promedio de las 8\\nzonas más sospechosas"];
  dec [label="p >= umbral?"];
  ok [label="Aprobar", fillcolor="#dcfce7"];
  aud [label="Auditar\\n(con zona marcada)", fillcolor="#fee2e2"];
  foto -> cnn -> pool -> dec; dec -> ok [label="no"]; dec -> aud [label="sí"];
}""")
    res = model_path.parent / "resultados"
    for f, cap in [("curva_valor.png", "Valor por umbral y curva ROC (conjunto de prueba sintético)"),
                   ("curvas.png", "Curvas de entrenamiento")]:
        if (res / f).exists():
            st.image(str(res / f), caption=cap)
    if (res / "metricas.json").exists():
        st.json(json.loads((res / "metricas.json").read_text()), expanded=False)
