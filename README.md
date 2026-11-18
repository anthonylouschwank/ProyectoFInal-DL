# FacturaSegura: detector de facturas alteradas para reembolsos de gastos

Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes.

Cuando un empleado pide el reembolso de un gasto, manda la foto de la factura. Editar esa foto en el celular (subir el total, cambiar la fecha o poner el NIT de la empresa en una compra personal) toma un minuto. Revisar cada factura a mano es caro, y no revisar ninguna deja pasar el fraude. **FacturaSegura** estima la probabilidad de que la foto haya sido alterada y marca la zona sospechosa. Solo las facturas por encima de un umbral, elegido según el valor de negocio, van a auditoría.

**Sin LLMs:** el núcleo es una CNN convolucional (EfficientNet-B0 preentrenada en ImageNet) ajustada por el grupo.

## Cómo funciona

```
Foto ─► EfficientNet-B0 (resolución completa) ─► mapa de sospecha (1/16) ─► promedio de las 8 zonas más sospechosas ─► p
                                                         │                                                              │
                                                         └──► zona marcada en la foto                     p ≥ umbral ─► auditar
                                                                                                          p < umbral ─► aprobar
```

- **Datos:** facturas FEL sintéticas (generador del Proyecto 3) fotografiadas con perspectiva, sombras, desteñido y ruido. Cada foto tiene una **gemela alterada** con exactamente la misma geometría. Las alteraciones imitan lo que haría alguien con una app del celular: reescribir el total, la fecha o un NIT, o copiar un dígito de la misma factura. Hay tres niveles de dificultad y cada una guarda su **máscara** y su **monto en riesgo**. Después ambas fotos pasan por recorte, brillo, reescalado y recompresión tipo WhatsApp, para que el modelo no use atajos.
- **Modelo:** en un clasificador estándar (promedio global + capa lineal) la alteración, que es un detalle chico dentro de la foto, se diluye; llega a AUC 0.95, pero con más falsas alarmas. El modelo final produce un **mapa de sospecha por zona**, supervisado con las máscaras, y clasifica con el promedio de las *k* zonas más sospechosas.
- **Entrenamiento:** recortes de 512×512 **a resolución completa**, tomando la misma ventana de la foto auténtica y de su gemela alterada. Así el modelo aprende la alteración y no el estilo de la factura. Al usarlo, recibe la foto completa (la red es totalmente convolucional). El modelo final (it5) suma un lote con más falsificaciones difíciles, y se elige por su recall con 1 % de falsas alarmas para no acusar a empleados honestos.
- **Modelo de valor:** fraude detenido − costo de auditar las marcadas − fricción por acusar a empleados honestos, por cada 1,000 facturas. El umbral se elige maximizando ese valor, con análisis de sensibilidad.

Resultados de las iteraciones: [RESULTADOS.md](RESULTADOS.md). Para seguir entrenando (configuración actual, qué se probó y qué intentar): [ITERACIONES.md](ITERACIONES.md).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Con GPU NVIDIA, instalar primero PyTorch con CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

El modelo entrenado (`models/final/`) no está en git por su tamaño; se comparte aparte o se reentrena (abajo).

## Uso

```bash
streamlit run app/app.py
```

Subir la foto de una factura. La app responde **Aprobar** o **Auditar**, con la probabilidad y la zona sospechosa en rojo. El umbral por defecto es el óptimo del modelo de valor (`models/final/umbral.json`) y se puede mover en la barra lateral. Todo corre local: las fotos no salen de la computadora.

## Reproducir

```bash
# 1. Datos: 3,000 pares de entrenamiento, 300 de validación y 300 de prueba (~12 min con 8 núcleos, ~1 GB)
python -m src.build_dataset

# 2. Entrenamiento (RTX 3070 Laptop: 30-40 min cada uno)
python -m train.train --head gap  --size 768 576 --out models/it1_gap        # it1: foto reducida, clasificador estándar
python -m train.train --head mapa --size 768 576 --out models/it2_mapa       # it2: foto reducida, mapa de sospecha
python -m train.train --head mapa --crop 512     --out models/it3_recortes   # it3: recortes a resolución completa

# Modelo final (it5): lote extra con más casos difíciles + menos falsas alarmas
python -m src.build_dataset --out data/tamper_dificil --n-train 3000 --n-val 0 --n-test 0 \
    --seed 4040 --pesos 0.3 0.35 0.35 --p-copiar 0.8
python -m train.train --head mapa --crop 512 --extra-train data/tamper_dificil/train --p-mask 0.5 \
    --select "recall@fpr1%" --out models/it5_menos_falsas

# 3. Evaluación en el conjunto de prueba: métricas, localización, líneas base, curva de valor y sensibilidad
python -m eval.evaluate --model models/it5_menos_falsas/best.pt --guardar-umbral

# 4. Fotos reales (CSV con archivo_imagen,label,monto_en_riesgo[,campo])
python -m eval.evaluate --model models/it5_menos_falsas/best.pt --real eval/real/real.csv

# Pruebas
pytest -q
```

## Estructura

```
src/synth_fel.py       Generador de facturas FEL (del Proyecto 3) con posición de cada campo
src/tamper.py          Alteraciones realistas, máscara, monto en riesgo y dificultad
src/build_dataset.py   Pares auténtica/alterada alineados píxel a píxel
src/model.py           EfficientNet-B0 con cabeza estándar o mapa de sospecha, Grad-CAM, inferencia
src/viz.py             Mapa de calor y zona sospechosa sobre la foto
src/validate.py        Reglas FEL (del Proyecto 3): línea base sin deep learning
train/train.py         Entrenamiento (foto reducida o recortes a resolución completa)
eval/evaluate.py       Métricas, localización, líneas base (reglas, ELA), valor y sensibilidad
app/app.py             App en Streamlit
tests/                 Pruebas (pytest)
```

## Privacidad y uso responsable

- Las fotos se procesan localmente. Las fotos reales de prueba (`eval/real/`) están fuera de git porque pueden tener NIT y nombres.
- Una factura marcada **no es prueba de fraude**: es una alerta para que una persona la revise. El sistema no debe usarse para sancionar sin auditoría humana.
- Los datos sintéticos son ficticios: no contienen información de personas o comercios reales.
