# Notas para seguir iterando el modelo

Guía corta para quien continúe el entrenamiento en otra computadora (persona o agente). Contiene la configuración actual, lo que ya se probó y qué intentar. El detalle de cada resultado está en [RESULTADOS.md](RESULTADOS.md).

## Estado actual

- **Modelo final: it3** (`models/final/best.pt`, también en `models/it3_recortes/`). Los pesos no están en git: copiarlos aparte o reentrenar (~30 min).
- **Prueba sintética:** AUC 0.961, recall con 1 % de falsas alarmas 90 %, recall en difíciles 66 %.
- **Valor:** **Q9,264 por cada 1,000 facturas** con umbral 0.98.
- **Techo de valor:** con estos supuestos el máximo posible es ≈ Q10,025, así que solo quedan ~Q760 por ganar. Se gana más **bajando falsas alarmas** que subiendo el AUC (ver it4).

## Configuración de it3

| Grupo | Parámetro | Valor |
|---|---|---|
| Datos | Generación | `python -m src.build_dataset` (semilla 2026; train/val/test usan 2026/2027/2028) |
| | Pares | 3,000 / 300 / 300 (train / val / test) |
| | Facturas carta | 25 % (`--carta-frac 0.25`) |
| | Dificultad fácil / media / difícil | 0.5 / 0.3 / 0.2 (`--pesos`) |
| | Copiar dígito (en difíciles) | 50 % (`--p-copiar 0.5`) |
| Modelo | Backbone | `efficientnet_b0` (timm, ImageNet), 4.1 M parámetros |
| | Cabeza | `mapa`: logits a 1/16 (fusión de stride 16 y 32) + promedio de las top-8 zonas |
| Entrenamiento | Recortes | `--crop 512` a resolución completa: misma ventana de la auténtica y de la alterada |
| | Ventana sobre la zona editada | 75 % de las veces (`p_mask`, en `CropPairs`) |
| | Etiqueta del recorte | "alterada" si contiene ≥ 60 px de máscara |
| | Pérdida | BCE de clasificación + 1.0 × BCE del mapa contra la máscara (`--seg-weight 1.0`) |
| | Objetivo del mapa | Máscara promediada a 1/16 × 4, recortada a [0, 1] (celda con ≥ 25 % editado = 1) |
| | Optimizador | AdamW, lr máx. 3e-4, weight decay 1e-4, OneCycle (10 % de calentamiento), clip de gradiente 2.0 |
| | Lote | 8 pares = 16 recortes (`--batch 8`) |
| | Épocas | 10 |
| | Precisión | bf16 autocast, **sin** `channels_last` (ver advertencias) |
| | Semilla | 42 |
| | Selección | Mejor AUC de validación (fotos completas) |
| Inferencia | Resolución | Foto completa, máx. 1600 px por lado, relleno a múltiplo de 128 |
| | Umbral | 0.98 (óptimo de valor, en `models/final/umbral.json`) |
| Valor | Supuestos | Prevalencia 3 %, Q10 por auditoría, Q20 de fricción por falsa alarma |

Tiempos en una RTX 3070 Laptop: ~2.8 min por época (3,000 pares), ~20 min de `eval/evaluate.py` (calcula Grad-CAM por foto), 3.6 GB de VRAM.

## Lo que ya se probó

| Iteración | Cambio | AUC prueba | Recall difícil | FPR @0.5 | Ahorro Q/1000 | Conclusión |
|---|---|---|---|---|---|---|
| it1 | Foto reducida 768×576, promedio global + lineal | 0.950 | 59 % | 3.7 % | 8,213 | Línea base |
| it2 | + mapa de sospecha + máscaras | 0.954 | 56 % | 1.7 % | 8,654 | Menos falsas alarmas, localiza |
| **it3** | **+ recortes 512 a resolución completa** | **0.961** | **66 %** | **1.7 %** | **9,264** | **Final** |
| it3b | it3 sin máscaras (`--seg-weight 0`) | 0.964 | 61 % | 0.0 % | 9,315 | Igual valor, pero el mapa ya no localiza (0 %) |
| it4 | + 3,000 pares con más difíciles (0.3/0.35/0.35, `--p-copiar 0.8`), 15 épocas | 0.984 | 80 % | 6.0 % | 9,059 | Mejor AUC, **más falsas alarmas → no más valor**. Mejor validación en la época 5 y luego baja |

El ruido de la evaluación es de ±Q200 (una o dos facturas del conjunto de prueba). Diferencias menores a eso no cuentan.

## Ideas para las siguientes iteraciones

Ordenadas por lo que esperamos que aporte. **Cambiar una cosa por iteración.**

1. **Elegir el checkpoint por valor y no por AUC.** it4 tuvo su mejor AUC en la época 5, pero siguió volviéndose desconfiado. Seleccionar por "recall con FPR ≤ 1 %" en validación, o por el valor directo, debería favorecer modelos bien calibrados. *Requiere código:* calcular esa métrica en `evaluate()` de `train/train.py` y usarla para guardar `best.pt`.
2. **Bajar falsas alarmas con más recortes de facturas auténticas.** Bajar `p_mask` (0.75 → 0.5) para que vea más zonas sin editar, o darle más peso a la clase auténtica en la BCE. Combinado con los datos de it4 podría quedarse con su recall en difíciles sin sus falsas alarmas. *Requiere código:* exponer `--p-mask` y/o `--peso-autenticas` en `train/train.py`.
3. **it4 con lr más baja** (1.5e-4) o menos épocas (6–8), porque sobreajustó después de la época 5. Solo flags: `--lr 1.5e-4 --epochs 8 --extra-train data/tamper_dificil/train`.
4. **Backbone más grande:** `--backbone efficientnet_b2` (o `convnext_nano`, `resnet50`). Cabe en 8 GB con recortes de 512 y batch 8. Solo flag.
5. **Aumentos de robustez para fotos reales:** recomprimir JPEG y reescalar 0.7–1.0 *igual* en los dos recortes de cada par, dentro de `CropPairs`. No debería cambiar mucho lo sintético, pero ayudaría con fotos reales. *Requiere código.*
6. **Variaciones de la cabeza:** `--topk 4` o `16`, y `--seg-weight 0.5` o `2`. Solo flags.
7. **Entrada forense:** agregar canales de residuo de alta frecuencia (filtros SRM) a la entrada RGB. Es el enfoque clásico de forensia de imágenes. *Requiere código* (en `Detector` y en `to_tensor`).

## Protocolo para comparar

```bash
python -m src.build_dataset                                       # mismos datos (semillas por defecto)
python -m src.build_dataset --out data/tamper_dificil --n-train 3000 --n-val 0 --n-test 0 \
    --seed 4040 --pesos 0.3 0.35 0.35 --p-copiar 0.8               # solo si se usa el lote de it4
pytest -q                                                         # incluye pruebas de determinismo
python -m train.train --head mapa --crop 512 --out models/it5_<cambio> [flags]
python -m eval.evaluate --model models/it5_<cambio>/best.pt
```

1. **No elegir nada mirando el conjunto de prueba.** Comparar en validación (`models/<it>/logs/epochs.csv`) y evaluar en prueba solo a los candidatos.
2. **El criterio de decisión es el ahorro en el umbral óptimo** (`results/<it>_test/metricas.json` → `valor.en_umbral_optimo.valor_neto_Q_1000`). El AUC, el recall en difíciles y la localización sirven para desempatar o para explicar.
3. **No usar un modelo que ya no localice** (`localizacion_mapa` cerca de 0, como it3b), aunque tenga buen valor: la app depende del mapa para explicarle al auditor.
4. Por cada iteración, agregar una fila a `docs/comparacion_iteraciones.csv` y a la tabla de `RESULTADOS.md`, y copiar `curvas.png` y `curva_valor.png` a `docs/`.
5. Si una iteración gana por más de ~Q200: copiar su `best.pt` a `models/final/`, correr `eval.evaluate` con `--guardar-umbral` y copiar `curva_valor.png`, `metricas.json` y `sensibilidad.csv` a `models/final/resultados/` (los usa la app).

## Advertencias técnicas

- **No usar `channels_last` con bf16:** cuDNN arma un plan nuevo por cada tamaño de imagen y la validación pasa de 35 ms a 1.2 s por foto.
- **Determinismo de los datos:** las imágenes dependen de la versión de Pillow y de las fuentes de `src/fonts/`. Con versiones distintas, las etiquetas y semillas son iguales pero los píxeles pueden variar un poco. Los números de `RESULTADOS.md` se obtuvieron con torch 2.11 + cu128, timm 1.0.31, Pillow 13.0 y numpy 2.5 en Linux. Para comparar con justicia, **reevaluar it3 en la máquina nueva** antes de comparar.
- **Windows:** los workers del DataLoader ya funcionan con `spawn` (`seed_worker` es una función de módulo). Si hay problemas, usar `--workers 0`.
- **Datos personales:** las fotos reales van en `eval/real/` (fuera de git). No subirlas.
