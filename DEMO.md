# Guía para la demo en vivo (laptop Windows)

Pasos para instalar FacturaSegura en otra computadora, probarlo antes de presentar y hacer la demo frente al profesor. Para la explicación del proyecto, ver [PRESENTACION.md](PRESENTACION.md).

## 1. Antes de la presentación (con internet, ~20 min)

Hacerlo **con tiempo** y no en el salón: la instalación descarga ~1 GB.

1. **Python 3.12** desde https://www.python.org/downloads/ (en el instalador, marcar **"Add python.exe to PATH"**).
2. **Git** desde https://git-scm.com/download/win (o descargar el repo como ZIP desde GitHub: *Code → Download ZIP*).
3. En **PowerShell**:

```powershell
cd $HOME\Documents
git clone https://github.com/anthonylouschwank/ProyectoFInal-DL.git
cd ProyectoFInal-DL
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Si `activate` da un error de "ejecución de scripts deshabilitada", correr antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y repetir la activación.
- En Windows, `pip` instala PyTorch para CPU, que alcanza: cada factura tarda ~1–2 s.

4. **Probar que funciona:**

```powershell
streamlit run app/app.py
```

Se abre el navegador en http://localhost:8501. Subir `demo/sinteticas/01_total_facil_alterada.jpg`: debe decir **Auditar** con el total marcado en rojo.

5. **Copiar las fotos reales de demo.** `demo/reales/` **no está en git** (la de la Shell tiene tu NIT y tu nombre). Pasarlas de la compu de entrenamiento a la laptop por USB, Drive o WhatsApp a ti mismo, dentro de `ProyectoFInal-DL\demo\reales\`.

6. Tener a mano el celular con una app que permita **escribir texto sobre una foto**: la galería de Samsung (*Editar → Dibujar / Texto*), Google Fotos (*Editar → Marcado*) o iPhone (*Editar → Marcación → +*). Practicar una vez.

## 2. Justo antes de presentar

```powershell
cd $HOME\Documents\ProyectoFInal-DL
.venv\Scripts\activate
streamlit run app/app.py
```

- **Subir una factura cualquiera antes de empezar,** para que el modelo ya esté cargado: la primera tarda unos segundos más.
- Dejar abiertas las carpetas `demo\sinteticas` y `demo\reales` en el Explorador de archivos.
- Si el profe va a dar una factura, tener el celular cargado y una forma rápida de pasar fotos a la laptop: WhatsApp Web abierto, correo o cable.

## 3. Guion de la demo (3–5 min)

| # | Qué hacer | Qué decir | Qué debería pasar |
|---|---|---|---|
| 1 | Subir `01_total_facil_autentica.jpg` | "Una factura auténtica de reembolso." | ✅ **Aprobar**, ~4 % |
| 2 | Subir `01_total_facil_alterada.jpg` | "La misma foto, pero el total se cambió de 139.40 a 339.40." | 🔎 **Auditar** ~100 %, total marcado en rojo |
| 3 | Subir `03_total_dificil_alterada.jpg` | "Alteración cuidadosa: misma fuente, misma tinta, mismo ruido." | 🔎 **Auditar**, marca Q.238.55 |
| 4 | Subir `04_nit_comprador_alterada.jpg` | "Fraude típico: poner el NIT de la empresa en una compra personal." | 🔎 **Auditar**, marca el NIT |
| 5 | Subir `demo\reales\fel_002_original.jpg` y `fel_002_editada.jpg` | "Fotos **reales** que el modelo nunca vio, editadas como lo haría alguien con el celular." | Original: ✅ **Aprobar** (0.78, algo cerca del umbral). Editada: 🔎 **Auditar**, marca "159.00" |
| 6 | (Opcional) `06_digito_copiado_se_escapa_alterada.jpg` | "Su límite: si copian un dígito de la misma factura (540.30 → 541.30), se le escapa." | ✅ Aprobar: **es el error esperado**, mostrarlo con honestidad |
| 7 | Pestaña **Cómo funciona y valor** | "El umbral sale del modelo de valor: Q9,162 de ahorro por cada 1,000 facturas." | Gráfica de valor y curvas |

Resultados medidos de los ejemplos:

| Ejemplo | Auténtica | Alterada |
|---|---|---|
| `01_total_facil` (139.40 → 339.40) | 0.04 | 1.00 |
| `02_fecha_media` (24/04 → 21/04) | 0.10 | 1.00 |
| `03_total_dificil` (188.55 → 238.55) | 0.01 | 1.00 |
| `04_nit_comprador` (C.F. → NIT) | 0.01 | 1.00 |
| `05_carta_total` (234.83 → 934.83) | 0.02 | 1.00 |
| `06_digito_copiado_se_escapa` (540.30 → 541.30) | 0.02 | **0.02 (no lo detecta)** |
| `reales/fel_001` Shell (270.51 → 870.51) | 0.58 | 1.00 |
| `reales/fel_002` &CAFE (59.00 → 159.00) | 0.78 | 1.00 |
| `reales/fel_003` parqueo (65.00 → 165.00) | 0.71 | 1.00 |

Medido en CPU, como en la laptop Windows, con umbral 0.88. En otra computadora pueden variar en el segundo decimal.

## 4. Prueba en vivo con una factura del profesor

La rúbrica pide al menos una entrada que no esté en el entrenamiento, idealmente propuesta por el catedrático:

1. Tomar la foto **de frente, con buena luz y el ticket completo**, que ocupe la mayor parte de la imagen.
2. Pasarla a la laptop y subirla: debería decir **Aprobar**.
3. En el celular, **editar la misma foto**: tapar el total con un rectángulo blanco y escribir otro número encima, o dejar que el profe elija qué cambiar.
4. Subir la versión editada: debería decir **Auditar** y marcar la zona.

## 5. Si algo sale distinto

| Situación | Qué hacer / qué decir |
|---|---|
| **Una factura real auténtica sale "Auditar"** | Es la limitación conocida: el modelo se entrenó con facturas sintéticas, y las fotos reales auténticas sacan probabilidades más altas (0.58–0.78 en nuestras pruebas contra ~0.1 en sintéticas). Mostrar que la probabilidad queda cerca del umbral (aviso de zona gris) y explicar que en producción se recalibraría el umbral con datos reales (plan de monitoreo). Se puede mover el umbral en la barra lateral para mostrar el efecto. |
| **Una edición real no se detecta** | Explicar qué tipo de edición fue. Si imita muy bien la fuente o es un solo dígito, es el punto débil medido (dígitos copiados: 63 % de recall). |
| **El recuadro rojo cae en otro lugar** | Cambiar en la barra lateral a **Grad-CAM** o mirar el mapa rojo completo. El modelo localiza la zona en 87 % de los casos sintéticos. |
| **Tarda mucho la primera vez** | Es la carga del modelo; las siguientes facturas tardan ~1–2 s en CPU. |
| **No se instaló o no funciona** | Plan B: el video de demostración (.mp4), las figuras de `docs/` (`ejemplo_deteccion.jpg`, `curva_valor_it5_menos_falsas.png`) y el informe PDF. |

## 6. Preguntas probables y respuestas cortas

- **¿Por qué deep learning y no reglas?** Las reglas fiscales (IVA 12/112, dígito del NIT), aun con lectura perfecta, detectan solo el 47 % con 14 % de falsas alarmas, y pierden dinero (−Q77 por cada 1,000 facturas). ELA, la técnica forense clásica, da AUC 0.51 (azar).
- **¿Es un LLM?** No. Es una CNN (EfficientNet-B0, 4.1 M de parámetros) que mira píxeles; nunca lee el texto.
- **¿De dónde salen los datos?** De un generador propio de facturas FEL fotografiadas, cada una con una gemela alterada idéntica salvo la edición. No existe un dataset público de facturas guatemaltecas alteradas.
- **¿Cómo saben que no hay fuga de información?** Cada conjunto usa una semilla distinta, las gemelas siempre van al mismo conjunto y el umbral se elige en validación, nunca en prueba.
- **¿Cuál fue la iteración más importante?** Entrenar con recortes a resolución completa (it3): la edición ocupa el 0.25 % de la foto y al reducirla se pierde.
- **¿Por qué no usaron el modelo con mejor AUC (it4: 0.984)?** Porque sospechaba de más facturas honestas, y su ahorro era menor (Q8,921 contra Q9,162).
- **¿Por qué el umbral es 0.88?** Es el que maximiza el ahorro en validación, con 3 % de facturas alteradas, Q10 por auditoría y Q20 de fricción. Con más fraude baja a 0.46, y con menos sube a 0.96.
- **¿Qué tan seguro es el ahorro?** Escala con la prevalencia, que es el supuesto más incierto: Q2,991 con 1 % de facturas alteradas y Q30,814 con 10 % (por cada 1,000 facturas). Auditar todo solo conviene a partir de ~8 % de fraude.
- **¿Qué falta para producción?** Medir con fotos reales (en las primeras 3 pruebas detectó las 3 ediciones y aprobó las 3 auténticas), recalibrar el umbral con datos propios de la empresa y verificar las marcadas contra la SAT antes de molestar al empleado.
