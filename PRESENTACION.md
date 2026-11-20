# FacturaSegura: qué es, cómo funciona y qué tan bien funciona

Resumen para la presentación: el problema, la solución, las iteraciones y los resultados. El detalle técnico completo está en [RESULTADOS.md](RESULTADOS.md) y en el informe (`informe/informe.pdf`). La guía de la demo está en [DEMO.md](DEMO.md).

## En una frase

**FacturaSegura revisa la foto de una factura de reembolso y estima si fue editada** (total, fecha o NIT cambiados), marcando dónde. Solo manda a auditoría las sospechosas, con un umbral elegido para **maximizar el ahorro de la empresa**.

## El problema

- Para cobrar un reembolso, el empleado manda la **foto** de la factura. Editarla en el celular toma un minuto.
- Hoy hay dos opciones malas, calculadas por cada 1,000 facturas con 3 % de fraude, Q10 por auditoría y Q20 de fricción por sospechar de alguien honesto:
  - **No revisar:** se pierden **Q10,293** en fraude.
  - **Revisar todo:** cuesta más de lo que ahorra, **−Q19,107**.
- **Las alternativas sin deep learning no sirven:**
  - Las reglas fiscales (IVA 12/112, dígito verificador del NIT), aun leyendo perfecto la factura, detectan el 47 % con 14 % de falsas alarmas (−Q77).
  - El análisis forense clásico ELA queda al azar (AUC 0.51).
- La edición ocupa en promedio el **0.25 % de la foto**: es un detalle visual chico, justo lo que una CNN aprende a ver.

## Cómo funciona

```
Foto ─► EfficientNet-B0 a resolución completa ─► mapa de sospecha por zona ─► promedio de las 8 zonas más sospechosas ─► p
                                                         └──► zona marcada en rojo                    p ≥ 0.88 → Auditar · p < 0.88 → Aprobar
```

1. **Datos propios.** Un generador crea facturas FEL guatemaltecas ficticias pero válidas (NIT con dígito verificador, serie y número del UUID, IVA), las "fotografía" (perspectiva, sombras, arrugas, desteñido, ruido, JPEG) y crea una **gemela alterada** de cada foto. Las alteraciones imitan apps del celular: reescribir un valor o copiar un dígito, en tres niveles de dificultad. Cada una guarda su máscara y su **monto en riesgo** (cuánto gana el que la alteró). En total son 6,000 pares de entrenamiento, 300 de validación y 300 de prueba.
2. **Modelo.** Una **CNN EfficientNet-B0** preentrenada en ImageNet y ajustada por nosotros (4.1 M de parámetros, 17 MB). **No es un LLM:** mira píxeles, no lee texto. En vez de promediar toda la foto, produce un **mapa de sospecha** y decide con las zonas más sospechosas. Así la edición no se diluye, y el mapa sirve como explicación.
3. **Entrenamiento.** Se usan **recortes a resolución completa** de la foto auténtica y de su gemela, en la misma ventana: la única diferencia es la edición, así que el modelo aprende la edición y no el estilo de la factura. El mapa se entrena con la máscara de la zona editada, para que señale el lugar correcto.
4. **Decisión por valor.** `ahorro = fraude detenido − costo de auditar − fricción con empleados honestos`. El umbral (0.88) es el que maximiza ese ahorro en validación.

## Iteraciones

| It. | Qué cambiamos | AUC | Ahorro (Q / 1,000) | Lección |
|---|---|---|---|---|
| 1 | Clasificador estándar, foto reducida | 0.950 | 7,791 | Funciona, pero con muchas falsas alarmas |
| 2 | + mapa de sospecha + máscaras | 0.954 | 8,643 | Mitad de falsas alarmas, y ahora señala dónde |
| 3 | + recortes a **resolución completa** | 0.961 | 9,000 | **El cambio más importante:** la edición se perdía al reducir la foto |
| 3b | (ablación) sin máscaras | 0.964 | 9,294 | Detecta igual, pero el mapa ya no señala la edición (0 %): sin explicación confiable |
| 4 | + lote con más fraudes difíciles | **0.984** | 8,921 | **Mejor AUC, menos dinero:** se volvió desconfiado con facturas honestas |
| **5** | Datos de it4 + más zonas sin editar + elegir por recall con 1 % de falsas alarmas | 0.981 | **9,162** | **Final:** detecta los fraudes difíciles sin sospechar de más |
| 6 | + pérdida ponderada por monto en riesgo | 0.980 | 8,968 | Pierde localización y falsas alarmas inestables: descartado |

**Las tres lecciones para contar:**
1. **La métrica técnica y el valor no siempre se mueven juntos.** it4 tuvo el mejor AUC y ahorró menos.
2. **La explicación importa:** sin máscaras el modelo detecta, pero no se sabe por qué. Las mantuvimos aunque no subieran el AUC.
3. **Rendimientos decrecientes:** el modelo final ya detiene el 94 % del fraude. Un modelo perfecto ganaría como mucho ~Q800 más; el resto depende de los supuestos (prevalencia) y del proceso (auditorías más rápidas, verificar contra la SAT).

En el camino también corregimos el protocolo: al principio elegíamos el umbral mirando el conjunto de prueba, lo que inflaba el resultado. Ahora se elige en validación.

## Qué tan exitoso es

**En el conjunto de prueba sintético** (600 fotos que el modelo nunca vio):

| Métrica | Valor |
|---|---|
| AUC | **0.981** |
| Detecta (con solo 1 % de falsas alarmas) | **93 %** de las alteraciones |
| Por dificultad: fácil / media / difícil | 99 % / 95 % / 80 % |
| Ubica la zona editada | **87 %** de las veces |
| Tickets y facturas carta | Igual de bien (AUC 0.981 en ambos) |
| Punto débil | Copiar un dígito de la misma factura: 63 % |

**En dinero** (por cada 1,000 facturas, umbral 0.88):

| Política | Ahorro neto |
|---|---|
| No revisar nada | 0 (se pierden Q10,293) |
| Auditar todo | −Q19,107 |
| Reglas fiscales con lectura perfecta | −Q77 |
| **FacturaSegura** | **+Q9,162**: detiene el 94 % del fraude auditando solo 34 facturas |

Para una empresa con 1,000 reembolsos al mes, eso son **~Q9,000 mensuales**. El ahorro escala con la cantidad de fraude: Q2,991 si solo el 1 % está alterado y Q30,814 si es el 10 %.

**En la operación:**
- Corre en cualquier laptop sin GPU (1–2 s por foto en CPU, 41 ms con GPU).
- Costo por factura ≈ Q0, y las fotos no salen de la computadora.
- Todo el entrenamiento cupo en ~5 horas de una GPU de laptop.

**Con fotos reales:** es una prueba chica pero alentadora. En 3 facturas reales (gasolinera, cafetería, parqueo) editadas como lo haría alguien con el celular, **detectó las 3 ediciones** (probabilidad 1.00, marcando el monto cambiado) y **aprobó las 3 auténticas**. Las auténticas reales sacan probabilidades más altas que las sintéticas (0.58–0.78), así que en producción habría que recalibrar el umbral con datos reales.

## Limitaciones (decirlas antes de que pregunten)

- **Todo el entrenamiento y la evaluación grande son con datos sintéticos.** Con fotos reales solo hay una prueba de 3 facturas.
- **Los supuestos del ahorro son estimaciones,** sobre todo el 3 % de fraude; por eso se presentan con análisis de sensibilidad.
- **Se le escapan las falsificaciones más cuidadosas:** un dígito copiado de la misma factura.
- **Es una herramienta de priorización, no una prueba de fraude:** toda factura marcada la revisa una persona.

## Estructura sugerida del pitch (8–10 min)

1. **El problema (1 min):** fraude en reembolsos con fotos editadas; no revisar pierde Q10,293 y revisar todo pierde Q19,107 por cada 1,000 facturas.
2. **Por qué deep learning (1 min):** la edición ocupa el 0.25 % de la foto; las reglas y ELA no sirven.
3. **Datos (1.5 min):** el generador, las gemelas alineadas, los niveles de dificultad y el monto en riesgo. Mostrar la figura de ejemplos.
4. **Modelo de valor (1.5 min):** la fórmula, los supuestos, la curva y el umbral 0.88.
5. **Iteraciones (2.5 min):** la tabla y las tres lecciones (resolución, it4 contra it5, máscaras).
6. **Resultados y cierre (1.5 min):** 93 % detectado con 1 % de falsas alarmas, Q9,162 por cada 1,000 facturas, prueba con fotos reales, limitaciones y siguientes pasos.

Después, la demo en vivo ([DEMO.md](DEMO.md)) y las preguntas.
