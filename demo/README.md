# Ejemplos para la demo

`sinteticas/`: pares del conjunto de **prueba** (el modelo nunca los vio). Cada par es la misma foto auténtica y alterada. Resultados esperados en [DEMO.md](../DEMO.md#3-guion-de-la-demo-35-min).

| Archivo | Qué se cambió | Dificultad |
|---|---|---|
| `01_total_facil_*` | Total 139.40 → 339.40 | Fácil |
| `02_fecha_media_*` | Fecha 24/04/2024 → 21/04/2024 | Media |
| `03_total_dificil_*` | Total Q.188.55 → Q.238.55 (misma fuente, tinta y ruido) | Difícil |
| `04_nit_comprador_*` | NIT del comprador C.F. → 44653378-5 | Fácil |
| `05_carta_total_*` | Total 234.83 → 934.83 (factura carta) | Fácil |
| `06_digito_copiado_se_escapa_*` | Total Q540.30 → Q541.30 (dígito copiado de la misma factura) | Difícil: **el modelo no lo detecta** |

`reales/` (no está en git, porque tiene datos personales): fotos reales de facturas y una versión editada de cada una, simulando una app del celular. Copiarla a mano a la computadora de la presentación.
