# INF-8239 · Unidad 03 · Sistemas recomendadores

Autor académico: Edwin Ramón José Nolasco

Proyecto inicial de LAB08 y LAB09. El dataset de muestra prueba el código; la evidencia final usa MovieLens.

## Inicio
```bash
uv python install 3.12
uv sync
uv run pytest -q
uv run python scripts/download_data.py
uv run python scripts/audit_data.py
```

## Laboratorios
```bash
uv run python scripts/lab08_content.py
uv run python scripts/lab08_baseline.py --test-size 0.2
uv run python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75
uv run streamlit run app/streamlit_app.py
```

## Evaluación
```bash
uv run python scripts/evaluate_models.py --factors 20 --epochs 12 --alpha 0.75 --k 10
```
Compara popularidad, contenido, factorización e híbrido con Precision@K, Recall@K,
NDCG@K, cobertura de catálogo y cobertura de usuarios, tanto en caliente como en
arranque en frío. El reporte queda en `reports/evaluation_metrics.json`.


## Pruebas
```bash
uv run pytest -q
```
Cubren la partición temporal sin fuga (`tests/test_split.py`), las métricas Top-N y
de cobertura (`tests/test_evaluation.py`, `tests/test_metrics.py`), el híbrido
(`tests/test_hybrid.py`) y el arranque en frío (`tests/test_cold_start.py`).

## Resultados finales
Modelos entrenados con `factors=20`, `epochs=12`, `alpha=0.75`, `k=10` sobre
MovieLens (`ml-latest-small`), con partición temporal (train 80 % / test 20 %).
Tabla principal (escenario **en caliente**, usuarios con historial en train):

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura catálogo | Cobertura usuarios | Entrenamiento (s) |
|---|---|---|---|---|---|---|
| Baseline (popularidad) | 0.08214 | 0.02197 | 0.09312 | 0.00559 | 1.00000 | 0.01 |
| Contenido | 0.02143 | 0.00506 | 0.02099 | **0.02237** | 1.00000 | 0.09 |
| Factorización | 0.10357 | 0.03588 | 0.11991 | 0.01703 | 1.00000 | 19.52 |
| **Híbrido** | **0.11786** | **0.04009** | **0.12747** | 0.01691 | 1.00000 | 20.07 |

Escenario de **arranque en frío** (88 usuarios que no existen en train):

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura usuarios |
|---|---|---|---|---|
| Baseline (popularidad) | 0.35455 | 0.04225 | 0.39389 | 1.00000 |
| Contenido | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Factorización | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| **Híbrido** | 0.35455 | 0.04225 | 0.39389 | **1.00000** |

El híbrido obtiene el mejor NDCG@10 en caliente y, vía su *fallback* a popularidad,
logra recomendar a **todos** los usuarios nuevos; contenido y factorización solos no
pueden servir a usuarios sin historial.

## Análisis de Pareto y decisión de producción
Analizamos el compromiso entre **utilidad** (NDCG@10 en caliente) y **esfuerzo
computacional** (tiempo de entrenamiento en la misma máquina):

| Modelo | Utilidad relativa (NDCG vs. híbrido) | Costo relativo | NDCG por segundo | ¿Frontera de Pareto? |
|---|---|---|---|---|
| Baseline (popularidad) | 73.1 % | 0.05 % | 9.31 | Sí |
| Contenido | 16.5 % | 0.45 % | 0.23 | No (dominado en *ranking*) |
| Factorización | 94.1 % | 97.3 % | 0.0061 | No (dominado por el híbrido) |
| Híbrido | 100 % | 100 % | 0.0064 | Sí |

- **Factorización está dominada.** El híbrido cuesta solo +0.55 s (+2.8 %) sobre la
  factorización, pero logra +6.3 % de NDCG y **cobertura total de arranque en frío**.
  La factorización pura nunca es la mejor elección.
- **Contenido está dominado en precisión** (NDCG 0.021, incluso por debajo del
  baseline), aunque conserva un nicho: **la mayor cobertura de catálogo (0.02237)**,
  útil para diversidad y para exponer ítems de cola larga.
- **Frontera de Pareto:** solo el **baseline** y el **híbrido** son no dominados.
  El baseline entrega ~73 % de la utilidad del híbrido con **~0 % del costo** y ya
  cubre el arranque en frío; el híbrido entrega la mejor utilidad a cambio de ~2 000×
  más cómputo.

**Aplicando el Principio de Pareto (80/20):** el baseline de popularidad captura
≈73 % de la calidad del mejor modelo por casi **nada** de cómputo, es decir, la mayor
parte del valor proviene de la fracción más barata del sistema. La personalización
que aporta la factorización es el ~27 % de utilidad restante a un costo 2 000× mayor.

**Decisión para producción:** usar el **sistema híbrido** como *ranker* principal,
porque (1) es el único no dominado en utilidad, (2) domina a la factorización en
calidad **y** en arranque en frío a costo casi idéntico, y (3) ya incorpora el
Principio de Pareto internamente: enruta a los usuarios nuevos (la mayoría de los
casos "baratos") al modelo de popularidad de costo casi nulo y reserva el cómputo
de factorización + contenido solo para usuarios con historial. Si la restricción
fuera latencia estricta en línea, sin infraestructura de reentrenamiento batch o un
MVP inicial, el **baseline de popularidad** es la elección racional por su relación
utilidad/costo. El modelo de contenido se recomienda solo como capa de diversidad,
no como *ranker* principal.

## Declaración de uso de IA
Durante el desarrollo de este proyecto se utilizó el modelo **Gemini 3.1 Pro** como consultor técnico y asistente de redacción en las siguientes actividades:

- **Desarrollo y Diseño:** Consultas teóricas para la implementación de la descarga idempotente, el *time-based split* sin fuga temporal, las arquitecturas base (popularidad, contenido, factorización e híbrido) y el marco de evaluación Top-N.
- **Depuración:** Asistencia para identificar el origen de errores técnicos, como el conflicto de compatibilidad con `np.matrix` en `scikit-learn` y el manejo de empates de *timestamp* en los cortes temporales.
- **Refactorización y Formato:** Apoyo en la optimización de código (ej. vectorización de métricas) y estructuración visual de las tablas de resultados y el análisis de Pareto en formato Markdown.

Todas las implementaciones y decisiones fueron validadas y adaptadas manualmente por el autor. Se adjunta una bitácora detallada (`AI_USE_DECLARATION.md`) como anexo.

Todas las decisiones de diseño, el análisis de resultados (incluido el de Pareto) y
la redacción fueron revisados y validados por el autor. **Se adjunta un log completo
de las sesiones de OpenCode en la entrega** como evidencia del uso de la herramienta.

## Interpretación
No presente una recomendación como verdad. Documente datos, candidatos, puntuación, métricas, fallback y límites.
