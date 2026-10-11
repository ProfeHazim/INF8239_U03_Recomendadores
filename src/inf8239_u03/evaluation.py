"""Utilidades de evaluación offline para el proyecto de recomendadores.

Separa el *cómo se construyen las recomendaciones* (dependiente del modelo) del
*cómo se miden* (métricas en :mod:`inf8239_u03.metrics`). Incluye el escenario de
arranque en frío con usuarios que no aparecen en entrenamiento.
"""

from __future__ import annotations

import pandas as pd

from .metrics import evaluate_top_n, user_coverage


def build_recommendations(
    model,
    users,
    seen_by_user: dict[int, set[int]],
    k: int = 10,
) -> dict[int, list[int]]:
    """Genera el Top-K por usuario, excluyendo lo ya visto en entrenamiento.

    Acepta cualquier modelo con la interfaz ``recommend(user_id, seen, k)``.
    """
    recommendations: dict[int, list[int]] = {}
    for user in users:
        user_id = int(user)
        frame = model.recommend(user_id, seen_by_user.get(user_id, set()), k)
        recommendations[user_id] = frame["movieId"].astype(int).tolist() if not frame.empty else []
    return recommendations


def warm_test_split(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Test evaluable en caliente: usuarios e ítems observados en entrenamiento."""
    known_users = set(train["userId"])
    known_items = set(train["movieId"])
    return test[test["userId"].isin(known_users) & test["movieId"].isin(known_items)].copy()


def cold_start_split(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Usuarios de prueba ausentes en entrenamiento y sus interacciones futuras."""
    cold_users = sorted(set(test["userId"]) - set(train["userId"]))
    cold_truth = test[test["userId"].isin(cold_users)].copy()
    return cold_users, cold_truth


def evaluate_warm(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Evalúa un modelo en el escenario en caliente (Top-N + cobertura)."""
    evaluable = warm_test_split(train, test)
    seen_by_user = train.groupby("userId")["movieId"].agg(set).to_dict()
    users = sorted(evaluable["userId"].unique())
    recommendations = build_recommendations(model, users, seen_by_user, k)
    metrics = evaluate_top_n(recommendations, evaluable, k, catalog_size=train["movieId"].nunique())
    metrics["user_coverage"] = user_coverage(recommendations, users)
    return metrics


def evaluate_cold_start(
    model,
    cold_users,
    cold_truth: pd.DataFrame,
    catalog_size: int,
    k: int = 10,
) -> dict[str, float]:
    """Evalúa un modelo para usuarios nuevos (sin historial de entrenamiento).

    ``user_coverage`` aquí es la fracción de usuarios nuevos que logran recibir
    al menos una recomendación: la prueba concreta de que el modelo tiene una
    estrategia de arranque en frío.
    """
    if not cold_users:
        return {"users": 0.0, "user_coverage": 0.0}
    recommendations = build_recommendations(model, cold_users, {}, k)
    metrics = evaluate_top_n(recommendations, cold_truth, k, catalog_size=catalog_size)
    metrics["user_coverage"] = user_coverage(recommendations, cold_users)
    return metrics
