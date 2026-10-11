from __future__ import annotations

from math import log2

import pandas as pd


def _relevance_by_user(truth: pd.DataFrame) -> dict[int, set[int]]:
    """Agrupa los ítems relevantes de cada usuario en un diccionario."""
    return truth.groupby("userId")["movieId"].agg(set).to_dict()


def hit_rate_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    hits = []
    for row in truth.itertuples():
        if row.userId in recommendations:
            hits.append(int(row.movieId in recommendations[row.userId][:k]))
    return sum(hits) / len(hits) if hits else 0.0


def precision_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    """Precisión media @K: aciertos / K para cada usuario con relevancia conocida."""
    relevant = _relevance_by_user(truth)
    if not relevant or k <= 0:
        return 0.0
    total = 0.0
    for user, items in relevant.items():
        recs = recommendations.get(user, [])[:k]
        total += len(set(recs) & items) / k
    return total / len(relevant)


def recall_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    """Recall medio @K: aciertos / (ítems relevantes del usuario)."""
    relevant = _relevance_by_user(truth)
    scores = []
    for user, items in relevant.items():
        if not items:
            continue
        recs = recommendations.get(user, [])[:k]
        scores.append(len(set(recs) & items) / len(items))
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    """NDCG medio @K con relevancia binaria (DCG / IDCG)."""
    relevant = _relevance_by_user(truth)
    scores = []
    for user, items in relevant.items():
        recs = recommendations.get(user, [])[:k]
        dcg = sum(1.0 / log2(rank + 1) for rank, item in enumerate(recs, start=1) if item in items)
        ideal_hits = min(k, len(items))
        idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def catalog_coverage(recommendations: dict[int, list[int]], catalog_size: int) -> float:
    recommended = {item for values in recommendations.values() for item in values}
    return len(recommended) / catalog_size if catalog_size else 0.0


def user_coverage(recommendations: dict[int, list[int]], user_ids) -> float:
    """Fracción de usuarios que reciben al menos una recomendación (no vacía)."""
    users = list(user_ids)
    if not users:
        return 0.0
    served = sum(1 for user in users if recommendations.get(user))
    return served / len(users)


def evaluate_top_n(
    recommendations: dict[int, list[int]],
    truth: pd.DataFrame,
    k: int = 10,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Resume las métricas Top-N (y cobertura de catálogo si se indica el tamaño)."""
    metrics: dict[str, float] = {
        "k": float(k),
        "users": float(truth["userId"].nunique()),
        "precision_at_k": precision_at_k(recommendations, truth, k),
        "recall_at_k": recall_at_k(recommendations, truth, k),
        "ndcg_at_k": ndcg_at_k(recommendations, truth, k),
        "hit_rate_at_k": hit_rate_at_k(recommendations, truth, k),
    }
    if catalog_size is not None:
        metrics["catalog_coverage"] = catalog_coverage(recommendations, catalog_size)
    return metrics
