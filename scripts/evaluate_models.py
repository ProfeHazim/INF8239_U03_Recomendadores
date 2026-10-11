from __future__ import annotations

import argparse
import json
from time import perf_counter

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.evaluation import cold_start_split, evaluate_cold_start, evaluate_warm
from inf8239_u03.recommenders import (
    ContentBasedRecommender,
    HybridRecommender,
    MatrixFactorization,
    PopularityRecommender,
    time_based_split,
)

parser = argparse.ArgumentParser(description="Evaluación rigurosa Top-N de los recomendadores")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--factors", type=int, default=20)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--alpha", type=float, default=0.75)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()

ratings, movies = load_movielens(MOVIELENS_DIR)
train, test = time_based_split(ratings, test_size=args.test_size)

# Fábricas para medir el costo de entrenamiento de cada modelo.
factories = {
    "popularity": lambda: PopularityRecommender().fit(train, movies),
    "content": lambda: ContentBasedRecommender().fit(train, movies),
    "matrix_factorization": lambda: MatrixFactorization(factors=args.factors, seed=42).fit(train, epochs=args.epochs, movies=movies),
    "hybrid": lambda: HybridRecommender(alpha=args.alpha, factors=args.factors, epochs=args.epochs, seed=42).fit(train, movies),
}

catalog_size = train["movieId"].nunique()
cold_users, cold_truth = cold_start_split(train, test)

models = {}
report = {
    "config": {
        "test_size": args.test_size,
        "factors": args.factors,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "k": args.k,
        "train_rows": len(train),
        "test_rows": len(test),
        "catalog_size": catalog_size,
        "cold_users": len(cold_users),
    },
    "warm": {},
    "cold_start": {},
    "cost": {},
}

for name, factory in factories.items():
    start = perf_counter()
    models[name] = factory()
    fit_seconds = perf_counter() - start

    warm = evaluate_warm(models[name], train, test, args.k)
    cold = evaluate_cold_start(models[name], cold_users, cold_truth, catalog_size, args.k)

    report["warm"][name] = warm
    report["cold_start"][name] = cold
    report["cost"][name] = {"fit_seconds": fit_seconds}

reports_dir = ROOT / "reports"
reports_dir.mkdir(exist_ok=True)
(reports_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

columns = ["precision_at_k", "recall_at_k", "ndcg_at_k", "catalog_coverage", "user_coverage"]
header = "modelo".ljust(22) + "".join(column.rjust(16) for column in columns)
print(f"\nEscenario en caliente (k={args.k})")
print(header)
for name, metrics in report["warm"].items():
    print(name.ljust(22) + "".join(f"{metrics.get(column, 0.0):16.5f}" for column in columns))

print(f"\nArranque en frío · {len(cold_users)} usuarios nuevos")
print(header)
for name, metrics in report["cold_start"].items():
    print(name.ljust(22) + "".join(f"{metrics.get(column, 0.0):16.5f}" for column in columns))

print("\nCosto de entrenamiento")
for name, cost in report["cost"].items():
    print(f"{name.ljust(22)}{cost['fit_seconds']:9.2f} s")

print(f"\nReporte guardado en {reports_dir / 'evaluation_metrics.json'}")
