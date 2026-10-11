from __future__ import annotations

import argparse
import json

from inf8239_u03.config import MOVIELENS_URL, ROOT
from inf8239_u03.data import ensure_movielens, load_movielens
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import PopularityRecommender, time_based_split

parser = argparse.ArgumentParser(description="LAB08 · baseline de popularidad suavizada")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--prior-count", type=float, default=None)
parser.add_argument("--prior-quantile", type=float, default=0.80)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()

# 1) Datos: descarga idempotente (data/raw está ignorado por git).
path = ensure_movielens(MOVIELENS_URL, ROOT / "data/raw")[0]
ratings, movies = load_movielens(path)

# 2) Partición temporal global sin fuga.
train, test = time_based_split(ratings, test_size=args.test_size)

# 3) Ajuste del baseline solo con train (popularidad suavizada).
model = PopularityRecommender(prior_count=args.prior_count, prior_quantile=args.prior_quantile).fit(train, movies)

# 4) Evaluación: mismo ranking para todos; se excluyen los ítems vistos en train.
catalog = set(train["movieId"])
known_users = set(train["userId"])
evaluable = test[test["userId"].isin(known_users) & test["movieId"].isin(catalog)]
seen_by_user = train.groupby("userId")["movieId"].agg(set).to_dict()
recommendations = {
    int(user): model.recommend(seen=seen_by_user.get(user, set()), k=args.k)["movieId"].astype(int).tolist()
    for user in evaluable["userId"].unique()
}

metrics = {
    "model": "smoothed_popularity",
    "prior_count": model.prior_count_,
    "prior_quantile": args.prior_quantile,
    "global_mean": model.global_mean_,
    "train_rows": len(train),
    "test_rows": len(test),
    "evaluable_rows": len(evaluable),
    "k": args.k,
    "hit_rate_at_k": hit_rate_at_k(recommendations, evaluable, args.k),
    "catalog_coverage": catalog_coverage(recommendations, len(catalog)),
    "leak_free": bool(train["timestamp"].max() < test["timestamp"].min()),
}

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
(reports / "popularity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
model.recommend(k=10).to_csv(reports / "popularity_top10.csv", index=False)
print(json.dumps(metrics, indent=2))
