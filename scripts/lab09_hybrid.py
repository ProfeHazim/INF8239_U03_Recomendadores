from __future__ import annotations

import argparse
import json
from time import perf_counter

from sklearn.metrics import root_mean_squared_error

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import (
    ContentBasedRecommender,
    MatrixFactorization,
    temporal_leave_one_out,
)

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, default=20)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--alpha", type=float, default=0.75)
args = parser.parse_args()

ratings, movies = load_movielens(MOVIELENS_DIR)
train, test = temporal_leave_one_out(ratings)

start = perf_counter()
collaborative_model = MatrixFactorization(factors=args.factors).fit(train, epochs=args.epochs, movies=movies)
content_model = ContentBasedRecommender().fit(train, movies)
train_seconds = perf_counter() - start

evaluable = test[test["userId"].isin(collaborative_model.user_index) & test["movieId"].isin(collaborative_model.item_index)]
predictions = [collaborative_model.predict(row.userId, row.movieId) for row in evaluable.itertuples()]
rmse = root_mean_squared_error(evaluable["rating"], predictions)

recommendations = {}
for user_id in evaluable["userId"].unique():
    seen = set(train.loc[train["userId"].eq(user_id), "movieId"])
    collaborative = collaborative_model.top_n(user_id, seen, 100)
    collaborative["normalized"] = (
        collaborative["collaborative_score"] - collaborative["collaborative_score"].min()
    ) / max(collaborative["collaborative_score"].max() - collaborative["collaborative_score"].min(), 1e-9)
    content_scores = content_model.score_items(user_id, collaborative["movieId"].tolist())
    content_min, content_max = content_scores.min(), content_scores.max()
    collaborative["content_score"] = (content_scores - content_min) / max(content_max - content_min, 1e-9)
    collaborative["hybrid_score"] = args.alpha * collaborative["normalized"] + (1 - args.alpha) * collaborative["content_score"]
    recommendations[int(user_id)] = collaborative.nlargest(10, "hybrid_score")["movieId"].astype(int).tolist()

metrics = {
    "rmse": float(rmse),
    "hit_rate_at_10": hit_rate_at_k(recommendations, evaluable, 10),
    "catalog_coverage": catalog_coverage(recommendations, len(collaborative_model.items)),
    "train_seconds": train_seconds,
    "factors": args.factors,
    "epochs": args.epochs,
    "alpha": args.alpha,
    "cold_start_policy": "popularidad por cantidad y media; no personalizada",
}

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/hybrid_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

popular = train.groupby("movieId")["rating"].agg(["mean", "count"]).sort_values(["count", "mean"], ascending=False)
cold_start_items = popular.index.tolist()[:10]
movies[movies["movieId"].isin(cold_start_items)].set_index("movieId").loc[cold_start_items].reset_index().to_csv(
    ROOT / "reports/cold_start_fallback.csv", index=False
)
print(json.dumps(metrics, indent=2))
