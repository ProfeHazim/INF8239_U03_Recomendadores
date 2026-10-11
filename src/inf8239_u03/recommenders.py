from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def smoothing_prior(counts: pd.Series, prior_count: float | None = None, prior_quantile: float = 0.80) -> float:
    """Prior de suavizado ``m`` para la media bayesiana.

    Si ``prior_count`` es None se usa el cuantil ``prior_quantile`` del número de
    valoraciones, un prior moderado habitual en sistemas de recomendación.
    """
    if prior_count is not None:
        return float(prior_count)
    return float(counts.quantile(prior_quantile))


def smoothed_popularity(
    ratings: pd.DataFrame,
    prior_count: float | None = None,
    prior_quantile: float = 0.80,
) -> pd.DataFrame:
    """Popularidad suavizada (Bayesian average / IMDb weighted rating).

    ``score = v/(v+m) * R + m/(v+m) * C`` donde ``v`` son las valoraciones del
    ítem, ``R`` su media, ``C`` la media global y ``m`` el prior de suavizado.
    Así los ítems con pocas valoraciones se contraen hacia la media global y no
    dominan el ranking por una única nota alta.
    """
    if ratings.empty:
        raise ValueError("ratings está vacío")
    stats = ratings.groupby("movieId")["rating"].agg(mean="mean", count="count")
    global_mean = float(ratings["rating"].mean())
    m = smoothing_prior(stats["count"], prior_count, prior_quantile)
    weight = stats["count"] / (stats["count"] + m)
    stats["smoothed_score"] = weight * stats["mean"] + (1.0 - weight) * global_mean
    stats.attrs["prior_count"] = m
    stats.attrs["global_mean"] = global_mean
    return stats.sort_values("smoothed_score", ascending=False)


class PopularityRecommender:
    """Baseline no personalizado de popularidad suavizada.

    Es el modelo de referencia del ejercicio y, a la vez, la política de
    *cold start*: como no usa el historial individual, recomienda lo mismo a
    cualquier usuario y solo debe excluir los ítems ya vistos.
    """

    def __init__(self, prior_count: float | None = None, prior_quantile: float = 0.80):
        self.prior_count = prior_count
        self.prior_quantile = prior_quantile

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> PopularityRecommender:
        self.scores_ = smoothed_popularity(ratings, self.prior_count, self.prior_quantile)
        self.prior_count_ = float(self.scores_.attrs["prior_count"])
        self.global_mean_ = float(self.scores_.attrs["global_mean"])
        self.ranking_ = self.scores_.index.astype(int).tolist()
        self.movies_ = movies
        return self

    def recommend(self, user_id: int | None = None, seen: set[int] | None = None, k: int = 10) -> pd.DataFrame:
        seen = set(seen or ())
        ranked = [item for item in self.ranking_ if item not in seen][:k]
        result = pd.DataFrame({"movieId": ranked})
        result["popularity_score"] = self.scores_.loc[ranked, "smoothed_score"].to_numpy()
        result["count"] = self.scores_.loc[ranked, "count"].to_numpy()
        if self.movies_ is not None:
            result = result.merge(self.movies_[["movieId", "title", "genres"]], on="movieId", how="left")
        return result.reset_index(drop=True)


def weighted_popularity(ratings: pd.DataFrame, movies: pd.DataFrame, quantile: float = 0.80) -> pd.DataFrame:
    """Compatibilidad con el flujo anterior: popularidad suavizada + filtro por conteo."""
    stats = smoothed_popularity(ratings, prior_quantile=quantile)
    stats["weighted_score"] = stats["smoothed_score"]
    minimum = float(stats["count"].quantile(quantile))
    return stats[stats["count"] >= minimum].join(movies.set_index("movieId")).sort_values("weighted_score", ascending=False)


def _fit_genre_tfidf(movies: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, object, dict[int, int]]:
    """Prepara el índice TF-IDF de géneros compartido por los modelos de contenido."""
    frame = movies.reset_index(drop=True).copy()
    frame["genres_text"] = frame["genres"].str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(frame["genres_text"])
    row_of = {int(movie_id): index for index, movie_id in enumerate(frame["movieId"])}
    return frame, vectorizer, matrix, row_of


class ContentRecommender:
    """Similitud ítem-ítem por géneros (TF-IDF + coseno)."""

    def fit(self, movies: pd.DataFrame) -> ContentRecommender:
        self.movies, self.vectorizer, self.matrix, self.row_of = _fit_genre_tfidf(movies)
        return self

    def recommend(self, title: str, k: int = 10) -> pd.DataFrame:
        matches = self.movies.index[self.movies["title"].eq(title)]
        if len(matches) == 0:
            raise KeyError(f"Título no encontrado: {title}")
        index = int(matches[0])
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        order = [value for value in scores.argsort()[::-1] if value != index][:k]
        result = self.movies.loc[order, ["movieId", "title", "genres"]].copy()
        result["content_score"] = scores[order]
        return result.reset_index(drop=True)


class ContentBasedRecommender:
    """Filtrado basado en contenido a nivel de usuario.

    Construye el perfil de cada usuario como la media ponderada de los vectores
    TF-IDF de géneros de las películas que valoró, con peso
    ``max(rating - neutral_rating, weight_floor)`` para dar más importancia a lo
    que le gustó. Cada candidato se puntúa por similitud coseno perfil–película.
    API compatible con el baseline: ``fit``/``recommend``.
    """

    def __init__(self, neutral_rating: float = 2.5, weight_floor: float = 0.1):
        self.neutral_rating = neutral_rating
        self.weight_floor = weight_floor

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> ContentBasedRecommender:
        self.movies_, self.vectorizer_, self.matrix_, self.row_of_ = _fit_genre_tfidf(movies)
        self.movie_ids_ = self.movies_["movieId"].to_numpy()
        history = ratings.loc[ratings["movieId"].isin(self.row_of_), ["userId", "movieId", "rating"]].copy()
        history["row"] = history["movieId"].map(self.row_of_).astype(int)
        self.history_ = history
        return self

    def user_profile(self, user_id: int):
        history = self.history_[self.history_["userId"].eq(user_id)]
        if history.empty:
            return None
        weights = np.clip(history["rating"].to_numpy() - self.neutral_rating, self.weight_floor, None)
        rows = history["row"].to_numpy()
        profile = self.matrix_[rows].multiply(weights[:, None]).sum(axis=0) / weights.sum()
        return np.asarray(profile)

    def score_items(self, user_id: int, movie_ids) -> np.ndarray:
        """Puntúa una lista concreta de ítems (útil para re-ranking híbrido).

        Devuelve un arreglo alineado con ``movie_ids``; los ítems desconocidos o
        los usuarios sin historial reciben 0.0.
        """
        ids = [int(movie_id) for movie_id in movie_ids]
        scores = np.zeros(len(ids), dtype=float)
        profile = self.user_profile(user_id)
        if profile is None:
            return scores
        pairs = [(position, self.row_of_[movie_id]) for position, movie_id in enumerate(ids) if movie_id in self.row_of_]
        if pairs:
            positions = [position for position, _ in pairs]
            matrix_rows = [row for _, row in pairs]
            scores[positions] = cosine_similarity(profile, self.matrix_[matrix_rows]).ravel()
        return scores

    def recommend(self, user_id: int, seen: set[int] | None = None, k: int = 10) -> pd.DataFrame:
        seen = set(seen or ())
        profile = self.user_profile(user_id)
        if profile is None:
            return pd.DataFrame(columns=["movieId", "content_score"])
        scores = cosine_similarity(profile, self.matrix_).ravel()
        ranked: list[tuple[int, float]] = []
        for index in scores.argsort()[::-1]:
            movie_id = int(self.movie_ids_[index])
            if movie_id in seen or scores[index] <= 0:
                continue
            ranked.append((movie_id, float(scores[index])))
            if len(ranked) == k:
                break
        result = pd.DataFrame(ranked, columns=["movieId", "content_score"])
        if not result.empty:
            result = result.merge(self.movies_[["movieId", "title", "genres"]], on="movieId", how="left")
        return result.reset_index(drop=True)

    def similar_to(self, title: str, k: int = 10) -> pd.DataFrame:
        matches = self.movies_.index[self.movies_["title"].eq(title)]
        if len(matches) == 0:
            raise KeyError(f"Título no encontrado: {title}")
        index = int(matches[0])
        scores = cosine_similarity(self.matrix_[index], self.matrix_).ravel()
        order = [value for value in scores.argsort()[::-1] if value != index][:k]
        result = self.movies_.loc[order, ["movieId", "title", "genres"]].copy()
        result["content_score"] = scores[order]
        return result.reset_index(drop=True)


class MatrixFactorization:
    """Filtrado colaborativo por factorización de matrices (SGD, sesgo global).

    API compatible con el baseline: ``fit(ratings, ...).recommend(user, seen, k)``.
    """

    def __init__(self, factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.05, seed: int = 42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed

    def fit(self, ratings: pd.DataFrame, epochs: int = 12, movies: pd.DataFrame | None = None) -> MatrixFactorization:
        self.users = sorted(ratings["userId"].unique())
        self.items = sorted(ratings["movieId"].unique())
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(self.users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(self.items), self.factors))
        self.global_mean = float(ratings["rating"].mean())
        observations = [(self.user_index[r.userId], self.item_index[r.movieId], float(r.rating)) for r in ratings.itertuples()]
        for _ in range(epochs):
            rng.shuffle(observations)
            for user, item, rating in observations:
                error = rating - self.predict_indices(user, item)
                previous_user = self.user_factors[user].copy()
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.regularization * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * previous_user - self.regularization * self.item_factors[item])
        self.movies_ = movies
        return self

    def predict_indices(self, user_index: int, item_index: int) -> float:
        return float(self.global_mean + self.user_factors[user_index] @ self.item_factors[item_index])

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.predict_indices(self.user_index[user_id], self.item_index[movie_id])

    def top_n(self, user_id: int, seen: set[int], k: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movieId", "collaborative_score"])
        all_items = np.asarray(self.items)
        mask = ~np.isin(all_items, np.asarray(list(seen), dtype=all_items.dtype))
        candidates = all_items[mask]
        all_scores = self.global_mean + self.item_factors @ self.user_factors[self.user_index[user_id]]
        scores = all_scores[mask]
        order = scores.argsort()[::-1][:k]
        return pd.DataFrame({"movieId": candidates[order], "collaborative_score": scores[order]})

    def recommend(self, user_id: int, seen: set[int] | None = None, k: int = 10) -> pd.DataFrame:
        table = self.top_n(user_id, set(seen or ()), k)
        if self.movies_ is not None and not table.empty:
            table = table.merge(self.movies_[["movieId", "title", "genres"]], on="movieId", how="left")
        return table.reset_index(drop=True)


class HybridRecommender:
    """Ensemble ponderado de filtrado colaborativo (MF) y contenido.

    Para usuarios conocidos genera un pool con ``MatrixFactorization`` y lo
    re-rankea combinando ambos puntajes normalizados:
    ``score = alpha * z(colaborativo) + (1 - alpha) * contenido``.

    Estrategia de arranque en frío en cascada:
    usuario sin historial pero con ítems en el catálogo → contenido;
    usuario totalmente nuevo → popularidad suavizada.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        factors: int = 20,
        epochs: int = 12,
        learning_rate: float = 0.01,
        regularization: float = 0.05,
        seed: int = 42,
        pool_size: int = 100,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar en [0, 1]")
        self.alpha = alpha
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.pool_size = pool_size

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> HybridRecommender:
        self.movies_ = movies
        self.content_ = ContentBasedRecommender().fit(ratings, movies)
        self.collaborative_ = MatrixFactorization(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            seed=self.seed,
        ).fit(ratings, epochs=self.epochs, movies=movies)
        self.popularity_ = PopularityRecommender().fit(ratings, movies)
        self.history_ = ratings.groupby("userId")["movieId"].agg(set).to_dict()
        return self

    @staticmethod
    def _normalize(values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        low, high = values.min(), values.max()
        if high - low < 1e-9:
            return np.ones_like(values)
        return (values - low) / (high - low)

    def recommend(self, user_id: int, seen: set[int] | None = None, k: int = 10) -> pd.DataFrame:
        seen = set(seen or ()) | set(self.history_.get(user_id, set()))
        if user_id in self.collaborative_.user_index:
            pool = self.collaborative_.top_n(user_id, seen, self.pool_size)
            if not pool.empty:
                pool = pool.copy()
                pool["content_score"] = self.content_.score_items(user_id, pool["movieId"].tolist())
                pool["hybrid_score"] = self.alpha * self._normalize(pool["collaborative_score"]) + (1 - self.alpha) * pool["content_score"]
                ranked = pool.nlargest(k, "hybrid_score")
                return ranked[["movieId", "hybrid_score", "collaborative_score", "content_score"]].reset_index(drop=True)

        content = self.content_.recommend(user_id, seen, k)
        if not content.empty:
            return content.rename(columns={"content_score": "hybrid_score"})[["movieId", "hybrid_score"]]

        popular = self.popularity_.recommend(user_id, seen, k)
        return popular.rename(columns={"popularity_score": "hybrid_score"})[["movieId", "hybrid_score"]]


def time_based_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición cronológica global sin fuga temporal (time-based split).

    Ordena todas las interacciones por fecha y reserva el ``test_size`` más
    reciente como conjunto de prueba. El corte es **estricto**:
    ``max(train.timestamp) < min(test.timestamp)``, por lo que ningún evento de
    prueba pudo informar el entrenamiento. A diferencia de
    :func:`temporal_leave_one_out`, aquí el corte es global y no por usuario, lo
    que refleja mejor un escenario causal de producción. Si muchos registros
    comparten el timestamp de corte, el conjunto de prueba puede quedar algo más
    grande que ``test_size`` (se prioriza cero fuga sobre el tamaño exacto).

    Parameters
    ----------
    ratings : DataFrame con ``userId``, ``movieId``, ``rating`` y ``timestamp``.
    test_size : fracción de interacciones reservada para prueba, en (0, 1).
    timestamp_col : nombre de la columna temporal (segundos epoch).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar en el intervalo abierto (0, 1)")
    if timestamp_col not in ratings.columns:
        raise ValueError(f"Falta la columna temporal '{timestamp_col}'")
    if len(ratings) < 2:
        raise ValueError("Se necesitan al menos 2 interacciones para partir")

    ordered = ratings.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n_test = min(max(1, round(len(ordered) * test_size)), len(ordered) - 1)
    split_at = len(ordered) - n_test
    # Timestamp del primer registro de prueba: el corte es estricto para que,
    # incluso con timestamps empatados, ningún evento de prueba entre en train.
    cutoff = ordered.loc[split_at, timestamp_col]
    train = ordered[ordered[timestamp_col] < cutoff]
    test = ordered[ordered[timestamp_col] >= cutoff]

    if train.empty or test.empty:
        raise ValueError("No se puede partir sin fuga: todos los timestamps coinciden")
    if float(train[timestamp_col].max()) >= float(test[timestamp_col].min()):
        raise ValueError("Partición con fuga temporal; revise los timestamps")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def temporal_leave_one_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ratings.sort_values(["userId", "timestamp"])
    test = ordered.groupby("userId", sort=False).tail(1)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)
