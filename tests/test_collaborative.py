import numpy as np

from inf8239_u03.recommenders import MatrixFactorization


def test_matrix_factorization_recommend_excludes_seen(sample_ratings, sample_movies):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2, movies=sample_movies)
    seen = {1, 2, 3}
    result = model.recommend(user_id=1, seen=seen, k=2)
    assert seen.isdisjoint(set(result["movieId"]))
    assert len(result) == 2
    assert "collaborative_score" in result.columns


def test_matrix_factorization_unknown_user_returns_empty(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    assert model.recommend(user_id=999, k=2).empty


def test_matrix_factorization_predict_is_finite(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    assert np.isfinite(model.predict(1, 1))
