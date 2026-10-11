import numpy as np

from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out


def test_temporal_split_keeps_last_event(sample_ratings):
    train, test = temporal_leave_one_out(sample_ratings)
    assert len(test) == sample_ratings["userId"].nunique()
    assert all(test.groupby("userId")["timestamp"].max() >= train.groupby("userId")["timestamp"].max())


def test_matrix_factorization_predicts_finite_value(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    assert np.isfinite(model.predict(1, 1))
