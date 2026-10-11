import pandas as pd

from inf8239_u03.metrics import ndcg_at_k, precision_at_k, recall_at_k, user_coverage


def test_precision_recall_ndcg_on_known_ranking():
    recommendations = {1: [1, 2, 3], 2: [4, 5, 6]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [1, 4]})
    assert precision_at_k(recommendations, truth, k=3) == 1 / 3
    assert recall_at_k(recommendations, truth, k=3) == 1.0
    assert ndcg_at_k(recommendations, truth, k=3) == 1.0


def test_ndcg_penalizes_low_rank_hit():
    recommendations = {1: [9, 1]}
    truth = pd.DataFrame({"userId": [1], "movieId": [1]})
    value = ndcg_at_k(recommendations, truth, k=2)
    assert 0.0 < value < 1.0


def test_user_coverage_counts_served_users():
    recommendations = {1: [10], 2: [], 3: [11]}
    assert user_coverage(recommendations, [1, 2, 3]) == 2 / 3
