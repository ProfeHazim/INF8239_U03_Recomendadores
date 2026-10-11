import pandas as pd

from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k


def test_ranking_metrics():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    assert hit_rate_at_k(recs, truth, 2) == 0.5
    assert catalog_coverage(recs, 5) == 0.8
