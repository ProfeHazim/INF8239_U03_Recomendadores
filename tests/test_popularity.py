import pandas as pd

from inf8239_u03.recommenders import PopularityRecommender, smoothed_popularity


def test_smoothing_shrinks_low_count_toward_global_mean():
    ratings = pd.DataFrame({
        "userId": [1, 2, 3, 4],
        "movieId": [1, 1, 1, 2],
        "rating": [4.0, 4.0, 4.0, 5.0],
        "timestamp": [1, 2, 3, 4],
    })
    scores = smoothed_popularity(ratings, prior_count=10.0)
    global_mean = float(ratings["rating"].mean())
    low_count_score = scores.loc[2, "smoothed_score"]
    assert abs(low_count_score - global_mean) < abs(5.0 - global_mean)
    assert low_count_score < 5.0


def test_popularity_recommender_excludes_seen(sample_ratings, sample_movies):
    model = PopularityRecommender().fit(sample_ratings, sample_movies)
    top = int(model.ranking_[0])
    result = model.recommend(seen={top}, k=3)
    assert top not in set(result["movieId"])
    assert len(result) == 3
