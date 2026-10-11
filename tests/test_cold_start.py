import pandas as pd

from inf8239_u03.evaluation import cold_start_split, evaluate_cold_start
from inf8239_u03.recommenders import PopularityRecommender


def test_cold_start_split_isolates_new_users(sample_ratings):
    new_user = pd.DataFrame({"userId": [99], "movieId": [1], "rating": [4.0], "timestamp": [10]})
    test = pd.concat([sample_ratings, new_user], ignore_index=True)
    cold_users, cold_truth = cold_start_split(sample_ratings, test)
    assert cold_users == [99]
    assert set(cold_truth["userId"]) == {99}


def test_popularity_serves_every_cold_start_user(sample_ratings, sample_movies):
    new_user = pd.DataFrame({"userId": [99], "movieId": [3], "rating": [4.0], "timestamp": [10]})
    test = pd.concat([sample_ratings, new_user], ignore_index=True)
    cold_users, cold_truth = cold_start_split(sample_ratings, test)
    model = PopularityRecommender().fit(sample_ratings, sample_movies)
    metrics = evaluate_cold_start(
        model, cold_users, cold_truth, catalog_size=sample_movies["movieId"].nunique(), k=3
    )
    assert metrics["user_coverage"] == 1.0
