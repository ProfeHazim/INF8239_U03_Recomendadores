from inf8239_u03.recommenders import ContentRecommender, weighted_popularity


def test_content_excludes_query_and_returns_unique_items(sample_movies):
    result = ContentRecommender().fit(sample_movies).recommend("Alpha", 3)
    assert "Alpha" not in set(result["title"])
    assert result["movieId"].is_unique


def test_popularity_contains_weighted_score(sample_ratings, sample_movies):
    result = weighted_popularity(sample_ratings, sample_movies, quantile=0.0)
    assert "weighted_score" in result.columns
