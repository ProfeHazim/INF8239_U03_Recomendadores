from inf8239_u03.recommenders import ContentBasedRecommender


def test_content_based_recommends_unseen_items(sample_ratings, sample_movies):
    model = ContentBasedRecommender().fit(sample_ratings, sample_movies)
    seen = {1, 2}
    result = model.recommend(user_id=1, seen=seen, k=3)
    assert not result.empty
    assert seen.isdisjoint(set(result["movieId"]))
    assert result["movieId"].is_unique
    assert "content_score" in result.columns


def test_content_based_unknown_user_returns_empty(sample_ratings, sample_movies):
    model = ContentBasedRecommender().fit(sample_ratings, sample_movies)
    result = model.recommend(user_id=999, k=3)
    assert result.empty


def test_content_based_similar_to_excludes_query(sample_ratings, sample_movies):
    model = ContentBasedRecommender().fit(sample_ratings, sample_movies)
    result = model.similar_to("Alpha", 3)
    assert "Alpha" not in set(result["title"])
