from inf8239_u03.recommenders import HybridRecommender


def test_hybrid_excludes_seen_items(sample_ratings, sample_movies):
    model = HybridRecommender(alpha=0.5, factors=3, epochs=2, seed=42).fit(sample_ratings, sample_movies)
    seen = {1, 2}
    result = model.recommend(1, seen, 3)
    assert not result.empty
    assert seen.isdisjoint(set(result["movieId"]))
    assert "hybrid_score" in result.columns


def test_hybrid_cold_start_user_falls_back(sample_ratings, sample_movies):
    model = HybridRecommender(alpha=0.5, factors=3, epochs=2, seed=42).fit(sample_ratings, sample_movies)
    result = model.recommend(999, set(), 3)
    assert not result.empty
