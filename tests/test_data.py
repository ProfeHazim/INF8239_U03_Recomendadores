import pytest

from inf8239_u03.data import validate_movielens


def test_valid_contract(sample_ratings, sample_movies):
    validate_movielens(sample_ratings, sample_movies)


def test_rejects_orphan_rating(sample_ratings, sample_movies):
    changed = sample_ratings.copy()
    changed.loc[0, "movieId"] = 999
    with pytest.raises(ValueError, match="sin película"):
        validate_movielens(changed, sample_movies)
