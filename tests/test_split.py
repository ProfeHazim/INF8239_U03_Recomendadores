from inf8239_u03.recommenders import time_based_split


def test_time_based_split_is_leak_free(sample_ratings):
    train, test = time_based_split(sample_ratings, test_size=0.2)
    assert not train.empty and not test.empty
    assert train["timestamp"].max() < test["timestamp"].min()


def test_time_based_split_conserves_all_rows(sample_ratings):
    train, test = time_based_split(sample_ratings, test_size=0.33)
    assert len(train) + len(test) == len(sample_ratings)
