from inf8239_u03.config import MOVIELENS_DIR
from inf8239_u03.data import load_movielens

ratings, movies = load_movielens(MOVIELENS_DIR)
possible = ratings["userId"].nunique() * movies["movieId"].nunique()
print("Ratings:", ratings.shape, "Movies:", movies.shape)
print("Usuarios:", ratings["userId"].nunique(), "Ítems valorados:", ratings["movieId"].nunique())
print("Rango temporal:", ratings["timestamp"].min(), ratings["timestamp"].max())
print("Densidad:", len(ratings) / possible)
print(ratings["rating"].describe())
