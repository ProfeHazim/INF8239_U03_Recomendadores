from inf8239_u03.config import MOVIELENS_URL, ROOT
from inf8239_u03.data import ensure_movielens

destination = ROOT / "data/raw"
root, digest = ensure_movielens(MOVIELENS_URL, destination)
print("Directorio de datos:", root)
print("SHA-256:", digest or "en caché (sin manifiesto)")
