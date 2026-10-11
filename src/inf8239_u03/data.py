from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


def download_zip(url: str, destination: Path) -> tuple[Path, str]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Ruta insegura en ZIP: {member.filename}")
        archive.extractall(destination)
    return destination, digest


def _find_movielens_root(destination: Path) -> Path | None:
    """Ubica la carpeta extraída del ZIP que contiene ``ratings.csv``."""
    candidates = sorted(Path(destination).glob("*/ratings.csv"))
    return candidates[0].parent if candidates else None


def ensure_movielens(url: str, destination: Path, force: bool = False) -> tuple[Path, str | None]:
    """Garantiza que MovieLens esté disponible localmente, descargándolo si falta.

    Es idempotente: si ``ratings.csv`` y ``movies.csv`` ya existen y ``force`` es
    False, no vuelve a descargar. Registra el SHA-256 del ZIP en
    ``destination/SHA256SUMS.txt`` para trazabilidad (el archivo queda ignorado
    por git porque ``data/raw/*`` está en ``.gitignore``).

    Returns
    -------
    (root, digest)
        Carpeta con los CSV y SHA-256 del ZIP (``None`` si se reutilizó la caché
        y no hay manifiesto previo).
    """
    destination = Path(destination)
    manifest = destination / "SHA256SUMS.txt"
    if not force:
        cached = _find_movielens_root(destination)
        if cached is not None and (cached / "movies.csv").exists():
            digest = None
            if manifest.exists():
                digest = manifest.read_text(encoding="utf-8").split()[0]
            return cached, digest
    _, digest = download_zip(url, destination)
    manifest.write_text(f"{digest}  {url}\n", encoding="utf-8")
    root = _find_movielens_root(destination)
    if root is None:
        raise FileNotFoundError(f"El ZIP descargado no contiene ratings.csv: {url}")
    return root, digest


def load_movielens(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(directory / "ratings.csv")
    movies = pd.read_csv(directory / "movies.csv")
    validate_movielens(ratings, movies)
    return ratings, movies


def validate_movielens(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}
    if not required_ratings <= set(ratings.columns):
        raise ValueError("ratings.csv no tiene las columnas requeridas")
    if not required_movies <= set(movies.columns):
        raise ValueError("movies.csv no tiene las columnas requeridas")
    if not ratings["movieId"].isin(movies["movieId"]).all():
        raise ValueError("Existen ratings sin película correspondiente")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Existen ratings fuera del rango 0.5–5.0")
