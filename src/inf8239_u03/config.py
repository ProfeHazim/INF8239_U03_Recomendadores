from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVIELENS_URL = os.getenv("MOVIELENS_URL", "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
MOVIELENS_DIR = ROOT / os.getenv("MOVIELENS_DIR", "data/raw/ml-latest-small")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
