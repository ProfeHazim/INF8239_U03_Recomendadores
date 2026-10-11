import streamlit as st

from inf8239_u03.config import MOVIELENS_DIR
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import ContentRecommender

st.title("INF-8239 · Recomendador académico")
st.caption("La lista expresa similitud de géneros; no garantiza preferencia individual.")
try:
    ratings, movies = load_movielens(MOVIELENS_DIR)
except FileNotFoundError:
    st.error("Descargue MovieLens con: uv run python scripts/download_data.py")
    st.stop()
model = ContentRecommender().fit(movies)
title = st.selectbox("Película de referencia", movies["title"].sort_values())
if st.button("Recomendar"):
    st.dataframe(model.recommend(title, 10), use_container_width=True)
    st.info("Método: TF-IDF de géneros y similitud coseno. Riesgo: sobre-especialización.")
