"""Two interchangeable embedders with the same interface: fit(texts) (optional) and embed(texts) -> L2-normalised matrix.

TfidfEmbedder is a lexical baseline: fast, offline, deterministic (used in unit tests and as an ablation).
MiniLMEmbedder is the semantic one (same model family as DocuMind); it needs `pip install sentence-transformers`.
"""
from __future__ import annotations

import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, texts):
        self._vec.fit(texts)
        return self

    def embed(self, texts) -> np.ndarray:
        return np.asarray(self._vec.transform(list(texts)).todense(), dtype=float)   # rows are already L2-normalised


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer        # imported lazily so tests do not need it
        self._model = SentenceTransformer(model_name)

    def embed(self, texts) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True), dtype=float)
