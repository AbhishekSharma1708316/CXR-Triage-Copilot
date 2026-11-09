"""A small, exact cosine-similarity retriever.

~30 cards do not need a vector database: a normalised matrix and one dot product are exact, fast, easy to test
and have no server to run. If the corpus grows to thousands of passages, swap this class for ChromaDB behind the same search().
"""
from __future__ import annotations

import numpy as np

from src.rag.cards import Card


class Retriever:
    def __init__(self, cards: list[Card], embedder):
        self.cards = list(cards)
        self.embedder = embedder
        texts = [c.search_text for c in self.cards]
        if hasattr(embedder, "fit"):
            embedder.fit(texts)
        self.matrix = embedder.embed(texts)                      # (n_cards, dim), L2-normalised rows
        self._index = {c.id: i for i, c in enumerate(self.cards)}

    def get(self, card_id: str) -> Card:
        return self.cards[self._index[card_id]]                  # KeyError for an unknown id, on purpose

    def search(self, query: str, k: int = 3, min_score: float = 0.0, exclude=()) -> list[tuple[Card, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q                                  # cosine similarity (both sides normalised)
        order = np.argsort(-scores, kind="stable")
        out = []
        for i in order:
            c = self.cards[i]
            if c.id in exclude or scores[i] < min_score:
                continue
            out.append((c, float(scores[i])))
            if len(out) == k:
                break
        return out
