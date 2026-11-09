import numpy as np
import pytest

from src.rag.cards import load_cards
from src.rag.embedders import TfidfEmbedder
from src.rag.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_cards(), TfidfEmbedder())


def test_results_are_sorted_and_limited(retriever):
    res = retriever.search("blue lips and cannot breathe", k=4)
    scores = [s for _, s in res]
    assert len(res) == 4 and scores == sorted(scores, reverse=True)


def test_a_card_title_finds_its_own_card(retriever):
    # Neighbouring cards can share words ("emergency", "help"), so the strict requirement is top-3,
    # and top-1 must still hold for nearly every card.
    top1 = 0
    for c in retriever.cards:
        ranked = [x.id for x, _ in retriever.search(c.title, k=3)]
        assert c.id in ranked, c.id
        top1 += ranked[0] == c.id
    assert top1 / len(retriever.cards) >= 0.9


def test_expected_clear_cases(retriever):
    assert retriever.search("What does the CRB-65 score mean", k=1)[0][0].id == "crb65-explained"
    assert "emergency-blue-skin" in [c.id for c, _ in retriever.search("my lips look blue", k=3)]


def test_exclude_and_min_score(retriever):
    top = retriever.search("blue lips", k=1)[0][0].id
    assert top not in [c.id for c, _ in retriever.search("blue lips", k=5, exclude={top})]
    assert retriever.search("blue lips", k=5, min_score=2.0) == []        # cosine similarity can never reach 2


def test_unrelated_query_scores_low(retriever):
    assert all(s < 0.2 for _, s in retriever.search("football league fixtures tomorrow", k=3))


def test_unknown_id_raises(retriever):
    with pytest.raises(KeyError):
        retriever.get("no-such-card")


class HashEmbedder:
    """A tiny deterministic embedder, to show the retriever works with any embedder that has embed()."""
    def embed(self, texts):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_retriever_accepts_any_embedder():
    r = Retriever(load_cards(), HashEmbedder())
    assert len(r.search("pneumonia", k=2)) == 2
