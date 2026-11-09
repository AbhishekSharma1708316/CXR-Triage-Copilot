import numpy as np
import pytest

from src.clinical.rules import Intake
from src.copilot.fusion import assess_case
from src.rag.cards import load_cards
from src.rag.context import ALWAYS, select_cards
from src.rag.embedders import TfidfEmbedder
from src.rag.retriever import Retriever

CONFIDENT_NORMAL = (0.97, 0.02, 0.01)
CONFIDENT_OPACITY = (0.02, 0.08, 0.90)


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_cards(), TfidfEmbedder())


def full(**kw):
    base = dict(age=40, confusion=False, resp_rate=16, systolic_bp=120, diastolic_bp=80, spo2=97,
                cannot_speak_full_sentences=False, blue_or_pale_lips_skin=False, coughing_blood=False, fainted_or_collapsed=False)
    base.update(kw)
    return Intake(**base)


def ids(case, retriever, **kw):
    return [c["id"] for c in select_cards(case, retriever, **kw)]


def test_emergency_sign_brings_its_card_and_how_to_get_help(retriever):
    got = ids(assess_case(CONFIDENT_NORMAL, "AP", full(blue_or_pale_lips_skin=True)), retriever)
    assert {"emergency-blue-skin", "emergency-how-to-get-help", "emergency-signs-overview"} <= set(got)


def test_imaging_urgent_without_emergency_signs(retriever):
    got = ids(assess_case(CONFIDENT_OPACITY, "AP", full()), retriever)
    assert "urgent-assessment" in got and "tool-heatmap" in got
    assert "emergency-how-to-get-help" not in got


def test_review_soon_and_routine(retriever):
    assert "review-soon-advice" in ids(assess_case(CONFIDENT_NORMAL, "AP", Intake(age=40)), retriever)
    routine = ids(assess_case(CONFIDENT_NORMAL, "AP", full()), retriever)
    assert "routine-advice" in routine and "tool-heatmap" not in routine


def test_conditional_cards(retriever):
    assert "incomplete-info" in ids(assess_case(CONFIDENT_NORMAL, "AP", Intake(age=40)), retriever)
    assert "children" in ids(assess_case(CONFIDENT_NORMAL, "AP", full(age=9)), retriever)
    assert "view-limit" in ids(assess_case(CONFIDENT_NORMAL, None, full()), retriever)
    assert "crb65-explained" in ids(assess_case(CONFIDENT_NORMAL, "AP", full(age=70)), retriever)


def test_required_cards_do_not_depend_on_retrieval(retriever):
    case = assess_case(CONFIDENT_NORMAL, "AP", full(coughing_blood=True))
    with_search = set(ids(case, retriever))
    without_search = set(ids(case, retriever, min_score=2.0))              # retrieval can find nothing at all
    assert {"emergency-coughing-blood", "emergency-how-to-get-help"} <= without_search
    assert without_search <= with_search


def test_every_card_is_cited_and_explained(retriever):
    for c in select_cards(assess_case(CONFIDENT_OPACITY, "PA", full(age=70)), retriever):
        assert c["source_url"].startswith("https://") and c["why"] and c["text"]


def test_property_always_cards_present_and_ids_unique(retriever):
    rng = np.random.default_rng(1)
    for _ in range(800):
        p = rng.dirichlet([1, 1, 1])
        view = rng.choice(["AP", "PA", None])
        b = lambda: rng.choice([True, False, None], p=[0.2, 0.6, 0.2])
        i = Intake(age=int(rng.integers(0, 100)), confusion=b(), cannot_speak_full_sentences=b(), blue_or_pale_lips_skin=b(),
                   coughing_blood=b(), fainted_or_collapsed=b(),
                   resp_rate=None if rng.random() < 0.3 else float(rng.integers(8, 45)),
                   spo2=None if rng.random() < 0.5 else float(rng.integers(80, 100)))
        case = assess_case(p, view, i)
        got = ids(case, retriever)
        assert len(got) == len(set(got))
        assert set(ALWAYS) <= set(got)
        if case["clinical"]["red_flags"]:
            assert "emergency-signs-overview" in got
