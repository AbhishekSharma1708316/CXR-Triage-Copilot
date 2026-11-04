import json

import numpy as np
import pytest

from src.clinical.rules import Intake
from src.copilot.fusion import assess_case

CONFIDENT_NORMAL = (0.97, 0.02, 0.01)
CONFIDENT_OPACITY = (0.02, 0.08, 0.90)


def full(**kw):
    base = dict(age=40, confusion=False, resp_rate=16, systolic_bp=120, diastolic_bp=80, spo2=97,
                cannot_speak_full_sentences=False, blue_or_pale_lips_skin=False, coughing_blood=False, fainted_or_collapsed=False)
    base.update(kw)
    return Intake(**base)


def test_clean_xray_can_never_lower_a_red_flag():
    r = assess_case(CONFIDENT_NORMAL, "AP", full(blue_or_pale_lips_skin=True))
    assert r["imaging"]["label"] == "Routine" and r["final"]["label"] == "Urgent" and r["escalated_by"] == "clinical"


def test_imaging_can_escalate_a_reassuring_intake():
    r = assess_case(CONFIDENT_OPACITY, "AP", full())
    assert r["final"]["label"] == "Urgent" and r["escalated_by"] == "imaging"


def test_routine_needs_clean_image_and_complete_intake():
    r = assess_case(CONFIDENT_NORMAL, "AP", full())
    assert r["final"]["label"] == "Routine" and r["escalated_by"] == "agree"


def test_incomplete_intake_blocks_routine():
    r = assess_case(CONFIDENT_NORMAL, "AP", Intake(age=40))
    assert r["final"]["label"] == "Review soon" and r["final"]["floored_for_incomplete_intake"]
    assert any("could not be confirmed" in s for s in r["limitations"])


def test_heatmap_only_for_imaging_urgent_cases():
    assert assess_case(CONFIDENT_OPACITY, "AP", full())["heatmap_allowed"] is True
    assert assess_case(CONFIDENT_NORMAL, "AP", full(blue_or_pale_lips_skin=True))["heatmap_allowed"] is False


def test_unknown_view_is_reported():
    assert any("view" in s for s in assess_case(CONFIDENT_NORMAL, None, full())["limitations"])


def test_output_is_json_serialisable_and_has_reasons():
    r = assess_case((0.3, 0.4, 0.3), "PA", full(age=70, resp_rate=31))
    json.dumps(r)
    assert r["reasons"] and r["limitations"]


def test_property_escalate_only_and_routine_requires_complete_intake():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        p = rng.dirichlet([1, 1, 1])
        view = rng.choice(["AP", "PA", None])
        b = lambda: rng.choice([True, False, None], p=[0.15, 0.6, 0.25])
        i = Intake(age=int(rng.integers(0, 100)), confusion=b(), cannot_speak_full_sentences=b(),
                   blue_or_pale_lips_skin=b(), coughing_blood=b(), fainted_or_collapsed=b(),
                   resp_rate=None if rng.random() < 0.3 else float(rng.integers(8, 45)),
                   systolic_bp=None if rng.random() < 0.3 else float(rng.integers(70, 180)),
                   diastolic_bp=None if rng.random() < 0.3 else float(rng.integers(30, 69)),
                   spo2=None if rng.random() < 0.5 else float(rng.integers(80, 100)))
        if i.systolic_bp is not None and i.diastolic_bp is not None and i.systolic_bp <= i.diastolic_bp:
            continue
        r = assess_case(p, view, i)
        assert r["final"]["level"] >= r["imaging"]["level"]            # rule 1: escalate-only
        assert r["final"]["level"] >= r["clinical"]["level"]
        if r["clinical"]["red_flags"]:
            assert r["final"]["level"] == 2
        if r["final"]["level"] == 0:                                    # rule 2: unknown is never normal
            assert r["clinical"]["intake_complete"]
