import pytest

from src.clinical.rules import EMERGENCY_FLAGS, Intake, clinical_assessment, crb65


def full(**kw):
    """A complete, reassuring intake that individual tests then modify."""
    base = dict(age=40, confusion=False, resp_rate=16, systolic_bp=120, diastolic_bp=80, spo2=97,
                cannot_speak_full_sentences=False, blue_or_pale_lips_skin=False, coughing_blood=False, fainted_or_collapsed=False)
    base.update(kw)
    return Intake(**base)


def test_reassuring_intake_is_routine_and_complete():
    a = clinical_assessment(full())
    assert a["label"] == "Routine" and a["intake_complete"] and a["red_flags"] == [] and a["crb65"]["score"] == 0


@pytest.mark.parametrize("kw,expected", [
    (dict(resp_rate=29), 0), (dict(resp_rate=30), 1),
    (dict(systolic_bp=91), 0), (dict(systolic_bp=90), 1),
    (dict(diastolic_bp=61), 0), (dict(diastolic_bp=60), 1),
    (dict(age=64), 0), (dict(age=65), 1),
    (dict(confusion=True), 1),
])
def test_crb65_item_boundaries(kw, expected):
    assert crb65(full(**kw))["score"] == expected


def test_crb65_levels():
    assert clinical_assessment(full(age=70))["label"] == "Review soon"                       # score 1
    assert clinical_assessment(full(age=70, resp_rate=32))["label"] == "Review soon"         # score 2
    assert clinical_assessment(full(age=70, resp_rate=32, systolic_bp=85))["label"] == "Urgent"   # score 3


def test_spo2_boundary():
    assert clinical_assessment(full(spo2=92))["label"] == "Routine"
    a = clinical_assessment(full(spo2=91))
    assert a["label"] == "Urgent" and a["red_flags"][0]["id"] == "spo2_below_92"


@pytest.mark.parametrize("field", [f for f, _, _ in EMERGENCY_FLAGS])
def test_each_emergency_flag_is_urgent(field):
    a = clinical_assessment(full(**{field: True}))
    assert a["label"] == "Urgent" and any(f["id"] == field for f in a["red_flags"])


def test_every_flag_has_a_source():
    assert all(src for _, _, src in EMERGENCY_FLAGS)
    assert all(f["source"] for f in clinical_assessment(full(blue_or_pale_lips_skin=True, spo2=90))["red_flags"])


def test_unknown_is_not_treated_as_normal():
    a = clinical_assessment(Intake(age=40))
    assert a["crb65"]["score"] == 0 and a["crb65"]["max_possible"] == 3
    assert not a["intake_complete"]
    assert set(a["unknown_key_items"]) == {"confusion", "respiratory_rate_ge_30", "low_blood_pressure"}


def test_blood_pressure_needs_both_values_to_count_as_known():
    assert "low_blood_pressure" in clinical_assessment(full(diastolic_bp=None))["unknown_key_items"]
    # but one abnormal value is enough to score the item
    assert crb65(full(diastolic_bp=None, systolic_bp=85))["items"]["low_blood_pressure"] is True


@pytest.mark.parametrize("kw", [dict(age=-1), dict(age=130), dict(spo2=120), dict(resp_rate=0),
                                dict(systolic_bp=80, diastolic_bp=90)])
def test_invalid_numbers_are_rejected(kw):
    with pytest.raises(ValueError):
        full(**kw)


def test_non_boolean_answers_are_rejected():
    with pytest.raises(TypeError):
        full(confusion="yes")
