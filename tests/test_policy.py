import numpy as np
import pytest

from src.imaging.policy import LEVELS, calibrate, imaging_level

CONFIDENT_NORMAL = (0.97, 0.02, 0.01)
CONFIDENT_OPACITY = (0.02, 0.08, 0.90)
UNCERTAIN = (0.4, 0.4, 0.2)


def test_calibrate_returns_probabilities():
    p = calibrate(UNCERTAIN)
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


def test_calibration_lowers_opacity_probability():
    # class-weighted training overstates P(opacity); the prior correction must pull it down
    assert calibrate((0.1, 0.2, 0.7))[2] < 0.7


@pytest.mark.parametrize("bad", [(0.5, 0.5), (0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), "abc"])
def test_calibrate_rejects_bad_input(bad):
    with pytest.raises((ValueError, TypeError)):
        calibrate(bad)


def test_confident_cases():
    for view in ("AP", "PA", None):
        assert imaging_level(CONFIDENT_OPACITY, view)["label"] == "Urgent"
        assert imaging_level(CONFIDENT_NORMAL, view)["label"] == "Routine"


def test_view_specific_thresholds():
    # the same uncertain image is only 'Review soon' on an AP film but 'Urgent' on a PA film (lower PA threshold)
    assert imaging_level(UNCERTAIN, "AP")["label"] == "Review soon"
    assert imaging_level(UNCERTAIN, "PA")["label"] == "Urgent"


def test_unknown_view_uses_strictest_thresholds():
    assert imaging_level(UNCERTAIN, None) == {**imaging_level(UNCERTAIN, "PA"), "view": None, "view_known": False}
    assert imaging_level(UNCERTAIN, "lateral")["view_known"] is False
    assert imaging_level(UNCERTAIN, "pa")["view"] == "PA"          # case-insensitive


def test_level_never_decreases_as_opacity_probability_rises():
    for view in ("AP", "PA", None):
        levels = [imaging_level(((1 - c) * 0.6, (1 - c) * 0.4, c), view)["level"] for c in np.linspace(0, 0.98, 99)]
        assert levels == sorted(levels)


def test_labels_match_levels():
    r = imaging_level(CONFIDENT_OPACITY, "AP")
    assert LEVELS[r["level"]] == r["label"]
