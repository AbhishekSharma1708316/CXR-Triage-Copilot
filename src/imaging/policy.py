"""Imaging triage policy, frozen in notebooks 03b and 05 (pure numpy, no torch needed).

Steps: raw softmax probabilities -> prior correction -> temperature scaling -> thresholds.
The constants below were copied from triage_config_v2.json and frozen BEFORE the test set was evaluated.
"""
from __future__ import annotations

import numpy as np

PRIOR = (0.3317, 0.443, 0.2253)            # training class frequencies: Normal, Not Normal, Lung Opacity
TEMPERATURE = 1.25
THRESHOLDS = {"AP": (0.39, 0.58), "PA": (0.12, 0.85)}   # (t_urgent, t_routine) when the X-ray view is known
THRESHOLDS_VIEW_UNKNOWN = (0.12, 0.85)                   # the strictest of the two
LEVELS = ("Routine", "Review soon", "Urgent")            # level index 0, 1, 2


def calibrate(p_raw) -> np.ndarray:
    """Return honest probabilities: correct for the class prior first, then apply temperature scaling."""
    p = np.asarray(p_raw, dtype=float)
    if p.shape != (3,):
        raise ValueError("expected 3 class probabilities (Normal, Not Normal, Lung Opacity)")
    if (p < 0).any() or abs(p.sum() - 1.0) > 1e-3:
        raise ValueError("probabilities must be non-negative and sum to 1")
    q = p * np.asarray(PRIOR)
    q = q / q.sum()
    z = np.log(np.clip(q, 1e-8, 1.0)) / TEMPERATURE
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def imaging_level(p_raw, view=None) -> dict:
    """Triage level from the image model alone. `view` is 'AP', 'PA' or None (unknown -> strictest thresholds)."""
    view = view.upper() if isinstance(view, str) else None
    if view not in THRESHOLDS:
        view = None
    t_urgent, t_routine = THRESHOLDS[view] if view else THRESHOLDS_VIEW_UNKNOWN
    p = calibrate(p_raw)
    if p[2] >= t_urgent:          # likely opacity -> Urgent (overrides everything else)
        level = 2
    elif p[0] >= t_routine:       # confidently normal -> Routine
        level = 0
    else:                         # everything uncertain -> Review soon
        level = 1
    return {"level": level, "label": LEVELS[level], "view": view, "view_known": view is not None,
            "p_normal": round(float(p[0]), 4), "p_not_normal": round(float(p[1]), 4), "p_opacity": round(float(p[2]), 4),
            "t_urgent": t_urgent, "t_routine": t_routine}
