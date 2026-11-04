"""Fuse the imaging result with the clinical rules into ONE structured, auditable object ('case facts').

Design rules (all unit-tested):
  1. Escalate-only: the final level is never lower than the imaging level or the clinical level.
  2. Unknown is never normal: 'Routine' cannot be confirmed while confusion, respiratory rate or blood pressure are missing.
  3. The language model later only EXPLAINS these facts; it never decides or prescribes.
"""
from __future__ import annotations

from src.clinical.rules import Intake, clinical_assessment
from src.imaging.policy import LEVELS, imaging_level

ALLOW_ROUTINE_WITH_INCOMPLETE_INTAKE = False

STANDARD_LIMITATIONS = [
    "Decision-support prototype: this is not a diagnosis and does not replace a clinician.",
    "The imaging probability means the scan resembles pneumonia-like opacity in the training data, not that pneumonia is present.",
]


def assess_case(p_raw, view, intake: Intake) -> dict:
    img = imaging_level(p_raw, view)
    clin = clinical_assessment(intake)

    final = max(img["level"], clin["level"])                  # rule 1: escalate-only
    reasons, limitations = [], list(STANDARD_LIMITATIONS)

    for f in clin["red_flags"]:
        reasons.append(f"Red flag: {f['text']} ({f['source']}).")
    crb = clin["crb65"]
    if crb["score"] > 0:
        reasons.append(f"CRB-65 score {crb['score']} from the information provided ({crb['band_from_known_items']}; {crb['source']}).")
    reasons.append(f"Imaging: estimated chance of pneumonia-like opacity {img['p_opacity']:.0%} "
                   f"(Urgent at {img['t_urgent']:.0%} or more) -> {img['label']}.")

    floored = False
    if final == 0 and not clin["intake_complete"] and not ALLOW_ROUTINE_WITH_INCOMPLETE_INTAKE:   # rule 2
        final, floored = 1, True
        limitations.append("Routine could not be confirmed because these were not provided: "
                           + ", ".join(clin["unknown_key_items"]) + ".")
    elif clin["unknown_key_items"]:
        limitations.append("Not provided (treated as unknown, never as normal): " + ", ".join(clin["unknown_key_items"]) + ".")
    if not img["view_known"]:
        limitations.append("The X-ray view (AP or PA) was unknown, so the strictest imaging thresholds were used.")

    if clin["level"] > img["level"]:
        escalated_by = "clinical"
    elif img["level"] > clin["level"]:
        escalated_by = "imaging"
    else:
        escalated_by = "agree"

    return {"final": {"level": final, "label": LEVELS[final], "floored_for_incomplete_intake": floored},
            "escalated_by": escalated_by, "imaging": img, "clinical": clin,
            "reasons": reasons, "limitations": limitations,
            # Grad-CAM audit (notebook 04): heatmaps were informative only for cases the model flags Urgent.
            "heatmap_allowed": img["level"] == 2}
