"""Transparent clinical rules: CRB-65 and red flags. Plain Python, no ML, fully unit-tested.

Every rule carries its source. Unknown answers are NEVER treated as normal: they are listed as unknown,
and the fusion step refuses to confirm 'Routine' while key items are missing.
This is a research prototype, not a medical device; thresholds must be reviewed by a clinician before any real use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SRC_BTS = "British Thoracic Society guideline for community-acquired pneumonia in adults (CRB65 and pulse oximetry recommendations)"
SRC_NHS = "NHS pneumonia guidance: when to call 999"
SRC_ALUK = "Asthma + Lung UK (formerly the British Lung Foundation): pneumonia, when to call 999"

LEVELS = ("Routine", "Review soon", "Urgent")

# (intake field, plain-language description, source). Any of these -> Urgent.
EMERGENCY_FLAGS = [
    ("cannot_speak_full_sentences", "Struggling to breathe or unable to speak in full sentences", SRC_NHS),
    ("blue_or_pale_lips_skin", "Blue, pale or blotchy lips, tongue or skin", SRC_NHS),
    ("confusion", "Sudden or new confusion, or becoming very sleepy", SRC_NHS),
    ("coughing_blood", "Coughing up blood", SRC_NHS),
    ("fainted_or_collapsed", "Fainting or collapse", SRC_ALUK),
]

_BOOL_FIELDS = ("confusion", "cannot_speak_full_sentences", "blue_or_pale_lips_skin", "coughing_blood", "fainted_or_collapsed")


@dataclass(frozen=True)
class Intake:
    """What the user tells us. Everything except age is optional; None means 'not provided'."""
    age: int
    confusion: Optional[bool] = None                       # new confusion or disorientation
    resp_rate: Optional[float] = None                      # breaths per minute
    systolic_bp: Optional[float] = None                    # mmHg
    diastolic_bp: Optional[float] = None                   # mmHg
    spo2: Optional[float] = None                           # % on room air (home pulse oximeter)
    cannot_speak_full_sentences: Optional[bool] = None
    blue_or_pale_lips_skin: Optional[bool] = None
    coughing_blood: Optional[bool] = None
    fainted_or_collapsed: Optional[bool] = None

    def __post_init__(self):
        if isinstance(self.age, bool) or not isinstance(self.age, (int, float)) or not 0 <= self.age <= 120:
            raise ValueError("age must be a number between 0 and 120")
        for name in _BOOL_FIELDS:
            v = getattr(self, name)
            if v is not None and not isinstance(v, bool):
                raise TypeError(f"{name} must be True, False or None")
        for name, lo, hi in (("resp_rate", 4, 80), ("systolic_bp", 40, 260), ("diastolic_bp", 20, 160), ("spo2", 50, 100)):
            v = getattr(self, name)
            if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} (or not provided)")
        if self.systolic_bp is not None and self.diastolic_bp is not None and self.systolic_bp <= self.diastolic_bp:
            raise ValueError("systolic blood pressure must be higher than diastolic")


def crb65(i: Intake) -> dict:
    """CRB-65: one point each for confusion, respiratory rate >= 30, low blood pressure, age >= 65 (BTS guideline).

    Items the user did not provide are 'unknown': they add no points, are listed, and widen max_possible.
    """
    bp_positive = (i.systolic_bp is not None and i.systolic_bp <= 90) or (i.diastolic_bp is not None and i.diastolic_bp <= 60)
    bp_known = bp_positive or (i.systolic_bp is not None and i.diastolic_bp is not None)
    items = {
        "confusion": i.confusion,
        "respiratory_rate_ge_30": None if i.resp_rate is None else i.resp_rate >= 30,
        "low_blood_pressure": True if bp_positive else (False if bp_known else None),
        "age_ge_65": i.age >= 65,
    }
    score = sum(1 for v in items.values() if v is True)
    unknown = [k for k, v in items.items() if v is None]
    band = "low risk" if score == 0 else ("increased risk" if score <= 2 else "high risk")
    return {"score": score, "max_possible": score + len(unknown), "items": items, "unknown": unknown,
            "band_from_known_items": band, "source": SRC_BTS}


def clinical_assessment(i: Intake) -> dict:
    """Clinical triage level from the intake alone: Urgent / Review soon / Routine, plus every reason."""
    crb = crb65(i)
    flags = [{"id": f, "text": text, "source": src, "tier": "emergency"}
             for f, text, src in EMERGENCY_FLAGS if getattr(i, f) is True]
    if i.spo2 is not None and i.spo2 < 92:
        flags.append({"id": "spo2_below_92", "text": f"Oxygen saturation {i.spo2:g}% (below 92%)", "source": SRC_BTS, "tier": "urgent"})

    if flags or crb["score"] >= 3:            # any red flag, or CRB-65 of 3 or more -> Urgent
        level = 2
    elif crb["score"] >= 1:                   # CRB-65 of 1-2 -> hospital assessment should be considered
        level = 1
    else:
        level = 0

    key_unknown = [k for k in ("confusion", "respiratory_rate_ge_30", "low_blood_pressure") if k in crb["unknown"]]
    return {"level": level, "label": LEVELS[level], "crb65": crb, "red_flags": flags,
            "intake_complete": not key_unknown, "unknown_key_items": key_unknown, "spo2_provided": i.spo2 is not None}
