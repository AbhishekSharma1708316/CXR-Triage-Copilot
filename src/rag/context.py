"""Choose the cards the LLM may use for ONE case.

Two layers, on purpose:
  1. REQUIRED cards are picked by plain rules from the case facts. Safety-critical advice (emergency signs, what to do next)
     never depends on a similarity search succeeding.
  2. SUPPORTING cards are retrieved by similarity from the case's own reasons, to add relevant explanation.
The LLM may cite only the ids returned here; the verifier (next step) enforces that.
"""
from __future__ import annotations

from src.rag.retriever import Retriever

# case-facts red-flag id -> the card that explains it
FLAG_CARDS = {
    "cannot_speak_full_sentences": "emergency-breathing",
    "blue_or_pale_lips_skin": "emergency-blue-skin",
    "confusion": "emergency-confusion",
    "coughing_blood": "emergency-coughing-blood",
    "fainted_or_collapsed": "emergency-collapse",
    "spo2_below_92": "low-oxygen",
}
ALWAYS = ["tool-limits-not-diagnosis", "tool-score-meaning", "levels-meaning"]


def select_cards(case: dict, retriever: Retriever, k_supporting: int = 2, min_score: float = 0.05) -> list[dict]:
    chosen: dict[str, str] = {}                                   # card id -> why it was included (insertion-ordered)

    def add(card_id, why):
        chosen.setdefault(card_id, why)

    final = case["final"]["label"]
    flags = case["clinical"]["red_flags"]

    # 1. required cards, by rule
    emergency_flags = [f for f in flags if f["tier"] == "emergency"]
    for f in flags:
        add(FLAG_CARDS[f["id"]], f"required: red flag {f['id']}")
    if emergency_flags:
        add("emergency-how-to-get-help", "required: an emergency sign is present")
    if final == "Urgent" and not emergency_flags:
        add("urgent-assessment", "required: final level Urgent without emergency signs")
    if final == "Review soon":
        add("review-soon-advice", "required: final level Review soon")
    if final == "Routine":
        add("routine-advice", "required: final level Routine")
    add("emergency-signs-overview", "required: always list the warning signs")
    for cid in ALWAYS:
        add(cid, "required: always included")
    if case["heatmap_allowed"]:
        add("tool-heatmap", "required: a heatmap is shown for this case")
    if case["final"]["floored_for_incomplete_intake"] or case["clinical"]["unknown_key_items"]:
        add("incomplete-info", "required: some intake items were not provided")
    if case["clinical"]["crb65"]["score"] > 0:
        add("crb65-explained", "required: CRB-65 points contributed")
    limitation_text = " ".join(case["limitations"])
    if "child" in limitation_text:
        add("children", "required: patient is a child")
    if not case["imaging"]["view_known"]:
        add("view-limit", "required: X-ray view unknown")

    # 2. supporting cards, by similarity to this case's own reasons
    query = " ".join(case["reasons"]) + f" {final} priority"
    for card, score in retriever.search(query, k=k_supporting, min_score=min_score, exclude=set(chosen)):
        add(card.id, f"retrieved (similarity {score:.2f})")

    return [{"id": cid, "title": retriever.get(cid).title, "text": retriever.get(cid).text,
             "source_name": retriever.get(cid).source_name, "source_url": retriever.get(cid).source_url, "why": why}
            for cid, why in chosen.items()]
