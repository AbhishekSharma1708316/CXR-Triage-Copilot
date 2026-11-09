"""Knowledge cards: short, paraphrased, cited statements the Copilot is allowed to rely on.

Why cards and not raw guideline pages: (1) copying whole web pages would raise copyright questions;
(2) a short, checked statement is easy to cite, to test and to verify against an LLM's answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"
TOPICS = {"emergency", "next_steps", "assessment", "background", "tool"}


@dataclass(frozen=True)
class Card:
    id: str
    topic: str
    title: str
    text: str
    source_name: str
    source_url: str
    last_checked: str
    tags: tuple

    @property
    def search_text(self) -> str:
        """What the retriever indexes: title, text and tags (tags carry the words patients actually use)."""
        return f"{self.title}. {self.text} {' '.join(self.tags)}"


def load_cards(path: Path | str | None = None) -> list[Card]:
    path = Path(path) if path else DATA_DIR / "cards.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cards = [Card(id=c["id"], topic=c["topic"], title=c["title"], text=c["text"], source_name=c["source_name"],
                  source_url=c["source_url"], last_checked=c["last_checked"], tags=tuple(c["tags"])) for c in raw]
    ids = [c.id for c in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate card ids")
    for c in cards:
        if c.topic not in TOPICS:
            raise ValueError(f"{c.id}: unknown topic {c.topic!r}")
        if not c.text.strip() or not c.source_url.startswith("https://"):
            raise ValueError(f"{c.id}: needs text and an https source_url")
    return cards
