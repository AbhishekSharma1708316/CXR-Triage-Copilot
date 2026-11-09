import json
import re

import pytest

from src.clinical.rules import EMERGENCY_FLAGS
from src.rag.cards import Card, load_cards
from src.rag.context import FLAG_CARDS

# The explainer must never recommend drugs or doses; the knowledge it works from must not contain them either.
MEDICATION_WORDS = re.compile(
    r"paracetamol|ibuprofen|amoxicillin|antibiotic|aspirin|steroid|prednisolone|azithromycin|doxycycline|clarithromycin|"
    r"\bmg\b|\bdose\b|\bdoses\b|tablet|capsule|syrup|inhaler", re.IGNORECASE)


@pytest.fixture(scope="module")
def cards():
    return load_cards()


def test_enough_cards_with_unique_ids(cards):
    assert len(cards) >= 25
    assert len({c.id for c in cards}) == len(cards)


def test_every_card_is_cited_and_short(cards):
    for c in cards:
        assert c.source_name and c.source_url.startswith("https://"), c.id
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.last_checked), c.id
        assert 20 <= len(c.text) <= 600, c.id
        assert c.title.strip() and c.tags, c.id


def test_cards_contain_no_medication_names_or_doses(cards):
    for c in cards:
        assert not MEDICATION_WORDS.search(c.title + " " + c.text), c.id


def test_every_red_flag_has_an_explaining_card(cards):
    ids = {c.id for c in cards}
    possible = {f for f, _, _ in EMERGENCY_FLAGS} | {"spo2_below_92"}
    assert possible <= set(FLAG_CARDS), "a red flag exists in rules.py without a card mapping"
    assert set(FLAG_CARDS.values()) <= ids


def test_loader_rejects_duplicates_and_bad_sources(tmp_path):
    base = {"id": "a", "topic": "tool", "title": "t", "text": "some text here that is long enough",
            "source_name": "s", "source_url": "https://example.org", "last_checked": "2026-11-04", "tags": ["x"]}

    def write(items):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(items))
        return p

    with pytest.raises(ValueError):
        load_cards(write([base, base]))                                   # duplicate id
    with pytest.raises(ValueError):
        load_cards(write([{**base, "source_url": "http://insecure"}]))    # not https
    with pytest.raises(ValueError):
        load_cards(write([{**base, "topic": "gossip"}]))                  # unknown topic
    assert isinstance(load_cards(write([base]))[0], Card)
