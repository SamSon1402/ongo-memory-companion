"""Entity & fact extraction contracts."""

from __future__ import annotations

from ongomemory.semantic.entities import extract_entities, extract_facts


def test_extracts_names() -> None:
    ents = extract_entities("I had a call with Joonatan and Vaishnavi.")
    assert "Joonatan" in ents
    assert "Vaishnavi" in ents


def test_extracts_times() -> None:
    ents = extract_entities("Let's meet at 4pm tomorrow.")
    assert "4pm" in [e.lower() for e in ents] or "4pm" in ents


def test_does_not_double_count() -> None:
    ents = extract_entities("Joonatan said Joonatan was busy.")
    assert ents.count("Joonatan") == 1


def test_fact_name_extraction() -> None:
    facts = extract_facts("Hey, I'm Sam.")
    keys = {f[0]: f[1] for f in facts}
    assert keys.get("name") == "Sam"


def test_fact_works_on() -> None:
    facts = extract_facts("I work on machine learning at InteractionLabs.")
    keys = {f[0]: f[1] for f in facts}
    assert "works_on" in keys
    assert "machine learning" in keys["works_on"].lower()


def test_fact_lives_in() -> None:
    facts = extract_facts("I live in Paris.")
    keys = {f[0]: f[1] for f in facts}
    assert keys.get("lives_in") == "Paris"


def test_no_facts_in_neutral_text() -> None:
    assert extract_facts("set a timer for five minutes") == []
