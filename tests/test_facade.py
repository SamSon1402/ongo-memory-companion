"""End-to-end tests through the Memory facade.

These prove the library API actually works — the thing the rest of
Ongo will import.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ongomemory import Memory
from ongomemory.core import Topic, UserId


def test_episode_then_recall(memory: Memory) -> None:
    memory.episode(text="I had a call with Joonatan at 4pm", topic=Topic.CALENDAR)
    memory.episode(text="went to the gym", topic=Topic.HEALTH)
    hits = memory.recall("when was the call with Joonatan?", k=3)
    assert hits
    assert "joonatan" in hits[0].episode.text.lower()


def test_facts_extracted_from_episode(memory: Memory) -> None:
    memory.episode(text="Hey, I'm Sam.", topic=Topic.PROFILE)
    memory.episode(text="I work on machine learning.", topic=Topic.WORK)
    facts = memory.facts()
    assert facts.get("name") == "Sam"
    assert "works_on" in facts


def test_direct_fact_writes(memory: Memory) -> None:
    memory.fact(key="lives_in", value="Paris", confidence=1.0)
    assert memory.facts()["lives_in"] == "Paris"


def test_per_user_isolation_via_facade(store, embedder) -> None:  # noqa: ANN001
    alice = Memory(user_id=UserId("alice"), store=store, embedder=embedder)
    bob = Memory(user_id=UserId("bob"), store=store, embedder=embedder)
    alice.episode(text="alice's secret", topic=Topic.PROFILE)
    bob.episode(text="bob's secret", topic=Topic.PROFILE)
    assert all(e.text == "alice's secret" for e in alice.all_episodes())
    assert all(e.text == "bob's secret" for e in bob.all_episodes())


def test_habit_emerges_from_pattern(memory: Memory) -> None:
    """Three coffee episodes at ~8:50 should surface a habit."""
    base = datetime.now(UTC) - timedelta(days=2)
    for d in range(3):
        when = (base + timedelta(days=d)).replace(hour=8, minute=50 + d - 1)
        memory.episode(text="coffee", topic=Topic.HEALTH, when=when)

    habits = memory.habits(refresh=True)
    assert len(habits) >= 1
    assert habits[0].observation_count >= 3
    assert habits[0].time_window is not None


def test_identity_stored(memory: Memory) -> None:
    memory.identify(face_id="f7a2", voice_id="v3c1", display_name="Sam")
    ident = memory.identity()
    assert ident is not None
    assert ident.face_id == "f7a2"
    assert ident.display_name == "Sam"


def test_recall_on_empty_returns_empty(memory: Memory) -> None:
    assert memory.recall("anything") == []


def test_recall_string_topic_works(memory: Memory) -> None:
    """Topic can be passed as a string for convenience."""
    memory.episode(text="some work", topic="work")  # type: ignore[arg-type]
    assert len(memory.all_episodes(topics=(Topic.WORK,))) == 1
