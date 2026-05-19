"""SqliteStore contracts.

Same behaviors as InMemoryStore — we *want* this duplication because
the storage backend is exactly where bugs hide.
"""

from __future__ import annotations

import numpy as np

from ongomemory.core import Episode, Fact, Habit, Identity, Topic, UserId
from ongomemory.store import SqliteStore


def _ep(text: str, vec=None) -> Episode:
    return Episode(
        episode_id=f"id-{hash(text) & 0xFFFFFF:x}",
        user_id=UserId("u1"),
        topic=Topic.WORK,
        text=text,
        vec=vec,
    )


def test_episode_roundtrip_in_memory() -> None:
    s = SqliteStore(db_path=":memory:")
    vec = (np.ones(8) / np.sqrt(8)).astype(np.float32)
    s.write_episode(_ep("hello", vec=vec))
    eps = s.episodes_for(UserId("u1"))
    assert len(eps) == 1
    assert eps[0].text == "hello"
    assert eps[0].vec is not None
    assert eps[0].vec.dtype == np.float32
    np.testing.assert_allclose(eps[0].vec, vec)


def test_duplicate_episode_id_is_idempotent() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_episode(_ep("dup"))
    s.write_episode(_ep("dup"))  # same id (content-addressed)
    assert len(s.episodes_for(UserId("u1"))) == 1


def test_topic_filter() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_episode(_ep("work thing"))
    s.write_episode(
        Episode(
            episode_id="health-ep",
            user_id=UserId("u1"),
            topic=Topic.HEALTH,
            text="gym",
        )
    )
    work = s.episodes_for(UserId("u1"), topics=[Topic.WORK])
    health = s.episodes_for(UserId("u1"), topics=[Topic.HEALTH])
    assert len(work) == 1 and len(health) == 1


def test_vector_search_skips_episodes_without_vec() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_episode(_ep("no vec here"))  # no vec
    vec = (np.ones(8) / np.sqrt(8)).astype(np.float32)
    s.write_episode(_ep("has vec", vec=vec))
    hits = s.vector_search(UserId("u1"), vec, k=5)
    assert len(hits) == 1
    assert hits[0][0].text == "has vec"


def test_facts_last_write_wins() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_fact(Fact(user_id=UserId("u1"), key="name", value="Sameer"))
    s.write_fact(Fact(user_id=UserId("u1"), key="name", value="Sam"))
    facts = s.latest_facts(UserId("u1"))
    assert facts["name"].value == "Sam"


def test_habit_replace_on_same_summary() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_habit(
        Habit(
            user_id=UserId("u1"),
            summary="morning coffee",
            topic=Topic.HEALTH,
            confidence=0.6,
            observation_count=3,
            supporting_episode_ids=("a", "b", "c"),
        )
    )
    s.write_habit(
        Habit(
            user_id=UserId("u1"),
            summary="morning coffee",
            topic=Topic.HEALTH,
            confidence=0.85,
            observation_count=4,
            supporting_episode_ids=("a", "b", "c", "d"),
        )
    )
    habits = s.habits_for(UserId("u1"))
    assert len(habits) == 1
    assert habits[0].observation_count == 4


def test_identity_roundtrip() -> None:
    s = SqliteStore(db_path=":memory:")
    s.write_identity(Identity(user_id=UserId("u1"), face_id="f7a2", display_name="Sam"))
    got = s.identity_for(UserId("u1"))
    assert got is not None
    assert got.face_id == "f7a2"
    assert got.display_name == "Sam"
