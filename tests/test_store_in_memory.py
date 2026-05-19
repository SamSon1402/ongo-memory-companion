"""InMemoryStore contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from ongomemory.core import Episode, Fact, Identity, Topic, UserId
from ongomemory.store import InMemoryStore


def _ep(text: str, user: str = "u1", topic: Topic = Topic.UNKNOWN, vec=None) -> Episode:
    return Episode(
        episode_id=f"id-{hash(text) & 0xFFFFFF:x}",
        user_id=UserId(user),
        topic=topic,
        text=text,
        vec=vec,
    )


def test_dedup_identical_text_same_user() -> None:
    s = InMemoryStore()
    s.write_episode(_ep("hello"))
    s.write_episode(_ep("hello"))   # dedup'd
    s.write_episode(_ep("HELLO"))   # normalized → also dedup'd
    eps = s.episodes_for(UserId("u1"))
    assert len(eps) == 1


def test_per_user_isolation() -> None:
    s = InMemoryStore()
    s.write_episode(_ep("hi", user="alice"))
    s.write_episode(_ep("hi", user="bob"))
    assert len(s.episodes_for(UserId("alice"))) == 1
    assert len(s.episodes_for(UserId("bob"))) == 1
    # Each user sees only their own:
    assert s.episodes_for(UserId("alice"))[0].user_id == "alice"


def test_topic_filter() -> None:
    s = InMemoryStore()
    s.write_episode(_ep("a", topic=Topic.WORK))
    s.write_episode(_ep("b", topic=Topic.HEALTH))
    s.write_episode(_ep("c", topic=Topic.WORK))
    work = s.episodes_for(UserId("u1"), topics=[Topic.WORK])
    assert len(work) == 2


def test_vector_search_returns_top_k() -> None:
    s = InMemoryStore()
    rng = np.random.default_rng(0)
    for i in range(10):
        v = rng.standard_normal(8).astype(np.float32)
        v /= np.linalg.norm(v)
        s.write_episode(_ep(f"ep {i}", vec=v))

    query = rng.standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)
    hits = s.vector_search(UserId("u1"), query, k=3)
    assert len(hits) == 3
    # Sims should be in descending order.
    sims = [h[1] for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_fact_last_write_wins() -> None:
    s = InMemoryStore()
    older = Fact(
        user_id=UserId("u1"), key="name", value="Sameer",
        confidence=0.85, created_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = Fact(
        user_id=UserId("u1"), key="name", value="Sam",
        confidence=0.95, created_at=datetime.now(UTC),
    )
    s.write_fact(older)
    s.write_fact(newer)
    assert s.latest_facts(UserId("u1"))["name"].value == "Sam"


def test_low_confidence_does_not_clobber_high() -> None:
    s = InMemoryStore()
    s.write_fact(Fact(user_id=UserId("u1"), key="name", value="Sam", confidence=0.95))
    # A shaky 0.3-conf fact arrives; should not overwrite.
    s.write_fact(Fact(user_id=UserId("u1"), key="name", value="Sham", confidence=0.30))
    assert s.latest_facts(UserId("u1"))["name"].value == "Sam"


def test_identity_roundtrip() -> None:
    s = InMemoryStore()
    s.write_identity(Identity(user_id=UserId("u1"), face_id="f7a2", display_name="Sam"))
    got = s.identity_for(UserId("u1"))
    assert got is not None and got.face_id == "f7a2"
    assert s.identity_for(UserId("does-not-exist")) is None
