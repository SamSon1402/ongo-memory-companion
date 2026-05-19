"""Ranker contracts.

If the ranker breaks, every "what was I doing yesterday?" turn breaks.
These tests pin down the multi-signal behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from ongomemory.core import Episode, RecallQuery, Topic, UserId
from ongomemory.recall import BlendedRanker, RankerWeights


def _ep(text: str, days_ago: float = 0.0, entities: tuple[str, ...] = ()) -> Episode:
    vec = np.zeros(8, dtype=np.float32)
    return Episode(
        episode_id=f"id-{abs(hash(text)) & 0xFFFFFF:x}",
        user_id=UserId("u1"),
        topic=Topic.WORK,
        text=text,
        entities=entities,
        created_at=datetime.now(UTC) - timedelta(days=days_ago),
        vec=vec,
    )


def _query(text: str = "what was I working on") -> RecallQuery:
    return RecallQuery(user_id=UserId("u1"), text=text, k=10)


def test_weights_normalized_when_not_summing_to_one() -> None:
    """The ranker should not require manually-normalized weights."""
    BlendedRanker(weights=RankerWeights(vec=2, recency=2, entity=0))  # ok


def test_invalid_weights_rejected() -> None:
    with pytest.raises(ValueError):
        RankerWeights(vec=-0.1)
    with pytest.raises(ValueError):
        RankerWeights(recency_half_life_days=0)


def test_recent_beats_old_when_vec_equal() -> None:
    """Two episodes with identical vec similarity: recency wins."""
    ranker = BlendedRanker()
    candidates = [
        (_ep("old", days_ago=14), 0.85),
        (_ep("new", days_ago=0.1), 0.85),
    ]
    hits = ranker.rank(_query(), np.zeros(8, dtype=np.float32), candidates)
    assert hits[0].episode.text == "new"


def test_entity_match_lifts_a_hit() -> None:
    """A query mentioning 'Joonatan' should pull episodes about Joonatan up."""
    # Tip the vec scores so the Joonatan episode would otherwise be #2.
    ranker = BlendedRanker(weights=RankerWeights(vec=0.4, recency=0.1, entity=0.5))
    candidates = [
        (_ep("random thought", days_ago=0.1), 0.95),
        (_ep("call with Joonatan", days_ago=0.1, entities=("Joonatan",)), 0.50),
    ]
    hits = ranker.rank(
        _query("did Joonatan say anything?"),
        np.zeros(8, dtype=np.float32),
        candidates,
    )
    assert "Joonatan" in hits[0].episode.text


def test_empty_candidates_returns_empty() -> None:
    ranker = BlendedRanker()
    assert ranker.rank(_query(), np.zeros(8, dtype=np.float32), []) == []


def test_decomposed_scores_are_reported() -> None:
    """Operators debug bad recalls by reading the decomposition."""
    ranker = BlendedRanker()
    candidates = [(_ep("hi", days_ago=1.0), 0.7)]
    hits = ranker.rank(_query("hi"), np.zeros(8, dtype=np.float32), candidates)
    h = hits[0]
    assert 0.0 <= h.vec_similarity <= 1.0
    assert 0.0 <= h.recency_score <= 1.0
    assert 0.0 <= h.entity_overlap <= 1.0
    assert 0.0 <= h.score <= 1.0


def test_respects_k() -> None:
    ranker = BlendedRanker()
    candidates = [(_ep(f"ep{i}", days_ago=i * 0.1), 0.9 - i * 0.05) for i in range(10)]
    q = RecallQuery(user_id=UserId("u1"), text="x", k=3)
    hits = ranker.rank(q, np.zeros(8, dtype=np.float32), candidates)
    assert len(hits) == 3
