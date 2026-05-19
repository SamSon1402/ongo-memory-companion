"""Domain type contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ongomemory.core import Episode, Fact, RecallQuery, Topic, UserId


class TestEpisode:
    def test_minimal_construction(self) -> None:
        ep = Episode(
            episode_id="abc123",
            user_id=UserId("u1"),
            topic=Topic.UNKNOWN,
            text="hi",
        )
        assert ep.confidence == 0.85
        assert ep.entities == ()

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            Episode(
                episode_id="abc", user_id=UserId("u1"), topic=Topic.UNKNOWN, text=""
            )

    def test_rejects_wrong_vec_dtype(self) -> None:
        bad_vec = np.zeros(384, dtype=np.float64)
        with pytest.raises(ValidationError):
            Episode(
                episode_id="abc",
                user_id=UserId("u1"),
                topic=Topic.UNKNOWN,
                text="hi",
                vec=bad_vec,
            )

    def test_accepts_valid_vec(self) -> None:
        vec = np.zeros(384, dtype=np.float32)
        ep = Episode(
            episode_id="abc",
            user_id=UserId("u1"),
            topic=Topic.UNKNOWN,
            text="hi",
            vec=vec,
        )
        assert ep.vec is not None
        assert ep.vec.shape == (384,)


class TestFact:
    def test_key_pattern(self) -> None:
        # Valid key.
        Fact(user_id=UserId("u1"), key="favorite_color", value="teal")
        # Invalid: dashes and capitals.
        with pytest.raises(ValidationError):
            Fact(user_id=UserId("u1"), key="Favorite-Color", value="teal")
        with pytest.raises(ValidationError):
            Fact(user_id=UserId("u1"), key="1name", value="Sam")


class TestRecallQuery:
    def test_k_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RecallQuery(user_id=UserId("u1"), text="hi", k=0)
        with pytest.raises(ValidationError):
            RecallQuery(user_id=UserId("u1"), text="hi", k=999)
