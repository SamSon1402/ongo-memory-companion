"""Domain types.

Episode  — one observed event ("user said X", "saw user enter room")
Fact     — a structured key/value about a user ("name=Sam")
Habit    — an inferred pattern from episodes ("coffee at ~08:50")
Identity — face_id + voice_id pair for a known user

Everything frozen. Episodes are append-only; the store never edits one
in place. Conflicting facts are resolved at *read* time by recency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, NewType, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── primitives ──────────────────────────────────────────────────────


# NewType, not a class — keeps the wire format a plain string.
UserId = NewType("UserId", str)


class Topic(StrEnum):
    """Topic taxonomy for episodes.

    Open-ended is tempting but bad — recall would have to do fuzzy
    matching on free-text topics. We pick a small set the assistant
    can map to from intent classification.
    """

    CALENDAR = "calendar"
    WORK = "work"
    HEALTH = "health"
    SOCIAL = "social"
    MEDIA = "media"
    HOME = "home"
    PROFILE = "profile"   # facts the user told us about themselves
    PROCEDURE = "procedure"  # how-to-do-something memories
    UNKNOWN = "unknown"


# ── Episode ─────────────────────────────────────────────────────────


class Episode(BaseModel):
    """One observation. Immutable. Vector is optional but recommended."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    episode_id: str = Field(min_length=1, max_length=64)
    user_id: UserId
    topic: Topic
    text: str = Field(min_length=1, max_length=2000)
    entities: tuple[str, ...] = Field(default_factory=tuple)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    vec: np.ndarray | None = None  # (D,) float32, optional

    @model_validator(mode="after")
    def _check_vec(self) -> Self:
        if self.vec is None:
            return self
        if self.vec.ndim != 1:
            raise ValueError(f"vec must be 1-D, got shape {self.vec.shape}")
        if self.vec.dtype != np.float32:
            raise ValueError(f"vec must be float32, got {self.vec.dtype}")
        return self


# ── Fact ────────────────────────────────────────────────────────────


class Fact(BaseModel):
    """A structured user attribute. Last-write-wins by created_at."""

    model_config = ConfigDict(frozen=True)

    user_id: UserId
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z_][a-z0-9_]*$")
    value: str = Field(min_length=1, max_length=500)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    source_episode_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Habit ───────────────────────────────────────────────────────────


class Habit(BaseModel):
    """A pattern inferred from episodes. Suggestion, not fact."""

    model_config = ConfigDict(frozen=True)

    user_id: UserId
    summary: str = Field(min_length=1, max_length=200)
    topic: Topic
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    observation_count: int = Field(ge=1)
    supporting_episode_ids: tuple[str, ...]
    # Optional time-of-day pattern: ('08:30', '09:15') if it has one.
    time_window: tuple[str, str] | None = None
    inferred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Identity ────────────────────────────────────────────────────────


class Identity(BaseModel):
    """The link between biometric ids and a user_id."""

    model_config = ConfigDict(frozen=True)

    user_id: UserId
    face_id: str | None = None  # opaque hash of face embedding
    voice_id: str | None = None
    display_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Recall ──────────────────────────────────────────────────────────


class RecallQuery(BaseModel):
    """A request to recall episodes."""

    model_config = ConfigDict(frozen=True)

    user_id: UserId
    text: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)
    topics: tuple[Topic, ...] | None = None  # optional filter
    since: datetime | None = None            # only episodes after this


class RecallHit(BaseModel):
    """One ranked result. Decomposed scores are kept for debugging."""

    model_config = ConfigDict(frozen=True)

    episode: Episode
    score: float
    vec_similarity: float
    recency_score: float
    entity_overlap: float
