"""Protocols.

Every pluggable component declares its interface here. Storage backend,
embedder, ranker, habit miner, identity resolver — each is a Protocol.

The split between `MemoryStore` and `Embedder` is intentional: the
embedder is *computational* (deterministic given the same input + model
weights), the store is *stateful* (mutable, persistent). Production
swaps them independently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ongomemory.core.types import (
    Episode,
    Fact,
    Habit,
    Identity,
    RecallHit,
    RecallQuery,
    Topic,
    UserId,
)


@runtime_checkable
class Embedder(Protocol):
    """Maps text → dense vector. Stateless."""

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Returns (N, D) float32. Default: loop, but real impls batch."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Storage. Hides the split between SQLite + Qdrant."""

    # ── episodes ────────────────────────────────────────────────────
    def write_episode(self, episode: Episode) -> None: ...
    def episodes_for(
        self, user_id: UserId, *, topics: Iterable[Topic] | None = None, limit: int = 100
    ) -> list[Episode]: ...
    def vector_search(
        self, user_id: UserId, vec: np.ndarray, k: int
    ) -> list[tuple[Episode, float]]:
        """Return (episode, cosine_similarity) pairs."""
        ...

    # ── facts ───────────────────────────────────────────────────────
    def write_fact(self, fact: Fact) -> None: ...
    def latest_facts(self, user_id: UserId) -> dict[str, Fact]:
        """Return key → latest fact, last-write-wins."""
        ...

    # ── habits ──────────────────────────────────────────────────────
    def write_habit(self, habit: Habit) -> None: ...
    def habits_for(self, user_id: UserId) -> list[Habit]: ...

    # ── identity ────────────────────────────────────────────────────
    def write_identity(self, identity: Identity) -> None: ...
    def identity_for(self, user_id: UserId) -> Identity | None: ...


@runtime_checkable
class Ranker(Protocol):
    """Turns raw vector hits into final RecallHits with multi-signal scoring."""

    def rank(
        self,
        query: RecallQuery,
        query_vec: np.ndarray,
        candidates: list[tuple[Episode, float]],
    ) -> list[RecallHit]: ...


@runtime_checkable
class HabitMiner(Protocol):
    """Scans recent episodes for patterns and produces Habits."""

    def mine(self, episodes: list[Episode], user_id: UserId) -> list[Habit]: ...


@runtime_checkable
class IdentityResolver(Protocol):
    """Maps biometric embeddings → user_id (or 'unknown')."""

    def resolve_face(self, face_embedding: np.ndarray) -> UserId | None: ...
    def resolve_voice(self, voice_embedding: np.ndarray) -> UserId | None: ...
