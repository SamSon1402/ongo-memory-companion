"""In-memory store.

Real implementation of the MemoryStore Protocol. Used by tests and the
demo. Vector search is brute-force cosine — fine for a few thousand
episodes (which is more than one user accumulates in months).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from ongomemory.core import (
    Episode,
    Fact,
    Habit,
    Identity,
    MemoryStore,
    Topic,
    UserId,
)
from ongomemory.core.errors import StoreError


class InMemoryStore:
    """Thread-safe in-memory implementation of MemoryStore."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._episodes: dict[UserId, list[Episode]] = defaultdict(list)
        self._facts: dict[UserId, dict[str, Fact]] = defaultdict(dict)
        self._habits: dict[UserId, list[Habit]] = defaultdict(list)
        self._identities: dict[UserId, Identity] = {}
        # Episode dedup window: only collapse repeats within this many minutes.
        # Beyond that, the same utterance on different days is a different episode.
        self._dedup_window_minutes = 10.0

    # ── episodes ────────────────────────────────────────────────────

    def write_episode(self, episode: Episode) -> None:
        with self._lock:
            normalized = _normalize(episode.text)
            # Walk recent episodes for this user looking for a close-in-time dup.
            for prior in reversed(self._episodes[episode.user_id]):
                if _normalize(prior.text) != normalized:
                    continue
                delta_min = abs(
                    (episode.created_at - prior.created_at).total_seconds() / 60.0
                )
                if delta_min <= self._dedup_window_minutes:
                    # Same content, same window → ignore.
                    return
                # Same content but well-separated in time → it's a new event.
                break
            self._episodes[episode.user_id].append(episode)

    def episodes_for(
        self,
        user_id: UserId,
        *,
        topics: Iterable[Topic] | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        with self._lock:
            eps = self._episodes.get(user_id, [])
            if topics is not None:
                allowed = set(topics)
                eps = [e for e in eps if e.topic in allowed]
            # Newest first.
            return sorted(eps, key=lambda e: e.created_at, reverse=True)[:limit]

    def vector_search(
        self, user_id: UserId, vec: np.ndarray, k: int
    ) -> list[tuple[Episode, float]]:
        if vec.ndim != 1:
            raise StoreError(f"query vec must be 1-D, got {vec.shape}")
        with self._lock:
            eps = [e for e in self._episodes.get(user_id, []) if e.vec is not None]

        if not eps:
            return []

        # Stack into a matrix once → one numpy call vs N python loops.
        mat = np.stack([e.vec for e in eps], axis=0)  # type: ignore[misc]
        # Vectors are L2-normalized at write time; cosine = dot.
        sims = mat @ vec
        idx = np.argsort(-sims)[:k]
        return [(eps[i], float(sims[i])) for i in idx]

    # ── facts ───────────────────────────────────────────────────────

    def write_fact(self, fact: Fact) -> None:
        with self._lock:
            existing = self._facts[fact.user_id].get(fact.key)
            # Last-write-wins, but a newer low-confidence fact does not
            # overwrite an older high-confidence one. Otherwise a single
            # bad ASR turn could clobber a known truth.
            if existing is not None:
                if fact.confidence < existing.confidence * 0.8:
                    return
            self._facts[fact.user_id][fact.key] = fact

    def latest_facts(self, user_id: UserId) -> dict[str, Fact]:
        with self._lock:
            return dict(self._facts.get(user_id, {}))

    # ── habits ──────────────────────────────────────────────────────

    def write_habit(self, habit: Habit) -> None:
        with self._lock:
            # Replace existing habit with same summary (re-mining is idempotent).
            existing = [h for h in self._habits[habit.user_id] if h.summary == habit.summary]
            for h in existing:
                self._habits[habit.user_id].remove(h)
            self._habits[habit.user_id].append(habit)

    def habits_for(self, user_id: UserId) -> list[Habit]:
        with self._lock:
            return sorted(
                self._habits.get(user_id, []), key=lambda h: h.confidence, reverse=True
            )

    # ── identity ────────────────────────────────────────────────────

    def write_identity(self, identity: Identity) -> None:
        with self._lock:
            self._identities[identity.user_id] = identity

    def identity_for(self, user_id: UserId) -> Identity | None:
        with self._lock:
            return self._identities.get(user_id)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# Static check: in case the Protocol grows a method.
_: MemoryStore = InMemoryStore()  # type: ignore[assignment, unused-ignore]
