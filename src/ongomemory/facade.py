"""The library API.

Aimed at the rest of the Ongo system. One class, a handful of methods,
sensible defaults. Anything more elaborate (custom embedder, custom
ranker weights, etc.) is reachable via the lower-level modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from ongomemory.core import (
    Embedder,
    Episode,
    Fact,
    Habit,
    Identity,
    MemoryStore,
    RecallHit,
    RecallQuery,
    Topic,
    UserId,
)
from ongomemory.embed import DeterministicEmbedder
from ongomemory.episodic import EpisodeWriter
from ongomemory.habits import TimeOfDayHabitMiner
from ongomemory.recall import BlendedRanker
from ongomemory.store import InMemoryStore, SqliteStore

log = structlog.get_logger(__name__)


class Memory:
    """Per-user memory facade.

    Default construction wires:
      * InMemoryStore  (override for SqliteStore in prod)
      * DeterministicEmbedder (override for MiniLMEmbedder in prod)
      * BlendedRanker with default weights
      * TimeOfDayHabitMiner

    Use the `for_user` classmethod for the common path; the constructor
    is there for dependency injection (tests, custom backends).
    """

    def __init__(
        self,
        *,
        user_id: UserId | str,
        store: MemoryStore,
        embedder: Embedder,
    ) -> None:
        self._user_id: UserId = UserId(str(user_id))
        self._store = store
        self._embedder = embedder
        self._writer = EpisodeWriter(store=store, embedder=embedder)
        self._ranker = BlendedRanker()
        self._miner = TimeOfDayHabitMiner()

    # ── factories ───────────────────────────────────────────────────

    @classmethod
    def for_user(
        cls,
        user_id: UserId | str,
        *,
        db_path: Path | str | None = None,
    ) -> "Memory":
        """The common case. SQLite on disk if `db_path` given, else in-memory."""
        store: MemoryStore
        if db_path is None:
            store = InMemoryStore()
        else:
            store = SqliteStore(db_path=db_path)
        return cls(
            user_id=user_id,
            store=store,
            embedder=DeterministicEmbedder(),
        )

    # ── write ───────────────────────────────────────────────────────

    def episode(
        self,
        *,
        text: str,
        topic: Topic | str = Topic.UNKNOWN,
        confidence: float = 0.85,
        when: datetime | None = None,
    ) -> Episode:
        """Record one observation."""
        if isinstance(topic, str):
            topic = Topic(topic)
        return self._writer.write(
            user_id=self._user_id,
            text=text,
            topic=topic,
            confidence=confidence,
            created_at=when,
        )

    def fact(self, *, key: str, value: str, confidence: float = 1.0) -> None:
        """Record a structured fact directly (bypasses episode text)."""
        self._store.write_fact(
            Fact(
                user_id=self._user_id,
                key=key,
                value=value,
                confidence=confidence,
                created_at=datetime.now(UTC),
            )
        )

    def identify(
        self,
        *,
        face_id: str | None = None,
        voice_id: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Register or update this user's biometric ids."""
        self._store.write_identity(
            Identity(
                user_id=self._user_id,
                face_id=face_id,
                voice_id=voice_id,
                display_name=display_name,
            )
        )

    # ── read ────────────────────────────────────────────────────────

    def recall(
        self,
        query_text: str,
        *,
        k: int = 5,
        topics: tuple[Topic, ...] | None = None,
    ) -> list[RecallHit]:
        """Return up to `k` ranked episodes most relevant to the query."""
        if not query_text.strip():
            return []
        query = RecallQuery(
            user_id=self._user_id, text=query_text, k=k, topics=topics
        )
        query_vec = self._embedder.embed(query_text)
        candidates = self._store.vector_search(self._user_id, query_vec, k * 3)
        # Optional topic filter — done post-vector-search since we want a
        # generous candidate pool.
        if topics is not None:
            allowed = set(topics)
            candidates = [(e, s) for e, s in candidates if e.topic in allowed]
        return self._ranker.rank(query, query_vec, candidates)

    def facts(self) -> dict[str, str]:
        """Flatten the latest_facts dict to key→value pairs."""
        return {k: f.value for k, f in self._store.latest_facts(self._user_id).items()}

    def all_episodes(
        self, *, topics: tuple[Topic, ...] | None = None, limit: int = 100
    ) -> list[Episode]:
        return self._store.episodes_for(self._user_id, topics=topics, limit=limit)

    def identity(self) -> Identity | None:
        return self._store.identity_for(self._user_id)

    # ── habits ──────────────────────────────────────────────────────

    def habits(self, *, refresh: bool = True) -> list[Habit]:
        """Return all known habits. If `refresh` is True, re-mine first."""
        if refresh:
            episodes = self._store.episodes_for(self._user_id, limit=1000)
            for habit in self._miner.mine(episodes, self._user_id):
                self._store.write_habit(habit)
        return self._store.habits_for(self._user_id)

    # ── introspection ───────────────────────────────────────────────

    @property
    def user_id(self) -> UserId:
        return self._user_id
