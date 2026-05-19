"""Episode writer.

The single entry point for adding observations. Composes the embedder,
the entity extractor, and the store.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import structlog

from ongomemory.core import (
    Embedder,
    Episode,
    Fact,
    MemoryStore,
    Topic,
    UserId,
)
from ongomemory.semantic.entities import extract_entities, extract_facts

log = structlog.get_logger(__name__)


class EpisodeWriter:
    """Writes Episodes (and any extractable Facts) to the store."""

    def __init__(self, *, store: MemoryStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def write(
        self,
        *,
        user_id: UserId,
        text: str,
        topic: Topic = Topic.UNKNOWN,
        confidence: float = 0.85,
        created_at: datetime | None = None,
        extract_facts_too: bool = True,
    ) -> Episode:
        text = text.strip()
        if not text:
            raise ValueError("episode text must be non-empty")

        created_at = created_at or datetime.now(UTC)
        episode_id = _stable_id(user_id, text, created_at)
        vec = self._embedder.embed(text)
        entities = extract_entities(text)

        episode = Episode(
            episode_id=episode_id,
            user_id=user_id,
            topic=topic,
            text=text,
            entities=tuple(entities),
            confidence=confidence,
            created_at=created_at,
            vec=vec,
        )
        self._store.write_episode(episode)

        if extract_facts_too:
            for key, value, conf in extract_facts(text):
                self._store.write_fact(
                    Fact(
                        user_id=user_id,
                        key=key,
                        value=value,
                        confidence=conf * confidence,
                        source_episode_id=episode.episode_id,
                        created_at=created_at,
                    )
                )

        log.info(
            "episode.written",
            user_id=str(user_id),
            episode_id=episode_id,
            topic=topic.value,
            n_entities=len(entities),
        )
        return episode


def _stable_id(user_id: UserId, text: str, when: datetime) -> str:
    """Episode IDs are content-addressed so writing the same thing twice
    yields the same id → idempotent writes."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(user_id).encode())
    h.update(text.lower().encode())
    h.update(when.isoformat(timespec="minutes").encode())
    return h.hexdigest()


def new_uuid_episode_id() -> str:
    """For cases where content-addressed ids aren't desired (writes from
    external sources, e.g. imports)."""
    return uuid.uuid4().hex[:16]
