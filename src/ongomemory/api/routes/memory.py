"""Memory endpoints.

POST   /users/{user_id}/episodes     write an episode
GET    /users/{user_id}/episodes     list episodes (optional topic filter)
POST   /users/{user_id}/recall       blended-rank recall query
GET    /users/{user_id}/facts        flat key→value of latest facts
GET    /users/{user_id}/habits       inferred habits (refreshes by default)
POST   /users/{user_id}/identity     register face/voice id

A single in-process registry holds one Memory per user_id. In a multi-
device deployment we'd index by (device_id, user_id); the surface
doesn't change.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ongomemory import Memory
from ongomemory.core import Topic

router = APIRouter()
log = structlog.get_logger(__name__)


# ── request / response schemas ──────────────────────────────────────


class WriteEpisodeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    topic: Topic = Topic.UNKNOWN
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class EpisodeResponse(BaseModel):
    episode_id: str
    topic: str
    text: str
    entities: list[str]
    confidence: float
    created_at: str


class RecallRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)
    topics: list[Topic] | None = None


class RecallHitResponse(BaseModel):
    episode_id: str
    text: str
    topic: str
    score: float
    vec_similarity: float
    recency_score: float
    entity_overlap: float
    created_at: str


class HabitResponse(BaseModel):
    summary: str
    topic: str
    confidence: float
    observation_count: int
    time_window: tuple[str, str] | None


class IdentityRequest(BaseModel):
    face_id: str | None = None
    voice_id: str | None = None
    display_name: str | None = None


# ── per-user registry ───────────────────────────────────────────────


class _Registry:
    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> Memory:
        async with self._lock:
            mem = self._memories.get(user_id)
            if mem is None:
                mem = Memory.for_user(user_id)
                self._memories[user_id] = mem
            return mem

    async def reset(self) -> None:
        async with self._lock:
            self._memories.clear()


_REGISTRY = _Registry()


def get_registry() -> _Registry:
    return _REGISTRY


# ── endpoints ───────────────────────────────────────────────────────


@router.post("/users/{user_id}/episodes", response_model=EpisodeResponse)
async def write_episode(
    user_id: str,
    body: WriteEpisodeRequest,
    registry: Annotated[_Registry, Depends(get_registry)],
) -> EpisodeResponse:
    mem = await registry.get(user_id)
    try:
        ep = mem.episode(text=body.text, topic=body.topic, confidence=body.confidence)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return _serialize_episode(ep)


@router.get("/users/{user_id}/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    user_id: str,
    registry: Annotated[_Registry, Depends(get_registry)],
    topic: Topic | None = None,
    limit: int = 50,
) -> list[EpisodeResponse]:
    if not 1 <= limit <= 500:
        raise HTTPException(422, detail="limit must be 1..500")
    mem = await registry.get(user_id)
    topics = (topic,) if topic is not None else None
    return [_serialize_episode(e) for e in mem.all_episodes(topics=topics, limit=limit)]


@router.post("/users/{user_id}/recall", response_model=list[RecallHitResponse])
async def recall(
    user_id: str,
    body: RecallRequest,
    registry: Annotated[_Registry, Depends(get_registry)],
) -> list[RecallHitResponse]:
    mem = await registry.get(user_id)
    topics = tuple(body.topics) if body.topics else None
    hits = mem.recall(body.text, k=body.k, topics=topics)
    return [
        RecallHitResponse(
            episode_id=h.episode.episode_id,
            text=h.episode.text,
            topic=h.episode.topic.value,
            score=round(h.score, 4),
            vec_similarity=round(h.vec_similarity, 4),
            recency_score=round(h.recency_score, 4),
            entity_overlap=round(h.entity_overlap, 4),
            created_at=h.episode.created_at.isoformat(),
        )
        for h in hits
    ]


@router.get("/users/{user_id}/facts")
async def facts(
    user_id: str,
    registry: Annotated[_Registry, Depends(get_registry)],
) -> dict[str, str]:
    mem = await registry.get(user_id)
    return mem.facts()


@router.get("/users/{user_id}/habits", response_model=list[HabitResponse])
async def habits(
    user_id: str,
    registry: Annotated[_Registry, Depends(get_registry)],
    refresh: bool = True,
) -> list[HabitResponse]:
    mem = await registry.get(user_id)
    return [
        HabitResponse(
            summary=h.summary,
            topic=h.topic.value,
            confidence=round(h.confidence, 3),
            observation_count=h.observation_count,
            time_window=h.time_window,
        )
        for h in mem.habits(refresh=refresh)
    ]


@router.post("/users/{user_id}/identity", status_code=204)
async def set_identity(
    user_id: str,
    body: IdentityRequest,
    registry: Annotated[_Registry, Depends(get_registry)],
) -> None:
    mem = await registry.get(user_id)
    mem.identify(
        face_id=body.face_id, voice_id=body.voice_id, display_name=body.display_name
    )


def _serialize_episode(ep) -> EpisodeResponse:  # noqa: ANN001
    return EpisodeResponse(
        episode_id=ep.episode_id,
        topic=ep.topic.value,
        text=ep.text,
        entities=list(ep.entities),
        confidence=round(ep.confidence, 3),
        created_at=ep.created_at.isoformat(),
    )
