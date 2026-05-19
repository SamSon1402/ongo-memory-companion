"""Blended ranker.

Pure cosine similarity over text embeddings is wrong for conversational
memory. Two real failure modes drove this design:

  1. *Stale-but-semantic-match*: a 3-month-old episode about "the call
     with Joonatan" matches "call with Joonatan" today better than
     yesterday's mention does, because semantic embeddings have no
     notion of time. Bad recall.

  2. *Entity-aware queries*: when the user mentions a specific name
     ("did Vaishnavi say anything about X?"), entity match should
     dominate over fuzzy semantic similarity to other people.

So we blend three signals:

  final = w_vec * cos_sim + w_recency * exp_decay + w_entity * jaccard

The weights are calibrated, not magic. Defaults below were eyeballed
against the demo script; the proper procedure is a small held-out set
of (query, expected_episode_id) pairs and a sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from ongomemory.core import Episode, RecallHit, RecallQuery
from ongomemory.semantic.entities import extract_entities


@dataclass(frozen=True)
class RankerWeights:
    """Per-signal weights in [0,1]. Sum doesn't have to be 1.0 — we
    normalize at the end if it isn't."""

    vec: float = 0.55
    recency: float = 0.25
    entity: float = 0.20

    # Recency half-life (in days). After this much elapsed time the
    # recency score drops to 0.5. Tuned for conversational memory —
    # a week-old episode is much weaker than yesterday's.
    recency_half_life_days: float = 3.0

    def __post_init__(self) -> None:
        for name in ("vec", "recency", "entity"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} weight must be non-negative, got {v}")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be > 0")


class BlendedRanker:
    """Multi-signal ranker. Implements the `Ranker` protocol."""

    def __init__(self, *, weights: RankerWeights | None = None) -> None:
        self._w = weights or RankerWeights()
        self._norm = self._w.vec + self._w.recency + self._w.entity
        if self._norm == 0:
            raise ValueError("at least one weight must be > 0")

    def rank(
        self,
        query: RecallQuery,
        query_vec: np.ndarray,  # noqa: ARG002 - kept for protocol shape
        candidates: list[tuple[Episode, float]],
    ) -> list[RecallHit]:
        if not candidates:
            return []

        now = datetime.now(UTC)
        query_entities = {e.lower() for e in extract_entities(query.text)}

        # Cosine sim is given by the store. We just rescale.
        max_cos = max((c[1] for c in candidates), default=1e-9) or 1e-9

        hits: list[RecallHit] = []
        for episode, cos_sim in candidates:
            vec_score = max(0.0, cos_sim) / max_cos

            age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            recency_score = math.exp(-age_days * math.log(2) / self._w.recency_half_life_days)

            # Jaccard on entity sets — a tiny set, so this is fine.
            ep_entities = {e.lower() for e in episode.entities}
            if query_entities and ep_entities:
                inter = len(query_entities & ep_entities)
                union = len(query_entities | ep_entities)
                entity_score = inter / union
            elif not query_entities and not ep_entities:
                # No entity signal either way — give a neutral score.
                entity_score = 0.5
            else:
                entity_score = 0.0

            blended = (
                self._w.vec * vec_score
                + self._w.recency * recency_score
                + self._w.entity * entity_score
            ) / self._norm

            hits.append(
                RecallHit(
                    episode=episode,
                    score=blended,
                    vec_similarity=float(cos_sim),
                    recency_score=recency_score,
                    entity_overlap=entity_score,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: query.k]
