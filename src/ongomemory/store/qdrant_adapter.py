"""Qdrant adapter (scaffolded).

For multi-user / large-corpus installs we don't want brute-force cosine
over a SQLite BLOB. Qdrant is a sensible default: it's small (50MB
binary), runs locally, has Python bindings, and supports payload
filters that map cleanly to our `topics` and `since` filters.

The Protocol shape matches what InMemoryStore / SqliteStore provide,
so the rest of the system doesn't change.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ongomemory.core import Episode, Topic, UserId


class QdrantAdapter:
    """Vector-side store delegating to a local Qdrant instance.

    Scaffolded — wire this in once we have Qdrant running locally:

        docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
    """

    COLLECTION = "ongo_episodes"

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6333,
        vector_dim: int = 384,
    ) -> None:
        self._host = host
        self._port = port
        self._vector_dim = vector_dim
        # self._client = QdrantClient(host=host, port=port)
        # self._client.recreate_collection(...) on first run
        raise NotImplementedError(
            "QdrantAdapter scaffolded — drop in once we have Qdrant in the "
            "edge image. See module docstring for the docker line."
        )

    def vector_search(
        self, user_id: UserId, vec: np.ndarray, k: int
    ) -> list[tuple[Episode, float]]:
        # NOTE for the real implementation:
        #   hits = self._client.search(
        #       collection_name=self.COLLECTION,
        #       query_vector=vec.tolist(),
        #       query_filter=Filter(must=[FieldCondition(
        #           key="user_id", match=MatchValue(value=str(user_id)))]),
        #       limit=k,
        #   )
        #   return [(self._rehydrate(h.payload), h.score) for h in hits]
        raise NotImplementedError

    def upsert_episode(self, episode: Episode) -> None:
        raise NotImplementedError
