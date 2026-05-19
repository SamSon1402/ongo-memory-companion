"""Core domain types and protocols."""

from ongomemory.core.errors import (
    EmbedError,
    IdentityError,
    MemoryError as OngoMemoryError,
    StoreError,
)
from ongomemory.core.protocols import (
    Embedder,
    HabitMiner,
    IdentityResolver,
    MemoryStore,
    Ranker,
)
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

__all__ = [
    # types
    "Episode",
    "Fact",
    "Habit",
    "Identity",
    "RecallHit",
    "RecallQuery",
    "Topic",
    "UserId",
    # protocols
    "Embedder",
    "HabitMiner",
    "IdentityResolver",
    "MemoryStore",
    "Ranker",
    # errors
    "EmbedError",
    "IdentityError",
    "OngoMemoryError",
    "StoreError",
]
