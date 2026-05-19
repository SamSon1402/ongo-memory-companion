"""Shared fixtures."""

from __future__ import annotations

import pytest

from ongomemory import Memory
from ongomemory.core import UserId
from ongomemory.embed import DeterministicEmbedder
from ongomemory.store import InMemoryStore


@pytest.fixture
def user_id() -> UserId:
    return UserId("test_user")


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def memory(user_id: UserId, store: InMemoryStore, embedder: DeterministicEmbedder) -> Memory:
    return Memory(user_id=user_id, store=store, embedder=embedder)
