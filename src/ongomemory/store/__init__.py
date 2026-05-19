"""Storage backends.

InMemoryStore — for tests and the smoke demo.
SqliteStore   — production default for the edge box.
QdrantAdapter — vector layer for production; SqliteStore handles structured side.
"""

from ongomemory.store.in_memory import InMemoryStore
from ongomemory.store.qdrant_adapter import QdrantAdapter
from ongomemory.store.sqlite_store import SqliteStore

__all__ = ["InMemoryStore", "QdrantAdapter", "SqliteStore"]
