"""Error taxonomy."""

from __future__ import annotations


class MemoryError(Exception):
    """Base class."""


class StoreError(MemoryError):
    """Storage failed (DB locked, disk full, schema mismatch)."""


class EmbedError(MemoryError):
    """Embedder failed (model not loaded, NaN output)."""


class IdentityError(MemoryError):
    """Identity resolution failed (no face_id provided, biometric mismatch)."""
