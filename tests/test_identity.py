"""Identity resolver contracts."""

from __future__ import annotations

import numpy as np
import pytest

from ongomemory.core import UserId
from ongomemory.core.errors import IdentityError
from ongomemory.identity import SimpleIdentityResolver


def _rand(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_resolves_enrolled_face() -> None:
    r = SimpleIdentityResolver(threshold=0.5)
    v = _rand(1)
    r.enroll_face(v, UserId("alice"))
    assert r.resolve_face(v) == "alice"


def test_unknown_face_returns_none() -> None:
    r = SimpleIdentityResolver(threshold=0.99)  # very strict
    r.enroll_face(_rand(1), UserId("alice"))
    # Different embedding, threshold tuned so it won't match.
    assert r.resolve_face(_rand(99)) is None


def test_resolves_to_closest() -> None:
    r = SimpleIdentityResolver(threshold=0.0)  # accept anything
    r.enroll_face(_rand(1), UserId("alice"))
    r.enroll_face(_rand(2), UserId("bob"))
    # A query closer to alice than to bob — should pick alice.
    q = _rand(1) + 0.05 * _rand(99)
    q /= np.linalg.norm(q)
    resolved = r.resolve_face(q)
    assert resolved == "alice"


def test_threshold_validation() -> None:
    # 0.0 is allowed (dev "accept all" mode), 1.0 is not (nothing would ever match).
    SimpleIdentityResolver(threshold=0.0)
    with pytest.raises(IdentityError):
        SimpleIdentityResolver(threshold=1.0)
    with pytest.raises(IdentityError):
        SimpleIdentityResolver(threshold=-0.1)


def test_returns_opaque_id_on_enroll() -> None:
    r = SimpleIdentityResolver()
    opaque = r.enroll_face(_rand(42), UserId("alice"))
    assert isinstance(opaque, str)
    assert len(opaque) >= 8  # blake2b 8-byte digest → 16 hex chars
