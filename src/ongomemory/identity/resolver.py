"""Identity resolver.

Maps a face_embedding / voice_embedding to a known user_id. Uses
cosine similarity against an enrolled set with a configurable
acceptance threshold.

Why a separate module: Ongo's face & voice recognition come from
separate upstream models (ArcFace + ECAPA-TDNN, or similar). The
*linking* — "this embedding belongs to user f7a2" — is a per-device
concern that has nothing to do with the recognition model itself.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

from ongomemory.core import UserId
from ongomemory.core.errors import IdentityError


class SimpleIdentityResolver:
    """In-memory resolver. Enroll(embedding, user_id), then resolve()."""

    def __init__(self, *, threshold: float = 0.62) -> None:
        if not 0.0 <= threshold < 1.0:
            raise IdentityError(f"threshold must be in [0,1), got {threshold}")
        self._threshold = threshold
        self._lock = threading.RLock()
        self._faces: list[tuple[np.ndarray, UserId]] = []
        self._voices: list[tuple[np.ndarray, UserId]] = []

    # ── enrollment ──────────────────────────────────────────────────

    def enroll_face(self, embedding: np.ndarray, user_id: UserId) -> str:
        return self._enroll(embedding, user_id, self._faces, kind="face")

    def enroll_voice(self, embedding: np.ndarray, user_id: UserId) -> str:
        return self._enroll(embedding, user_id, self._voices, kind="voice")

    def _enroll(
        self, emb: np.ndarray, user_id: UserId, target: list, *, kind: str
    ) -> str:
        if emb.ndim != 1:
            raise IdentityError(f"{kind} embedding must be 1-D, got {emb.shape}")
        normalized = _l2_normalize(emb.astype(np.float32))
        with self._lock:
            target.append((normalized, user_id))
        return _opaque_id(normalized)

    # ── resolution ──────────────────────────────────────────────────

    def resolve_face(self, face_embedding: np.ndarray) -> UserId | None:
        return self._resolve(face_embedding, self._faces)

    def resolve_voice(self, voice_embedding: np.ndarray) -> UserId | None:
        return self._resolve(voice_embedding, self._voices)

    def _resolve(self, emb: np.ndarray, enrolled: list) -> UserId | None:
        if emb.ndim != 1:
            return None
        with self._lock:
            if not enrolled:
                return None
            q = _l2_normalize(emb.astype(np.float32))
            best_user: UserId | None = None
            best_sim = -1.0
            for stored, uid in enrolled:
                sim = float(stored @ q)
                if sim > best_sim:
                    best_sim = sim
                    best_user = uid
            if best_sim >= self._threshold:
                return best_user
            return None

    @property
    def n_enrolled_faces(self) -> int:
        return len(self._faces)

    @property
    def n_enrolled_voices(self) -> int:
        return len(self._voices)


# ── helpers ─────────────────────────────────────────────────────────


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _opaque_id(normalized: np.ndarray) -> str:
    """Stable, non-reversible id derived from a normalized embedding.

    We share these with downstream systems instead of raw embeddings
    so a leaked id can't be reversed back to a biometric.
    """
    return hashlib.blake2b(normalized.tobytes(), digest_size=8).hexdigest()
