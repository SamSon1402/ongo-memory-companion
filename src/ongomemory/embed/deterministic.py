"""Deterministic embedder for tests + the demo loop.

Hash-and-bit-spread embedder. Produces an L2-normalized 384-dim vector
where similar texts (sharing tokens) have correlated bits. Not a real
semantic embedder — but stable, fast, no dependencies, and good enough
that the recall pipeline can be tested end-to-end.

Production swaps `MiniLMEmbedder` in.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\b\w+\b")


class DeterministicEmbedder:
    """Token-hash + signed-bit projection. ~Stable, ~useful, no deps."""

    def __init__(self, *, dim: int = 384, seed: int = 0xC0FFEE) -> None:
        if dim <= 0 or dim % 8 != 0:
            raise ValueError(f"dim must be a positive multiple of 8, got {dim}")
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"deterministic-h{self._dim}"

    def embed(self, text: str) -> np.ndarray:
        """L2-normalized vector for one text."""
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec

        for tok in tokens:
            # Two independent hashes per token → fixed positions, signed contribution.
            h1 = hashlib.blake2b(
                tok.encode(), digest_size=4, person=b"ongo_h1"
            ).digest()
            h2 = hashlib.blake2b(
                tok.encode(), digest_size=4, person=b"ongo_h2"
            ).digest()
            pos = int.from_bytes(h1, "little") % self._dim
            sign = 1.0 if (int.from_bytes(h2, "little") & 1) else -1.0
            vec[pos] += sign

        # L2 normalize so cosine similarity is just a dot product.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self.embed(t)
        return out
