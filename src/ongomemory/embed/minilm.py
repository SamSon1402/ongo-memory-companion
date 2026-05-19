"""Production embedder: sentence-transformers/all-MiniLM-L6-v2.

Why MiniLM:
  * 384 dims — small index, fast cosine
  * 80 MB on disk, runs on a Pi 5 / Coral CPU in ~15ms per text
  * Strong on short conversational text (which is what we have)

Scaffolded — needs `pip install sentence-transformers` and the model
weights cached. The Protocol shape is identical to DeterministicEmbedder
so swapping is a config change.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ongomemory.core.errors import EmbedError


class MiniLMEmbedder:
    """Production embedder. Scaffolded — wire in once weights are cached."""

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
        normalize: bool = True,
    ) -> None:
        self._device = device
        self._cache_dir = cache_dir
        self._normalize = normalize
        self._model = None  # lazy
        # The constructor doesn't raise; we lazy-load on first embed() so
        # this class can be imported without the dependency installed.

    @property
    def dim(self) -> int:
        return 384  # fixed by the model

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    def embed(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        vec = model.encode(text, normalize_embeddings=self._normalize)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        out = model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedError(
                "sentence-transformers not installed. "
                "`pip install 'ongomemory[embed]'`."
            ) from exc
        self._model = SentenceTransformer(
            self.MODEL_ID,
            device=self._device,
            cache_folder=str(self._cache_dir) if self._cache_dir else None,
        )
        return self._model
