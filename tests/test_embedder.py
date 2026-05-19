"""Embedder contracts."""

from __future__ import annotations

import numpy as np
import pytest

from ongomemory.embed import DeterministicEmbedder


def test_dim_must_be_positive_multiple_of_8() -> None:
    with pytest.raises(ValueError):
        DeterministicEmbedder(dim=7)
    with pytest.raises(ValueError):
        DeterministicEmbedder(dim=0)
    DeterministicEmbedder(dim=384)  # ok


def test_empty_returns_zero_vector() -> None:
    e = DeterministicEmbedder()
    v = e.embed("")
    assert v.shape == (384,)
    assert np.linalg.norm(v) == 0.0


def test_deterministic(embedder: DeterministicEmbedder) -> None:
    a = embedder.embed("hello world")
    b = embedder.embed("hello world")
    np.testing.assert_array_equal(a, b)


def test_l2_normalized(embedder: DeterministicEmbedder) -> None:
    v = embedder.embed("the quick brown fox")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_similar_texts_more_correlated_than_unrelated(
    embedder: DeterministicEmbedder,
) -> None:
    a = embedder.embed("the quick brown fox jumps over the lazy dog")
    b = embedder.embed("a quick brown fox leaps over a lazy dog")  # shares many tokens
    c = embedder.embed("photosynthesis converts light into glucose")
    # Cosine = dot product on L2-normalized vectors.
    sim_ab = float(a @ b)
    sim_ac = float(a @ c)
    assert sim_ab > sim_ac


def test_dtype_is_float32(embedder: DeterministicEmbedder) -> None:
    v = embedder.embed("hi")
    assert v.dtype == np.float32


def test_batch_embed_shape(embedder: DeterministicEmbedder) -> None:
    texts = ["a", "ab cd", "longer phrase here"]
    mat = embedder.embed_batch(texts)
    assert mat.shape == (3, 384)
    assert mat.dtype == np.float32
