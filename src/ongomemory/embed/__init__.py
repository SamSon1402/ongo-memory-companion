"""Embedders. Two implementations:
  * DeterministicEmbedder — hashing + sign bits. For tests, ~no deps.
  * MiniLMEmbedder        — sentence-transformers/all-MiniLM-L6-v2. Production.
"""

from ongomemory.embed.deterministic import DeterministicEmbedder
from ongomemory.embed.minilm import MiniLMEmbedder

__all__ = ["DeterministicEmbedder", "MiniLMEmbedder"]
