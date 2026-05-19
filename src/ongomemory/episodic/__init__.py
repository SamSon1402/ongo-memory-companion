"""Episodic memory: the write path.

Turning a raw observation into a stored Episode involves:
  1. Generating a stable episode_id
  2. Computing the embedding
  3. Extracting entities (delegated to semantic/)
  4. Dedup check (handled by the store)
  5. Optionally extracting Facts from the episode text
"""

from ongomemory.episodic.writer import EpisodeWriter

__all__ = ["EpisodeWriter"]
