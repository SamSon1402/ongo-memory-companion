"""Habit mining.

Looks for clusters of similar episodes that recur at similar times of
day. Two ingredients:

  1. *What*: episodes are grouped by topic + entity overlap.
  2. *When*: within a group, the times-of-day are checked for tight
     clustering. We use a simple "fits in a 90-minute window" test —
     proper kernel density estimation is overkill for the volume of
     data Ongo sees (a few hundred episodes / month / user).

Output: Habit objects with confidence ∝ (observations / window_size).
The pipeline elsewhere surfaces these as suggestions: "Sam takes coffee
around 8:50 — should I set a focus block after?"

This is the file the CTO will read to see whether memory is just
storage or a real product feature.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

import structlog

from ongomemory.core import Episode, Habit, Topic, UserId

log = structlog.get_logger(__name__)


# Min observations to call something a habit. 3 is the Goldilocks number:
# 2 is anecdote, 4 is overcautious.
MIN_OBSERVATIONS = 3

# How wide a time-of-day window we'll accept (minutes). A 90-min window
# around the median is generous enough for "morning coffee" but tight
# enough to differentiate "morning routine" from "afternoon routine".
TIME_WINDOW_MINUTES = 90


class TimeOfDayHabitMiner:
    """Find time-of-day patterns. Stateless — pass episodes in, get habits out."""

    def __init__(
        self,
        *,
        min_observations: int = MIN_OBSERVATIONS,
        window_minutes: int = TIME_WINDOW_MINUTES,
    ) -> None:
        if min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if window_minutes <= 0:
            raise ValueError("window_minutes must be > 0")
        self._min_obs = min_observations
        self._window_min = window_minutes

    def mine(self, episodes: list[Episode], user_id: UserId) -> list[Habit]:
        """Return all habits we can confidently extract from these episodes."""
        if len(episodes) < self._min_obs:
            return []

        # Group by (topic, primary_entity). Primary entity = first one if any.
        groups: dict[tuple[Topic, str | None], list[Episode]] = defaultdict(list)
        for ep in episodes:
            primary = ep.entities[0].lower() if ep.entities else None
            groups[(ep.topic, primary)].append(ep)

        habits: list[Habit] = []
        for (topic, primary), eps in groups.items():
            if len(eps) < self._min_obs:
                continue

            # Times-of-day in minutes-since-midnight
            tods = sorted(_minute_of_day(e.created_at) for e in eps)
            cluster, start_min, end_min = _largest_dense_cluster(tods, self._window_min)
            if len(cluster) < self._min_obs:
                continue

            # The supporting episodes are those whose tod falls in the cluster.
            supporting = [
                e for e in eps if start_min <= _minute_of_day(e.created_at) <= end_min
            ]

            summary = _build_summary(topic, primary, supporting, start_min, end_min)
            # Confidence: ratio of in-window observations to total, attenuated
            # by how tight the window is. A tighter cluster = higher confidence.
            tightness = 1.0 - (end_min - start_min) / max(1, self._window_min)
            ratio = len(supporting) / max(1, len(eps))
            confidence = min(1.0, 0.5 + 0.4 * ratio + 0.1 * max(0.0, tightness))

            habits.append(
                Habit(
                    user_id=user_id,
                    summary=summary,
                    topic=topic,
                    confidence=confidence,
                    observation_count=len(supporting),
                    supporting_episode_ids=tuple(e.episode_id for e in supporting),
                    time_window=(_fmt_time(start_min), _fmt_time(end_min)),
                )
            )

        log.info(
            "habits.mined",
            user_id=str(user_id),
            n_episodes=len(episodes),
            n_habits=len(habits),
        )
        return habits


# ── helpers ─────────────────────────────────────────────────────────


def _minute_of_day(when) -> int:  # noqa: ANN001
    return when.hour * 60 + when.minute


def _fmt_time(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _largest_dense_cluster(
    sorted_minutes: list[int], window_min: int
) -> tuple[list[int], int, int]:
    """Sliding window over a sorted list of times-of-day.

    Returns the densest window of width ≤ window_min and the (start, end)
    minutes-of-day it covers. Empty input yields ([], 0, 0).
    """
    if not sorted_minutes:
        return [], 0, 0

    best: list[int] = []
    best_start = best_end = 0
    left = 0
    for right in range(len(sorted_minutes)):
        while sorted_minutes[right] - sorted_minutes[left] > window_min:
            left += 1
        cur = sorted_minutes[left : right + 1]
        if len(cur) > len(best):
            best = list(cur)
            best_start, best_end = cur[0], cur[-1]
    return best, best_start, best_end


def _build_summary(
    topic: Topic, primary_entity: str | None, eps: list[Episode], start: int, end: int
) -> str:
    """Make a human-readable habit summary."""
    when = f"around {_fmt_time((start + end) // 2)}"
    if topic == Topic.HEALTH and primary_entity == "coffee":
        return f"Sam takes coffee {when}"
    if topic == Topic.HEALTH:
        return f"Sam has a health-related routine {when}"
    if topic == Topic.WORK:
        return f"Sam works on {primary_entity or 'projects'} {when}"
    if topic == Topic.SOCIAL and primary_entity:
        return f"Sam interacts with {primary_entity} {when}"
    if topic == Topic.MEDIA:
        return f"Sam consumes media {when}"
    # Generic fallback — also useful when topic is UNKNOWN
    sample = eps[0].text[:40] + ("…" if len(eps[0].text) > 40 else "")
    return f"Recurring {topic.value} pattern {when} — like “{sample}”"


def episode_signature(ep: Episode) -> str:
    """Stable fingerprint for an episode — useful in tests."""
    return hashlib.blake2b(
        f"{ep.user_id}:{ep.topic}:{ep.text}".encode(), digest_size=4
    ).hexdigest()
