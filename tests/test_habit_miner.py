"""Habit miner contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ongomemory.core import Episode, Topic, UserId
from ongomemory.habits import TimeOfDayHabitMiner


def _ep(text: str, days_ago: int, hour: int, minute: int, topic: Topic, entities=()) -> Episode:
    when = datetime.now(UTC) - timedelta(days=days_ago)
    when = when.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return Episode(
        episode_id=f"id-d{days_ago}-{hour}-{minute}",
        user_id=UserId("u1"),
        topic=topic,
        text=text,
        entities=entities,
        created_at=when,
    )


def test_returns_no_habits_below_threshold() -> None:
    miner = TimeOfDayHabitMiner(min_observations=3)
    eps = [_ep("coffee", 0, 8, 50, Topic.HEALTH, entities=("coffee",))]
    assert miner.mine(eps, UserId("u1")) == []


def test_min_observations_validated() -> None:
    with pytest.raises(ValueError):
        TimeOfDayHabitMiner(min_observations=1)


def test_finds_tight_time_cluster() -> None:
    miner = TimeOfDayHabitMiner(min_observations=3, window_minutes=90)
    eps = [
        _ep("coffee", 2, 8, 49, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 1, 8, 51, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 0, 8, 53, Topic.HEALTH, entities=("coffee",)),
    ]
    habits = miner.mine(eps, UserId("u1"))
    assert len(habits) == 1
    h = habits[0]
    assert h.topic == Topic.HEALTH
    assert h.observation_count == 3
    assert h.confidence >= 0.7
    assert h.time_window is not None


def test_ignores_unclustered_observations() -> None:
    """3 episodes spread across the day shouldn't surface a habit."""
    miner = TimeOfDayHabitMiner(min_observations=3, window_minutes=90)
    eps = [
        _ep("coffee", 2, 8, 0, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 1, 14, 0, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 0, 20, 0, Topic.HEALTH, entities=("coffee",)),
    ]
    assert miner.mine(eps, UserId("u1")) == []


def test_separates_morning_and_evening_routines() -> None:
    """If the user has both, we should produce TWO habits, not one wide blob."""
    miner = TimeOfDayHabitMiner(min_observations=3, window_minutes=60)
    eps = [
        # Morning cluster
        _ep("coffee", 2, 8, 50, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 1, 8, 49, Topic.HEALTH, entities=("coffee",)),
        _ep("coffee", 0, 8, 51, Topic.HEALTH, entities=("coffee",)),
        # Evening cluster (different topic + entity)
        _ep("gym", 2, 18, 0, Topic.WORK, entities=("gym",)),
        _ep("gym", 1, 18, 5, Topic.WORK, entities=("gym",)),
        _ep("gym", 0, 17, 58, Topic.WORK, entities=("gym",)),
    ]
    habits = miner.mine(eps, UserId("u1"))
    assert len(habits) == 2
    summaries = " ".join(h.summary.lower() for h in habits)
    assert "coffee" in summaries
