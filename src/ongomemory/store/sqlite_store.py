"""SQLite store — production default for the structured side.

Schema:
  episodes(episode_id PK, user_id, topic, text, entities_json, confidence,
           created_at, vec_blob)
  facts(user_id, key, value, confidence, source_episode_id, created_at)
  habits(habit_id PK, user_id, summary, topic, confidence, count,
         supporting_ids_json, window_start, window_end, inferred_at)
  identities(user_id PK, face_id, voice_id, display_name, created_at)

Vector search here is brute-force over a BLOB column — fine up to a
few thousand episodes per user, which is months of normal use. For
multi-user installs or a multi-tenant edge deployment we delegate
vector search to `QdrantAdapter` and use SQLite only for the structured
side.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ongomemory.core import (
    Episode,
    Fact,
    Habit,
    Identity,
    Topic,
    UserId,
)
from ongomemory.core.errors import StoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    topic         TEXT NOT NULL,
    text          TEXT NOT NULL,
    entities_json TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL,
    created_at    TEXT NOT NULL,
    vec_blob      BLOB
);
CREATE INDEX IF NOT EXISTS idx_episodes_user_created ON episodes(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_user_topic ON episodes(user_id, topic);

CREATE TABLE IF NOT EXISTS facts (
    user_id            TEXT NOT NULL,
    key                TEXT NOT NULL,
    value              TEXT NOT NULL,
    confidence         REAL NOT NULL,
    source_episode_id  TEXT,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (user_id, key, created_at)
);
CREATE INDEX IF NOT EXISTS idx_facts_user_key ON facts(user_id, key, created_at DESC);

CREATE TABLE IF NOT EXISTS habits (
    habit_id            TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    summary             TEXT NOT NULL,
    topic               TEXT NOT NULL,
    confidence          REAL NOT NULL,
    observation_count   INTEGER NOT NULL,
    supporting_ids_json TEXT NOT NULL,
    window_start        TEXT,
    window_end          TEXT,
    inferred_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_habits_user ON habits(user_id, confidence DESC);

CREATE TABLE IF NOT EXISTS identities (
    user_id      TEXT PRIMARY KEY,
    face_id      TEXT,
    voice_id     TEXT,
    display_name TEXT,
    created_at   TEXT NOT NULL
);
"""


class SqliteStore:
    """SQLite-backed MemoryStore."""

    def __init__(self, *, db_path: Path | str = ":memory:") -> None:
        self._db_path = str(db_path)
        # check_same_thread=False so the FastAPI worker pool can share us;
        # the RLock below ensures we serialize properly.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── episodes ────────────────────────────────────────────────────

    def write_episode(self, episode: Episode) -> None:
        vec_blob = (
            episode.vec.tobytes() if episode.vec is not None else None
        )
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO episodes "
                    "(episode_id, user_id, topic, text, entities_json, confidence, "
                    " created_at, vec_blob) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        episode.episode_id,
                        str(episode.user_id),
                        episode.topic.value,
                        episode.text,
                        json.dumps(list(episode.entities)),
                        episode.confidence,
                        episode.created_at.isoformat(),
                        vec_blob,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"write_episode failed: {exc}") from exc

    def episodes_for(
        self,
        user_id: UserId,
        *,
        topics: Iterable[Topic] | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        sql = (
            "SELECT episode_id, user_id, topic, text, entities_json, confidence, "
            "created_at, vec_blob FROM episodes WHERE user_id = ?"
        )
        params: list[object] = [str(user_id)]
        if topics is not None:
            topic_list = [t.value for t in topics]
            if not topic_list:
                return []
            placeholders = ",".join("?" * len(topic_list))
            sql += f" AND topic IN ({placeholders})"
            params.extend(topic_list)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            return [_row_to_episode(row) for row in cur.fetchall()]

    def vector_search(
        self, user_id: UserId, vec: np.ndarray, k: int
    ) -> list[tuple[Episode, float]]:
        if vec.ndim != 1 or vec.dtype != np.float32:
            raise StoreError(f"query vec must be 1-D float32, got {vec.shape}/{vec.dtype}")
        with self._lock:
            cur = self._conn.execute(
                "SELECT episode_id, user_id, topic, text, entities_json, confidence, "
                "created_at, vec_blob FROM episodes "
                "WHERE user_id = ? AND vec_blob IS NOT NULL",
                (str(user_id),),
            )
            rows = cur.fetchall()

        if not rows:
            return []

        episodes = [_row_to_episode(r) for r in rows]
        mat = np.stack([e.vec for e in episodes], axis=0)  # type: ignore[misc]
        sims = mat @ vec
        idx = np.argsort(-sims)[:k]
        return [(episodes[i], float(sims[i])) for i in idx]

    # ── facts ───────────────────────────────────────────────────────

    def write_fact(self, fact: Fact) -> None:
        # Same protection as in-memory: don't let a shaky fact overwrite a strong one.
        with self._lock:
            cur = self._conn.execute(
                "SELECT confidence FROM facts WHERE user_id=? AND key=? "
                "ORDER BY created_at DESC LIMIT 1",
                (str(fact.user_id), fact.key),
            )
            row = cur.fetchone()
            if row and fact.confidence < row[0] * 0.8:
                return
            self._conn.execute(
                "INSERT INTO facts (user_id, key, value, confidence, source_episode_id, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (
                    str(fact.user_id),
                    fact.key,
                    fact.value,
                    fact.confidence,
                    fact.source_episode_id,
                    fact.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def latest_facts(self, user_id: UserId) -> dict[str, Fact]:
        from datetime import datetime as _dt
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, value, confidence, source_episode_id, created_at "
                "FROM facts WHERE user_id=? "
                "GROUP BY key HAVING MAX(created_at)",
                (str(user_id),),
            )
            rows = cur.fetchall()
        return {
            row[0]: Fact(
                user_id=user_id,
                key=row[0],
                value=row[1],
                confidence=row[2],
                source_episode_id=row[3],
                created_at=_dt.fromisoformat(row[4]),
            )
            for row in rows
        }

    # ── habits ──────────────────────────────────────────────────────

    def write_habit(self, habit: Habit) -> None:
        habit_id = f"{habit.user_id}:{habit.summary}"[:128]
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO habits "
                "(habit_id, user_id, summary, topic, confidence, observation_count, "
                " supporting_ids_json, window_start, window_end, inferred_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    habit_id,
                    str(habit.user_id),
                    habit.summary,
                    habit.topic.value,
                    habit.confidence,
                    habit.observation_count,
                    json.dumps(list(habit.supporting_episode_ids)),
                    habit.time_window[0] if habit.time_window else None,
                    habit.time_window[1] if habit.time_window else None,
                    habit.inferred_at.isoformat(),
                ),
            )
            self._conn.commit()

    def habits_for(self, user_id: UserId) -> list[Habit]:
        from datetime import datetime as _dt
        with self._lock:
            cur = self._conn.execute(
                "SELECT summary, topic, confidence, observation_count, "
                "supporting_ids_json, window_start, window_end, inferred_at "
                "FROM habits WHERE user_id=? ORDER BY confidence DESC",
                (str(user_id),),
            )
            rows = cur.fetchall()
        return [
            Habit(
                user_id=user_id,
                summary=r[0],
                topic=Topic(r[1]),
                confidence=r[2],
                observation_count=r[3],
                supporting_episode_ids=tuple(json.loads(r[4])),
                time_window=(r[5], r[6]) if r[5] and r[6] else None,
                inferred_at=_dt.fromisoformat(r[7]),
            )
            for r in rows
        ]

    # ── identity ────────────────────────────────────────────────────

    def write_identity(self, identity: Identity) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO identities "
                "(user_id, face_id, voice_id, display_name, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    str(identity.user_id),
                    identity.face_id,
                    identity.voice_id,
                    identity.display_name,
                    identity.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def identity_for(self, user_id: UserId) -> Identity | None:
        from datetime import datetime as _dt
        with self._lock:
            cur = self._conn.execute(
                "SELECT face_id, voice_id, display_name, created_at "
                "FROM identities WHERE user_id=?",
                (str(user_id),),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Identity(
            user_id=user_id,
            face_id=row[0],
            voice_id=row[1],
            display_name=row[2],
            created_at=_dt.fromisoformat(row[3]),
        )


def _row_to_episode(row: tuple) -> Episode:
    from datetime import datetime as _dt
    (eid, uid, topic, text, entities_json, conf, created_at, vec_blob) = row
    vec = (
        np.frombuffer(vec_blob, dtype=np.float32).copy() if vec_blob else None
    )
    return Episode(
        episode_id=eid,
        user_id=UserId(uid),
        topic=Topic(topic),
        text=text,
        entities=tuple(json.loads(entities_json)),
        confidence=conf,
        created_at=_dt.fromisoformat(created_at),
        vec=vec,
    )
