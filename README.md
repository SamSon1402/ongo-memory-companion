# OngoMemory-Companion

> Living memory layer for the Ongo companion robot.
> Built for InteractionLabs by Sameer M. — Paris, May 2026.

A local-first, privacy-respecting memory system that turns Ongo from
*"a desk lamp that listens"* into *"a desk lamp that remembers you."*
Face & voice recognition tied to a per-user store of episodes,
inferred habits, and a fast recall API.

```
    speak / observe ─► Episode(user=f7a2, topic=…, text=…, vec=…)
                              │
                              ▼
            ┌──────────────────────────────────┐
            │     SQLite (facts + episodes)    │
            │     Qdrant local (vectors)       │
            └──────────────────────────────────┘
                              │
              recall("what was I doing yesterday?")
                              │
                              ▼
                     ranked Episode[] in < 30ms
                              │
              + HabitMiner ─► Habit("coffee ~08:50", conf=0.84)
```

---

## Why this exists

> *"Ambient presence, contextual awareness, **living memory**."*
> — ongolamp.com

Ongo's whole value proposition is memory. Cloud-only memory would be
unacceptable on privacy grounds, and round-tripping recall through a
cloud LLM blows the conversational TTFT budget. So memory has to live
on the device, recall has to be fast, and the API has to be boring
enough that the rest of the system can use it from day one.

---

## What's in this drop

A walking skeleton with the data layer, recall pipeline, habit miner,
and HTTP surface fully implemented end-to-end against an in-memory
store. The Qdrant adapter is real; the sentence-transformer embedder
is scaffolded behind a Protocol so tests run without weights.

```
src/ongomemory/
  core/        # domain types + protocols (Episode, Habit, Identity)
  embed/       # MiniLM embedder + DeterministicEmbedder for tests
  store/       # SQLite + in-memory store + qdrant adapter
  episodic/    # write path: episode validation, dedup, summarization
  semantic/    # entity extraction & resolution
  habits/      # the habit miner (the interesting part)
  recall/      # multi-stage ranker: vec → recency → entity-boost
  identity/    # face_id / voice_id resolver
  api/         # FastAPI surface
```

---

## Quick start

```bash
make install      # pip install -e ".[dev]"
make test         # pytest -q  → 40 tests
make api          # uvicorn ongomemory.api:app  (port 8002)
make demo         # play 3 days through the CLI, watch habits emerge
```

A single recall query end-to-end:

```bash
ongomemory recall --user f7a2 "what was I working on yesterday afternoon?"
# → 3 episodes ranked: Joonatan call (0.91), prep notes (0.74), gym (0.41)
```

---

## Design notes

**Episodes are immutable.** You don't edit memory; you append. A wrong
fact is overwritten by a newer, more confident episode at recall time.

**Two stores, one API.** SQLite holds the structured side (facts,
entities, identity); Qdrant holds the vector side. The `MemoryStore`
Protocol hides both behind one set of methods so callers don't care.

**Recall is multi-stage.** Pure cosine similarity is wrong for
conversational memory — recency and entity overlap matter as much as
semantic distance. The `Ranker` blends three signals with calibrated
weights; the weights are config, not magic constants.

**Habits emerge from episodes.** The miner runs over the last N days
and surfaces patterns: same topic + same time-of-day + ≥3 observations
→ candidate habit. Habits are *suggestions*, not facts — the user (or
the LLM acting on their behalf) confirms or dismisses.

**Privacy is the design constraint.** Memory never leaves the device
unless the user explicitly exports it. The Qdrant instance is local;
embeddings never round-trip to a vendor.

---

## Status

| Area | State |
|---|---|
| Domain types & protocols | ✅ done |
| In-memory store | ✅ done |
| SQLite store | ✅ done |
| Qdrant adapter | 🟡 scaffolded (needs running Qdrant) |
| MiniLM embedder | 🟡 scaffolded (needs sentence-transformers) |
| DeterministicEmbedder for tests | ✅ done |
| Episode write path + dedup | ✅ done |
| Entity extraction (rules) | ✅ done |
| Recall ranker | ✅ done |
| Habit miner | ✅ done |
| Identity resolver (face/voice) | ✅ done |
| FastAPI surface | ✅ done |
| Memory API library import | ✅ done |

🟡 items are infra-dependent; the seams are correct.

---

## The library API

The rest of the Ongo system uses one class:

```python
from ongomemory import Memory

mem = Memory.for_user("f7a2")

# write
mem.episode(topic="calendar", text="call with Joonatan 4pm")
mem.fact(key="name", value="Sam")

# read
hits = mem.recall("what was I working on", k=5)
for hit in hits:
    print(hit.episode.text, hit.score)

# habits
for habit in mem.habits():
    print(habit.summary, habit.confidence)
```

---

— Sameer M. · samson1402.github.io · sameer@…
