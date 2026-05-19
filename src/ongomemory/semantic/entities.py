"""Entity extraction.

Two functions:
  * extract_entities(text)   → list of strings ("Joonatan", "Tuesday")
  * extract_facts(text)      → list of (key, value, confidence)

Implementation is rule-based. In production these get supplemented by
a small NER model — but rules cover the actual long tail of Ongo's
turn distribution (proper nouns from a known set + a handful of
time / date patterns).

The point of *this file* is the interface. Switching to spaCy or a
fine-tuned distilbert is a one-line change in `EpisodeWriter`.
"""

from __future__ import annotations

import re

# Person names: capitalized words. We rely on the stoplist below to
# filter common false positives like "I", "Ongo", day/month names.
_PERSON_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Day-of-week
_DAY_RE = re.compile(
    r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b", re.IGNORECASE
)

# Time-of-day like "4pm", "16:00", "08:30"
_TIME_RE = re.compile(r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s?[ap]m)\b", re.IGNORECASE)


def extract_entities(text: str) -> list[str]:
    """Pull person-ish + time-ish entities. Deduped, preserves order."""
    seen: set[str] = set()
    out: list[str] = []

    for m in _PERSON_RE.finditer(text):
        word = m.group(0)
        if word.lower() in _STOPLIST:
            continue
        if word not in seen:
            seen.add(word)
            out.append(word)

    for m in _DAY_RE.finditer(text):
        w = m.group(0)
        if w not in seen:
            seen.add(w)
            out.append(w)

    for m in _TIME_RE.finditer(text):
        w = m.group(0)
        if w not in seen:
            seen.add(w)
            out.append(w)

    return out


# ── fact extraction ─────────────────────────────────────────────────


# (regex, fact_key_template, value_group, confidence)
# fact_key_template lets us emit dynamic keys, e.g. "lives_in" vs "works_at".
_FACT_RULES: list[tuple[re.Pattern[str], str, int, float]] = [
    # "I'm Sam" / "I am Sam" / "my name is Sam"
    (re.compile(r"\b(?:i(?:'m| am)|my name is)\s+([A-Z][a-zA-Z]+)\b", re.IGNORECASE), "name", 1, 0.92),
    # "I work on / at X"
    (re.compile(r"\bi\s+work\s+(?:on|at|in)\s+([a-zA-Z][\w\s\-]{2,40})", re.IGNORECASE), "works_on", 1, 0.85),
    # "I live in Paris"
    (re.compile(r"\bi\s+live\s+in\s+([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)", re.IGNORECASE), "lives_in", 1, 0.9),
    # "my favorite X is Y"
    (re.compile(r"\bmy\s+favou?rite\s+(\w+)\s+is\s+([a-zA-Z][\w\s\-]{1,40})", re.IGNORECASE),
     "favorite_{0}", 2, 0.82),
]


def extract_facts(text: str) -> list[tuple[str, str, float]]:
    """Pull structured facts.

    Returns (key, value, confidence) tuples. Multiple rules can fire;
    the caller (EpisodeWriter) writes each to the store, where the
    last-write-wins resolution kicks in.
    """
    out: list[tuple[str, str, float]] = []
    for pattern, key_tpl, value_group, conf in _FACT_RULES:
        for match in pattern.finditer(text):
            value = match.group(value_group).strip().rstrip(".,!?")
            if not value:
                continue
            # Some rules have format placeholders in their key template.
            if "{" in key_tpl:
                key = key_tpl.format(match.group(1).lower())
            else:
                key = key_tpl
            # Sanitize key to match the Fact.key validator.
            key = re.sub(r"[^a-z0-9_]", "_", key.lower())
            if not re.match(r"^[a-z_]", key):
                continue
            out.append((key, value, conf))
    return out


# Common capitalized words that aren't names.
_STOPLIST = frozenset({
    "i", "i'm", "im", "ill", "ive", "we", "we're",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "ongo", "hi", "hello", "hey", "ok", "okay", "yes", "no", "yeah",
})
