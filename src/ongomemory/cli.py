"""Command line interface."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ongomemory import Memory
from ongomemory.core import Topic

app = typer.Typer(name="ongomemory", no_args_is_help=True)
console = Console()


@app.command()
def recall(
    query: str = typer.Argument(...),
    user: str = typer.Option("f7a2", "--user", "-u"),
    k: int = typer.Option(5, "--k", "-k"),
) -> None:
    """Run a recall query against an existing in-memory store."""
    mem = Memory.for_user(user)
    # No prior writes, so this will be empty unless invoked from `demo`.
    hits = mem.recall(query, k=k)
    if not hits:
        console.print("[yellow]no episodes in memory for this user yet[/yellow]")
        console.print("hint: run `ongomemory demo` first")
        return
    _print_hits(query, hits)


@app.command()
def demo(
    user: str = typer.Option("f7a2", "--user", "-u"),
) -> None:
    """Run the scripted three-day demo. Builds memory, mines habits, runs recall."""
    console.print(Panel.fit("OngoMemory · 3-day demo · user [cyan]f7a2[/cyan]", style="bold"))

    mem = Memory.for_user(user)
    # Anchor at 00:00 UTC three days ago so timeltamps line up with the
    # intended hour-of-day (otherwise we'd be offset by datetime.now's clock).
    base = (datetime.now(UTC) - timedelta(days=3)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    days = [
        # (day_offset, hour, minute, topic, text)
        (0, 9, 14, Topic.PROFILE, "Hey, I'm Sam."),
        (0, 9, 16, Topic.WORK, "I work on machine learning."),
        (0, 14, 32, Topic.CALENDAR, "Call with Joonatan at 4pm."),
        (0, 18, 1, Topic.HOME, "Goodnight Ongo."),

        # Day 1: morning Coffee + evening Gym
        (1, 8, 49, Topic.HEALTH, "Coffee, the usual."),
        (1, 8, 50, Topic.SOCIAL, "Good morning."),
        (1, 14, 0, Topic.CALENDAR, "Had the call with Joonatan yesterday at 4pm."),
        (1, 17, 45, Topic.HEALTH, "Gym — calisthenics again."),

        # Day 2: more Coffee + Gym at the same hours → habits emerge
        (2, 8, 51, Topic.HEALTH, "Coffee in hand."),
        (2, 10, 22, Topic.WORK, "Prepping for the InteractionLabs interview."),
        (2, 15, 30, Topic.WORK, "Two hours of focus, feeling drained."),
        (2, 17, 58, Topic.HEALTH, "Gym before dinner."),

        # Day 3: third Coffee + third Gym → habit miner threshold met
        (3, 8, 50, Topic.HEALTH, "Coffee to start the day."),
        (3, 18, 5, Topic.HEALTH, "Gym again, calisthenics."),
    ]

    for day, h, m, topic, text in days:
        when = base + timedelta(days=day, hours=h, minutes=m)
        mem.episode(text=text, topic=topic, when=when)

    console.print(f"[green]✓[/green] wrote {len(days)} episodes\n")

    # Facts that emerged from the text
    facts = mem.facts()
    if facts:
        t = Table(title="Extracted facts", show_header=True)
        t.add_column("key", style="cyan")
        t.add_column("value")
        for k, v in facts.items():
            t.add_row(k, v)
        console.print(t)
        console.print()

    # Habits
    habits = mem.habits(refresh=True)
    if habits:
        t = Table(title="Inferred habits", show_header=True)
        t.add_column("summary", style="magenta")
        t.add_column("conf", justify="right")
        t.add_column("obs", justify="right")
        t.add_column("window", style="dim")
        for h in habits:
            window = " — ".join(h.time_window) if h.time_window else "—"
            t.add_row(h.summary, f"{h.confidence:.2f}", str(h.observation_count), window)
        console.print(t)
        console.print()
    else:
        console.print("[dim]no habits surfaced (need more episodes)[/dim]\n")

    # Recall demo
    query = "what was I working on yesterday?"
    hits = mem.recall(query, k=4)
    _print_hits(query, hits)


def _print_hits(query: str, hits) -> None:  # noqa: ANN001
    t = Table(title=f'Recall: "{query}"', show_header=True)
    t.add_column("score", justify="right", style="green")
    t.add_column("vec", justify="right", style="dim")
    t.add_column("rec", justify="right", style="dim")
    t.add_column("ent", justify="right", style="dim")
    t.add_column("topic")
    t.add_column("text", style="cyan")
    for h in hits:
        t.add_row(
            f"{h.score:.3f}",
            f"{h.vec_similarity:.2f}",
            f"{h.recency_score:.2f}",
            f"{h.entity_overlap:.2f}",
            h.episode.topic.value,
            h.episode.text,
        )
    console.print(t)


if __name__ == "__main__":
    app()
