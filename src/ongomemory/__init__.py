"""ongomemory — living memory layer for the Ongo companion robot.

Public API:

    from ongomemory import Memory

    mem = Memory.for_user("f7a2")
    mem.episode(topic="calendar", text="call w/ Joonatan 4pm")
    hits = mem.recall("what was I working on", k=5)

The `Memory` class is a thin facade over the store, writer, ranker,
and habit miner. Most code shouldn't import anything else from this
package.
"""

from ongomemory.facade import Memory

__version__ = "0.1.0"
__all__ = ["Memory"]
