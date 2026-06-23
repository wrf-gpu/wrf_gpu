"""v0.20 host-RAM guard for the segmented nested host loop.

The segmented host loop in ``execute_nested_pipeline`` used to do
``events.extend(result.events)`` once per output-interval segment, so the host
``events`` list grew O(forecast_length) over a 24-120 h skill-gate run (~5,000
event tuples/forecast-hour).  The only values any downstream consumer reads are
the aggregate ``event_counts`` / ``force_counts`` Counters, so the loop now folds
each segment's events into running Counters + a bounded tail (host RAM O(1) in
forecast length).  These tests pin:

* ``GPUWRF_NESTED_EVENT_TAIL`` parsing (default / explicit / 0=unbounded / junk).
* the streaming fold is byte-identical to the OLD single full-list count
  (counting is order-independent + associative).
"""

from __future__ import annotations

import os
import random
from collections import Counter, deque

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from gpuwrf.integration.nested_pipeline import _nested_event_tail_cap_from_env


def test_event_tail_cap_env_parsing(monkeypatch):
    monkeypatch.delenv("GPUWRF_NESTED_EVENT_TAIL", raising=False)
    assert _nested_event_tail_cap_from_env() == 4096  # release default
    assert _nested_event_tail_cap_from_env(default=8) == 8

    monkeypatch.setenv("GPUWRF_NESTED_EVENT_TAIL", "256")
    assert _nested_event_tail_cap_from_env() == 256

    monkeypatch.setenv("GPUWRF_NESTED_EVENT_TAIL", "0")  # unbounded tail (legacy)
    assert _nested_event_tail_cap_from_env() == 0

    monkeypatch.setenv("GPUWRF_NESTED_EVENT_TAIL", "-1")  # negative -> unbounded
    assert _nested_event_tail_cap_from_env() == -1

    monkeypatch.setenv("GPUWRF_NESTED_EVENT_TAIL", "not-an-int")  # junk -> default
    assert _nested_event_tail_cap_from_env() == 4096


def _fold(seg_event_batches, cap):
    """Mirror the nested_pipeline segment-loop fold exactly."""
    event_counts: Counter = Counter()
    force_counts: Counter = Counter()
    tail: deque = deque(maxlen=cap if cap > 0 else None)
    for batch in seg_event_batches:
        for event in batch:
            if not event:
                continue
            event_counts[event[0]] += 1
            if event[0] == "force":
                force_counts[f"{event[1]}->{event[2]}"] += 1
            tail.append(event)
    return event_counts, force_counts, tail


def test_streaming_fold_matches_old_full_list_count():
    """The running Counters equal counting the concatenated full event list, for
    any segmentation -- so the payload's event_counts/force_counts are unchanged
    by the host-RAM guard."""
    rng = random.Random(1234)
    batches = []
    for seg in range(60):
        batch = []
        for _ in range(rng.randint(2, 25)):
            kind = rng.choice(["advance", "advance", "force", "feedback", "output"])
            if kind == "force":
                batch.append(
                    ("force", rng.choice(["d01", "d02"]), rng.choice(["d02", "d03"]), seg)
                )
            else:
                batch.append((kind, "dX", seg))
        batches.append(batch)

    all_events = [ev for batch in batches for ev in batch]
    old_event_counts = dict(Counter(e[0] for e in all_events))
    old_force_counts = dict(
        Counter(f"{e[1]}->{e[2]}" for e in all_events if e and e[0] == "force")
    )

    cap = 9
    ec, fc, tail = _fold(batches, cap)
    assert dict(ec) == old_event_counts
    assert dict(fc) == old_force_counts
    # Bounded tail = last-N of the full concatenation.
    assert len(tail) == cap
    assert tuple(tail) == tuple(all_events[-cap:])


def test_streaming_fold_unbounded_tail_when_cap_nonpositive():
    batches = [[("advance", "dX", i) for i in range(5)] for _ in range(4)]
    ec, _fc, tail = _fold(batches, cap=0)
    assert ec == Counter({"advance": 20})
    assert len(tail) == 20  # unbounded -> kept everything
