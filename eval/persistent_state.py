"""Thread-safe, throttled persistence for the incremental benchmark runner.

``PersistentState`` owns the in-memory list of parent items that the
generation loop writes into, and periodically flushes a stripped copy of
that list to disk so that long runs survive crashes / interruptions.

Kept in its own module so the IOAA runner and any other future benchmark
runner can share the same save-throttling logic without duplicating it.
"""

from __future__ import annotations

import copy
import os
import threading
import time

from judge_utils import write_json


__all__ = ["PersistentState"]


def _strip_internal(item: dict) -> dict:
    """Return a deep copy of ``item`` with runner-internal keys removed."""
    out = copy.deepcopy(item)
    out.pop("_source_index", None)
    return out


class PersistentState:
    """Thread-safe wrapper that periodically flushes the in-memory item list.

    - ``save_async()`` is the cheap call workers make after each unit of
      work; it performs a real disk write at most once every
      ``save_interval_sec`` seconds.
    - ``save()`` always writes (used at shutdown / after joining workers).
    """

    def __init__(
        self,
        items: list[dict],
        output_path: str,
        save_interval_sec: float = 5.0,
    ):
        self.items = items
        self.output_path = output_path
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._save_interval = save_interval_sec

    def save_async(self) -> None:
        now = time.time()
        with self._lock:
            if now - self._last_save < self._save_interval:
                return
            self._last_save = now
        self.save()

    def save(self) -> None:
        # Hold the lock for the entire write so two concurrent saves cannot
        # both stream into ``output_path`` and produce a truncated JSON.
        # We also write to a ``.tmp`` sibling first and ``os.replace`` into
        # place so readers / resumes never observe a half-written file.
        with self._lock:
            data = [_strip_internal(it) for it in self.items]
            self._last_save = time.time()
            tmp_path = f"{self.output_path}.tmp"
            write_json(data, tmp_path)
            os.replace(tmp_path, self.output_path)
