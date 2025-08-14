# backend/app/utils/debug_buffer.py
from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List

# Simple in-memory ring buffer for recent provider I/O
_MAX = 200  # keep last 200 entries
_buf: Deque[Dict[str, Any]] = deque(maxlen=_MAX)
_lock = Lock()


def push(item: Dict[str, Any]) -> None:
    """
    Append a debug item to the ring buffer.
    The item can be any dict (raw text, parsed json, errors, etc).
    """
    with _lock:
        _buf.appendleft(
            {
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                **item,
            }
        )


def recent(n: int = 5) -> List[Dict[str, Any]]:
    """Return the most recent n items (default 5)."""
    if n <= 0:
        return []
    with _lock:
        return list(list(_buf)[: min(n, len(_buf))])
