from __future__ import annotations
from collections import deque
from typing import Any, Dict, List

# Keep the last 50 entries
_BUF: deque = deque(maxlen=50)

def record(entry: Dict[str, Any]) -> None:
    """
    Append a compact, JSON-serializable snapshot.
    Callers should avoid secrets and truncate large blobs before passing them.
    """
    try:
        _BUF.append(entry)
    except Exception:
        # Never break functional code because of debug logging
        pass

def recent(n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the most recent n snapshots, newest last.
    """
    n = max(1, min(int(n or 5), len(_BUF)))
    return list(_BUF)[-n:]
