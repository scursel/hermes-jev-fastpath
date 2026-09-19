"""Turn-scoped decision cache keyed by session ID, turn ID, and text hash.

A decision is reusable only when all three identifiers match and the entry is younger
than the TTL; entries are bounded and every operation is lock-guarded.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from .types import Decision


@dataclass(frozen=True)
class CacheKey:
    session_id: str
    turn_id: str
    text_hash: str


class DecisionCache:
    """Bounded LRU + TTL cache of typed :class:`Decision` objects."""

    def __init__(self, ttl_seconds: float = 900.0, max_entries: int = 1024, monotonic=time.monotonic):
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._entries: OrderedDict[CacheKey, tuple[float, Decision]] = OrderedDict()

    @staticmethod
    def key(session_id: str, turn_id: str, text: str) -> CacheKey:
        digest = hashlib.sha256(str(text or "").encode("utf-8", "replace")).hexdigest()
        return CacheKey(str(session_id or ""), str(turn_id or ""), digest)

    def _prune(self, now: float) -> None:
        expired = [key for key, (stamp, _decision) in self._entries.items() if now - stamp >= self._ttl]
        for key in expired:
            self._entries.pop(key, None)

    def get(self, key: CacheKey) -> Decision | None:
        if not key.session_id or not key.turn_id:
            return None
        now = self._monotonic()
        with self._lock:
            self._prune(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            stamp, decision = entry
            if now - stamp >= self._ttl:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return decision

    def put(self, key: CacheKey, decision: Decision) -> None:
        if not key.session_id or not key.turn_id:
            return
        now = self._monotonic()
        with self._lock:
            self._prune(now)
            self._entries.pop(key, None)
            self._entries[key] = (now, decision)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
