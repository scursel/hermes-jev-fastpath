"""Turn-identity TTL cache: key isolation, expiry, bounding, and thread safety."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

from jev_fastpath.cache import CacheKey, DecisionCache
from jev_fastpath.types import Decision


def _decision(handler="calculator", confidence=0.97):
    return Decision(
        handler_id=handler, confidence=confidence, short_circuit_probability=0.95, latency_ms=10,
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_key_hashes_text_and_keeps_identity():
    key = DecisionCache.key("s1", "t1", "2 + 2")
    assert key.session_id == "s1"
    assert key.turn_id == "t1"
    assert key.text_hash == hashlib.sha256(b"2 + 2").hexdigest()


def test_identity_and_text_isolate_entries():
    a = DecisionCache.key("s1", "t1", "2 + 2")
    assert DecisionCache.key("s1", "t1", "2 + 2") == a
    assert DecisionCache.key("s2", "t1", "2 + 2") != a
    assert DecisionCache.key("s1", "t2", "2 + 2") != a
    assert DecisionCache.key("s1", "t1", "3 + 3") != a


def test_put_then_get_roundtrip():
    cache = DecisionCache()
    key = DecisionCache.key("s1", "t1", "2 + 2")
    cache.put(key, _decision())
    assert cache.get(key) == _decision()


def test_get_miss_returns_none():
    cache = DecisionCache()
    key = DecisionCache.key("s1", "t1", "2 + 2")
    assert cache.get(key) is None


def test_empty_identity_is_not_cacheable():
    cache = DecisionCache()
    empty_session = DecisionCache.key("", "t1", "2 + 2")
    empty_turn = DecisionCache.key("s1", "", "2 + 2")
    cache.put(empty_session, _decision())
    cache.put(empty_turn, _decision())
    assert cache.get(empty_session) is None
    assert cache.get(empty_turn) is None


def test_expiry_at_ttl(monkeypatch=None):
    clock = FakeClock()
    cache = DecisionCache(ttl_seconds=900.0, monotonic=clock)
    key = DecisionCache.key("s1", "t1", "2 + 2")
    cache.put(key, _decision())
    clock.now = 899.9
    assert cache.get(key) is not None
    clock.now = 900.0
    assert cache.get(key) is None
    clock.now = 900.1
    assert cache.get(key) is None


def test_expired_entries_are_pruned():
    clock = FakeClock()
    cache = DecisionCache(ttl_seconds=100.0, monotonic=clock)
    for turn in range(5):
        cache.put(DecisionCache.key("s1", f"t{turn}", "text"), _decision())
    clock.now = 100.0
    cache.put(DecisionCache.key("s1", "fresh", "text"), _decision())
    for turn in range(5):
        assert cache.get(DecisionCache.key("s1", f"t{turn}", "text")) is None


def test_put_replaces_exact_key():
    clock = FakeClock()
    cache = DecisionCache(ttl_seconds=100.0, monotonic=clock)
    key = DecisionCache.key("s1", "t1", "2 + 2")
    cache.put(key, _decision(handler="calculator"))
    clock.now = 10.0
    cache.put(key, _decision(handler="clock"))
    assert cache.get(key).handler_id == "clock"


def test_cache_is_bounded_and_evicts_oldest():
    clock = FakeClock()
    cache = DecisionCache(max_entries=3, monotonic=clock)
    keys = [DecisionCache.key("s1", f"t{turn}", "text") for turn in range(4)]
    for key in keys:
        cache.put(key, _decision())
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None
    assert cache.get(keys[3]) is not None


def test_hit_promotes_against_eviction():
    clock = FakeClock()
    cache = DecisionCache(max_entries=2, monotonic=clock)
    a = DecisionCache.key("s1", "ta", "text")
    b = DecisionCache.key("s1", "tb", "text")
    c = DecisionCache.key("s1", "tc", "text")
    cache.put(a, _decision())
    cache.put(b, _decision())
    assert cache.get(a) is not None  # promote a above b
    cache.put(c, _decision())
    assert cache.get(a) is not None
    assert cache.get(b) is None
    assert cache.get(c) is not None


def test_concurrent_get_put_stays_consistent():
    cache = DecisionCache(max_entries=64)
    keys = [DecisionCache.key(f"s{turn}", "t1", "text") for turn in range(64)]

    def _work(turn):
        key = keys[turn]
        cache.put(key, _decision(confidence=turn / 100))
        return cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_work, range(64)))
    assert all(result is not None for result in results)
