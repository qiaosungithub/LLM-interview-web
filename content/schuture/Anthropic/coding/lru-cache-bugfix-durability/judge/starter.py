from collections import OrderedDict


class Memoize:
    """Wraps func so that repeated calls with the same arguments return a cached result instead of
    calling func again. Holds at most capacity entries; a call that would add a new one beyond that
    evicts the least recently used entry first. Both a hit and a miss count as a "use" of the key
    they touch, for the purpose of deciding which entry is least recently used."""

    def __init__(self, func, capacity):
        self.func = func
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def generate_key(self, *args, **kwargs):
        key = args + tuple(kwargs.items())
        if len(key) == 1:
            return key[0]
        return key

    def __call__(self, *args, **kwargs):
        key = self.generate_key(*args, **kwargs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        result = self.func(*args, **kwargs)
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


class DurableMemoize(Memoize):
    def __init__(self, func, capacity, path: str | None = None) -> None:
        """path is the base path this cache persists to and recovers from; None (the default)
        disables persistence entirely. Recovery, if path names an existing store, happens here,
        before __init__ returns."""
        raise NotImplementedError

    def compact(self) -> None:
        """Rewrites the stored state so that its size no longer depends on call history, only on
        the entries currently cached. Safe to call at any time, including right before a crash."""
        raise NotImplementedError


class ThreadSafeMemoize(DurableMemoize):
    def __init__(self, func, capacity, path: str | None = None) -> None:
        """Same contract as DurableMemoize.__init__."""
        raise NotImplementedError
