# expect: WA
# The cache is keyed by the name as passed to resolve, so other spellings of the same name miss.
import threading
from concurrent.futures import Future


class DNSResolutionError(Exception):
    """Base class for every error resolve can raise."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class NameNotFoundError(DNSResolutionError):
    """name has no record anywhere it was looked up."""


class ARecord:
    def __init__(self, addresses: list[str], ttl: int) -> None:
        self.addresses = list(addresses)
        self.ttl = ttl


class CNAMERecord:
    def __init__(self, target: str, ttl: int) -> None:
        self.target = target
        self.ttl = ttl


def _normalize(name: str) -> str:
    return name.lower().rstrip(".") + "."


def _normalized_zone(zone):
    out = {}
    for name, record in zone.items():
        out[_normalize(name)] = record
    return out


class Resolver:
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        self._zone = _normalized_zone(zone)
        self._fallback_zone = _normalized_zone(fallback_zone) if fallback_zone is not None else None
        self.max_chain_length = max_chain_length

    def resolve(self, name: str) -> list[str]:
        current = _normalize(name)
        record = self._zone.get(current)
        if record is None:
            raise NameNotFoundError(current)
        return list(record.addresses)


class ResolutionCycleError(DNSResolutionError):
    """a step's target is a name already visited earlier in this resolution."""


class ChainTooLongError(DNSResolutionError):
    """reaching an A record needed more than max_chain_length CNAME hops."""


class Resolver(Resolver):
    def _follow_chain(self, zone: dict, name: str) -> tuple[list[str], int]:
        origin = _normalize(name)
        current = origin
        visited = {current}
        ttl = None
        hops = 0
        while True:
            record = zone.get(current)
            if record is None:
                raise NameNotFoundError(current)
            ttl = record.ttl if ttl is None else min(ttl, record.ttl)
            if isinstance(record, ARecord):
                return list(record.addresses), ttl
            target = _normalize(record.target)
            if target in visited:
                raise ResolutionCycleError(target)
            hops += 1
            if hops > self.max_chain_length:
                raise ChainTooLongError(origin)
            visited.add(target)
            current = target

    def resolve(self, name: str) -> list[str]:
        addresses, _ttl = self._follow_chain(self._zone, name)
        return addresses


class Resolver(Resolver):
    def resolve(self, name: str) -> list[str]:
        try:
            addresses, _ttl = self._follow_chain(self._zone, name)
            return addresses
        except NameNotFoundError:
            if self._fallback_zone is None:
                raise
            addresses, _ttl = self._follow_chain(self._fallback_zone, name)
            return addresses


class _CacheEntry:
    __slots__ = ("addresses", "expires_at")

    def __init__(self, addresses: list[str], expires_at: int) -> None:
        self.addresses = addresses
        self.expires_at = expires_at


class Resolver(Resolver):
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        super().__init__(zone, fallback_zone, max_chain_length=max_chain_length)
        self._now = 0
        self._cache: dict[str, _CacheEntry] = {}

    def advance_clock(self, seconds: int) -> None:
        self._now += seconds

    def _resolve_uncached(self, name: str) -> tuple[list[str], int]:
        try:
            return self._follow_chain(self._zone, name)
        except NameNotFoundError:
            if self._fallback_zone is None:
                raise
            return self._follow_chain(self._fallback_zone, name)

    def resolve(self, name: str) -> list[str]:
        key = _normalize(name)
        entry = self._cache.get(key)
        if entry is not None and self._now < entry.expires_at:
            return list(entry.addresses)
        addresses, ttl = self._resolve_uncached(key)
        self._cache[key] = _CacheEntry(list(addresses), self._now + ttl)
        return list(addresses)


class Resolver(Resolver):
    def __init__(self, zone, fallback_zone=None, *, max_chain_length=8):
        super().__init__(zone, fallback_zone, max_chain_length=max_chain_length)
        self._lock = threading.Lock()
        self._in_flight: dict[str, Future] = {}

    def advance_clock(self, seconds: int) -> None:
        with self._lock:
            self._now += seconds

    def resolve(self, name: str) -> list[str]:
        key = _normalize(name)
        with self._lock:
            entry = self._cache.get(name)
            if entry is not None and self._now < entry.expires_at:
                return list(entry.addresses)
            future = self._in_flight.get(key)
            if future is None:
                future = self._in_flight[key] = Future()
                leader = True
            else:
                leader = False
        if not leader:
            return list(future.result())
        try:
            addresses, ttl = self._resolve_uncached(key)
        except Exception as exc:
            with self._lock:
                del self._in_flight[key]
            future.set_exception(exc)
            raise
        with self._lock:
            self._cache[name] = _CacheEntry(list(addresses), self._now + ttl)
            del self._in_flight[key]
        future.set_result(list(addresses))
        return list(addresses)
