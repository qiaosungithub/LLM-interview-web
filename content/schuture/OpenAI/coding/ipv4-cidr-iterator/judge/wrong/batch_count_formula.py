# expect: WA
# next_batch counts the remaining addresses as (distance + 1) // step instead of distance // step + 1.
import re
from itertools import islice

_SEG = r"(0|[1-9][0-9]{0,2})"
_FORM = re.compile(rf"{_SEG}\.{_SEG}\.{_SEG}\.{_SEG}(?:/(0|[1-9][0-9]?))?")


def _dotted(value):
    return ".".join(str(b) for b in value.to_bytes(4, "big"))


class IPV4Iterator:
    def __init__(self, ip_or_cidr, reverse=False, step=1):
        m = _FORM.fullmatch(ip_or_cidr)
        if m is None:
            raise ValueError(f"not an address or CIDR block: {ip_or_cidr!r}")
        octets = [int(g) for g in m.groups()[:4]]
        if max(octets) > 255:
            raise ValueError("segment above 255")
        addr = int.from_bytes(bytes(octets), "big")
        if m.group(5) is None:
            lo, hi = 0, 2 ** 32 - 1
        else:
            plen = int(m.group(5))
            if plen > 32:
                raise ValueError("prefix above 32")
            span = 1 << (32 - plen)
            lo = addr // span * span
            hi = lo + span - 1
        if step <= 0:
            raise ValueError("step must be positive")
        self._walk = self._addresses(addr, lo, hi, -step if reverse else step)
        self._cur, self._lo, self._hi, self._step, self._rev = addr, lo, hi, step, reverse

    @staticmethod
    def _addresses(cur, lo, hi, delta):
        while lo <= cur <= hi:
            yield _dotted(cur)
            cur += delta

    def __iter__(self):
        return self

    def __next__(self):
        if not self._lo <= self._cur <= self._hi:
            raise StopIteration
        out = _dotted(self._cur)
        self._cur += -self._step if self._rev else self._step
        return out

    def next_batch(self, size):
        if size < 0:
            raise ValueError("size must be non-negative")
        if not self._lo <= self._cur <= self._hi:
            return []
        dist = self._cur - self._lo if self._rev else self._hi - self._cur
        k = min(size, (dist + 1) // self._step)
        return [next(self) for _ in range(k)]
