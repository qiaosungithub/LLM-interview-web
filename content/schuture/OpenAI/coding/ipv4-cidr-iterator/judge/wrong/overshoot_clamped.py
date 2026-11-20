# expect: WA
# When a step would jump past the end of the range, lands on the end address instead of stopping.
import re

_OCTET_RE = re.compile(r"\A(0|[1-9][0-9]{0,2})\Z")


def _parse_ip(s):
    parts = s.split(".")
    if len(parts) != 4:
        raise ValueError(f"expected 4 dot-separated octets, got {s!r}")
    value = 0
    for p in parts:
        if not _OCTET_RE.match(p):
            raise ValueError(f"bad octet {p!r} in {s!r}: must be digits with no leading zero")
        octet = int(p)
        if octet > 255:
            raise ValueError(f"octet {octet} out of range 0-255 in {s!r}")
        value = (value << 8) | octet      # NOTE: big-endian -- the leftmost octet is the most significant byte
    return value


def ip_to_int(ip):
    return _parse_ip(ip)


def int_to_ip(value):
    # NOTE: shifts 24,16,8,0 mirror _parse_ip's order -- swap either one and addresses would compare backwards
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


MAX_IP = (1 << 32) - 1   # 255.255.255.255


class IPV4Iterator:
    def __init__(self, ip_or_cidr):
        self._cursor = ip_to_int(ip_or_cidr)
        self._hi = MAX_IP

    def __iter__(self):
        return self

    def __next__(self):
        if self._cursor > self._hi:          # NOTE: > not >=, so MAX_IP itself is still produced
            raise StopIteration
        result = int_to_ip(self._cursor)
        self._cursor += 1
        return result

def __init__(self, ip_or_cidr, reverse=False):
    self._cursor = ip_to_int(ip_or_cidr)
    self._reverse = reverse
    self._lo, self._hi = 0, MAX_IP


def __next__(self):
    # NOTE: self._cursor is a plain Python int, so decrementing past 0 just makes it negative --
    # it does not wrap around the way a fixed-width unsigned counter would, and this comparison
    # alone is enough to catch the underflow, no separate "would this go negative" check needed
    if self._cursor < self._lo or self._cursor > self._hi:
        raise StopIteration
    result = int_to_ip(self._cursor)
    self._cursor += -1 if self._reverse else 1
    return result


IPV4Iterator.__init__ = __init__   # attach to the class defined in Part 1
IPV4Iterator.__next__ = __next__

_PREFIX_RE = re.compile(r"\A(0|[1-9][0-9]?)\Z")


def _parse_prefix(s):
    if not _PREFIX_RE.match(s):
        raise ValueError(f"bad prefix {s!r}: must be digits with no leading zero")
    value = int(s)
    if not (0 <= value <= 32):
        raise ValueError(f"prefix {value} out of range 0-32")
    return value


def parse_ip_or_cidr(s):
    if s.count("/") > 1:
        raise ValueError(f"bad CIDR notation: {s!r}")
    if "/" in s:
        ip_part, prefix_part = s.split("/")
        return _parse_ip(ip_part), _parse_prefix(prefix_part)
    return _parse_ip(s), None


def network_and_broadcast(start, prefix):
    host_bits = 32 - prefix
    # NOTE: prefix == 0 shifts by 32 -- fine for a Python int (its low 32 bits come out 0), but a
    # 32-bit int in C/Java shifted by 32 is undefined or unchanged, and /0 comes out wrong
    mask = (0xFFFFFFFF << host_bits) & 0xFFFFFFFF
    network = start & mask
    broadcast = network + (1 << host_bits) - 1
    return network, broadcast


def __init__(self, ip_or_cidr, reverse=False):
    start, prefix = parse_ip_or_cidr(ip_or_cidr)
    self._cursor = start
    self._reverse = reverse
    if prefix is None:
        self._lo, self._hi = 0, MAX_IP
    else:
        self._lo, self._hi = network_and_broadcast(start, prefix)


IPV4Iterator.__init__ = __init__      # attach to the class of Part 1; Part 2's __next__ needs no change

def __init__(self, ip_or_cidr, reverse=False, step=1):
    start, prefix = parse_ip_or_cidr(ip_or_cidr)
    if step <= 0:
        raise ValueError(f"step must be a positive integer, got {step}")
    self._cursor = start
    self._reverse = reverse
    self._step = step
    if prefix is None:
        self._lo, self._hi = 0, MAX_IP
    else:
        self._lo, self._hi = network_and_broadcast(start, prefix)


def __next__(self):
    if self._cursor < self._lo or self._cursor > self._hi:
        raise StopIteration
    result = int_to_ip(self._cursor)
    nxt = self._cursor + (-self._step if self._reverse else self._step)
    end = self._lo if self._reverse else self._hi
    if self._cursor != end and not (self._lo <= nxt <= self._hi):
        nxt = end                       # do not skip the last address of the range
    self._cursor = nxt
    return result


IPV4Iterator.__init__ = __init__
IPV4Iterator.__next__ = __next__

def next_batch(self, size):
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    out = []
    for _ in range(size):
        try:
            out.append(next(self))
        except StopIteration:
            break
    return out


IPV4Iterator.next_batch = next_batch
