"""A model of the statement for tests.py, written from its rules with no regular expressions and no ipaddress module.

parse(s) gives (start, lo, hi), or raises ValueError with the reason in Chinese; addresses() lists what the
iterator produces. In 自定义运行:
    from ref import addresses
    print(addresses("203.0.113.100/28", reverse=True))
"""
TOP = 2 ** 32 - 1           # 255.255.255.255
DIGITS = "0123456789"


def _number(text, largest, what):
    """The value of one segment or prefix: ASCII digits, no leading zero unless it is "0", at most `largest`."""
    if not text:
        raise ValueError(f"{what}是空的")
    if any(ch not in DIGITS for ch in text):
        raise ValueError(f"{what} {text!r} 只能由数字 0-9 组成")
    if len(text) > 1 and text[0] == "0":
        raise ValueError(f"{what} {text!r} 有前导零")
    value = 0
    for ch in text:
        value = value * 10 + DIGITS.index(ch)
    if value > largest:
        raise ValueError(f"{what} {text!r} 超过 {largest}")
    return value


def parse(s):
    """(start, lo, hi): the address ip_or_cidr names and the range iteration is confined to."""
    if s.count("/") > 1:
        raise ValueError("最多只能有一个 '/'")
    address, slash, prefix = s.partition("/")
    parts = address.split(".")
    if len(parts) != 4:
        raise ValueError(f"地址要正好分成四段，这里有 {len(parts)} 段")
    start = 0
    for i, part in enumerate(parts, 1):
        start = start * 256 + _number(part, 255, f"第 {i} 段")
    if not slash:
        return start, 0, TOP
    size = 2 ** (32 - _number(prefix, 32, "前缀"))
    lo = start - start % size
    return start, lo, lo + size - 1


def text(value):
    """The dotted form of a 32-bit integer."""
    parts = []
    for _ in range(4):
        value, octet = divmod(value, 256)
        parts.append(str(octet))
    return ".".join(reversed(parts))


def addresses(s, reverse=False, step=1, count=None):
    """What IPV4Iterator(s, reverse, step) produces, at most `count` of it."""
    start, lo, hi = parse(s)
    out, cur = [], start
    while lo <= cur <= hi and (count is None or len(out) < count):
        out.append(text(cur))
        cur += -step if reverse else step
    return out


def remaining(s, reverse=False, step=1):
    """How many addresses IPV4Iterator(s, reverse, step) produces in all."""
    start, lo, hi = parse(s)
    return (start - lo if reverse else hi - start) // step + 1


class Model:
    """The iterator itself, for sequences that mix next() and next_batch()."""

    def __init__(self, s, reverse=False, step=1):
        self.cur, self.lo, self.hi = parse(s)
        self.move = -step if reverse else step

    def take(self, size):
        out = []
        while len(out) < size and self.lo <= self.cur <= self.hi:
            out.append(text(self.cur))
            self.cur += self.move
        return out
