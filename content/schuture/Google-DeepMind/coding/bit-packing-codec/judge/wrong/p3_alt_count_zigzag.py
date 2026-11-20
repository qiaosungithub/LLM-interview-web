# expect: WA
# Part 3: zigzags the count in the varint alternative, though the statement says the count is a plain varint.
def _check_k(k):
    if not isinstance(k, int) or not 1 <= k <= 32:
        raise ValueError("bad k")


def pack(values: list[int], k: int) -> bytes:
    _check_k(k)
    big = 0
    for v in values:
        if v < 0 or v >> k:
            raise ValueError("value does not fit")
        big = (big << k) | v
    nbits = k * len(values)
    nbytes = (nbits + 7) // 8
    big <<= nbytes * 8 - nbits
    return big.to_bytes(nbytes, "big")


def unpack(data: bytes, k: int, count: int) -> list[int]:
    _check_k(k)
    nbits = k * count
    if len(data) != (nbits + 7) // 8:
        raise ValueError("bad length")
    big = int.from_bytes(data, "big") >> (len(data) * 8 - nbits)
    mask = (1 << k) - 1
    return [(big >> (k * (count - 1 - i))) & mask for i in range(count)]


def _uv(v: int) -> bytes:
    if v < 0:
        raise ValueError("negative")
    groups = []
    while True:
        groups.append(v & 0x7F)
        v >>= 7
        if not v:
            break
    return bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])


def _uv_len(v: int) -> int:
    return 1 if v == 0 else (v.bit_length() + 6) // 7


def encode_uvarints(values: list[int]) -> bytes:
    out = bytearray()
    for v in values:
        out += _uv(v)
    return bytes(out)


def _read(data: bytes, i: int):
    val = 0
    shift = 0
    while True:
        if i >= len(data):
            raise ValueError("truncated")
        b = data[i]
        i += 1
        val += (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return val, i


def decode_uvarints(data: bytes) -> list[int]:
    out, i = [], 0
    while i < len(data):
        v, i = _read(data, i)
        out.append(v)
    return out


def zigzag(n: int) -> int:
    return (n << 1) ^ -(n < 0)


def unzigzag(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _mw(values):
    if len(values) == 0:
        raise ValueError("empty")
    lo, hi = min(values), max(values)
    w = (hi - lo).bit_length()
    if w > 32:
        raise ValueError("too wide")
    return lo, w


def encode_for(values: list[int]) -> bytes:
    m, w = _mw(values)
    out = _uv(zigzag(m)) + bytes([w]) + _uv(len(values))
    if w:
        out += pack([v - m for v in values], w)
    return out


def decode_for(data: bytes) -> list[int]:
    zm, i = _read(data, 0)
    if i >= len(data):
        raise ValueError("no width byte")
    w = data[i]
    count, i = _read(data, i + 1)
    m = unzigzag(zm)
    rest = data[i:]
    if w == 0:
        if rest:
            raise ValueError("trailing")
        return [m] * count
    return [m + d for d in unpack(rest, w, count)]


def for_size(values: list[int]) -> int:
    m, w = _mw(values)
    n = len(values)
    return _uv_len(zigzag(m)) + 1 + _uv_len(n) + (w * n + 7) // 8


def best_encoding(values: list[int]) -> str:
    f = for_size(values)
    alt = _uv_len(zigzag(len(values))) + sum(_uv_len(zigzag(v)) for v in values)
    return "for" if f <= alt else "varint"
