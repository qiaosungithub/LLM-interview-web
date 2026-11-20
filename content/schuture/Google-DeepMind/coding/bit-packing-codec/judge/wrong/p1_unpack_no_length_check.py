# expect: WA
# Part 1 (the page's pitfall): unpack never checks len(data), so an overlong buffer silently succeeds.
def _packed_size(k: int, count: int) -> int:
    """Bytes pack(values, k) uses for count values: ceil(k * count / 8), via integer arithmetic."""
    return (k * count + 7) // 8


def pack(values: list[int], k: int) -> bytes:
    if not (1 <= k <= 32):
        raise ValueError(f"k must satisfy 1 <= k <= 32, got {k}")
    out = bytearray()
    acc = 0                    # unflushed bits, right-aligned in the low nbits positions
    nbits = 0
    limit = 1 << k
    for v in values:
        if not (0 <= v < limit):
            raise ValueError(f"value {v} does not fit in {k} unsigned bits")   # NOTE: checked before packing
        acc = (acc << k) | v
        nbits += k
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1     # NOTE: drop the bits already flushed, so acc stays a handful of bits
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)    # NOTE: zero-pad the final byte's low bits
    return bytes(out)


def unpack(data: bytes, k: int, count: int) -> list[int]:
    if not (1 <= k <= 32):
        raise ValueError(f"k must satisfy 1 <= k <= 32, got {k}")
    if len(data) < _packed_size(k, count):
        raise ValueError("data too short")
    values = []
    acc = 0
    nbits = 0
    idx = 0
    mask = (1 << k) - 1
    for _ in range(count):
        while nbits < k:               # NOTE: pull in whole bytes until there are enough bits for one field
            acc = (acc << 8) | data[idx]
            idx += 1
            nbits += 8
        nbits -= k
        values.append((acc >> nbits) & mask)
        acc &= (1 << nbits) - 1
    return values


def _uvarint_size(v: int) -> int:
    """Bytes _encode_uvarint(v) uses, from the size formula alone: max(1, ceil(bitlen(v) / 7))."""
    return max(1, -(-v.bit_length() // 7))


def _encode_uvarint(v: int) -> bytes:
    if v < 0:
        raise ValueError(f"unsigned varint expects v >= 0, got {v}")
    out = bytearray()
    while True:
        group = v & 0x7F
        v >>= 7
        if v:
            out.append(group | 0x80)       # NOTE: high bit set -- at least one more byte follows
        else:
            out.append(group)              # last group: high bit stays 0
            return bytes(out)


def encode_uvarints(values: list[int]) -> bytes:
    return b"".join(_encode_uvarint(v) for v in values)


def _read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    """Reads one unsigned varint starting at data[pos]. Returns (value, position right after it)."""
    v = 0
    shift = 0
    n = len(data)
    while True:
        if pos >= n:
            raise ValueError("truncated varint data")   # NOTE: ran out of bytes with a continuation pending
        byte = data[pos]
        pos += 1
        v |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return v, pos
        shift += 7


def decode_uvarints(data: bytes) -> list[int]:
    values = []
    pos = 0
    n = len(data)
    while pos < n:
        v, pos = _read_uvarint(data, pos)
        values.append(v)
    return values


def zigzag(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def unzigzag(z: int) -> int:
    return z // 2 if z % 2 == 0 else -(z + 1) // 2


def _for_header_width(values: list[int]) -> tuple[int, int]:
    """(m, w) shared by encode_for and for_size, after validating the block is non-empty and its
    range still fits pack's own k <= 32 limit."""
    if not values:
        raise ValueError("a frame-of-reference block needs at least one value")
    m = min(values)
    w = (max(values) - m).bit_length()
    if w > 32:
        raise ValueError(f"block range needs {w} bits, more than pack's 32-bit limit")   # NOTE: propagate early
    return m, w


def encode_for(values: list[int]) -> bytes:
    m, w = _for_header_width(values)
    header = _encode_uvarint(zigzag(m)) + bytes([w]) + _encode_uvarint(len(values))
    payload = pack([v - m for v in values], w) if w else b""   # NOTE: w == 0 -- every delta is 0, skip pack
    return header + payload


def decode_for(data: bytes) -> list[int]:
    zm, pos = _read_uvarint(data, 0)
    m = unzigzag(zm)
    if pos >= len(data):
        raise ValueError("truncated frame-of-reference header")
    w = data[pos]
    pos += 1
    count, pos = _read_uvarint(data, pos)
    if w == 0:
        if len(data) != pos:      # NOTE: w == 0 carries no payload -- nothing may follow the header
            raise ValueError("trailing bytes after an all-equal frame-of-reference block")
        return [m] * count
    return [m + d for d in unpack(data[pos:], w, count)]


def for_size(values: list[int]) -> int:
    m, w = _for_header_width(values)
    header = _uvarint_size(zigzag(m)) + 1 + _uvarint_size(len(values))
    payload = _packed_size(w, len(values)) if w else 0
    return header + payload


def _varint_block_size(values: list[int]) -> int:
    """Bytes the alternative scheme uses: a count varint, then one zigzag varint per value."""
    return _uvarint_size(len(values)) + sum(_uvarint_size(zigzag(v)) for v in values)


def best_encoding(values: list[int]) -> str:
    # NOTE: for_size raises on an empty block, so best_encoding does too -- no separate check needed
    return "for" if for_size(values) <= _varint_block_size(values) else "varint"
