"""Helpers for tests.py: straightforward restatements of the three formats, written from the statement with
'0'/'1' strings and divmod (no bit accumulator), plus random input generators."""


# ---------------------------------------------------------------- Part 1

def packed_size(k, count):
    return (k * count + 7) // 8


def pack(values, k):
    bits = "".join(format(v, f"0{k}b") for v in values)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def unpack(data, k, count):
    bits = "".join(format(b, "08b") for b in data)
    return [int(bits[i:i + k], 2) for i in range(0, k * count, k)]


# ---------------------------------------------------------------- Part 2

def uvarint(v):
    groups = []
    while True:
        v, rem = divmod(v, 128)
        groups.append(rem)
        if v == 0:
            break
    return bytes(g | 0x80 if i < len(groups) - 1 else g for i, g in enumerate(groups))


def uvarints(values):
    return b"".join(uvarint(v) for v in values)


def zigzag(n):
    return 2 * n if n >= 0 else -2 * n - 1


def unzigzag(z):
    return z // 2 if z % 2 == 0 else -(z + 1) // 2


def random_uvarint_value(r):
    kind = r.randrange(4)
    if kind == 0:
        return r.randint(0, 300)
    if kind == 1:
        g = r.randint(1, 12)                      # right at a 7-bit group boundary
        return (1 << (7 * g)) + r.choice([-1, 0, 1])
    if kind == 2:
        return r.randrange(1 << r.randint(1, 64))
    return r.randrange(1 << r.randint(65, 200))


# ---------------------------------------------------------------- Part 3

def encode_for(values):
    m, big = min(values), max(values)
    w = (big - m).bit_length()
    header = uvarint(zigzag(m)) + bytes([w]) + uvarint(len(values))
    return header + (pack([v - m for v in values], w) if w else b"")


def varint_block(values):
    return uvarint(len(values)) + b"".join(uvarint(zigzag(v)) for v in values)


def best_encoding(values):
    return "for" if len(encode_for(values)) <= len(varint_block(values)) else "varint"


def random_block(r):
    kind = r.randrange(6)
    n = r.randint(1, 40) if r.random() < 0.85 else r.randint(100, 300)
    if kind == 0:                                   # narrow range around a large offset
        base = r.randrange(1 << r.randint(10, 60)) * r.choice([1, -1])
        return [base + r.randint(0, 40) for _ in range(n)]
    if kind == 1:                                   # small values and one outlier
        vals = [r.randint(-5, 5) for _ in range(n)]
        vals[r.randrange(n)] = r.choice([1, -1]) * r.randint(1 << 20, 1 << 30)
        return vals
    if kind == 2:                                   # uniform over a random span that still fits 32 bits
        lo = r.randint(-(1 << 31), 1 << 31)
        span = (1 << r.randint(0, 32)) - 1
        return [lo + r.randint(0, span) for _ in range(n)]
    if kind == 3:                                   # every value equal: w == 0
        return [r.randint(-1000, 1000)] * n
    if kind == 4:                                   # m near 0, 64 or 128 where the zigzag varint changes size
        m = r.choice([0, 0, -1, 63, 64, -64, -65, 8191, 8192, -8192])
        return [m] + [m + r.randint(0, 9) for _ in range(n - 1)]
    return [r.randint(-200, 200) for _ in range(n)]
