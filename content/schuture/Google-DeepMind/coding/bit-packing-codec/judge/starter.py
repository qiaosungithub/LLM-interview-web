def pack(values: list[int], k: int) -> bytes:
    """Write each value as a k-bit field, most significant bit first, zero-padding the last byte."""
    raise NotImplementedError


def unpack(data: bytes, k: int, count: int) -> list[int]:
    """Read back count consecutive k-bit fields written by pack."""
    raise NotImplementedError


def encode_uvarints(values: list[int]) -> bytes:
    """Concatenate the unsigned LEB128 varint encoding of every value."""
    raise NotImplementedError


def decode_uvarints(data: bytes) -> list[int]:
    """Read varints back to back until no bytes remain."""
    raise NotImplementedError


def zigzag(n: int) -> int:
    """Map 0, -1, 1, -2, 2, ... to 0, 1, 2, 3, 4, ..."""
    raise NotImplementedError


def unzigzag(z: int) -> int:
    """The inverse of zigzag."""
    raise NotImplementedError


def encode_for(values: list[int]) -> bytes:
    """Frame-of-reference block: zigzag varint m, one byte w, varint count, then the packed offsets."""
    raise NotImplementedError


def decode_for(data: bytes) -> list[int]:
    """The exact inverse of encode_for."""
    raise NotImplementedError


def for_size(values: list[int]) -> int:
    """Exact size in bytes of encode_for(values), computed by formula, without encoding."""
    raise NotImplementedError


def best_encoding(values: list[int]) -> str:
    """"for" or "varint", whichever encodes values shorter; ties -> "for"."""
    raise NotImplementedError
