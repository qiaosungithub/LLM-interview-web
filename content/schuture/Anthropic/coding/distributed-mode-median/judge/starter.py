def global_mode(rank: int, shard: list[int], send, recv, barrier) -> int | None:
    """Every rank calls this once; rank 0's return value is the global mode of the ten shards combined."""
    raise NotImplementedError


def global_median(rank: int, shard: list[int], send, recv, barrier) -> int | None:
    """Same calling convention as global_mode; rank 0's return value is the (lower) median."""
    raise NotImplementedError


def sample_sort(rank: int, shard: list[int], send, recv, barrier) -> list[int]:
    """Every rank calls this once; rank r's return value is result[r] as defined in the statement."""
    raise NotImplementedError
