from dataclasses import dataclass


@dataclass
class Shard:
    id: str        # distinct among the shards passed to one call
    start: int
    end: int


def rebalance(limit: int, shards: list[Shard]) -> list[Shard]:
    """limit >= 1. Returns the surviving shards. A surviving shard's start is >= its original
    start and its end is >= its original end -- never the other way around."""
    raise NotImplementedError


class ShardRouter:
    def __init__(self):
        """Part 2: assigns every integer key to one shard of a set that changes one shard at a time."""

    def add_shard(self, shard_id: str) -> None:
        """ValueError if already present."""
        raise NotImplementedError

    def remove_shard(self, shard_id: str) -> None:
        """ValueError if absent."""
        raise NotImplementedError

    def locate(self, key: int) -> str:
        """Owner's id; LookupError if there are no shards."""
        raise NotImplementedError
