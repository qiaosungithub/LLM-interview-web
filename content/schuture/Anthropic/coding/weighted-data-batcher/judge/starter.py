class DataRegistry:
    # The interface the statement gives. The tests pass their own registry (ref.Registry) to DataBatcher.
    def get_iterator(self, name: str, offset: int = 0):
        """Returns a fresh iterator over dataset `name`, opened directly at position `offset`;
        opening is fast at any offset, it never walks through the examples it skips. Writing L for
        the number of examples in `name`, the iterator yields the example at index offset % L, then
        (offset + 1) % L, then (offset + 2) % L, and so on forever: once it passes the last example
        it wraps back to the first and keeps going. It never raises StopIteration."""
        raise NotImplementedError


class DataBatcher:
    def __init__(self, registry: DataRegistry, weights: dict[str, int], batch_size: int,
                 allocation: str = "deterministic", seed: int = 0) -> None:
        """allocation is "deterministic" or "stochastic"; seed is used only when allocation == "stochastic"."""
        raise NotImplementedError

    def next_batch(self) -> list:
        """Returns batch_size examples drawn from the datasets in proportion to weights."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        """A small, JSON-serializable dict sufficient to resume from here."""
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        """Moves this batcher to the position state describes."""
        raise NotImplementedError
