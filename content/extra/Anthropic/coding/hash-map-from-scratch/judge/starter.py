class HashMap:
    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        """hash_fn(key) -> int replaces hash(key); eq_fn(a, b) -> bool replaces a == b; both are optional."""

    def put(self, key, value) -> None:
        """Insert key, or overwrite its value if it is already present."""
        raise NotImplementedError

    def get(self, key, default=None):
        """The value stored for key, or default if key is absent."""
        raise NotImplementedError

    def __contains__(self, key) -> bool:
        """Whether key is stored."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of keys stored."""
        raise NotImplementedError

    def delete(self, key) -> bool:
        """Remove key if it is present. Returns whether it was present."""
        raise NotImplementedError

    def items(self):
        """Iterates over (key, value) for every stored key, in slot order."""
        raise NotImplementedError

    def keys(self):
        """Iterates over every stored key, in slot order."""
        raise NotImplementedError

    def __iter__(self):
        """Same as keys()."""
        raise NotImplementedError
