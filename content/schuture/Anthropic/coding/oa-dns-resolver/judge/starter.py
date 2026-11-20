class ARecord:
    def __init__(self, addresses: list[str], ttl: int) -> None:
        """addresses is one or more IPv4 addresses, kept in the order given; ttl is a positive integer (Level 4)."""
        self.addresses = addresses
        self.ttl = ttl


class CNAMERecord:
    def __init__(self, target: str, ttl: int) -> None:
        """target is another name, not necessarily normalised; ttl is a positive integer (Level 4)."""
        self.target = target
        self.ttl = ttl


class DNSResolutionError(Exception):
    """Base class for every error resolve can raise; name is whichever name each subclass names."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class NameNotFoundError(DNSResolutionError):
    """Level 1: a name reached while resolving is not a key of the zone (name: that normalised name)."""


class ResolutionCycleError(DNSResolutionError):
    """Level 2: a step's target was already visited in this call (name: the repeated name)."""


class ChainTooLongError(DNSResolutionError):
    """Level 2: a step would make the chain longer than max_chain_length (name: the normalised query)."""


class Resolver:
    def __init__(self, zone: dict[str, ARecord | CNAMERecord],
                 fallback_zone: dict[str, ARecord | CNAMERecord] | None = None,
                 *, max_chain_length: int = 8) -> None:
        """Built once from a zone (and, from Level 3, an optional fallback zone); answers many resolve calls."""

    def resolve(self, name: str) -> list[str]:
        """Returns the IPv4 addresses name resolves to, in the order Level 2 defines."""
        raise NotImplementedError

    def advance_clock(self, seconds: int) -> None:
        """Level 4: adds seconds (a non-negative integer) to the resolver's clock."""
        raise NotImplementedError
