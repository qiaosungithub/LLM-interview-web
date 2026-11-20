class CountingProbe:
    """Wraps is_supported and counts how many times it is actually called."""
    def __init__(self, is_supported):
        self._is_supported = is_supported
        self.calls = 0

    def __call__(self, version: str) -> bool:
        self.calls += 1
        return self._is_supported(version)


class Registry:
    """The available packages: name -> version -> requires."""
    def __init__(self):
        self._versions = {}

    def add_version(self, name: str, version: str, requires: dict[str, str] | None = None) -> None:
        """Registers one available (name, version) and its direct dependencies."""
        self._versions.setdefault(name, {})[version] = dict(requires or {})

    def versions(self, name: str) -> list[str]:
        """All versions registered for name (empty if the name is unknown)."""
        return list(self._versions.get(name, {}))

    def requires(self, name: str, version: str) -> dict[str, str]:
        """The constraints declared by exactly this (name, version)."""
        return self._versions[name][version]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parses "{major}.{minor}.{patch}" into (major, minor, patch)."""
    raise NotImplementedError


def find_earliest_supported_monotone(versions: list[str], is_supported) -> str | None:
    """Part 1: is_supported is monotone once versions are ordered numerically. Returns the earliest supported
    version, or None."""
    raise NotImplementedError


def find_earliest_supported(versions: list[str], is_supported) -> str | None:
    """Part 2: same contract as find_earliest_supported_monotone, but is_supported need not be monotone."""
    raise NotImplementedError


def find_earliest_supported_hierarchical(versions: list[str], is_supported) -> str | None:
    """Part 3: same contract, assuming A1-A3; O(log M + log N_minor + log P) calls to is_supported, never
    probing the same version string twice."""
    raise NotImplementedError


def resolve(registry: Registry, root: str) -> list[str] | None:
    """Part 4: one version for root and every package the chosen versions require, all constraints holding,
    as "name@version" strings in an install order (dependencies first); None if no valid choice exists."""
    raise NotImplementedError
