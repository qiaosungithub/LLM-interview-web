import dataclasses

DELETE = ...  # sentinel; usable as a value anywhere in a config


class ConfigError(Exception):
    """Base class for every error raised below."""


class ConfigNotFoundError(ConfigError):
    """name, or a name in some _base_ list, is not in the store."""


class ConfigCycleError(ConfigError):
    """The _base_ graph has a cycle reachable from name."""


def deep_merge(base: dict, overlay: dict) -> dict:
    """Returns a new dict combining base with overlay on top; see the rule above."""
    raise NotImplementedError


def resolve_bases(store: dict[str, dict], name: str) -> dict:
    """Returns name's fully composed config. The result has no _base_ key."""
    raise NotImplementedError


class OverrideSyntaxError(ConfigError):
    """An override string does not match path=value or +path=value."""


class OverrideKeyError(ConfigError):
    """A path segment is missing without '+', or an existing segment is not a dict."""


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Applies every override in order to a copy of config; returns the copy."""
    raise NotImplementedError


class InterpolationError(ConfigError):
    """A ${...} reference addresses a missing path, or embeds a non-scalar in a string."""


class InterpolationCycleError(ConfigError):
    """Resolving some path's ${...} references needs that path's own value."""


def resolve_interpolations(config: dict) -> dict:
    """Returns a copy of config with every ${a.b} reference replaced by its value."""
    raise NotImplementedError


@dataclasses.dataclass
class FieldError:
    path: str      # full dotted path from the config's root, e.g. "trainer.lr"
    message: str


def validate_config(config: dict, schema: type) -> list[FieldError]:
    """Returns every mismatch between config and schema, collected rather than raised."""
    raise NotImplementedError


def freeze_config(config: dict):
    """Returns an immutable snapshot: nested dicts -> MappingProxyType, nested lists -> tuple."""
    raise NotImplementedError


def config_hash(config: dict) -> str:
    """Returns a stable content hash of config, independent of key order."""
    raise NotImplementedError
