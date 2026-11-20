# expect: WA
# Wrong: deep_merge copies base only shallowly and stores overlay values as they are, so the result shares lists and dicts with its arguments.
import dataclasses
import hashlib
import json
import re
import typing
from types import MappingProxyType


class ConfigError(Exception):
    """Base class for every error raised below."""


class ConfigNotFoundError(ConfigError):
    pass


class ConfigCycleError(ConfigError):
    pass


class _DeleteType:
    def __repr__(self) -> str:
        return "DELETE"


DELETE = _DeleteType()


def _clone(value):
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if value is DELETE:
            result.pop(key, None)
        elif isinstance(value, dict):
            prior = result.get(key)
            # NOTE: recurse against {}, not a wholesale _clone(value), even when base has no dict
            # here yet -- otherwise a DELETE nested inside value would never take effect.
            result[key] = deep_merge(prior if isinstance(prior, dict) else {}, value)
        else:
            result[key] = value
    return result


def resolve_bases(store: dict, name: str) -> dict:
    return _resolve(store, name, [], {})


def _resolve(store, name, visiting, memo):
    if name in memo:                  # already resolved earlier in this call -- a diamond base
        return memo[name]
    if name not in store:
        raise ConfigNotFoundError(f"no config named {name!r} in the store")
    if name in visiting:               # NOTE: visiting is the current _base_ path, not the whole call
        chain = " -> ".join(visiting[visiting.index(name):] + [name])
        raise ConfigCycleError(f"cycle in _base_: {chain}")
    visiting.append(name)
    merged = {}
    for base_name in store[name].get("_base_", []):
        merged = deep_merge(merged, _resolve(store, base_name, visiting, memo))
    own_keys = {k: v for k, v in store[name].items() if k != "_base_"}
    merged = deep_merge(merged, own_keys)
    visiting.pop()
    memo[name] = merged
    return merged


class OverrideSyntaxError(ConfigError):
    pass


class OverrideKeyError(ConfigError):
    pass


_PATH_RE = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')
_INT_RE = re.compile(r'-?\d+', re.ASCII)
_FLOAT_RE = re.compile(r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+', re.ASCII)
_STRING_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _parse_scalar(token: str):
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if _FLOAT_RE.fullmatch(token):
        return float(token)
    if _INT_RE.fullmatch(token):
        return int(token)
    m = _STRING_RE.fullmatch(token)
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    raise OverrideSyntaxError(f"not a valid value: {token!r}")


def _split_list_items(inner: str) -> list:
    if not inner.strip():
        return []
    items, quote, start = [], None, 0
    for i, ch in enumerate(inner):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ",":                    # NOTE: a comma inside a quoted element is not a separator
            items.append(inner[start:i])
            start = i + 1
    items.append(inner[start:])
    return items


def _parse_value(token: str):
    if token.startswith("[") and token.endswith("]"):
        return [_parse_scalar(item.strip()) for item in _split_list_items(token[1:-1])]
    return _parse_scalar(token)


def parse_override(text: str):
    create = text.startswith("+")
    body = text[1:] if create else text
    path_part, sep, value_part = body.partition("=")
    if not sep or not _PATH_RE.fullmatch(path_part):
        raise OverrideSyntaxError(f"not a valid override: {text!r}")
    return create, path_part.split("."), _parse_value(value_part)


def _walk_and_set(config: dict, path: list, value, create: bool, text: str) -> None:
    node = config
    for segment in path[:-1]:
        if segment not in node:
            if not create:
                raise OverrideKeyError(f"{text}: {'.'.join(path)} does not exist (use '+' to create it)")
            node[segment] = {}
        elif not isinstance(node[segment], dict):
            raise OverrideKeyError(f"{text}: {segment!r} is not a mapping, cannot descend into it")
        node = node[segment]
    last = path[-1]
    if last not in node and not create:
        raise OverrideKeyError(f"{text}: {'.'.join(path)} does not exist (use '+' to create it)")
    node[last] = value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = _clone(config)
    for text in overrides:
        create, path, value = parse_override(text)
        _walk_and_set(result, path, value, create, text)
    return result


class InterpolationError(ConfigError):
    pass


class InterpolationCycleError(ConfigError):
    pass


_REF_RE = re.compile(r'\$\{([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\}')


def _get_path(root: dict, path: list):
    node = root
    for segment in path:
        if not isinstance(node, dict) or segment not in node:
            raise InterpolationError(f"unknown path: {'.'.join(path)}")
        node = node[segment]
    return node


def _resolve_ref(root, path_str, memo, visiting):
    if path_str in memo:
        return memo[path_str]
    if path_str in visiting:
        chain = " -> ".join(visiting[visiting.index(path_str):] + [path_str])
        raise InterpolationCycleError(f"circular interpolation: {chain}")
    visiting.append(path_str)
    value = _resolve_value(root, _get_path(root, path_str.split(".")), memo, visiting)
    visiting.pop()
    memo[path_str] = value
    return value


def _resolve_value(root, value, memo, visiting):
    if isinstance(value, str):
        whole = _REF_RE.fullmatch(value)
        if whole:                                         # the entire string is one reference
            return _resolve_ref(root, whole.group(1), memo, visiting)

        def substitute(m):
            target = _resolve_ref(root, m.group(1), memo, visiting)
            if isinstance(target, bool):                   # NOTE: check bool before int -- bool is an int
                return "true" if target else "false"
            if isinstance(target, (int, float, str)):
                return str(target)
            raise InterpolationError(f"cannot embed {m.group(1)!r} (a {type(target).__name__}) in a string")

        return _REF_RE.sub(substitute, value)
    if isinstance(value, dict):
        return {k: _resolve_value(root, v, memo, visiting) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(root, v, memo, visiting) for v in value]
    return value


def resolve_interpolations(config: dict) -> dict:
    return _resolve_value(config, config, {}, [])


@dataclasses.dataclass
class FieldError:
    path: str
    message: str


def _is_schema(tp) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _required(f: dataclasses.Field) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _check_type(value, tp, path: str) -> list[FieldError]:
    if typing.get_origin(tp) is list:
        elem_type = typing.get_args(tp)[0]
        if not isinstance(value, list):
            return [FieldError(path, f"expected list, got {type(value).__name__}")]
        return [err for i, item in enumerate(value) for err in _check_type(item, elem_type, f"{path}.{i}")]
    if _is_schema(tp):
        return validate_config(value, tp, path)
    if tp is float:                                 # NOTE: an int widens to float; a bool never does
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif tp in (int, bool, str):
        ok = isinstance(value, tp) and not (tp is int and isinstance(value, bool))
    else:
        return [FieldError(path, f"unsupported schema type {tp!r}")]
    return [] if ok else [FieldError(path, f"expected {tp.__name__}, got {type(value).__name__}")]


def validate_config(config: dict, schema: type, _path: str = "") -> list[FieldError]:
    if not isinstance(config, dict):
        return [FieldError(_path or "<root>", f"expected a mapping, got {type(config).__name__}")]
    hints = typing.get_type_hints(schema)
    fields = {f.name: f for f in dataclasses.fields(schema)}
    errors = []
    for name, f in fields.items():
        full = f"{_path}.{name}" if _path else name
        if name not in config:
            if _required(f):
                errors.append(FieldError(full, "required field is missing"))
            continue
        errors.extend(_check_type(config[name], hints[name], full))
    for name in config:
        if name not in fields:
            full = f"{_path}.{name}" if _path else name
            errors.append(FieldError(full, "unexpected field, not declared in the schema"))
    return errors


def freeze_config(config):
    if isinstance(config, dict):
        return MappingProxyType({k: freeze_config(v) for k, v in config.items()})
    if isinstance(config, list):
        return tuple(freeze_config(v) for v in config)
    return config


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
