"""Reference answer from the page (Parts 1-3), joined into one file."""
from collections import deque


def _split(path: str) -> list[str]:
    return [tok for tok in path.split("/") if tok != ""]     # NOTE: drops "" from //, a leading /, or a trailing /


def _apply(stack: list[str], token: str) -> None:
    if token in ("", "."):
        return
    if token == "..":
        if stack:                # NOTE: popping an empty stack is the root clamp, not an error
            stack.pop()
        return
    stack.append(token)


def cd_relative(cwd: str, destination: str) -> str:
    stack = _split(cwd)
    for token in destination.split("/"):        # NOTE: split, not _split -- "" tokens must reach _apply
        _apply(stack, token)
    return "/" + "/".join(stack)                 # NOTE: build the leading "/" here; join([]) is "", not "/"


def _start(cwd: str, destination: str, home: str) -> tuple[str, str]:
    if destination == "~" or destination.startswith("~/"):
        return home, destination[2:]      # NOTE: "~"[2:] is "", so a bare "~" lands exactly on home
    if destination.startswith("~"):
        raise ValueError(f"unsupported destination: {destination!r} (~user is out of scope)")
    if destination.startswith("/"):
        return "/", destination           # NOTE: destination's own leading "/" becomes an "" token, dropped by _apply
    return cwd, destination               # NOTE: only a leading "~" is special; "a/~b" falls through to here


def cd_absolute(cwd: str, destination: str, home: str) -> str:
    base, rest = _start(cwd, destination, home)
    stack = _split(base)
    for token in rest.split("/"):
        _apply(stack, token)
    return "/" + "/".join(stack)


class SymlinkLoopError(Exception):
    pass


MAX_SYMLINK_HOPS = 20   # NOTE: a choice of this problem; Linux (kernel lookup and glibc realpath) fails with ELOOP after 40


def cd(cwd: str, destination: str, home: str, symlinks: dict[str, str]) -> str:
    base, rest = _start(cwd, destination, home)
    stack = _split(base)                  # physical already: cwd and home never lie under a link
    pending = deque(rest.split("/"))
    hops = 0
    while pending:
        token = pending.popleft()
        if token in ("", ".", ".."):
            _apply(stack, token)          # NOTE: ".." pops the PHYSICAL stack -- already past any earlier link
            continue
        target = symlinks.get("/" + "/".join(stack + [token]))   # the path this token would create
        if target is None:
            stack.append(token)
            continue
        hops += 1
        if hops > MAX_SYMLINK_HOPS:
            raise SymlinkLoopError(f"more than {MAX_SYMLINK_HOPS} symlink expansions resolving {destination!r}")
        if target.startswith("/"):
            stack = []
        # NOTE: otherwise `stack` is still the directory that contains the link -- the base of a relative target
        pending.extendleft(reversed(target.split("/")))
    return "/" + "/".join(stack)
