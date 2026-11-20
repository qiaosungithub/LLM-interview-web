# expect: WA
# Wrong for Part 3: resolves a relative target from inside the link (as if the link were the directory to start
# from) instead of from the directory that holds the link. Passes the sample (its target is absolute).
MAX_SYMLINK_HOPS = 20


class SymlinkLoopError(Exception):
    pass


def _walk(parts, path):
    for comp in path.split("/"):
        if comp == "..":
            if parts:
                parts.pop()
        elif comp and comp != ".":
            parts.append(comp)
    return parts


def _as_path(parts):
    return "/" + "/".join(parts)


def cd_relative(cwd, destination):
    return _as_path(_walk(cwd.split("/")[1:] if cwd != "/" else [], destination))


def _origin(cwd, destination, home):
    if destination[:1] == "~":
        if len(destination) > 1 and destination[1] != "/":
            raise ValueError("~user not supported")
        return home, destination[1:]
    if destination[:1] == "/":
        return "/", destination
    return cwd, destination


def cd_absolute(cwd, destination, home):
    base, rest = _origin(cwd, destination, home)
    return _as_path(_walk(_walk([], base), rest))


def cd(cwd, destination, home, symlinks):
    base, rest = _origin(cwd, destination, home)
    here = _walk([], base)
    work = [iter(rest.split("/"))]      # stack of component iterators; a link pushes its target's components
    hops = 0
    while work:
        comp = next(work[-1], None)
        if comp is None:
            work.pop()
            continue
        if comp in ("", "."):
            continue
        if comp == "..":
            if here:
                here.pop()
            continue
        candidate = _as_path(here + [comp])
        if candidate in symlinks:
            hops += 1
            if hops > MAX_SYMLINK_HOPS:
                raise SymlinkLoopError(destination)
            target = symlinks[candidate]
            if target.startswith("/"):
                here = []
            else:
                here = here + [comp]     # starts from the link itself
            work.append(iter(target.split("/")))
        else:
            here.append(comp)
    return _as_path(here)
