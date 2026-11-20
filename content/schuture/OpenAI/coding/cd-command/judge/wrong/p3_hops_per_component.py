# expect: WA
# Wrong for Part 3: restarts the expansion count at each component of destination, so a call whose expansions are
# spread over several components never raises even with more than 20 in total. Real cycles still raise.
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
        if len(work) == 1:
            hops = 0            # a new top-level component: count afresh
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
            work.append(iter(target.split("/")))
        else:
            here.append(comp)
    return _as_path(here)
