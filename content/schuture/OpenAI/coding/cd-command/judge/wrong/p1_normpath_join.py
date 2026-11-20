# expect: WA
# Wrong for Parts 1-2: joins cwd + "/" + destination and calls posixpath.normpath, which keeps exactly two
# leading slashes, so from cwd "/" (or home "/") the result starts with "//". Passes every sample.
import posixpath

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
    return posixpath.normpath(cwd + "/" + destination)


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
    if base == "/":
        return posixpath.normpath(rest) if rest.startswith("/") else posixpath.normpath("/" + rest)
    return posixpath.normpath(base + "/" + rest)


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
            work.append(iter(target.split("/")))
        else:
            here.append(comp)
    return _as_path(here)
