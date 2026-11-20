"""A direct, recursive reading of the statement's rules, and random input generators for the tests."""
import itertools

MAX_HOPS = 20


class _TooMany(Exception):
    pass


def resolve(cwd, destination, home="/", symlinks=None):
    """Where `cd destination` lands. Follows the components left to right; whenever the path so far is a key, the
    link's target is followed from the link's directory (or from "/" if absolute) by the same rules. Returns
    (path, or None after more than 20 expansions; number of expansions). Raises ValueError for ~user forms."""
    links = symlinks or {}
    if destination == "~" or destination.startswith("~/"):
        start, rest = home, destination[1:]
    elif destination.startswith("~"):
        raise ValueError(destination)
    elif destination.startswith("/"):
        start, rest = "/", destination
    else:
        start, rest = cwd, destination
    hops = 0

    def walk(parts, path):
        nonlocal hops
        for comp in path.split("/"):
            if comp in ("", "."):
                continue
            if comp == "..":
                parts = parts[:-1]            # [][:-1] is []: ".." stops at "/"
                continue
            here = "/" + "/".join(parts + [comp])
            if here not in links:
                parts = parts + [comp]
                continue
            hops += 1
            if hops > MAX_HOPS:
                raise _TooMany
            target = links[here]
            parts = walk([] if target.startswith("/") else parts, target)
        return parts

    try:
        return "/" + "/".join(walk([c for c in start.split("/") if c], rest)), hops
    except _TooMany:
        return None, hops


# ---------------------------------------------------------------- Parts 1-2 inputs

NAMES = ["a", "b", "c", "...", ".x", "x y", "文档"]


def random_dir(r, depth=3):
    """A normalized absolute path (possibly "/")."""
    return "/" + "/".join(r.choice(NAMES) for _ in range(r.randint(0, depth)))


def random_rest(r, max_len=6):
    """A relative path that does not start with "/" or "~": names, ".", "..", doubled and trailing slashes."""
    tokens = [r.choice(NAMES + [".", "..", ".."])]
    tokens += [r.choice(NAMES + [".", "..", "..", ""]) for _ in range(r.randint(0, max_len))]
    return "/".join(tokens)


# ---------------------------------------------------------------- Part 3 inputs

DIRS = ["/"] + ["/" + "/".join(p) for n in (1, 2, 3) for p in itertools.product("pq", repeat=n)]
# absolute and relative targets; targets that climb (even past "/"), point at their own directory, end at or pass
# through another link; together they also make genuine cycles
TARGETS = ["/q/p", "/p/q/x", "/q/x/p", "/p/y/../x", "..", "../..", "../../../../q", ".", "y", "x", "../x/q",
           "x/..", "./y/p", "p/x", "../p"]
WORDS = ["p", "q", "x", "y", "x", "y", ".", "..", ""]


def random_links(r):
    """Links named x and y in random directories of the p/q tree. No key lies under another key."""
    density = r.uniform(0.2, 0.6)
    links = {}
    for d in DIRS:
        for name in ("x", "y"):
            if r.random() < density:
                links[d.rstrip("/") + "/" + name] = r.choice(TARGETS)
    return links


def random_cd_input(r):
    """(cwd, destination, home, symlinks) over the p/q tree."""
    links = random_links(r)
    cwd, home = r.choice(DIRS), r.choice(DIRS)
    rest = "/".join(r.choice(WORDS) for _ in range(r.randint(1, 6)))
    kind = r.choice(["relative", "relative", "absolute", "home"])
    if kind == "home":
        destination = "~" if r.random() < 0.1 else "~/" + rest
    elif kind == "absolute":
        destination = "/" + rest.lstrip("/")
    else:
        destination = rest.lstrip("/") or "."
    return cwd, destination, home, links
