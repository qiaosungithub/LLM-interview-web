def cd_relative(cwd: str, destination: str) -> str:
    """Part 1: the normalized absolute path reached by following the relative destination from cwd."""
    raise NotImplementedError


def cd_absolute(cwd: str, destination: str, home: str) -> str:
    """Part 2: destination may also be absolute or start with ~ (the home directory); ~user raises ValueError."""
    raise NotImplementedError


class SymlinkLoopError(Exception):
    """Raised by cd when one call needs more than MAX_SYMLINK_HOPS symlink expansions."""


MAX_SYMLINK_HOPS = 20


def cd(cwd: str, destination: str, home: str, symlinks: dict[str, str]) -> str:
    """Part 3: the physical path (as with cd -P), following the symlinks exactly by the rules of the statement."""
    raise NotImplementedError
