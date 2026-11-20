# expect: WA
# Trusts (size, prefix hash) and skips the full read, so files that differ after the first 4096 bytes are merged.
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

CHUNK_SIZE = 1 << 20   # 1 MiB per streamed read, so a huge file is never loaded into memory at once


def _iter_files(root_path: str):
    """Yields every regular, non-symlink file under root_path. A symbolic link -- to a file, to a
    directory, or dangling -- is never followed and never yielded, so a directory reached only through
    one is not walked into either."""
    for dirpath, _dirnames, filenames in os.walk(root_path):   # followlinks=False by default
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue    # NOTE: islink() first -- isfile() alone would follow the symlink to decide
            yield path


def _hash_file(path: str) -> bytes | None:
    """Streams path through SHA-256 in fixed-size chunks. Returns None if it cannot be read."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                hasher.update(chunk)
    except OSError:
        return None      # NOTE: permission error or similar -- skip this file, never crash the walk
    return hasher.digest()


def find_duplicate_files(root_path: str) -> list[list[str]]:
    by_hash = defaultdict(list)
    for path in _iter_files(root_path):
        digest = _hash_file(path)
        if digest is not None:
            by_hash[digest].append(path)
    groups = [sorted(paths) for paths in by_hash.values() if len(paths) >= 2]
    groups.sort(key=lambda g: g[0])   # NOTE: groups ordered by their own (already-sorted) first path
    return groups


PREFIX_SIZE = 4096   # bytes sampled for the cheap pre-filter -- a filesystem block is a natural size


def _hash_prefix(path: str) -> bytes | None:
    """Hashes only the first PREFIX_SIZE bytes of path (the whole file, if it is shorter). Returns
    None if it cannot be read."""
    try:
        with open(path, "rb") as f:
            data = f.read(PREFIX_SIZE)
    except OSError:
        return None
    return hashlib.sha256(data).digest()


def find_duplicate_files_fast(root_path: str) -> list[list[str]]:
    by_size = defaultdict(list)
    for path in _iter_files(root_path):
        try:
            size = os.path.getsize(path)          # NOTE: metadata the OS already has -- no content read
        except OSError:
            continue                               # gone, or unreadable metadata: treat as unreadable
        by_size[size].append(path)

    by_prefix = defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue                               # NOTE: tier 1 -- a size nothing else shares can't collide
        for path in paths:
            digest = _hash_prefix(path)
            if digest is not None:
                by_prefix[(size, digest)].append(path)

    groups = []
    for (_size, _prefix_digest), candidates in by_prefix.items():
        if len(candidates) < 2:
            continue                               # NOTE: tier 2 -- distinct prefixes already rule these out
        groups.append(candidates)

    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    return groups


def _hash_many(paths, hash_fn, max_workers, pool_cls) -> dict[str, bytes]:
    """Applies hash_fn to every path using a worker pool; returns {path: digest}, silently omitting
    paths that could not be read."""
    if not paths:
        return {}
    with pool_cls(max_workers=max_workers) as pool:
        digests = pool.map(hash_fn, paths)   # NOTE: map() preserves input order, so the zip below is safe
        return {path: digest for path, digest in zip(paths, digests) if digest is not None}


def find_duplicate_files_concurrent(root_path: str, max_workers: int | None = None,
                                     use_processes: bool = False) -> list[list[str]]:
    pool_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    by_size = defaultdict(list)
    for path in _iter_files(root_path):
        try:
            by_size[os.path.getsize(path)].append(path)
        except OSError:
            continue
    tier1 = [(size, p) for size, paths in by_size.items() if len(paths) >= 2 for p in paths]

    prefixes = _hash_many([p for _, p in tier1], _hash_prefix, max_workers, pool_cls)
    by_prefix = defaultdict(list)
    for size, path in tier1:
        if path in prefixes:
            by_prefix[(size, prefixes[path])].append(path)

    tier3 = [(size, p) for (size, _pfx), paths in by_prefix.items() if len(paths) >= 2 for p in paths]
    fulls = _hash_many([p for _, p in tier3], _hash_file, max_workers, pool_cls)
    by_full = defaultdict(list)
    for size, path in tier3:
        if path in fulls:
            by_full[(size, fulls[path])].append(path)

    groups = [sorted(paths) for paths in by_full.values() if len(paths) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups
