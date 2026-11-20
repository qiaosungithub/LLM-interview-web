# expect: WA
# Part 2 hashes every file in full with FileIO.readall(), even when every size is different.
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

PREFIX = 4096
CHUNK = 1 << 16


def _scan(root_path):
    """(path, size) for every regular file under root_path, never following a symlink; explicit stack."""
    stack = [root_path]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        yield e.path, e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue


def _ordered(groups):
    out = [sorted(g) for g in groups if len(g) >= 2]
    out.sort(key=lambda g: g[0])
    return out


def _whole(path):
    try:
        with open(path, "rb", buffering=0) as f:
            return hashlib.sha256(f.readall()).digest()
    except OSError:
        return None


def find_duplicate_files(root_path):
    by_digest = defaultdict(list)
    for path, _size in _scan(root_path):
        d = _whole(path)
        if d is not None:
            by_digest[d].append(path)
    return _ordered(by_digest.values())


def _head(path):
    try:
        with open(path, "rb") as f:
            return f.read(PREFIX)
    except OSError:
        return None


def _candidates(root_path, read_head):
    """Buckets of paths that share size and first PREFIX bytes (each bucket has >= 2 readable files)."""
    by_size = defaultdict(list)
    for path, size in _scan(root_path):
        by_size[size].append(path)
    out = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        heads = defaultdict(list)
        for path, head in read_head(paths):
            if head is not None:
                heads[head].append(path)
        for bucket in heads.values():
            if len(bucket) >= 2:
                out.append((size, bucket))
    return out


def _serial_heads(paths):
    return [(p, _head(p)) for p in paths]


def find_duplicate_files_fast(root_path):
    return find_duplicate_files(root_path)
    groups = []
    for size, bucket in _candidates(root_path, _serial_heads):
        if size <= PREFIX:          # the head is the whole file
            groups.append(bucket)
            continue
        by_digest = defaultdict(list)
        for p in bucket:
            d = _whole(p)
            if d is not None:
                by_digest[d].append(p)
        groups.extend(by_digest.values())
    return _ordered(groups)


def find_duplicate_files_concurrent(root_path, max_workers=None, use_processes=False):
    pool_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with pool_cls(max_workers=max_workers) as pool:
        def pooled_heads(paths):
            futs = {pool.submit(_head, p): p for p in paths}
            return [(futs[f], f.result()) for f in as_completed(futs)]

        groups = []
        jobs = {}
        for size, bucket in _candidates(root_path, pooled_heads):
            if size <= PREFIX:
                groups.append(bucket)
                continue
            for p in bucket:
                jobs[pool.submit(_whole, p)] = (size, p)
        by_digest = defaultdict(list)
        for f in as_completed(jobs):
            d = f.result()
            if d is not None:
                size, p = jobs[f]
                by_digest[(size, d)].append(p)
        groups.extend(by_digest.values())
    return _ordered(groups)
