# expect: WA
# Orders checkpoints by file modification time instead of by step, which agrees only while every save has a larger step than the last.
import glob
import json
import logging
import os
import random
import re
import tempfile

import numpy as np

CHECKPOINT_GLOB = "ckpt_*.npz"
DEFAULT_KEEP = 3
_CKPT_RE = re.compile(r"^ckpt_(\d+)\.npz$")
_PAYLOAD_KEY = "__payload_json__"

logger = logging.getLogger(__name__)


def _checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, f"ckpt_{step}.npz")


def _step_of(path: str) -> int:
    m = _CKPT_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"not a checkpoint file: {path}")
    return int(m.group(1))


def _checkpoint_files_by_step(directory: str) -> list:
    """Every checkpoint file in `directory`, oldest first -- sorted on the parsed integer step,
    not the filename string (ckpt_9.npz must sort before ckpt_10.npz)."""
    return sorted(glob.glob(os.path.join(directory, CHECKPOINT_GLOB)), key=os.path.getmtime)   # newest last


def save_checkpoint(directory: str, step: int, state: dict, metadata: dict | None = None) -> None:
    """Writes `state` (a dict of numpy arrays and/or JSON-safe scalars) to ckpt_<step>.npz in
    `directory`, tagged with `metadata`. Atomic: a crash at any point leaves either the previous
    checkpoint or nothing at the final path, never a truncated ckpt_<step>.npz."""
    os.makedirs(directory, exist_ok=True)
    metadata = dict(metadata) if metadata is not None else {}   # NOTE: a fresh dict every call --
    metadata["step"] = step                                     # never mutate a shared default
    arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    plain = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    payload = json.dumps({"state": plain, "metadata": metadata}).encode("utf-8")
    final_path = _checkpoint_path(directory, step)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".tmp-ckpt_{step}-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            # NOTE: allow_pickle stays off on load below, so only plain arrays belong here --
            # everything else (dicts, RNG state, strings) travels through the JSON side channel.
            np.savez(f, **arrays, **{_PAYLOAD_KEY: np.frombuffer(payload, dtype=np.uint8)})
            f.flush()
            os.fsync(f.fileno())          # NOTE: bytes are durably on disk before the rename below
        os.replace(tmp_path, final_path)  # NOTE: atomic on a POSIX filesystem -- a reader sees either
                                           #       the old file or the complete new one, never a partial write
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)           # NOTE: never leave a half-written temp file behind
        raise


def _load_one(path: str) -> dict:
    with np.load(path, allow_pickle=False) as npz:  # NOTE: allow_pickle=False -- a corrupt or hostile
        raw = bytes(npz[_PAYLOAD_KEY]).decode("utf-8")  # file cannot execute code by being loaded
        payload = json.loads(raw)
        arrays = {k: npz[k] for k in npz.files if k != _PAYLOAD_KEY}
    return {**arrays, **payload["state"]}


def load_latest(directory: str):
    """Returns the state of the newest readable checkpoint in `directory`, or None if there is
    none at all. A checkpoint that fails to load (partial write, disk corruption) is skipped with
    a logged warning and the next-newest one is tried, rather than silently treated the same as
    "no checkpoint"; if every checkpoint present fails to load, the last error is raised instead
    of returning None, since that is a disk problem the caller must not mistake for a fresh start."""
    files = _checkpoint_files_by_step(directory)
    last_error = None
    for path in reversed(files):
        try:
            return _load_one(path)
        except Exception as e:
            logger.warning("checkpoint %s failed to load (%s); falling back to an older one", path, e)
            last_error = e
    if files:
        raise RuntimeError(f"no checkpoint in {directory!r} could be read") from last_error
    return None


def apply_retention(directory: str, keep: int = DEFAULT_KEEP) -> None:
    """Deletes all but the newest `keep` checkpoints in `directory`."""
    for path in _checkpoint_files_by_step(directory)[:-keep]:
        os.remove(path)
        logger.info("removed old checkpoint %s", path)


class ShardedSampler:
    """Shuffles range(num_examples) with `seed` and `epoch`, then splits it across `world_size`
    data-parallel workers so every index is used exactly once. The first `num_examples %
    world_size` workers by rank get one extra example each, so split sizes differ by at most one.
    indices() returns worker `rank`'s full list for the epoch; to resume partway through an
    epoch, slice it -- indices()[position:] -- since the permutation is a pure function of
    (seed, epoch), so an integer position is all a caller needs to save to resume exactly."""

    def __init__(self, num_examples: int, rank: int, world_size: int, seed: int, epoch: int):
        self.num_examples = num_examples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch

    def indices(self) -> list:
        rng = random.Random(self.seed + self.epoch)
        perm = list(range(self.num_examples))
        rng.shuffle(perm)
        base, remainder = divmod(self.num_examples, self.world_size)
        # NOTE: ranks [0, remainder) get one extra example each, so sizes sum to num_examples exactly
        start = self.rank * base + min(self.rank, remainder)
        size = base + (1 if self.rank < remainder else 0)
        return perm[start:start + size]
