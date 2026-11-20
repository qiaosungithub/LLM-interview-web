"""checkpointing.py -- resumable training: save every N steps, keep the last 3, resume from the latest.

Usage:
    save_checkpoint(run_dir, step, state)          # called every N training steps
    apply_retention(run_dir, keep=3)                # called right after, to bound disk usage
    state = load_latest(run_dir)                    # called once, at process start, to resume
"""
import glob
import os
import pickle
import random

CHECKPOINT_GLOB = "ckpt_*.pkl"
DEFAULT_KEEP = 3


def _checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, f"ckpt_{step}.pkl")


def _checkpoint_files(directory: str) -> list:
    return glob.glob(os.path.join(directory, CHECKPOINT_GLOB))


def save_checkpoint(directory: str, step: int, state: dict, metadata: dict = {}) -> None:
    """Writes `state` (model/optimizer arrays, kept opaque here) to ckpt_<step>.pkl in `directory`,
    tagged with `metadata` -- small caller-supplied bookkeeping, e.g. which steps have been saved
    so far, for debugging retention."""
    os.makedirs(directory, exist_ok=True)
    metadata["step"] = step
    metadata.setdefault("history", []).append(step)
    payload = {"state": state, "metadata": metadata}
    with open(_checkpoint_path(directory, step), "wb") as f:
        pickle.dump(payload, f)


def load_latest(directory: str):
    """Returns the state of the newest checkpoint in `directory`, or None if there is none."""
    files = sorted(_checkpoint_files(directory))
    if not files:
        return None
    try:
        with open(files[-1], "rb") as f:
            payload = pickle.load(f)
        return payload["state"]
    except Exception:
        return None


def apply_retention(directory: str, keep: int = DEFAULT_KEEP) -> None:
    """Deletes all but the newest `keep` checkpoints in `directory`."""
    files = sorted(_checkpoint_files(directory))
    for path in files[:-keep]:
        os.remove(path)


class ShardedSampler:
    """Shuffles range(num_examples) with `seed` and `epoch`, then splits it across `world_size`
    data-parallel workers. indices() returns the list of example indices worker `rank` processes
    this epoch."""

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
        per_worker = self.num_examples // self.world_size
        start = self.rank * per_worker
        return perm[start:start + per_worker]
