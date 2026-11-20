"""Helpers for tests.py: a simulated 10-rank cluster that counts bytes, plain references and checkers.

`Cluster().run(fn, shards)` has the same interface as the page's verification code, so an answer can also
use it in a custom run (`from ref import Cluster`)."""
import copy
import operator
import pickle
import threading
import time
from collections import Counter, deque

from judgelib import Mismatch

P = 10


class _Abort(BaseException):
    """Raised inside a rank's send/recv/barrier once the run is over, so a blocked rank thread unwinds."""


class Cluster:
    """Ten ranks on real threads. send() never blocks and delivers a pickled copy; recv(src) blocks until a
    message from src is there; barrier() releases once all ranks have called it. Every send costs
    len(pickle.dumps(obj)) bytes, counted per sending and per receiving rank. When every rank that has not
    returned is blocked with nothing that could ever wake it, the run stops and reports a deadlock."""

    def __init__(self, num_ranks: int = P):
        self.num_ranks = n = num_ranks
        self._cond = threading.Condition()
        self._box = [[deque() for _ in range(n)] for _ in range(n)]      # [dst][src]
        self._state = ["run"] * n                                         # run | ("recv", src) | barrier | done
        self._at_barrier = 0
        self._generation = 0
        self._over = False
        self.deadlock = None
        self.bytes_sent = 0
        self.sent = [0] * n            # bytes each rank sent
        self.received = [0] * n        # bytes sent to each rank
        self.bytes_received = self.received
        self.messages = 0

    # ------------------------------------------------------------ primitives
    def _rank_arg(self, x, what: str) -> int:
        try:
            r = operator.index(x)
        except TypeError:
            raise TypeError(f"{what} 应当是 0 到 {self.num_ranks - 1} 的整数 rank，得到 {x!r}") from None
        if not 0 <= r < self.num_ranks:
            raise ValueError(f"{what} 应当是 0 到 {self.num_ranks - 1} 的整数 rank，得到 {x!r}")
        return r

    def _make_send(self, rank: int):
        def send(dst, obj):
            if self._over:
                raise _Abort
            dst = self._rank_arg(dst, "send 的 dst")
            data = pickle.dumps(obj)
            try:
                item = pickle.loads(data)      # the receiver gets a copy, as over a real network
            except Exception:
                item = copy.deepcopy(obj)
            with self._cond:
                self.sent[rank] += len(data)
                self.received[dst] += len(data)
                self.bytes_sent += len(data)
                self.messages += 1
                self._box[dst][rank].append(item)
                self._cond.notify_all()
        return send

    def _make_recv(self, rank: int):
        def recv(src):
            src = self._rank_arg(src, "recv 的 src")
            with self._cond:
                box = self._box[rank][src]
                while not box:
                    if self._over:
                        raise _Abort
                    self._state[rank] = ("recv", src)
                    self._check_stuck()
                    if self._over:
                        raise _Abort
                    self._cond.wait(0.5)
                self._state[rank] = "run"
                return box.popleft()
        return recv

    def _make_barrier(self, rank: int):
        def barrier():
            with self._cond:
                if self._over:
                    raise _Abort
                generation = self._generation
                self._at_barrier += 1
                if self._at_barrier == self.num_ranks:
                    self._at_barrier = 0
                    self._generation += 1
                    for r in range(self.num_ranks):
                        if self._state[r] == "barrier":
                            self._state[r] = "run"
                    self._cond.notify_all()
                    return
                self._state[rank] = "barrier"
                self._check_stuck()
                while self._generation == generation:
                    if self._over:
                        raise _Abort
                    self._cond.wait(0.5)
        return barrier

    def _check_stuck(self) -> None:
        """Called with the lock held after a rank blocks or returns."""
        if self._over:
            return                          # already stopped; keep the first diagnosis
        waiting = []
        for r, st in enumerate(self._state):
            if st == "run":
                return
            if st == "done":
                continue
            if st != "barrier" and self._box[r][st[1]]:
                return                      # its message is already there; it is about to wake
            waiting.append(r)
        if not waiting:
            return
        done = [r for r, st in enumerate(self._state) if st == "done"]
        parts = []
        for r in waiting:
            st = self._state[r]
            if st == "barrier":
                parts.append(f"rank {r} 在 barrier() 里等待")
            else:
                src = st[1]
                tail = "（它已经返回了）" if src in done else ""
                parts.append(f"rank {r} 在 recv({src}) 里等待 rank {src} 的消息{tail}")
        text = "；".join(parts)
        if done:
            text += f"；已经返回的 rank：{', '.join(map(str, done))}"
        if any(self._state[r] == "barrier" for r in waiting) and done:
            text += "（barrier() 要等全部 10 个 rank 都调用才会放行）"
        self.deadlock = "死锁：还没返回的 rank 全部在等待，再也不会有新消息。" + text
        self._over = True
        self._cond.notify_all()

    # ------------------------------------------------------------ running
    def run(self, fn, shards: list, wait: float = 8.0) -> list:
        n = self.num_ranks
        results = [None] * n
        errors = []

        def worker(rank: int) -> None:
            try:
                results[rank] = fn(rank, shards[rank], self._make_send(rank), self._make_recv(rank),
                                   self._make_barrier(rank))
            except _Abort:
                pass
            except BaseException as e:      # noqa: BLE001 -- reported from the main thread below
                with self._cond:
                    errors.append((rank, e))
            finally:
                with self._cond:
                    self._state[rank] = "done"
                    self._check_stuck()

        threads = [threading.Thread(target=worker, args=(r,), daemon=True, name=f"rank-{r}") for r in range(n)]
        try:
            for t in threads:
                t.start()
            deadline = time.monotonic() + wait
            for t in threads:
                while t.is_alive() and time.monotonic() < deadline:
                    t.join(min(0.2, max(0.0, deadline - time.monotonic())))
        finally:
            with self._cond:            # unblock every rank still waiting, also when the case timed out
                self._over = True
                self._cond.notify_all()
        if errors:
            rank, e = errors[0]
            raise e
        if self.deadlock:
            raise Mismatch(self.deadlock, show=False)
        late = [r for r, t in enumerate(threads) if t.is_alive()]
        if late:
            raise Mismatch(f"{wait:g} 秒后 rank {', '.join(map(str, late))} 还没有返回", show=False)
        return results


def run(fn, shards: list) -> tuple:
    """Runs fn on a fresh cluster with copies of the shards; returns (results, cluster)."""
    c = Cluster()
    results = c.run(fn, [list(s) for s in shards])
    return results, c


# ---------------------------------------------------------------- plain references

def mode_of(values: list) -> int:
    counts = Counter(values)
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def median_of(values: list) -> int:
    s = sorted(values)
    return s[(len(s) - 1) // 2]


def histogram_bytes(shards: list) -> int:
    """H: the bytes to ship every shard's local histogram once, as a list of (value, count) tuples."""
    return sum(len(pickle.dumps(sorted(Counter(s).items()))) for s in shards)


def data_bytes(shards: list) -> int:
    """W: the bytes of the whole dataset pickled as one list."""
    return len(pickle.dumps([v for s in shards for v in s]))


def sort_problem(results: list, flat_sorted: list):
    """None when results is a legal sample_sort output for the dataset, else a Chinese sentence."""
    if not isinstance(results, list) or len(results) != P:
        return "内部错误：结果数量不对"
    for r, part in enumerate(results):
        if not isinstance(part, (list, tuple)):
            return f"rank {r} 的返回值应当是 list，得到 {type(part).__name__}"
    for r, part in enumerate(results):
        for i in range(len(part) - 1):
            if part[i + 1] < part[i]:
                return f"result[{r}] 本身不是有序的：下标 {i} 是 {part[i]!r}，下标 {i + 1} 是 {part[i + 1]!r}"
    last, last_rank = None, None
    for r, part in enumerate(results):
        if part:
            if last is not None and part[0] < last:
                return (f"result[{last_rank}] 的最后一个元素 {last!r} 大于 result[{r}] 的第一个元素 {part[0]!r}，"
                        f"按 rank 顺序拼接后不是非降序")
            last, last_rank = part[-1], r
    flat = [v for part in results for v in part]
    if flat != flat_sorted:
        got, want = Counter(flat), Counter(flat_sorted)
        extra = sorted((got - want).elements())
        missing = sorted((want - got).elements())
        bits = []
        if missing:
            bits.append(f"少了 {len(missing)} 个元素（例如 {missing[:5]}）")
        if extra:
            bits.append(f"多了 {len(extra)} 个元素（例如 {extra[:5]}）")
        return "拼接后的元素和原数据不一致：" + "，".join(bits)
    return None
