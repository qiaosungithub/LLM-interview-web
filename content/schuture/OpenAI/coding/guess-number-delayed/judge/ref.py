"""The other side of the game for tests.py: Judge (check), BatchJudge (check_batch) and RoundJudge (check_round)
enforce every rule of the statement against a hidden secret and count the calls.

With secret=None a judge is an adversary: every comparison it reveals is consistent with some secret, chosen so
that the largest group of candidates survives; the secret is fixed only when the answer returns, as one that agrees
with everything revealed. A deterministic answer sees exactly what it would see if that secret had been fixed from
the start.

They can be used in 自定义运行 to replay a test, for example:
    from ref import Judge
    judge = Judge(7, 5)
    print(find_secret(7, judge.check), judge.transcript())
"""
import operator


class Rejected(ValueError):
    """Raised when a call breaks the rules. The first reason is kept in judge.problem."""


def ceil_log(n, base):
    """ceil(log_base(n)) by integer arithmetic."""
    k, v = 0, 1
    while v < n:
        v, k = v * base, k + 1
    return k


def as_int(x):
    """x as an int, or None if it is not an integer (an integral float counts as one)."""
    if isinstance(x, float):
        return int(x) if x.is_integer() else None
    try:
        return operator.index(x)
    except TypeError:
        return None


class Game:
    """One secret in [1, n]: the interval [lo, hi] of candidates consistent with the comparisons revealed so far, and
    the guesses used."""

    def __init__(self, n, secret=None, lean=1):
        self.n, self.secret, self.lean = n, secret, lean
        self.lo, self.hi = 1, n
        self.used = set()

    def determined(self):
        return self.lo == self.hi

    def bad_guess(self, x, seen=()):
        """None if x is a legal new guess, else the reason in Chinese."""
        v = as_int(x)
        if v is None:
            return f"猜测必须是整数，得到 {x!r}"
        if not 1 <= v <= self.n:
            return f"猜测 {v} 不在 [1, {self.n}] 内"
        if v in self.used or v in seen:
            return f"猜测 {v} 在这局里已经用过，不能重复"
        return None

    def reveal(self, xs):
        """The comparisons of the guesses xs against the secret, in order; narrows [lo, hi]."""
        if self.secret is None:
            a, b = self._choose(xs)
            out = [-1 if x < a else (1 if x > b else 0) for x in xs]
        else:
            out = [-1 if x < self.secret else (1 if x > self.secret else 0) for x in xs]
        for x, r in zip(xs, out):
            if r == -1:
                self.lo = max(self.lo, x + 1)
            elif r == 1:
                self.hi = min(self.hi, x - 1)
            else:
                self.lo = self.hi = x
        return out

    def _choose(self, xs):
        """The adversary's answer: the largest of the groups the guesses split [lo, hi] into (a guess itself is a
        group of one, chosen last); ties go to the highest group if lean > 0, else the lowest."""
        cuts = sorted(x for x in set(xs) if self.lo <= x <= self.hi)
        groups, start = [], self.lo
        for x in cuts:
            if start < x:
                groups.append((x - start, 1, start * self.lean, (start, x - 1)))
            groups.append((1, 0, x * self.lean, (x, x)))
            start = x + 1
        if start <= self.hi:
            groups.append((self.hi - start + 1, 1, start * self.lean, (start, self.hi)))
        return max(groups)[3]

    def settle(self, got):
        """Fixes the secret when the answer returns got (an adversary picks a candidate other than got if it can)."""
        if self.secret is None:
            v = as_int(got)
            self.secret = self.hi if v == self.lo and self.lo < self.hi else self.lo
        return self.secret


_REJECTED = object()        # the result logged for a rejected call


def _format(x):
    return repr(x) if as_int(x) is None or isinstance(x, (bool, float)) else str(as_int(x))


class _Judge:
    name = ""

    def __init__(self, limit):
        self.limit, self.calls, self.pending = limit, 0, None
        self.log = []                   # (argument as shown, result or _REJECTED)
        self.problem = None

    def _reject(self, shown, reason):
        self.log.append((shown, _REJECTED))
        if self.problem is None:
            self.problem = f"第 {len(self.log)} 次调用 {self.name}({shown}) 被拒绝：{reason}"
        raise Rejected(reason)

    def transcript(self, last=12):
        lines = [f"{self.name}({a}) -> {'被拒绝' if r is _REJECTED else repr(r)}" for a, r in self.log]
        if len(lines) > last:
            lines = [f"…（前 {len(lines) - last} 次调用省略）"] + lines[-last:]
        return lines


class Judge(_Judge):
    """Part 1: check(x) for one secret in [1, n]."""
    name = "check"

    def __init__(self, n, secret=None, lean=1):
        super().__init__(2 * ceil_log(n, 2) + 1)
        self.n, self.game = n, Game(n, secret, lean)

    def check(self, x):
        shown = _format(x)
        if self.game.determined():
            self._reject(shown, "n = 1 时第一次调用之前秘密数字就已经确定，一次都不能调用 check" if self.n == 1 else
                         f"已经揭示的比较结果只剩下一个可能的 secret（{self.game.lo}），不能再调用 check")
        if self.calls >= self.limit:
            self._reject(shown, f"check 最多只能调用 2⌈log₂ n⌉ + 1 = {self.limit} 次")
        reason = self.game.bad_guess(x)
        if reason:
            self._reject(shown, reason)
        x = as_int(x)
        self.calls += 1
        self.game.used.add(x)
        result = None if self.pending is None else self.game.reveal([self.pending])[0]
        self.pending = x
        self.log.append((shown, result))
        return result


class BatchJudge(_Judge):
    """Part 2: check_batch(xs) with one or two guesses per call, for one secret in [1, n]."""
    name = "check_batch"

    def __init__(self, n, secret=None, lean=1):
        super().__init__(2 * ceil_log(n, 3) + 1)
        self.n, self.game = n, Game(n, secret, lean)

    def check_batch(self, xs):
        try:
            xs = list(xs)
        except TypeError:
            self._reject(repr(xs), "xs 应当是猜测的列表")
        shown = "[" + ", ".join(_format(x) for x in xs) + "]"
        if self.game.determined():
            self._reject(shown, "n = 1 时第一次调用之前秘密数字就已经确定，一次都不能调用 check_batch" if self.n == 1
                         else f"已经揭示的比较结果只剩下一个可能的 secret（{self.game.lo}），不能再调用 check_batch")
        if self.calls >= self.limit:
            self._reject(shown, f"check_batch 最多只能调用 2⌈log₃ n⌉ + 1 = {self.limit} 次")
        if not 1 <= len(xs) <= 2:
            self._reject(shown, f"每次调用要带一个或两个猜测，这次带了 {len(xs)} 个")
        seen = set()
        for x in xs:
            reason = self.game.bad_guess(x, seen)
            if reason:
                self._reject(shown, reason)
            seen.add(as_int(x))
        xs = [as_int(x) for x in xs]
        self.calls += 1
        self.game.used.update(xs)
        result = None if self.pending is None else self.game.reveal(self.pending)
        self.pending = xs
        self.log.append((shown, result))
        return None if result is None else list(result)


class RoundJudge(_Judge):
    """Part 3: check_round(guesses) for several games behind one channel."""
    name = "check_round"

    def __init__(self, bounds, secrets=None, lean=1):
        super().__init__(2 * ceil_log(max(bounds.values()), 2) + 1)
        self.games = {g: Game(n, None if secrets is None else secrets[g], lean) for g, n in bounds.items()}

    def check_round(self, guesses):
        try:
            guesses = dict(guesses)
        except (TypeError, ValueError):
            self._reject(repr(guesses), "guesses 应当是把游戏编号映射到猜测的字典")
        shown = "{" + ", ".join(f"{g!r}: {_format(x)}" for g, x in guesses.items()) + "}"
        if self.calls >= self.limit:
            self._reject(shown, f"check_round 最多只能调用 2⌈log₂(max n)⌉ + 1 = {self.limit} 次")
        for g, x in guesses.items():
            game = self.games.get(g)
            if game is None:
                self._reject(shown, f"没有编号为 {g!r} 的游戏")
            if game.determined():
                self._reject(shown, f"游戏 {g!r} 的秘密数字已经确定（{game.lo}），不能再为它提交猜测")
            reason = game.bad_guess(x)
            if reason:
                self._reject(shown, f"游戏 {g!r}：{reason}")
        guesses = {g: as_int(x) for g, x in guesses.items()}
        self.calls += 1
        for g, x in guesses.items():
            self.games[g].used.add(x)
        result = None if self.pending is None else {g: self.games[g].reveal([x])[0] for g, x in self.pending.items()}
        self.pending = guesses
        self.log.append((shown, result))
        return None if result is None else dict(result)


# ---------------------------------------------------------------- inputs for Part 3

def many_games(count, seed, high=10 ** 9):
    """(bounds, secrets) for count games with ids "g0", "g1", ...: about one in ten has n <= 3, one in ten has
    n = high, the rest are spread on a log scale up to high."""
    import random
    r = random.Random(seed)
    bounds = {}
    for i in range(count):
        u = r.random()
        bounds[f"g{i}"] = (r.randint(1, min(3, high)) if u < 0.1 else high if u < 0.2 else
                           r.randint(1, min(high, 10 ** r.randint(1, 9))))
    return bounds, {g: r.randint(1, n) for g, n in bounds.items()}
