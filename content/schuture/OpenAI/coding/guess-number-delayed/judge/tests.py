from judgelib import calling, case, expect, expect_true, info, rng
import ref

# The judges in ref.py play the other side: they enforce every rule of the statement (guesses in range, never
# repeated, no call once the secret is determined, the call budget, one or two guesses per batch) and count calls.
# secret=None makes them an adversary that answers so the largest group of candidates survives and fixes the
# secret only when the answer returns.

PART = {
    1: ("find_secret", "Judge", "check"),
    2: ("find_secret_batched", "BatchJudge", "check_batch"),
}


def _n(n):
    return "10⁹" if n == 10 ** 9 else str(n)


def _snippet(part, n, secret, lean):
    fn, cls, method = PART[part]
    args = f"{n}, {secret}" if secret is not None else f"{n}, None, {lean}"
    note = f"n = {_n(n)}，secret = {secret}" if secret is not None else f"n = {_n(n)}，secret 由对手在你返回时选定"
    return (f"from ref import {cls}    # 评测程序扮演的 {method}（/judge/ref.py）\n"
            f"judge = {cls}({args})    # {note}\n"
            f"print({fn}({n}, judge.{method}))")


def _with_log(head, judge):
    lines = [line if len(line) <= 240 else line[:240] + " …" for line in judge.transcript()]
    return head + "\n# 调用记录：\n" + "\n".join("#   " + line for line in lines) if lines else head + "\n# 没有调用过"


def _run(fn, head, judge, *args):
    """Calls the answer; a rule broken on the way fails the case, whatever the answer did afterwards."""
    calling(head)
    try:
        got = fn(*args)
    except Exception:
        if judge.problem is None:
            calling(_with_log(head, judge))
            raise
        got = None
    if judge.problem:
        expect_true(False, judge.problem, call=_with_log(head, judge))
    return got


def play(sol, part, n, secret=None, lean=1):
    """One game of Part 1 or Part 2. Returns the number of calls used."""
    fn, cls, method = PART[part]
    judge = getattr(ref, cls)(n, secret, lean)
    head = _snippet(part, n, secret, lean)
    got = _run(getattr(sol, fn), head, judge, n, getattr(judge, method))
    lo, hi = judge.game.lo, judge.game.hi
    want = judge.game.settle(got)
    if ref.as_int(got) != want or isinstance(got, bool):
        msg = f"secret 是 {want}，返回了 {got!r}"
        if lo < hi:
            msg += (f"；返回时已经揭示的比较结果还不能确定 secret：[{lo}, {hi}] 里的 {hi - lo + 1} 个数都与它们相符"
                    + ("，评测在你返回时从中选定了 secret" if secret is None else ""))
        expect(got, want, call=_with_log(head, judge), msg=msg)
    return judge.calls


def play_games(sol, bounds, secrets=None, lean=1, head=None):
    """One run of Part 3. Returns the number of rounds used."""
    judge = ref.RoundJudge(bounds, secrets, lean)
    if head is None:
        what = f"{bounds!r}, {secrets!r}" if secrets is not None else f"{bounds!r}, None, {lean}"
        head = (f"from ref import RoundJudge    # 评测程序扮演的 check_round（/judge/ref.py）\n"
                f"judge = RoundJudge({what})\n"
                f"print(solve_games({bounds!r}, judge.check_round))")
    got = _run(sol.solve_games, head, judge, dict(bounds), judge.check_round)
    call = _with_log(head, judge)
    try:
        got = dict(got)
    except Exception:
        expect_true(False, f"应当返回把每个游戏编号映射到秘密数字的字典，得到 {type(got).__name__}", call=call, got=got)
    missing = [g for g in bounds if g not in got]
    extra = [g for g in got if g not in bounds]
    expect_true(not missing, f"返回的字典缺少游戏 {missing[:5]!r}", call=call, got=got)
    expect_true(not extra, f"返回的字典里有不存在的游戏编号 {extra[:5]!r}", call=call, got=got)
    for g in bounds:
        game = judge.games[g]
        lo, hi = game.lo, game.hi
        want = game.settle(got[g])
        if ref.as_int(got[g]) != want or isinstance(got[g], bool):
            msg = f"游戏 {g!r} 的 secret 是 {want}，返回了 {got[g]!r}"
            if lo < hi:
                msg += (f"；返回时已经揭示的比较结果还不能确定它：[{lo}, {hi}] 里的 {hi - lo + 1} 个数都与它们相符"
                        + ("，评测在你返回时从中选定了 secret" if secrets is None else ""))
            expect_true(False, msg, call=call, got=got)
    return judge.calls


# ---------------------------------------------------------------- sizes for the worst-case tests

SIZES = sorted({v for e in range(1, 30) for v in (2 ** e - 1, 2 ** e, 2 ** e + 1)}
               | {3 ** e + d for e in range(1, 19) for d in (-1, 0, 1)}
               | {10 ** e for e in range(1, 10)} | {10 ** 9 - 1, 999999937})
SIZES = [v for v in SIZES if 2 <= v <= 10 ** 9]


def worst_case(sol, part):
    """The adversary against every size in SIZES, both tie-breaking directions."""
    most = 0
    for n in SIZES:
        for lean in (1, -1):
            calls = play(sol, part, n, None, lean)
            if n == 10 ** 9:
                most = max(most, calls)
    base = "₂" if part == 1 else "₃"
    limit = 2 * ref.ceil_log(10 ** 9, 2 if part == 1 else 3) + 1
    info(f"n = 10⁹ 的最坏情况用了 {most} 次调用（上限 2⌈log{base} n⌉ + 1 = {limit}）")


def every_secret(sol, part, sizes):
    for n in sizes:
        for secret in range(1, n + 1):
            play(sol, part, n, secret)


def random_secrets(sol, part, salt):
    r = rng(salt)
    for _ in range(300):
        n = r.choice([r.randint(65, 300), r.randint(300, 5000), r.randint(5000, 10 ** 6)])
        play(sol, part, n, r.choice([1, n, r.randint(1, n)]))


def big_secrets(sol, part):
    n = 10 ** 9
    r = rng("big")
    for secret in [1, 2, n // 2, n // 2 + 1, n - 1, n] + [r.randint(1, n) for _ in range(40)]:
        play(sol, part, n, secret)
    for n in (2 ** 29, 2 ** 29 + 1, 3 ** 18, 3 ** 18 + 1):
        play(sol, part, n, r.randint(1, n))


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 n = 7")
def p1_example(sol):
    play(sol, 1, 7, 5)


@case("Part 1", title="n = 1：不能调用 check")
def p1_n1(sol):
    play(sol, 1, 1, 1)


@case("Part 1", title="n = 2、3")
def p1_n2_n3(sol):
    every_secret(sol, 1, [2, 3])


@case("Part 1", title="确定后不能再调用")
def p1_stop_once_determined(sol):
    for n, secret in [(7, 5), (2, 2), (8, 8), (100, 1), (100, 100), (1000, 377)]:
        play(sol, 1, n, secret)


@case("Part 1", title="n ≤ 12 的每个秘密数字")
def p1_tiny(sol):
    every_secret(sol, 1, range(4, 13))


@case("Part 1", title="n ≤ 64 的每个秘密数字")
def p1_small(sol):
    every_secret(sol, 1, range(13, 65))


@case("Part 1", title="随机 n 和秘密数字")
def p1_random(sol):
    random_secrets(sol, 1, "p1")


@case("Part 1", title="n = 10⁹")
def p1_big(sol):
    big_secrets(sol, 1)


@case("Part 1", title="最坏情况：调用次数上限")
def p1_worst_case(sol):
    worst_case(sol, 1)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 n = 13")
def p2_example(sol):
    play(sol, 2, 13, 9)


@case("Part 2", title="n = 1：不能调用 check_batch")
def p2_n1(sol):
    play(sol, 2, 1, 1)


@case("Part 2", title="n = 2、3")
def p2_n2_n3(sol):
    every_secret(sol, 2, [2, 3])


@case("Part 2", title="n ≤ 12 的每个秘密数字")
def p2_tiny(sol):
    every_secret(sol, 2, range(4, 13))


@case("Part 2", title="n ≤ 64 的每个秘密数字")
def p2_small(sol):
    every_secret(sol, 2, range(13, 65))


@case("Part 2", title="随机 n 和秘密数字")
def p2_random(sol):
    random_secrets(sol, 2, "p2")


@case("Part 2", title="n = 10⁹")
def p2_big(sol):
    big_secrets(sol, 2)


@case("Part 2", title="最坏情况：调用次数上限")
def p2_worst_case(sol):
    worst_case(sol, 2)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：两局游戏")
def p3_example(sol):
    play_games(sol, {"A": 5, "B": 20}, {"A": 4, "B": 15})


@case("Part 3", title="只有一局")
def p3_one_game(sol):
    for n, secret in [(1, 1), (2, 1), (2, 2), (7, 5), (100, 64), (10 ** 9, 123456789)]:
        play_games(sol, {"only": n}, {"only": secret})


@case("Part 3", title="n = 1 的游戏不能提交猜测")
def p3_n1_games(sol):
    play_games(sol, {"a": 1, "b": 50, "c": 1}, {"a": 1, "b": 17, "c": 1})
    play_games(sol, {"x": 1, "y": 1}, {"x": 1, "y": 1})


@case("Part 3", title="已确定的游戏不能再猜")
def p3_finished_games(sol):
    play_games(sol, {"tiny": 3, "small": 10, "big": 10 ** 9}, {"tiny": 2, "small": 7, "big": 987654321})
    play_games(sol, {"游戏1": 4, "游戏2": 1000}, {"游戏1": 4, "游戏2": 1})


@case("Part 3", title="随机小规模")
def p3_random_small(sol):
    r = rng("p3")
    for _ in range(300):
        bounds = {f"g{i}": r.randint(1, 10) for i in range(r.randint(1, 5))}
        play_games(sol, bounds, {g: r.randint(1, n) for g, n in bounds.items()})


@case("Part 3", title="300 局：轮数不随局数增长")
def p3_many_games(sol):
    seed = rng("many").randrange(10 ** 6)
    bounds, secrets = ref.many_games(300, seed)
    head = (f"from ref import RoundJudge, many_games    # 评测程序扮演的 check_round（/judge/ref.py）\n"
            f"bounds, secrets = many_games(300, {seed})    # 300 局，n 最大 10⁹\n"
            f"judge = RoundJudge(bounds, secrets)\n"
            f"print(solve_games(dict(bounds), judge.check_round))")
    rounds = play_games(sol, bounds, secrets, head=head)
    info(f"300 局一起用了 {rounds} 轮（上限 2⌈log₂(max n)⌉ + 1 = {2 * ref.ceil_log(max(bounds.values()), 2) + 1}）")


@case("Part 3", title="最坏情况：轮数上限")
def p3_worst_case(sol):
    r = rng("worst")
    for count, high, lean in [(40, 10 ** 9, 1), (40, 10 ** 9, -1), (20, 1000, 1), (20, 1000, -1), (5, 2, 1)]:
        seed = r.randrange(10 ** 6)
        bounds, _ = ref.many_games(count, seed, high)
        head = (f"from ref import RoundJudge, many_games    # 评测程序扮演的 check_round（/judge/ref.py）\n"
                f"bounds, _ = many_games({count}, {seed}, {high})    # {count} 局，n 最大 {_n(high)}\n"
                f"judge = RoundJudge(bounds, None, {lean})    # 每局的 secret 由对手在你返回时选定\n"
                f"print(solve_games(dict(bounds), judge.check_round))")
        play_games(sol, bounds, None, lean, head=head)
