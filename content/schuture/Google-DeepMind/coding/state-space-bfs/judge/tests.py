from math import comb

from judgelib import calling, case, expect, rng, show_call
import ref

MOD = 1_000_000_007

GRID_A = ["S..AT", "##.##", "..a.."]           # Part 1 example: 8 moves, a unique shortest path
GRID_LOCKED = ["S.AT", ".#.#", ".#a#"]         # Part 1 example: the key sits behind its own door
GRID_C = ["S..AT", ".#.##", "..a.."]           # Part 2 example: 8 moves, 2 shortest paths
GRID_D = ["S~~T", "...."]                      # Part 3 example: cost 5, 2 cheapest paths

# A one-row corridor with six keys: a is right of S, b left behind A, c right behind B, and so on, so the robot
# walks back and forth: 1 + 3 + 5 + 7 + 9 + 11 + 13 = 49 moves.
CHAIN = ["fEdCbASaBcDeFT"]


def _mod_hint(fn, got, want, extra):
    """A hint when a count is right except that it was not reduced modulo mod."""
    mod = extra[0] if extra else MOD
    if fn.startswith("count") and type(got) is int and got != want and got % mod == want:
        return f"路径条数本身没错，但返回值还要对 mod = {mod} 取模（应在 0 到 mod - 1 之间）"
    return ""


def check(sol, fn, grid, want, *extra, msg=""):
    """Calls sol.fn(copy of grid, *extra) and compares with want; the call shown can be pasted as is."""
    call = calling(show_call(fn, grid, *extra))
    got = getattr(sol, fn)(list(grid), *extra)
    expect(got, want, call, msg or _mod_hint(fn, got, want, extra))


def check_text(sol, fn, grid, text, want, *extra):
    """The same for a big grid, shown as the Python expression `text` that builds it."""
    call = calling(f"{fn}({text}{''.join(', ' + repr(x) for x in extra)})")
    got = getattr(sol, fn)(list(grid), *extra)
    expect(got, want, call, _mod_hint(fn, got, want, extra))


def open_grid(rows, cols, fill="."):
    g = [fill * cols for _ in range(rows)]
    g[0] = "S" + g[0][1:]
    g[-1] = g[-1][:-1] + "T"
    return g


def open_text(rows, cols, fill="."):
    return (f'["S" + "{fill}" * {cols - 1}] + ["{fill}" * {cols}] * {rows - 2} + ["{fill}" * {cols - 1} + "T"]')


def small_grids(r, count, mud):
    out = []
    for _ in range(count):
        g = ref.random_grid(r, r.randint(1, 6), r.randint(2, 7), r.randint(0, 4),
                            wall=r.choice([0.1, 0.2, 0.3]), mud=mud and r.choice([0.2, 0.4]), extra=r.randint(0, 3))
        out.append(g)
    return out


def medium_grids(r, count, mud):
    return [ref.random_grid(r, r.randint(8, 12), r.randint(10, 14), r.randint(3, 6), wall=0.22, mud=mud, extra=2)
            for _ in range(count)]


# Medium grids and their brute-force answers are built at import time, which is not charged to any case.
_med = rng("medium")
MEDIUM = [(g, ref.brute(g)) for g in medium_grids(_med, 12, 0.0)]
MEDIUM_MUD = [(g, ref.brute(g, True)) for g in medium_grids(_med, 12, 0.25)]


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：绕路拿钥匙，8 步")
def p1_example(sol):
    check(sol, "shortest_path", GRID_A, 8)


@case("Part 1", sample=True, title="题目例子：钥匙在门后，不可达")
def p1_example_unreachable(sol):
    check(sol, "shortest_path", GRID_LOCKED, -1)


@case("Part 1", title="S 和 T 相邻；只有一列的网格")
def p1_adjacent(sol):
    check(sol, "shortest_path", ["ST"], 1)
    check(sol, "shortest_path", ["TS"], 1)
    check(sol, "shortest_path", ["S", "T"], 1)
    check(sol, "shortest_path", ["S", ".", ".", "T"], 3)


@case("Part 1", title="被墙隔开；不能斜着走")
def p1_walls(sol):
    check(sol, "shortest_path", ["S.#T"], -1)
    check(sol, "shortest_path", ["S#", "#T"], -1)
    check(sol, "shortest_path", ["S.#", "#.#", "##T"], -1)
    check(sol, "shortest_path", ["S#..", "...#", "#..T"], 5)


@case("Part 1", title="拿到钥匙后要回头经过 S")
def p1_back_through_start(sol):
    check(sol, "shortest_path", ["aS.AT"], 5)
    check(sol, "shortest_path", ["S.AT", "a###"], 5)


@case("Part 1", title="没有对应钥匙的门永远打不开")
def p1_door_without_key(sol):
    check(sol, "shortest_path", ["SBT"], -1)
    check(sol, "shortest_path", ["S.BT", "...."], 5)
    check(sol, "shortest_path", ["S.AT", "b###"], -1)


@case("Part 1", title="同一扇门过两次：钥匙不会用掉")
def p1_door_twice(sol):
    check(sol, "shortest_path", ["SaA.b", "##B##", "##T##"], 8)


@case("Part 1", title="六把钥匙，来回取")
def p1_key_chain(sol):
    check(sol, "shortest_path", CHAIN, 49)


@case("Part 1", title="随机小网格（对照暴力解）")
def p1_random(sol):
    for g in small_grids(rng(), 150, 0.0):
        check(sol, "shortest_path", g, ref.brute(g)[0])


@case("Part 1", title="随机中等网格，3 到 6 种钥匙")
def p1_medium(sol):
    for g, (d, _) in MEDIUM:
        check(sol, "shortest_path", g, d)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：两条最短路径")
def p2_example(sol):
    check(sol, "count_shortest_paths", GRID_C, 2)


@case("Part 2", title="唯一的最短路径；不可达返回 0")
def p2_unique_and_unreachable(sol):
    check(sol, "count_shortest_paths", GRID_A, 1)
    check(sol, "count_shortest_paths", GRID_LOCKED, 0)
    check(sol, "count_shortest_paths", ["S#T"], 0)


@case("Part 2", title="空地：路径数是组合数")
def p2_open_grid(sol):
    check(sol, "count_shortest_paths", ["ST"], 1)
    check(sol, "count_shortest_paths", ["S.", ".T"], 2)
    check(sol, "count_shortest_paths", ["S..", "...", "..T"], 6)
    check(sol, "count_shortest_paths", ["S...", "....", "...T"], 10)
    check(sol, "count_shortest_paths", open_grid(6, 6), 252)


@case("Part 2", title="回头的路线不同，也是不同的路径")
def p2_different_returns(sol):
    # key a: 2 moves away by one route; back to (0, 1) in 3 moves by three routes; then through A into T
    check(sol, "count_shortest_paths", ["S.AT", "..##", "a.##"], 3)


@case("Part 2", title="到达 T 时持有的钥匙可以不同")
def p2_keys_at_target(sol):
    check(sol, "count_shortest_paths", ["S.", "aT"], 2)
    check(sol, "count_shortest_paths", ["Sa.", ".b.", "..T"], 6)


@case("Part 2", title="只数最短的：更长的路线不算")
def p2_longer_routes_ignored(sol):
    check(sol, "count_shortest_paths", ["S.T", "a.."], 1)
    check(sol, "count_shortest_paths", ["aS.T"], 1)


@case("Part 2", title="传入的 mod")
def p2_custom_mod(sol):
    check(sol, "count_shortest_paths", open_grid(6, 6), 52, 100)
    check(sol, "count_shortest_paths", open_grid(6, 6), 0, 7)
    check(sol, "count_shortest_paths", GRID_C, 0, 1)
    check(sol, "count_shortest_paths", GRID_C, 2, 5)
    check(sol, "count_shortest_paths", GRID_A, 0, 1)            # one path: 1 % 1 == 0
    check(sol, "count_shortest_paths", ["S.", "aT"], 0, 2)      # one path with key a, one without: 2 % 2 == 0


@case("Part 2", title="随机小网格（对照暴力解）")
def p2_random(sol):
    for g in small_grids(rng(), 150, 0.0):
        check(sol, "count_shortest_paths", g, ref.brute(g)[1] % MOD)


@case("Part 2", title="随机中等网格，3 到 6 种钥匙")
def p2_medium(sol):
    for g, (_, n) in MEDIUM:
        check(sol, "count_shortest_paths", g, n % MOD)


@case("Part 2", cpu=2, title="40×40 空地：路径数巨大，要取模")
def p2_big_open(sol):
    g = open_grid(40, 40)
    check_text(sol, "count_shortest_paths", g, open_text(40, 40), comb(78, 39) % MOD)
    check_text(sol, "count_shortest_paths", g, open_text(40, 40), comb(78, 39) % 998_244_353, 998_244_353)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：穿过泥地和绕路一样贵")
def p3_example(sol):
    check(sol, "shortest_path_weighted", GRID_D, 5)
    check(sol, "count_shortest_paths_weighted", GRID_D, 2)


@case("Part 3", title="绕开泥地更便宜")
def p3_mud_detour(sol):
    check(sol, "shortest_path_weighted", ["S~~~T", "....."], 6)
    check(sol, "count_shortest_paths_weighted", ["S~~~T", "....."], 1)
    check(sol, "shortest_path_weighted", ["S~T"], 3)
    check(sol, "count_shortest_paths_weighted", ["S~T", "..."], 1)


@case("Part 3", title="没有泥地时和 Part 1、2 一样")
def p3_no_mud(sol):
    for g, d, n in ((GRID_A, 8, 1), (GRID_C, 8, 2), (GRID_LOCKED, -1, 0), (["S.", "aT"], 2, 2), (CHAIN, 49, 1)):
        check(sol, "shortest_path_weighted", g, d)
        check(sol, "count_shortest_paths_weighted", g, n)


@case("Part 3", title="泥地、钥匙和门一起")
def p3_mud_keys(sol):
    check(sol, "shortest_path_weighted", ["S~AT", "a..."], 5)
    check(sol, "count_shortest_paths_weighted", ["S~AT", "a..."], 2)
    check(sol, "shortest_path_weighted", ["S~AT", "~#.#", "~#a#"], -1)
    check(sol, "count_shortest_paths_weighted", ["S~AT", "~#.#", "~#a#"], 0)


@case("Part 3", title="传入的 mod")
def p3_custom_mod(sol):
    g = open_grid(6, 6, "~")
    check(sol, "count_shortest_paths_weighted", g, 52, 100)
    check(sol, "count_shortest_paths_weighted", GRID_D, 0, 1)
    check(sol, "count_shortest_paths_weighted", ["S~T"], 0, 1)
    check(sol, "count_shortest_paths_weighted", ["S.", "aT"], 0, 2)   # cost 2 by either route: 2 % 2 == 0


@case("Part 3", title="随机小网格（对照暴力解）")
def p3_random(sol):
    for g in small_grids(rng(), 150, 1):
        d, n = ref.brute(g, True)
        check(sol, "shortest_path_weighted", g, d)
        check(sol, "count_shortest_paths_weighted", g, n % MOD)


@case("Part 3", title="随机中等网格，带泥地")
def p3_medium(sol):
    for g, (d, n) in MEDIUM_MUD:
        check(sol, "shortest_path_weighted", g, d)
        check(sol, "count_shortest_paths_weighted", g, n % MOD)


@case("Part 3", cpu=2, title="40×40 全是泥地：要取模")
def p3_big_mud(sol):
    g = open_grid(40, 40, "~")
    text = open_text(40, 40, "~")
    check_text(sol, "shortest_path_weighted", g, text, 2 * 77 + 1)
    check_text(sol, "count_shortest_paths_weighted", g, text, comb(78, 39) % MOD)
