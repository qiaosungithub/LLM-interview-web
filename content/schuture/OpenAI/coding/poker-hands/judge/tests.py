from judgelib import calling, case, expect, expect_true, rng, show_call
import ref

# Part 1 ignores suits and Part 2 adds the flush, and one file can follow only one of them. Part 1 tests therefore
# never use a hand whose three cards share a suit: on every other hand the two parts agree.
# Parts 1 and 2 allow N up to 1e5 players, so cards repeat across players; within one hand the cards are distinct.

STRATEGIES = ("never_fold", "fold_unless_pair", "fold_unless_flush_draw")


def need(sol, *names):
    for name in names:
        expect_true(callable(getattr(sol, name, None)), f"没有找到 {name}（名字要和题目一致）")


def check_type(sol, cards, want, why=""):
    need(sol, "hand_type")
    call = calling(show_call("hand_type", cards))
    expect(sol.hand_type(list(cards)), want, call=call, msg=f"牌型应当是 {want}" + (f"（{why}）" if why else ""))


def check_compare(sol, a, b, want, why=""):
    """compare_hands(a, b) == want, and the reverse call gives -want."""
    need(sol, "compare_hands")
    for x, y, w in ((a, b, want), (b, a, -want)):
        call = calling(show_call("compare_hands", x, y))
        got = sol.compare_hands(list(x), list(y))
        expect(got, w, call=call, msg=f"compare_hands 应当返回 {w}" + (f"（{why}）" if why else ""))


def _as_list(got):
    if isinstance(got, (str, bytes, dict, set)) or not hasattr(got, "__iter__"):
        return got
    return list(got)


def check_rank(sol, hands, want, why="", call=None):
    need(sol, "rank_players")
    call = calling(call or show_call("rank_players", hands))
    got = _as_list(sol.rank_players([list(h) for h in hands]))
    msg = "获胜者不对" + (f"（{why}）" if why else "")
    if isinstance(got, list) and sorted(got) == want and got != want:
        msg = "获胜者对了，但要按下标从小到大返回"
    expect(got, want, call=call, msg=msg)


def random_compares(sol, r, n, flush, pool):
    for _ in range(n):
        a = ref.random_hand(r, pool, allow_flush=flush) if not flush else flush_heavy_hand(r, pool)
        b = ref.random_hand(r, pool, allow_flush=flush) if not flush else flush_heavy_hand(r, pool)
        check_compare(sol, a, b, ref.compare(a, b, flush))


def random_ranks(sol, r, n, flush, pool, players=(2, 12)):
    for _ in range(n):
        k = r.randint(*players)
        hands = [flush_heavy_hand(r, pool) if flush else ref.random_hand(r, pool, allow_flush=False)
                 for _ in range(k)]
        check_rank(sol, hands, ref.winners(hands, flush))


def flush_heavy_hand(r, pool):
    """About a third of these hands are flushes (three distinct ranks of one suit drawn from the pool)."""
    if r.random() < 0.35:
        suit = r.choice(ref.SUITS)
        same = [c for c in pool if c[1] == suit]
        if len(same) >= 3:
            return r.sample(same, 3)
    return ref.random_hand(r, pool)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：两手对子并列")
def p1_example(sol):
    p0, p1, p2 = ["7H", "7D", "4C"], ["7S", "7C", "4D"], ["KD", "QS", "JC"]
    check_type(sol, p0, "Pair")
    check_type(sol, p1, "Pair")
    check_type(sol, p2, "High Card")
    check_rank(sol, [p0, p1, p2], [0, 1])


@case("Part 1", title="三种牌型，对子在任意位置")
def p1_hand_type(sol):
    check_type(sol, ["9H", "TC", "9D"], "Pair", "两张 9 不相邻也是对子")
    check_type(sol, ["4C", "7H", "7D"], "Pair")
    check_type(sol, ["7H", "4C", "7D"], "Pair", "两张 7 不相邻也是对子")
    check_type(sol, ["QS", "QD", "2C"], "Pair")
    check_type(sol, ["AS", "AD", "AH"], "Three of a Kind")
    check_type(sol, ["2C", "2D", "2S"], "Three of a Kind")
    check_type(sol, ["2C", "3D", "AH"], "High Card")
    check_type(sol, ["TC", "JD", "QH"], "High Card", "连续的点数在这里不是牌型")
    check_type(sol, ["AH", "KH", "QD"], "High Card")


@case("Part 1", title="牌型更强的永远赢")
def p1_type_order(sol):
    check_compare(sol, ["2C", "2D", "2H"], ["AC", "AD", "KH"], 1, "三条 2 胜过对子 A")
    check_compare(sol, ["2C", "2D", "3H"], ["AC", "KD", "QH"], 1, "对子 2 胜过高牌 A K Q")
    check_compare(sol, ["3S", "3D", "3C"], ["KS", "QH", "JC"], 1)
    check_compare(sol, ["5C", "4D", "5H"], ["AS", "KH", "JD"], 1)


@case("Part 1", title="对子：先比对子，再比单牌")
def p1_pair_compare(sol):
    check_compare(sol, ["8C", "8D", "2H"], ["7C", "7D", "AH"], 1, "先比对子的点数")
    check_compare(sol, ["7H", "7D", "4C"], ["7S", "7C", "5D"], -1, "对子相同时比单牌")
    check_compare(sol, ["5D", "7H", "7C"], ["7S", "4C", "7D"], 1, "对子相同时比单牌")
    check_compare(sol, ["QC", "2D", "QH"], ["QS", "QD", "AC"], -1, "对子相同时比单牌")
    check_compare(sol, ["9C", "9D", "AH"], ["9H", "9S", "2C"], 1, "对子相同时比单牌")
    check_compare(sol, ["7H", "7D", "4C"], ["4S", "7C", "7S"], 0, "对子和单牌都相同就并列")


@case("Part 1", title="高牌：从大到小逐张比较")
def p1_high_card_compare(sol):
    check_compare(sol, ["AC", "3D", "2H"], ["KC", "QD", "JH"], 1, "先比最大的一张")
    check_compare(sol, ["AC", "KD", "2H"], ["AD", "QH", "JC"], 1, "最大的相同时比中间的")
    check_compare(sol, ["AC", "KD", "3H"], ["AD", "KH", "2C"], 1, "前两张相同时比最小的")
    check_compare(sol, ["2C", "AD", "KH"], ["KS", "QD", "AH"], -1, "先把点数从大到小排好再比")
    check_compare(sol, ["9C", "5D", "3H"], ["3D", "9H", "5S"], 0, "点数全相同就并列")
    check_compare(sol, ["JC", "8D", "4H"], ["JS", "8C", "6D"], -1)


@case("Part 1", title="三条的比较；花色不打破并列")
def p1_trips_and_suits(sol):
    check_compare(sol, ["2C", "2D", "2H"], ["AC", "AD", "AH"], -1)
    check_compare(sol, ["KC", "KD", "KH"], ["QS", "QD", "QC"], 1)
    check_compare(sol, ["KC", "KD", "KH"], ["KS", "KD", "KC"], 0, "三条点数相同就并列，花色不参与比较")
    check_compare(sol, ["AS", "KH", "QH"], ["AC", "KD", "QS"], 0, "花色不参与比较")
    check_compare(sol, ["TS", "TH", "3S"], ["TD", "TC", "3C"], 0, "花色不参与比较")


@case("Part 1", title="rank_players：并列与位置")
def p1_rank_players(sol):
    check_rank(sol, [["2C", "3D", "5H"], ["2D", "3H", "5C"]], [0, 1], "两手牌点数相同，并列")
    check_rank(sol, [["2C", "3D", "5H"], ["2D", "3H", "6C"]], [1])
    check_rank(sol, [["AS", "AD", "KC"], ["QH", "QS", "QD"], ["AH", "AC", "KD"]], [1])
    check_rank(sol, [["9C", "4D", "2H"], ["KD", "KH", "3S"], ["8S", "8C", "AH"], ["KS", "3D", "KC"],
                     ["JD", "TH", "9S"]], [1, 3], "并列的玩家要全部返回")
    check_rank(sol, [["6C", "6D", "TH"], ["TS", "6H", "6S"], ["6C", "TD", "6H"]], [0, 1, 2], "全部并列")
    check_rank(sol, [["3C", "4D", "6H"], ["3D", "4H", "6S"], ["2C", "4C", "6D"], ["AS", "2D", "2H"]], [3])


@case("Part 1", title="随机比较（对照逐条规则的实现）")
def p1_random(sol):
    r = rng()
    random_compares(sol, r, 1500, False, ref.FULL_DECK)
    random_compares(sol, r, 1500, False, ref.NARROW_DECK)
    random_ranks(sol, r, 200, False, ref.FULL_DECK)
    random_ranks(sol, r, 200, False, ref.NARROW_DECK)


BIG_N, BIG_TOP = 100_000, 1_000
BIG_CALL = f"rank_players(<{BIG_N} 手牌，其中 {BIG_TOP} 手随机放置的三条 A>)"


@case("Part 1", cpu=15, title="10⁵ 位玩家，大量并列：不能 O(N²)")
def p1_large(sol):
    r = rng("big")
    hands = [ref.random_hand(r, allow_flush=False) for _ in range(BIG_N)]
    aces = (["AC", "AD", "AH"], ["AC", "AD", "AS"], ["AC", "AH", "AS"], ["AD", "AH", "AS"])
    for i in r.sample(range(BIG_N), BIG_TOP):
        hands[i] = list(r.choice(aces))
    want = ref.winners(hands, False)
    check_rank(sol, hands, want, call=BIG_CALL)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：同花胜过对子")
def p2_example(sol):
    p0, p1, p2 = ["9C", "7C", "3C"], ["JD", "JH", "8S"], ["AS", "QD", "6H"]
    check_type(sol, p0, "Flush")
    check_type(sol, p1, "Pair")
    check_type(sol, p2, "High Card")
    check_rank(sol, [p0, p1, p2], [0])


@case("Part 2", title="同花的识别")
def p2_hand_type(sol):
    check_type(sol, ["AH", "KH", "QH"], "Flush")
    check_type(sol, ["2D", "5D", "3D"], "Flush")
    check_type(sol, ["TS", "JS", "QS"], "Flush")
    check_type(sol, ["AH", "KH", "QD"], "High Card", "只有两张同花色不算同花")
    check_type(sol, ["9S", "9H", "2S"], "Pair")
    check_type(sol, ["4C", "4D", "4H"], "Three of a Kind")


@case("Part 2", title="三条 > 同花 > 对子 > 高牌")
def p2_type_order(sol):
    check_compare(sol, ["2C", "2D", "2H"], ["AS", "KS", "QS"], 1, "三条胜过同花")
    check_compare(sol, ["5H", "3H", "2H"], ["AC", "AD", "KH"], 1, "同花胜过对子")
    check_compare(sol, ["5H", "3H", "2H"], ["AC", "KD", "QH"], 1, "同花胜过高牌")
    check_compare(sol, ["3C", "3D", "3S"], ["4H", "6H", "8H"], 1, "三条胜过同花")
    check_rank(sol, [["AS", "KS", "QS"], ["2C", "2D", "2H"], ["AC", "AD", "KH"]], [1], "三条胜过同花")


@case("Part 2", title="同花之间按高牌规则比较")
def p2_flush_compare(sol):
    check_compare(sol, ["AC", "3C", "2C"], ["KD", "QD", "JD"], 1, "先比最大的一张")
    check_compare(sol, ["AH", "KH", "3H"], ["AS", "KS", "2S"], 1, "前两张相同时比最小的")
    check_compare(sol, ["JS", "9S", "2S"], ["JH", "9H", "4H"], -1)
    check_compare(sol, ["9C", "7C", "3C"], ["9D", "7D", "3D"], 0, "花色不打破并列")
    check_compare(sol, ["7D", "QD", "5D"], ["QH", "7H", "5H"], 0, "花色不打破并列")


@case("Part 2", title="rank_players 用新的顺序")
def p2_rank_players(sol):
    check_rank(sol, [["QC", "QD", "AH"], ["7S", "4S", "2S"], ["KH", "KD", "KS"], ["AD", "KD", "JD"]], [2])
    check_rank(sol, [["9C", "7C", "3C"], ["AS", "AH", "QD"], ["9H", "7H", "3H"]], [0, 2], "两手同花并列")
    check_rank(sol, [["2H", "3H", "5H"], ["AC", "AD", "KS"], ["AS", "KH", "QD"]], [0])


@case("Part 2", title="随机比较（对照逐条规则的实现）")
def p2_random(sol):
    r = rng()
    random_compares(sol, r, 1500, True, ref.FULL_DECK)
    random_compares(sol, r, 1500, True, ref.NARROW_DECK)
    random_ranks(sol, r, 200, True, ref.FULL_DECK)
    random_ranks(sol, r, 200, True, ref.NARROW_DECK)


# ---------------------------------------------------------------- Part 3
KEEP = 3   # deal_card lines printed in the replay; earlier ones are folded into a loop


def _replay(head, lines, current):
    out = [head]
    k = len(lines)
    if k > KEEP:
        out += [f"for _ in range({k - KEEP}):", "    game.deal_card()"]
    out += [f"print(game.deal_card())   # -> {line!r}" for line in lines[max(0, k - KEEP):]]
    return "\n".join(out + [current])


def _same_bool(got, want):
    return isinstance(got, (bool, int)) and bool(got) == want


def run_round(sol, n, deck, names, want=None, why=""):
    """Plays one round step by step: is_over() before every deal_card(), then is_over() and check_result()."""
    need(sol, "PokerRound", *names)
    want = want or ref.play(n, deck, names)
    head = f"game = PokerRound({n}, {deck!r}, [{', '.join(names)}])"
    calling(head)
    game = sol.PokerRound(n, list(deck), [getattr(sol, name) for name in names])
    tail = f"。这个测试检查的是：{why}" if why else ""
    for k in range(3 * n):
        here = calling(_replay(head, want[:k], "print(game.is_over())   # <- 这一步，应当打印 False"))
        got = game.is_over()
        expect_true(_same_bool(got, False), f"第 {k + 1} 次 deal_card 之前 is_over() 应当是 False（一共要发 {3 * n} 次）",
                    call=here, got=got)
        here = calling(_replay(head, want[:k], f"print(game.deal_card())   # <- 这一步，应当打印 {want[k]!r}"))
        expect(game.deal_card(), want[k], call=here, msg=f"第 {k + 1} 次 deal_card 的返回不对{tail}")
    here = calling(_replay(head, want[:3 * n], "print(game.is_over())   # <- 这一步，应当打印 True"))
    got = game.is_over()
    expect_true(_same_bool(got, True), f"{3 * n} 次 deal_card 之后 is_over() 应当是 True", call=here, got=got)
    here = calling(_replay(head, want[:3 * n], f"print(game.check_result())   # <- 这一步，应当打印 {want[-1]!r}"))
    expect(game.check_result(), want[-1], call=here, msg=f"check_result 的返回不对{tail}")


@case("Part 3", sample=True, title="题目例子：弃牌不消耗牌")
def p3_example(sol):
    run_round(sol, 3, ["3S", "TD", "8H", "QC", "TS", "4H", "2D", "6H", "7C"],
              ["fold_unless_pair", "never_fold", "fold_unless_flush_draw"], want=[
                  "Player 0 is dealt 3S.", "Player 1 is dealt TD.", "Player 2 is dealt 8H.",
                  "Player 0 is dealt QC.", "Player 1 is dealt TS.", "Player 2 is dealt 4H.",
                  "Player 0 folds.", "Player 1 is dealt 2D.", "Player 2 is dealt 6H.",
                  "Winner: Player 2",
              ])


@case("Part 3", title="三种弃牌规则")
def p3_fold_rules(sol):
    for name, cards, want in (("never_fold", ["2C", "7D"], False), ("never_fold", ["AS", "AH"], False),
                              ("never_fold", ["9H", "4H"], False),
                              ("fold_unless_pair", ["9H", "9D"], False), ("fold_unless_pair", ["9H", "TH"], True),
                              ("fold_unless_pair", ["2C", "AD"], True),
                              ("fold_unless_flush_draw", ["9H", "2H"], False),
                              ("fold_unless_flush_draw", ["9H", "9D"], True),
                              ("fold_unless_flush_draw", ["KC", "QS"], True)):
        need(sol, name)
        call = calling(show_call(name, cards))
        expect(getattr(sol, name)(list(cards)), want, call=call,
               msg=f"{name} 应当返回 {want}（True 表示弃牌）")


@case("Part 3", title="连续弃牌后，下一位拿下一张牌")
def p3_fold_takes_no_card(sol):
    run_round(sol, 4, ["2C", "9D", "KH", "5S", "7H", "4C", "KD", "5D", "AS", "5H", "QS", "JS"],
              ["fold_unless_pair", "fold_unless_pair", "never_fold", "never_fold"],
              why="弃牌不消耗牌堆里的牌，后面的玩家拿的是下一张尚未发出的牌")
    run_round(sol, 3, ["8C", "3D", "JH", "8D", "7S", "QH", "8H", "AH", "KC"],
              ["never_fold", "fold_unless_pair", "fold_unless_flush_draw"],
              why="弃牌不消耗牌堆里的牌，后面的玩家拿的是下一张尚未发出的牌")


@case("Part 3", title="所有人都弃牌")
def p3_all_fold(sol):
    run_round(sol, 3, ["2C", "5D", "9H", "3D", "7S", "JC", "AS", "KS", "QS"],
              ["fold_unless_pair", "fold_unless_flush_draw", "fold_unless_pair"],
              why="全部弃牌时没有获胜者")
    run_round(sol, 2, ["AH", "AD", "KC", "QS", "JC", "TC"], ["fold_unless_flush_draw", "fold_unless_pair"],
              why="全部弃牌时没有获胜者")


@case("Part 3", title="只剩一人：直接获胜")
def p3_one_left(sol):
    run_round(sol, 3, ["2C", "AS", "AD", "3D", "KS", "QC", "5H", "9S", "TH"],
              ["never_fold", "fold_unless_pair", "fold_unless_flush_draw"],
              why="只剩一位没弃牌的玩家时，他直接获胜")
    run_round(sol, 4, ["AC", "AD", "KH", "2S", "QD", "KS", "JH", "3S", "TD", "4S", "9D", "8C"],
              ["fold_unless_flush_draw", "fold_unless_pair", "fold_unless_pair", "fold_unless_flush_draw"],
              why="只剩一位没弃牌的玩家时，他直接获胜")


@case("Part 3", title="摆牌比较用 Part 2 的规则")
def p3_showdown(sol):
    run_round(sol, 3, ["4H", "QS", "AD", "9H", "QC", "KC", "2H", "3D", "JS"],
              ["fold_unless_flush_draw", "fold_unless_pair", "never_fold"], why="同花胜过对子")
    run_round(sol, 3, ["4H", "QS", "AD", "9H", "QC", "KC", "2H", "QD", "JS"],
              ["fold_unless_flush_draw", "fold_unless_pair", "never_fold"], why="三条胜过同花")
    run_round(sol, 2, ["AS", "2C", "KD", "3C", "QH", "4C"], ["never_fold", "fold_unless_flush_draw"],
              why="同花胜过高牌")


@case("Part 3", title="并列获胜的写法")
def p3_tie(sol):
    run_round(sol, 4, ["AH", "QC", "AS", "5D", "KD", "JD", "KC", "4D", "7C", "7D", "3H", "AC"],
              ["never_fold", "fold_unless_pair", "never_fold", "never_fold"], why="点数相同的两手牌并列")
    run_round(sol, 5, ["AC", "2S", "AD", "3C", "AH", "KD", "7H", "KH", "4D", "KS", "5H", "5S", "5C", "9D", "TC"],
              ["never_fold", "fold_unless_pair", "never_fold", "fold_unless_flush_draw", "never_fold"],
              why="三位玩家并列，下标从小到大")


@case("Part 3", title="牌堆比 3N 长；17 位玩家")
def p3_deck_sizes(sol):
    r = rng()
    for n, extra in ((2, 46), (17, 0), (17, 1), (5, 10)):
        deck = r.sample(ref.FULL_DECK, 3 * n + extra)
        run_round(sol, n, deck, [r.choice(STRATEGIES) for _ in range(n)])


@case("Part 3", title="随机牌局（对照逐条规则的实现）")
def p3_random(sol):
    r = rng()
    for _ in range(150):
        n = r.randint(2, 17)
        deck = r.sample(ref.FULL_DECK, r.randint(3 * n, 52))
        run_round(sol, n, deck, [r.choice(STRATEGIES) for _ in range(n)])
    for _ in range(150):
        n = r.randint(2, 4)
        deck = r.sample(ref.NARROW_DECK, r.randint(3 * n, 12))
        run_round(sol, n, deck, [r.choice(STRATEGIES) for _ in range(n)])
