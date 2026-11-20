"""Helpers for tests.py: the rules spelled out one by one (rank counts, a tie-break per type), and random hands.

flush=False is Part 1 (suits ignored), flush=True is Parts 2 and 3.
"""
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "CDHS"
FULL_DECK = [r + s for r in RANKS for s in SUITS]
NARROW_DECK = [r + s for r in "789" for s in SUITS]   # 12 cards: ties, pairs and flush draws are common
TYPES = {False: ["High Card", "Pair", "Three of a Kind"], True: ["High Card", "Pair", "Flush", "Three of a Kind"]}


def label(cards, flush):
    counts = Counter(c[0] for c in cards)
    if len(counts) == 1:
        return "Three of a Kind"
    if flush and len({c[1] for c in cards}) == 1:
        return "Flush"
    return "Pair" if len(counts) == 2 else "High Card"


def tie_break(cards, name):
    values = sorted((RANKS.index(c[0]) for c in cards), reverse=True)
    if name == "Three of a Kind":
        return [values[0]]
    if name == "Pair":
        pair = next(v for v, k in Counter(values).items() if k == 2)
        return [pair, next(v for v in values if v != pair)]
    return values                                  # High Card and Flush: all three, high to low


def key(cards, flush):
    name = label(cards, flush)
    return (TYPES[flush].index(name), tie_break(cards, name))


def compare(a, b, flush):
    ka, kb = key(a, flush), key(b, flush)
    return (ka > kb) - (ka < kb)


def winners(hands, flush):
    keys = [key(h, flush) for h in hands]
    best = max(keys)
    return [i for i, k in enumerate(keys) if k == best]


def monosuited(cards):
    return len({c[1] for c in cards}) == 1


def random_hand(r, pool=FULL_DECK, allow_flush=True):
    """Three distinct cards; with allow_flush=False never all of one suit (Part 1 tests avoid such hands)."""
    while True:
        hand = r.sample(pool, 3)
        if allow_flush or not monosuited(hand):
            return hand


def should_fold(name, first_two):
    if name == "never_fold":
        return False
    if name == "fold_unless_pair":
        return first_two[0][0] != first_two[1][0]
    return first_two[0][1] != first_two[1][1]        # fold_unless_flush_draw


def play(n, deck, names):
    """The round's lines: 3n deal-or-fold lines, then the result line."""
    hands = [[] for _ in range(n)]
    folded = [False] * n
    pos = 0
    log = []
    for pass_no in range(3):
        for p in range(n):
            if pass_no == 2 and should_fold(names[p], hands[p]):
                folded[p] = True
                log.append(f"Player {p} folds.")
            else:
                hands[p].append(deck[pos])
                pos += 1
                log.append(f"Player {p} is dealt {hands[p][-1]}.")
    active = [p for p in range(n) if not folded[p]]
    if not active:
        log.append("No winner: all players folded.")
    elif len(active) == 1:
        log.append(f"Winner: Player {active[0]}")
    else:
        best = [active[i] for i in winners([hands[p] for p in active], True)]
        log.append(f"Winner: Player {best[0]}" if len(best) == 1 else
                   "Winner: Tie (Players " + ", ".join(map(str, best)) + ")")
    return log
