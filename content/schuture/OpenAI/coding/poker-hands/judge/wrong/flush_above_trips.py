# expect: WA
# Part 2 puts the flush on top, above Three of a Kind.
RANKS = "23456789TJQKA"  # index doubles as the rank's numeric value
TYPE_NAMES = ["High Card", "Pair", "Three of a Kind"]  # indexed by the priority inside a hand key


def _rank_value(card):
    return RANKS.index(card[0])


def _hand_key(cards):
    ranks = sorted((_rank_value(c) for c in cards), reverse=True)
    if ranks[0] == ranks[1] == ranks[2]:
        return (2, (ranks[0],))
    if ranks[0] == ranks[1]:                # NOTE: after a descending sort the pair is always adjacent,
        return (1, (ranks[0], ranks[2]))    #       at [0]/[1] or at [1]/[2] -- it can never be split
    if ranks[1] == ranks[2]:
        return (1, (ranks[1], ranks[0]))
    return (0, tuple(ranks))


def hand_type(cards):
    return TYPE_NAMES[_hand_key(cards)[0]]


def compare_hands(cards_a, cards_b):
    ka, kb = _hand_key(cards_a), _hand_key(cards_b)
    return 0 if ka == kb else (1 if ka > kb else -1)  # NOTE: tuples compare lexicographically, so the
                                                      #       priority always dominates the tie-break


def rank_players(hands):
    keys = [_hand_key(h) for h in hands]
    best = max(keys)
    return [i for i, k in enumerate(keys) if k == best]


_ranks_only_key = _hand_key                                     # Part 1's key, suits ignored
TYPE_NAMES = ["High Card", "Pair", "Three of a Kind", "Flush"]  # flush goes on top


def _hand_key(cards):  # supersedes Part 1's version
    priority, tie_break = _ranks_only_key(cards)
    if priority == 2:
        return (2, tie_break)
    if len({c[1] for c in cards}) == 1:
        return (3, tuple(sorted((_rank_value(c) for c in cards), reverse=True)))
    return (priority, tie_break)


def never_fold(first_two):
    return False


def fold_unless_pair(first_two):
    return first_two[0][0] != first_two[1][0]


def fold_unless_flush_draw(first_two):
    return first_two[0][1] != first_two[1][1]


class PokerRound:
    def __init__(self, num_players, deck, strategies):
        assert len(deck) >= 3 * num_players and len(set(deck)) == len(deck)
        self.num_players = num_players
        self.deck = deck
        self.strategies = strategies
        self.hands = [[] for _ in range(num_players)]
        self.folded = [False] * num_players
        self.next_card = 0        # NOTE: one running pointer -- a fold never advances it
        self.calls_made = 0

    def is_over(self):
        return self.calls_made >= 3 * self.num_players

    def deal_card(self):
        pass_no, player = divmod(self.calls_made, self.num_players)  # NOTE: pass_no is 0, 1 or 2
        self.calls_made += 1
        if pass_no == 2 and self.strategies[player](self.hands[player]):
            self.folded[player] = True
            return f"Player {player} folds."
        card = self.deck[self.next_card]
        self.next_card += 1
        self.hands[player].append(card)
        return f"Player {player} is dealt {card}."

    def check_result(self):
        active = [i for i in range(self.num_players) if not self.folded[i]]
        if not active:
            return "No winner: all players folded."
        if len(active) == 1:
            return f"Winner: Player {active[0]}"
        keys = {i: _hand_key(self.hands[i]) for i in active}
        best = max(keys.values())
        winners = [i for i in active if keys[i] == best]
        if len(winners) == 1:
            return f"Winner: Player {winners[0]}"
        return "Winner: Tie (Players " + ", ".join(str(i) for i in winners) + ")"


def play_round(num_players, deck, strategies):
    game = PokerRound(num_players, deck, strategies)
    log = []
    while not game.is_over():
        log.append(game.deal_card())
    log.append(game.check_result())
    return log
