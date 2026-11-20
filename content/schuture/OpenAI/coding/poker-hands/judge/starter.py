def hand_type(cards: list[str]) -> str:
    """Returns "Three of a Kind", "Flush" (from Part 2), "Pair" or "High Card" for a 3-card hand like ["9H", "TC", "9D"]."""
    raise NotImplementedError


def compare_hands(cards_a: list[str], cards_b: list[str]) -> int:
    """Returns 1 if cards_a ranks strictly above cards_b, -1 if strictly below, 0 if the two hands tie."""
    raise NotImplementedError


def rank_players(hands: list[list[str]]) -> list[int]:
    """Returns, in increasing order, the 0-indexed players whose hand does not rank below any other player's hand."""
    raise NotImplementedError


def never_fold(first_two: list[str]) -> bool:
    """Never folds."""
    raise NotImplementedError


def fold_unless_pair(first_two: list[str]) -> bool:
    """Folds unless the first two cards are a pair."""
    raise NotImplementedError


def fold_unless_flush_draw(first_two: list[str]) -> bool:
    """Folds unless the first two cards are a flush draw (same suit)."""
    raise NotImplementedError


class PokerRound:
    def __init__(self, num_players: int, deck: list[str], strategies: list):
        """strategies[i] is player i's fold rule, one of the three functions above."""

    def deal_card(self) -> str:
        """Advances the round by exactly one deal-or-fold and returns "Player {i} is dealt {card}." or "Player {i} folds."."""
        raise NotImplementedError

    def is_over(self) -> bool:
        """True once all 3 * num_players deals-or-folds have happened."""
        raise NotImplementedError

    def check_result(self) -> str:
        """Returns "Winner: Player {i}", "Winner: Tie (Players {i}, {j}, ...)" or "No winner: all players folded."."""
        raise NotImplementedError
