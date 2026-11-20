from enum import Enum


class ElementType(Enum):
    EMBER = "Ember"
    TIDE = "Tide"
    BRAMBLE = "Bramble"
    SPARK = "Spark"


class Monster:
    def __init__(self, name: str, hp: int, attack: int, element: ElementType):
        """hp and attack are positive integers; element is added in Part 2."""

    def is_alive(self) -> bool:
        """Alive while HP is above 0; eliminated once HP drops to 0 or below."""
        raise NotImplementedError

    def take_damage(self, damage: int) -> None:
        """Lowers HP by damage."""
        raise NotImplementedError

    def calculate_damage(self, defender: "Monster") -> int:
        """Damage this monster deals to defender, after the type chart and the rounding rule above."""
        raise NotImplementedError


class Team:
    def __init__(self, name: str, monsters: list[Monster]):
        """monsters is the attack/defense order; it is non-empty."""

    def first_alive(self) -> "Monster | None":
        """The first living monster in the list, or None."""
        raise NotImplementedError

    def is_defeated(self) -> bool:
        """True once every monster in the list is eliminated."""
        raise NotImplementedError

    def best_attacker_against(self, defender: Monster) -> "Monster | None":
        """The living monster on this team with the highest calculate_damage(defender); ties go to the first."""
        raise NotImplementedError


def run_battle(team_a: Team, team_b: Team) -> list[str]:
    """Simulates the battle and returns the event log, one string per line."""
    raise NotImplementedError
