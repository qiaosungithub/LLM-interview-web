# expect: WA
# A monster with exactly 0 HP still counts as alive (>= instead of >).
from enum import Enum
import math


class Monster:
    def __init__(self, name, hp, attack):
        self.name = name
        self.hp = hp
        self.attack = attack

    def is_alive(self):
        return self.hp >= 0

    def take_damage(self, damage):
        self.hp -= damage

    def calculate_damage(self, defender):
        return self.attack


class Team:
    def __init__(self, name, monsters):
        self.name = name
        self.monsters = monsters

    def first_alive(self):
        return next((m for m in self.monsters if m.is_alive()), None)

    def is_defeated(self):
        return self.first_alive() is None


def default_choose_attacker(attacking_team, defender):
    return attacking_team.first_alive()      # NOTE: `defender` is unused here, only in Part 3's strategy


def default_describe_hit(attacker, defender):
    return ""


def run_battle(team_a, team_b, choose_attacker=default_choose_attacker, describe_hit=default_describe_hit):
    log = [f"Match start: {team_a.name} takes on {team_b.name}"]
    attacking, defending = team_a, team_b
    while attacking.first_alive() is not None and defending.first_alive() is not None:
        defender = defending.first_alive()
        attacker = choose_attacker(attacking, defender)
        damage = attacker.calculate_damage(defender)
        tag = describe_hit(attacker, defender)
        defender.take_damage(damage)              # NOTE: apply damage before reading is_alive()/hp below
        if defender.is_alive():
            log.append(f"{attacker.name} hits {defender.name} for {damage} damage{tag}. "
                        f"{defender.name} has {defender.hp} HP left.")
        else:
            log.append(f"{attacker.name} hits {defender.name} for {damage} damage{tag}. "
                        f"{defender.name} is defeated!")
        attacking, defending = defending, attacking
    winner = team_a if team_a.first_alive() is not None else team_b
    log.append(f"Match over: {winner.name} wins!")
    return log


class ElementType(Enum):
    EMBER = "Ember"
    TIDE = "Tide"
    BRAMBLE = "Bramble"
    SPARK = "Spark"


TYPE_CHART = {
    (ElementType.EMBER, ElementType.BRAMBLE): 2.0,
    (ElementType.BRAMBLE, ElementType.EMBER): 0.5,
    (ElementType.BRAMBLE, ElementType.TIDE): 2.0,
    (ElementType.TIDE, ElementType.BRAMBLE): 0.5,
    (ElementType.TIDE, ElementType.SPARK): 2.0,
    (ElementType.SPARK, ElementType.TIDE): 0.5,
    (ElementType.SPARK, ElementType.EMBER): 2.0,
    (ElementType.EMBER, ElementType.SPARK): 0.5,
}


def type_multiplier(attacker_element, defender_element):
    return TYPE_CHART.get((attacker_element, defender_element), 1.0)


def format_multiplier(multiplier):
    return f"{int(multiplier)}x" if multiplier == int(multiplier) else f"{multiplier}x"


def type_effect_tag(attacker, defender):
    return f" ({format_multiplier(type_multiplier(attacker.element, defender.element))})"


def _init_with_element(self, name, hp, attack, element):
    self.name = name
    self.hp = hp
    self.attack = attack
    self.element = element


def calculate_damage_by_type(self, defender):
    multiplier = type_multiplier(self.element, defender.element)
    return max(1, math.floor(self.attack * multiplier))   # NOTE: floor(1 * 0.5) is 0; the rules say a hit
                                                            #       always deals at least 1


Monster.__init__ = _init_with_element          # attach to the Monster class of Part 1
Monster.calculate_damage = calculate_damage_by_type


def best_attacker_against(self, defender):
    living = [m for m in self.monsters if m.is_alive()]
    # NOTE: max() keeps the FIRST element on a tie (checked: max(['a', 'b'], key={'a': 5, 'b': 5}.get) == 'a')
    return max(living, key=lambda m: m.calculate_damage(defender), default=None)


Team.best_attacker_against = best_attacker_against   # attach to the Team class of Part 1


def smart_choose_attacker(attacking_team, defender):
    return attacking_team.best_attacker_against(defender)
