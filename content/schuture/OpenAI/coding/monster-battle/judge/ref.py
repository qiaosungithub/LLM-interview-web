"""Helpers for tests.py: a direct simulator of the three parts' rules, and random teams.

A team is (team name, [(monster name, hp, attack, element name), ...]); element names are "Ember", "Tide",
"Bramble", "Spark". The chart is the statement's table, kept as exact fractions.
"""
from fractions import Fraction

ELEMENTS = ("Ember", "Tide", "Bramble", "Spark")
MEMBER = {"Ember": "EMBER", "Tide": "TIDE", "Bramble": "BRAMBLE", "Spark": "SPARK"}
# rows: attacker, columns: defender, in the order of ELEMENTS (the statement's table)
TABLE = {"Ember": ("1", "1", "2", "0.5"), "Tide": ("1", "1", "0.5", "2"),
         "Bramble": ("0.5", "2", "1", "1"), "Spark": ("2", "0.5", "1", "1")}


def multiplier_text(att, dfn):
    return TABLE[att][ELEMENTS.index(dfn)]


def damage(attack, att, dfn):
    f = Fraction(multiplier_text(att, dfn))
    return max(1, attack * f.numerator // f.denominator)


def battle(team_a, team_b, chart, tags, smart):
    """The event log. chart: damage uses the table; tags: hit lines carry " (2x)"; smart: Part 3's attacker."""
    teams = [(name, [[n, hp, atk, el] for n, hp, atk, el in ms]) for name, ms in (team_a, team_b)]
    log = [f"Match start: {teams[0][0]} takes on {teams[1][0]}"]
    side = 0
    while all(any(m[1] > 0 for m in ms) for _, ms in teams):
        defender = next(m for m in teams[1 - side][1] if m[1] > 0)

        def hit(m):
            return damage(m[2], m[3], defender[3]) if chart else m[2]

        alive = [m for m in teams[side][1] if m[1] > 0]
        attacker = alive[0]
        if smart:
            for m in alive[1:]:
                if hit(m) > hit(attacker):          # strict: a tie keeps the earlier monster
                    attacker = m
        dmg = hit(attacker)
        defender[1] -= dmg
        tag = f" ({multiplier_text(attacker[3], defender[3])}x)" if tags else ""
        line = f"{attacker[0]} hits {defender[0]} for {dmg} damage{tag}. "
        log.append(line + (f"{defender[0]} has {defender[1]} HP left." if defender[1] > 0
                           else f"{defender[0]} is defeated!"))
        side = 1 - side
    winner = teams[0][0] if any(m[1] > 0 for m in teams[0][1]) else teams[1][0]
    log.append(f"Match over: {winner} wins!")
    return log


def random_team(r, name, size, hp=(1, 20), attack=(1, 6), elements=ELEMENTS):
    """Monster names are the team name plus an index (A0, A1, ...), so lines from the two teams never look alike."""
    return (name, [(f"{name}{i}", r.randint(*hp), r.randint(*attack), r.choice(elements)) for i in range(size)])
