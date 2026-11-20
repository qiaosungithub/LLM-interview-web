import ast
import inspect

from judgelib import Mismatch, calling, case, expect_true, rng
import ref

# The statement does not say how one file yields each part's battle, so two conventions are accepted:
#   plain: run_battle(a, b) follows the newest rules the code implements (Part 2 adds the multiplier to every
#          hit line, Part 3 changes the attacker), as the statement's "now" / "no longer" read literally;
#   hooks: the reference answer's run_battle(a, b, choose_attacker=..., describe_hit=...), with run_battle(a, b)
#          staying Part 1 and the page's type_effect_tag / smart_choose_attacker passed for Parts 2 and 3.
# Part 1 tests give every monster the element Ember when Monster takes one (all multipliers are then 1x).

E = ref.MEMBER


class Text(str):
    """Shown as-is (no quotes) in the expected / actual lines. A str subclass, so judgelib shows a long log in full
    instead of cutting its middle out as it does for other objects longer than 400 characters."""

    def __repr__(self):
        return str.__str__(self)


def stage(sol):
    """How far the code goes: 2 once it defines ElementType, 3 once Team also has best_attacker_against."""
    if not hasattr(sol, "ElementType"):
        return 1
    return 3 if hasattr(getattr(sol, "Team", None), "best_attacker_against") else 2


def takes_element(sol):
    """Whether Monster(...) is built with an element: the code defines ElementType and Monster takes a 4th argument."""
    if not hasattr(sol, "ElementType"):
        return False
    try:
        params = list(inspect.signature(sol.Monster).parameters.values())
    except (TypeError, ValueError):
        return True
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 4


def need(sol, *names):
    for name in names:
        expect_true(hasattr(sol, name), f"没有找到 {name}（名字要和题目一致）")


def need_methods(sol, cls, *names):
    need(sol, cls)
    for name in names:
        expect_true(callable(getattr(getattr(sol, cls), name, None)), f"{cls} 没有 {name} 方法")


def need_best(sol):
    need(sol, "ElementType")
    need_methods(sol, "Team", "best_attacker_against")


# ---------------------------------------------------------------- building teams

def monster_code(m, with_element):
    name, hp, attack, element = m
    return f"Monster({name!r}, {hp}, {attack}, ElementType.{E[element]})" if with_element else \
        f"Monster({name!r}, {hp}, {attack})"


def team_code(var, team, with_element):
    name, monsters = team
    return f"{var} = Team({name!r}, [{', '.join(monster_code(m, with_element) for m in monsters)}])"


def build(sol, team, with_element):
    name, monsters = team
    kinds = sol.ElementType if with_element else None
    made = [sol.Monster(n, hp, atk, getattr(kinds, E[el])) if with_element else sol.Monster(n, hp, atk)
            for n, hp, atk, el in monsters]
    return sol.Team(name, made)


def _params(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def convention(sol, part):
    """(how run_battle is called, "hooks" or "plain") for this part; see the comment at the top."""
    params = _params(sol.run_battle)
    has = lambda name: callable(getattr(sol, name, None))
    if part == 2 and "describe_hit" in params and has("type_effect_tag"):
        return "run_battle(a, b, describe_hit=type_effect_tag)", "hooks"
    if part == 3 and {"describe_hit", "choose_attacker"} <= params and has("type_effect_tag") \
            and has("smart_choose_attacker"):
        return "run_battle(a, b, choose_attacker=smart_choose_attacker, describe_hit=type_effect_tag)", "hooks"
    return "run_battle(a, b)", "plain"


def tagged(lines):
    """The statement's Part 1 log with " (1x)" after "damage" on every hit line (all monsters share an element)."""
    return [x.replace(" damage. ", " damage (1x). ", 1) for x in lines]


def accepted(sol, part, a, b, conv, want):
    """The logs that count as right: this part's rules first, then later parts' rules the code may follow."""
    st = stage(sol)
    if part == 1:
        logs = [want or ref.battle(a, b, chart=False, tags=False, smart=False)]
        if st >= 2:
            logs.append(tagged(logs[0]))
        if st >= 3:
            logs.append(ref.battle(a, b, chart=True, tags=True, smart=True))
    elif part == 2:
        logs = [want or ref.battle(a, b, chart=True, tags=True, smart=False)]
        if st >= 3 and conv == "plain":
            logs.append(ref.battle(a, b, chart=True, tags=True, smart=True))
    else:
        logs = [want or ref.battle(a, b, chart=True, tags=True, smart=True)]
    return logs


# ---------------------------------------------------------------- comparing logs

def _prefix(x, y):
    n = 0
    while n < len(x) and n < len(y) and x[n] == y[n]:
        n += 1
    return n


def _window(lines, k, total):
    if total <= 30:
        lo, hi = 0, len(lines)
    else:
        lo, hi = max(0, k - 3), min(len(lines), k + 3)
    out = [f"{i + 1:>3}: {lines[i]}" for i in range(lo, hi)]
    if lo > 0:
        out.insert(0, f"     …（前 {lo} 行相同）")
    if hi < len(lines):
        out.append(f"     …（后面还有 {len(lines) - hi} 行）")
    return Text("\n".join(out) if out else "（空）")


def normalize(got):
    if isinstance(got, str):
        return "应当返回字符串列表（每行一个字符串），而不是一整个字符串"
    if not isinstance(got, (list, tuple)):
        return f"应当返回字符串列表，得到的是 {type(got).__name__}"
    for i, x in enumerate(got):
        if not isinstance(x, str):
            return f"日志第 {i + 1} 行应当是字符串，得到的是 {type(x).__name__}：{x!r:.80}"
    return list(got)


def check_battle(sol, part, a, b, want=None, why=""):
    """Runs one battle on fresh objects and compares the log with every accepted log."""
    need(sol, "Monster", "Team", "run_battle")
    if part >= 2:
        need(sol, "ElementType")
    with_element = part >= 2 or takes_element(sol)
    written, conv = convention(sol, part)
    code = "\n".join([team_code("a", a, with_element), team_code("b", b, with_element),
                      f"for line in {written}:", "    print(line)"])
    calling(code)
    logs = accepted(sol, part, a, b, conv, want)
    ta, tb = build(sol, a, with_element), build(sol, b, with_element)
    got = sol.run_battle(ta, tb) if conv == "plain" else \
        sol.run_battle(ta, tb, describe_hit=sol.type_effect_tag) if part == 2 else \
        sol.run_battle(ta, tb, choose_attacker=sol.smart_choose_attacker, describe_hit=sol.type_effect_tag)
    lines = normalize(got)
    if isinstance(lines, str):
        raise Mismatch(lines, code, None, got)
    if lines in logs:
        return
    best = max(logs, key=lambda log: _prefix(log, lines))
    k = _prefix(best, lines)
    if k == len(lines):
        msg = f"日志只有 {len(lines)} 行，从第 {k + 1} 行起缺了"
    elif k == len(best):
        msg = f"日志在应有的 {len(best)} 行之后还多出了内容"
    else:
        msg = f"日志第 {k + 1} 行不对"
    if len(logs) > 1:
        msg += "（按可接受的几种写法里最接近的一种比较）"
    if why:
        msg += f"。这个测试检查的是：{why}"
    total = max(len(best), len(lines))
    raise Mismatch(msg, code, _window(best, k, total), _window(lines, k, total))


# ---------------------------------------------------------------- scripted method calls

def _literal(text):
    return ast.literal_eval(text)


KEEP = 6   # earlier checks dropped from the replay (they have no side effects); every statement is kept


def _replay(done, current):
    statements = [x for x in done if not x.startswith("print(")]
    checks = [x for x in done if x.startswith("print(")][-KEEP:]
    return "\n".join([x for x in done if x in statements or x in checks] + [current])


def run_script(sol, script, why_default=""):
    """Lines are statements, or  expr -> literal  checks; "X is name -> True" checks X returns that very object
    and names the monster actually returned. {E} expands to ", ElementType.EMBER" when Monster takes an element."""
    need(sol, "Monster", "Team")
    with_element = takes_element(sol)
    ns = {"Monster": sol.Monster, "Team": sol.Team}
    if hasattr(sol, "ElementType"):
        ns["ElementType"] = sol.ElementType
    done = []
    for raw in script.strip().splitlines():
        code, _, why = raw.strip().partition("#")
        code = code.strip().replace("{E}", ", ElementType.EMBER" if with_element else "")
        why = why.strip() or why_default
        expr, arrow, want_text = code.partition(" -> ")
        if not arrow:
            calling(_replay(done, f"{code}   # <- 这一步"))
            exec(code, ns)
            done.append(code)
            continue
        want = _literal(want_text.strip())
        left, is_, right = expr.partition(" is ")
        line = f"print({expr})"
        here = calling(_replay(done, f"{line}   # <- 这一步，应当打印 {want!r}"))
        if is_:
            obj = eval(left, ns)
            ok = obj is ns[right.strip()]
            got = Text(_who(obj, ns))
            want_shown = Text(f"返回 {right.strip()}")
            msg = f"{left.strip()} 应当返回 {right.strip()}"
        else:
            got = eval(expr, ns)
            ok = _same(got, want)
            want_shown = want
            msg = f"{expr} 应当是 {want!r}"
        if not ok:
            raise Mismatch(msg + (f"（{why}）" if why else ""), here, want_shown, got)
        done.append(f"{line}   # -> {want!r}")


def _who(obj, ns):
    for name, value in ns.items():
        if value is obj and not name.startswith("_") and name not in ("Monster", "Team", "ElementType"):
            return f"返回了 {name}"
    return f"返回了 {obj!r:.80}"


def _same(got, want):
    if want is None:
        return got is None
    try:
        if isinstance(want, bool):
            return isinstance(got, (bool, int)) and bool(got == want)
        return bool(got == want) and not isinstance(got, bool)
    except Exception:
        return False


def random_battles(sol, part, n, sizes=(1, 5), hp=(1, 20), attack=(1, 6)):
    r = rng()
    elements = ("Ember",) if part == 1 else ref.ELEMENTS
    for _ in range(n):
        a = ref.random_team(r, "A", r.randint(*sizes), hp, attack, elements)
        b = ref.random_team(r, "B", r.randint(*sizes), hp, attack, elements)
        check_battle(sol, part, a, b)


# ---------------------------------------------------------------- Part 1
VANGUARD = ("Vanguard", [("Talonis", 25, 14, "Ember")])
MARSH = ("Marsh", [("Sable", 10, 5, "Ember"), ("Thornback", 20, 6, "Ember")])


@case("Part 1", sample=True, title="题目例子：Vanguard 对 Marsh")
def p1_example(sol):
    check_battle(sol, 1, VANGUARD, MARSH, want=[
        "Match start: Vanguard takes on Marsh",
        "Talonis hits Sable for 14 damage. Sable is defeated!",
        "Thornback hits Talonis for 6 damage. Talonis has 19 HP left.",
        "Talonis hits Thornback for 14 damage. Thornback has 6 HP left.",
        "Thornback hits Talonis for 6 damage. Talonis has 13 HP left.",
        "Talonis hits Thornback for 14 damage. Thornback is defeated!",
        "Match over: Vanguard wins!",
    ])


@case("Part 1", title="is_alive、first_alive、is_defeated")
def p1_methods(sol):
    need_methods(sol, "Monster", "is_alive", "take_damage")
    need_methods(sol, "Team", "first_alive", "is_defeated")
    run_script(sol, """
        pip = Monster('Pip', 3, 1{E})
        bo = Monster('Bo', 5, 2{E})
        cy = Monster('Cy', 4, 7{E})
        team = Team('T', [pip, bo, cy])
        pip.is_alive() -> True
        team.first_alive() is pip -> True
        team.is_defeated() -> False
        pip.take_damage(2)
        pip.is_alive() -> True               # HP 还剩 1
        pip.take_damage(1)
        pip.is_alive() -> False              # HP 恰好降到 0 就被淘汰
        team.first_alive() is bo -> True     # pip 已被淘汰，第一个存活的是 bo
        bo.take_damage(9)
        bo.is_alive() -> False               # HP 降到 0 以下也是淘汰
        team.first_alive() is cy -> True
        team.is_defeated() -> False
        cy.take_damage(4)
        team.first_alive() -> None           # 全部被淘汰
        team.is_defeated() -> True
    """)


@case("Part 1", title="HP 恰好降到 0：被淘汰")
def p1_exact_zero(sol):
    check_battle(sol, 1, ("Embers", [("Ash", 10, 5, "Ember")]),
                 ("Bluffs", [("Bolt", 5, 3, "Ember"), ("Cove", 8, 4, "Ember")]),
                 why="HP 恰好为 0 时应当写 is defeated!")
    check_battle(sol, 1, ("North", [("Nix", 6, 2, "Ember")]), ("South", [("Sol", 6, 3, "Ember")]))


@case("Part 1", title="淘汰对手后照样交换攻守")
def p1_turn_after_kill(sol):
    check_battle(sol, 1, ("Hunters", [("Ash", 30, 10, "Ember")]),
                 ("Herd", [("Bolt", 5, 1, "Ember"), ("Cove", 5, 1, "Ember"), ("Dusk", 50, 2, "Ember")]),
                 why="每次攻击之后都交换攻防，不论有没有淘汰谁")
    check_battle(sol, 1, ("Pair", [("Pa", 5, 4, "Ember"), ("Pb", 5, 4, "Ember")]),
                 ("Trio", [("Ta", 5, 4, "Ember"), ("Tb", 5, 4, "Ember"), ("Tc", 5, 4, "Ember")]))


@case("Part 1", title="攻击者是第一个存活的怪兽")
def p1_first_alive_attacker(sol):
    check_battle(sol, 1, ("Odd", [("Weak", 3, 1, "Ember"), ("Strong", 40, 9, "Ember")]),
                 ("Wall", [("Cole", 40, 2, "Ember")]),
                 why="下标 0 的怪兽存活时由它出手，而不是攻击力更高的那个")
    check_battle(sol, 1, ("Line", [("L0", 2, 3, "Ember"), ("L1", 5, 1, "Ember"), ("L2", 9, 6, "Ember")]),
                 ("Row", [("R0", 7, 2, "Ember"), ("R1", 3, 5, "Ember")]))


@case("Part 1", title="队伍 B 获胜、一击结束")
def p1_short_and_b_wins(sol):
    check_battle(sol, 1, ("Glass", [("Ash", 5, 1, "Ember")]), ("Stone", [("Bolt", 20, 5, "Ember")]))
    check_battle(sol, 1, ("Quick", [("Zap", 1, 100, "Ember")]), ("Slow", [("Lump", 100, 1, "Ember")]))
    check_battle(sol, 1, ("Few", [("F0", 3, 2, "Ember")]),
                 ("Many", [("M0", 1, 1, "Ember"), ("M1", 1, 1, "Ember"), ("M2", 9, 1, "Ember")]))


@case("Part 1", title="随机对战（对照参考实现）")
def p1_random(sol):
    random_battles(sol, 1, 300)


@case("Part 1", title="很长的对战")
def p1_long(sol):
    r = rng()
    a = ref.random_team(r, "A", 6, (150, 300), (1, 3), ("Ember",))
    b = ref.random_team(r, "B", 6, (150, 300), (1, 3), ("Ember",))
    check_battle(sol, 1, a, b)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：Wildfire 对 Thicket")
def p2_example(sol):
    check_battle(sol, 2, ("Wildfire", [("Cinder", 30, 9, "Ember")]), ("Thicket", [("Mossken", 26, 7, "Bramble")]),
                 want=[
                     "Match start: Wildfire takes on Thicket",
                     "Cinder hits Mossken for 18 damage (2x). Mossken has 8 HP left.",
                     "Mossken hits Cinder for 3 damage (0.5x). Cinder has 27 HP left.",
                     "Cinder hits Mossken for 18 damage (2x). Mossken is defeated!",
                     "Match over: Wildfire wins!",
                 ])


def _chart_script(attack):
    names = {el: f"{el.lower()}{attack}" for el in ref.ELEMENTS}
    lines = [f"{names[el]} = Monster('{el}{attack}', 50, {attack}, ElementType.{E[el]})" for el in ref.ELEMENTS]
    for att in ref.ELEMENTS:
        for dfn in ref.ELEMENTS:
            lines.append(f"{names[att]}.calculate_damage({names[dfn]}) -> {ref.damage(attack, att, dfn)}"
                         f"   # {att} 打 {dfn} 是 {ref.multiplier_text(att, dfn)}x")
    return "\n".join(lines)


@case("Part 2", title="属性表的 16 种组合")
def p2_chart(sol):
    need(sol, "ElementType")
    need_methods(sol, "Monster", "calculate_damage")
    run_script(sol, _chart_script(8))
    run_script(sol, _chart_script(7))


@case("Part 2", title="向下取整，且至少为 1")
def p2_floor_and_clamp(sol):
    need(sol, "ElementType")
    need_methods(sol, "Monster", "calculate_damage")
    run_script(sol, """
        spark = Monster('Spark', 9, 1, ElementType.SPARK)
        tide = Monster('Tide', 9, 1, ElementType.TIDE)
        ember3 = Monster('Ember3', 9, 3, ElementType.EMBER)
        ember5 = Monster('Ember5', 9, 5, ElementType.EMBER)
        bramble9 = Monster('Bramble9', 9, 9, ElementType.BRAMBLE)
        spark.calculate_damage(tide) -> 1      # floor(1 × 0.5) = 0，小于 1 时取 1
        tide.calculate_damage(spark) -> 2
        spark.calculate_damage(spark) -> 1
        ember3.calculate_damage(spark) -> 1    # floor(3 × 0.5) = 1
        ember5.calculate_damage(spark) -> 2    # floor(5 × 0.5) = 2
        bramble9.calculate_damage(ember3) -> 4 # floor(9 × 0.5) = 4
    """)
    check_battle(sol, 2, ("Sparks", [("Wisp", 3, 1, "Spark")]), ("Tides", [("Moss", 2, 1, "Tide")]),
                 why="0.5x 的伤害向下取整后小于 1 时取 1")


@case("Part 2", title="倍率写成 2x、0.5x、1x")
def p2_multiplier_text(sol):
    check_battle(sol, 2, ("Mixed", [("Fern", 30, 4, "Bramble"), ("Kiln", 30, 6, "Ember")]),
                 ("Others", [("Reef", 12, 5, "Tide"), ("Volt", 30, 3, "Spark"), ("Char", 30, 2, "Ember")]),
                 why="倍率去掉多余的 .0：2x、0.5x、1x")


@case("Part 2", title="随机对战（对照参考实现）")
def p2_random(sol):
    random_battles(sol, 2, 300)


@case("Part 2", title="攻击力为 1 的长对战")
def p2_long(sol):
    r = rng()
    a = ref.random_team(r, "A", 6, (100, 200), (1, 2), ref.ELEMENTS)
    b = ref.random_team(r, "B", 6, (100, 200), (1, 2), ref.ELEMENTS)
    check_battle(sol, 2, a, b)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：Skyfleet 对 Depths")
def p3_example(sol):
    check_battle(sol, 3, ("Skyfleet", [("Talos", 20, 10, "Spark"), ("Draketh", 25, 12, "Ember")]),
                 ("Depths", [("Coraline", 22, 8, "Tide")]), want=[
                     "Match start: Skyfleet takes on Depths",
                     "Draketh hits Coraline for 12 damage (1x). Coraline has 10 HP left.",
                     "Coraline hits Talos for 16 damage (2x). Talos has 4 HP left.",
                     "Draketh hits Coraline for 12 damage (1x). Coraline is defeated!",
                     "Match over: Skyfleet wins!",
                 ])


@case("Part 3", title="按伤害数值选，不是按克制选")
def p3_highest_damage(sol):
    need_best(sol)
    run_script(sol, """
        ten = Monster('Ten', 20, 10, ElementType.EMBER)
        tri = Monster('Tri', 20, 3, ElementType.SPARK)
        target = Monster('Target', 50, 1, ElementType.EMBER)
        team = Team('T', [tri, ten])
        team.best_attacker_against(target) is ten -> True   # ten 造成 10（1x），tri 只造成 6（2x）
    """)
    run_script(sol, """
        big = Monster('Big', 20, 10, ElementType.EMBER)
        six = Monster('Six', 20, 6, ElementType.TIDE)
        target = Monster('Target', 50, 1, ElementType.SPARK)
        team = Team('T', [big, six])
        team.best_attacker_against(target) is six -> True   # six 造成 12（2x），big 只造成 5（0.5x）
        team.best_attacker_against(Monster('Mid', 50, 1, ElementType.TIDE)) is big -> True   # 10 对 6
    """)
    check_battle(sol, 3, ("Picker", [("Tri", 20, 3, "Spark"), ("Ten", 20, 10, "Ember")]),
                 ("Embers", [("Coal", 25, 2, "Ember"), ("Pyre", 9, 4, "Ember")]),
                 why="攻击者是对当前防御者伤害最高的存活怪兽")


@case("Part 3", title="伤害并列时选排在前面的")
def p3_ties(sol):
    need_best(sol)
    run_script(sol, """
        first = Monster('First', 30, 10, ElementType.EMBER)
        second = Monster('Second', 30, 10, ElementType.EMBER)
        target = Monster('Target', 100, 1, ElementType.SPARK)
        Team('Tied', [first, second]).best_attacker_against(target) is first -> True
        Team('Tied', [second, first]).best_attacker_against(target) is second -> True
        quad = Monster('Quad', 30, 4, ElementType.SPARK)
        octo = Monster('Octo', 30, 8, ElementType.TIDE)
        ember = Monster('Ember', 100, 1, ElementType.EMBER)
        Team('Mix', [quad, octo]).best_attacker_against(ember) is quad -> True   # 4 × 2 = 8 × 1
        Team('Mix', [octo, quad]).best_attacker_against(ember) is octo -> True
    """, why_default="伤害并列时应当选列表里排在更前面的那个")
    check_battle(sol, 3, ("Twins", [("Ta", 30, 6, "Tide"), ("Tb", 30, 6, "Tide"), ("Tc", 30, 3, "Spark")]),
                 ("Lone", [("Ox", 40, 5, "Bramble")]), why="伤害并列时由排在前面的出手")


@case("Part 3", title="跳过已淘汰的怪兽；全灭时返回 None")
def p3_skips_eliminated(sol):
    need_best(sol)
    run_script(sol, """
        ace = Monster('Ace', 5, 20, ElementType.EMBER)
        low = Monster('Low', 5, 2, ElementType.EMBER)
        mid = Monster('Mid', 5, 4, ElementType.EMBER)
        target = Monster('Target', 50, 1, ElementType.TIDE)
        team = Team('T', [ace, low, mid])
        team.best_attacker_against(target) is ace -> True
        ace.take_damage(5)
        team.best_attacker_against(target) is mid -> True   # ace 已被淘汰
        mid.take_damage(6)
        low.take_damage(5)
        team.best_attacker_against(target) -> None
    """)


@case("Part 3", title="防御者仍是第一个存活的怪兽")
def p3_defender_rule(sol):
    check_battle(sol, 3, ("Strikers", [("Fizz", 30, 5, "Spark"), ("Blaze", 30, 6, "Ember")]),
                 ("Guard", [("Wall", 20, 1, "Bramble"), ("Wave", 4, 9, "Tide")]),
                 why="防御者始终是防守方第一个存活的怪兽，攻击者不能挑目标")
    check_battle(sol, 3, ("Swap", [("S0", 9, 2, "Tide"), ("S1", 9, 3, "Bramble"), ("S2", 9, 4, "Spark")]),
                 ("Foes", [("F0", 6, 3, "Spark"), ("F1", 6, 3, "Ember"), ("F2", 6, 3, "Tide")]))


@case("Part 3", title="随机对战（对照参考实现）")
def p3_random(sol):
    random_battles(sol, 3, 300)


@case("Part 3", title="大队伍的长对战")
def p3_long(sol):
    r = rng()
    a = ref.random_team(r, "A", 8, (100, 200), (1, 4), ref.ELEMENTS)
    b = ref.random_team(r, "B", 8, (100, 200), (1, 4), ref.ELEMENTS)
    check_battle(sol, 3, a, b)
