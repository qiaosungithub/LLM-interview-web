# The judge runs every answer with the checker from the statement, copied verbatim below (parse, run, PUZZLES,
# check, report). Helpers for tests.py follow it, under "judge helpers".

"""Validity checker for the queue machine. This file is all the tooling you get."""
import random
from collections import deque

LIMIT = 99            # values are integers 0..LIMIT
CAP = 16              # the queue holds at most CAP values
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
NEED = {"ADD": 2, "SUB": 2, "DUP": 1, "ROT": 1, "DROP": 1,     # values an instruction pops
        "INC": 1, "DEC": 1, "ZERO": 0, "JMP": 0, "JZ": 1}


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment. Returns [(op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        op = parts[0]
        if op not in NEED or len(parts) != (2 if op in ("JMP", "JZ") else 1):
            raise ValueError(f"bad instruction: {line.strip()!r}")
        prog.append((op, int(parts[1]) if len(parts) == 2 else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(arg is not None and not 0 <= arg <= len(prog) for _, arg in prog):
        raise ValueError("jump target out of range")      # target len(prog) means "halt"
    return prog


def run(prog, values):
    """Final queue (front first), or None on overflow or after MAX_STEPS steps."""
    q = deque(values)
    pc = 0
    for _ in range(MAX_STEPS):
        if pc == len(prog):
            return list(q)                    # ran off the end (or jumped to len(prog))
        op, arg = prog[pc]
        if len(q) < NEED[op]:
            return list(q)                    # starved: halt without executing it
        pc += 1
        if op == "ADD":
            a = q.popleft(); b = q.popleft(); q.append(min(a + b, LIMIT))
        elif op == "SUB":
            a = q.popleft(); b = q.popleft(); q.append(max(a - b, 0))
        elif op == "DUP":
            a = q.popleft(); q.append(a); q.append(a)
        elif op == "ROT":
            q.append(q.popleft())
        elif op == "DROP":
            q.popleft()
        elif op == "INC":
            q.append(min(q.popleft() + 1, LIMIT))
        elif op == "DEC":
            q.append(max(q.popleft() - 1, 0))
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ":
            if q.popleft() == 0:
                pc = arg
        if len(q) > CAP:
            return None                       # overflow
    return None


# ---- the four puzzles: a reference function (the spec) and the test inputs ----

PAIRS = [[a, b] for a in range(LIMIT + 1) for b in range(LIMIT + 1)]
PAIRS_98 = [[a, b] for a in range(LIMIT) for b in range(LIMIT)]       # Part 4: both inputs <= 98
_rng = random.Random(2026)
LISTS = [list(range(1, k + 1)) for k in range(1, CAP)] + [[LIMIT] * (CAP - 1)] + [
    [_rng.randint(1, LIMIT) for _ in range(_rng.randint(1, CAP - 1))] for _ in range(300)]

PUZZLES = {
    "q1": (lambda q: [max(q)], PAIRS),
    "q2": (lambda q: [abs(q[0] - q[1])], PAIRS),
    "q3": (lambda q: [len(q)], LISTS),
    "q4": (lambda q: [int(q[0] == q[1])], PAIRS_98),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right queue."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, t) == spec(t) for t in tests), len(prog)


def report(submission):
    """submission: {puzzle name: program text}. Prints one line per puzzle, returns the total score."""
    total = 0
    for name in PUZZLES:
        valid, count = check(name, submission[name]) if name in submission else (False, 0)
        total += count if valid else 20
        print(f"{name}: {'valid' if valid else 'INVALID':7s} {count:2d} instructions")
    print(f"total score: {total}")
    return total


# ---- judge helpers (not part of the statement) ----

def why_invalid(prog, values):
    """Why run(prog, values) returned None, as a Chinese sentence: the queue overflowed (at which step, with how
    many values) or the run hit MAX_STEPS. Only used for messages; validity is always decided by run() above."""
    q = deque(values)
    pc = 0
    for step in range(1, MAX_STEPS + 1):
        if pc == len(prog):
            break
        op, arg = prog[pc]
        if len(q) < NEED[op]:
            break
        pc += 1
        if op == "ADD":
            a = q.popleft(); b = q.popleft(); q.append(min(a + b, LIMIT))
        elif op == "SUB":
            a = q.popleft(); b = q.popleft(); q.append(max(a - b, 0))
        elif op == "DUP":
            a = q.popleft(); q.append(a); q.append(a)
        elif op == "ROT":
            q.append(q.popleft())
        elif op == "DROP":
            q.popleft()
        elif op == "INC":
            q.append(min(q.popleft() + 1, LIMIT))
        elif op == "DEC":
            q.append(max(q.popleft() - 1, 0))
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ":
            if q.popleft() == 0:
                pc = arg
        if len(q) > CAP:        # only DUP and ZERO grow the queue, and both fall through: the one that ran is pc - 1
            return (f"队列溢出：执行到第 {step:,} 步（第 {pc - 1} 条指令 {op}）时队列里有 {len(q)} 个数，"
                    f"超过上限 {CAP}（超过就无效）")
    else:
        return f"跑满 {MAX_STEPS:,} 步还没有停机（每个测试最多 {MAX_STEPS:,} 步，超过就无效）"
    return "检查器判为无效"          # not reached: run() returned None, so the loop above did not halt either


def stats(prog, values):
    """(steps, longest queue) of one run that halts, counted like the statement's example: the instruction that
    starves is not a step. Only used to report numbers with info()."""
    q = deque(values)
    pc = 0
    peak = len(q)
    for step in range(MAX_STEPS + 1):
        if pc == len(prog):
            return step, peak
        op, arg = prog[pc]
        if len(q) < NEED[op]:
            return step, peak
        pc += 1
        if op == "ADD":
            a = q.popleft(); b = q.popleft(); q.append(min(a + b, LIMIT))
        elif op == "SUB":
            a = q.popleft(); b = q.popleft(); q.append(max(a - b, 0))
        elif op == "DUP":
            a = q.popleft(); q.append(a); q.append(a)
        elif op == "ROT":
            q.append(q.popleft())
        elif op == "DROP":
            q.popleft()
        elif op == "INC":
            q.append(min(q.popleft() + 1, LIMIT))
        elif op == "DEC":
            q.append(max(q.popleft() - 1, 0))
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ":
            if q.popleft() == 0:
                pc = arg
        peak = max(peak, len(q))
    return None, peak


def random_list(rng, length):
    """A Part 3 input: `length` numbers in 1..99, small and large values mixed."""
    return [rng.choice([rng.randint(1, 9), rng.randint(1, LIMIT), LIMIT]) for _ in range(length)]
