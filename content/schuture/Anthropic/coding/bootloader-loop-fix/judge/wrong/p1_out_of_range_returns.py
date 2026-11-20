# expect: WA
# returns (False, acc, None) for an out-of-range jump instead of raising ProgramError
class ProgramError(Exception):
    """Raised when the program counter would move outside 0 .. len(program)."""


def run_program(program: list[tuple[str, int]]) -> tuple[bool, int, int | None]:
    n = len(program)
    pc, acc, visited = 0, 0, set()
    while True:
        if pc == n:
            return True, acc, None
        if pc < 0 or pc > n:
            return False, acc, None
        if pc in visited:
            return False, acc, pc
        visited.add(pc)
        op, value = program[pc]
        if op == "plus":
            acc += value
            pc += 1
        elif op == "next":
            pc += 1
        elif op == "jump":
            pc += value
        else:
            raise ValueError(f"unknown op {op!r}")

FLIPPED = {"jump": "next", "next": "jump"}


def find_fix_brute_force(program: list[tuple[str, int]]) -> tuple[int, int]:
    for line, (op, value) in enumerate(program):
        if op not in FLIPPED:
            continue                                    # NOTE: plus is never the corrupted line
        candidate = list(program)
        candidate[line] = (FLIPPED[op], value)
        try:
            terminated, acc, _ = run_program(candidate)
        except ProgramError:
            continue                                     # this flip sends pc out of range -- not the fix
        if terminated:
            return line, acc
    raise ValueError("no single flip terminates this program")

def find_fix(program: list[tuple[str, int]]) -> tuple[int, int]:
    n = len(program)

    def succ(i: int) -> int:
        op, value = program[i]
        return i + value if op == "jump" else i + 1

    predecessors = {j: [] for j in range(n + 1)}
    for i in range(n):
        s = succ(i)
        if 0 <= s <= n:
            predecessors[s].append(i)

    reaches_end = [False] * (n + 1)
    reaches_end[n] = True
    stack = [n]
    while stack:
        j = stack.pop()
        for i in predecessors[j]:
            if not reaches_end[i]:
                reaches_end[i] = True
                stack.append(i)

    pc, acc, seen = 0, 0, set()
    while pc not in seen and 0 <= pc < n:
        seen.add(pc)
        op, value = program[pc]
        if op in FLIPPED:
            alternative = pc + 1 if op == "jump" else pc + value
            if 0 <= alternative <= n and reaches_end[alternative]:
                return pc, acc + _run_from(program, alternative)
        if op == "plus":
            acc += value
        pc = succ(pc)
    raise ValueError("no single flip terminates this program")


def _run_from(program: list[tuple[str, int]], start: int) -> int:
    """Sums the plus values from start to termination, following the UNFLIPPED program -- valid
    because membership in reaches_end already guarantees this chain runs to n without a repeat."""
    n, acc, pc = len(program), 0, start
    while pc != n:
        op, value = program[pc]
        if op == "plus":
            acc += value
        pc = pc + value if op == "jump" else pc + 1
    return acc

import re

_SIGNED_INT = re.compile(r"^[+-]\d+$")
_OPS = {"plus", "next", "jump"}


class ParseError(ValueError):
    def __init__(self, line_number: int, raw_line: str):
        super().__init__(f"line {line_number}: cannot parse {raw_line.strip()!r} as '<op> <signed integer>'")
        self.line_number = line_number


def parse_program(text: str) -> list[tuple[str, int]]:
    program = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        code = raw_line.split("#", 1)[0].strip()   # NOTE: strip the comment BEFORE checking for blank
        if not code:
            continue
        tokens = code.split()
        if len(tokens) != 2 or tokens[0] not in _OPS or not _SIGNED_INT.match(tokens[1]):
            raise ParseError(line_number, raw_line)
        program.append((tokens[0], int(tokens[1])))
    return program
