# expect: WA
# reports the line that jumped back into visited code as loop_line instead of the line about to run a second time.
import re


class ProgramError(Exception):
    pass


class ParseError(ValueError):
    def __init__(self, line_number, message=""):
        super().__init__(f"line {line_number}: {message}")
        self.line_number = line_number


def _trace(program):
    """('end', acc, None) | ('loop', acc, line) | ('out', acc, None)."""
    n = len(program)
    visited = [False] * n
    pc = acc = 0
    last = None
    while pc != n:
        if pc < 0 or pc > n:
            return "out", acc, None
        if visited[pc]:
            return "loop", acc, last
        visited[pc] = True
        last = pc
        op, value = program[pc]
        if op == "jump":
            pc += value
        else:
            if op == "plus":
                acc += value
            pc += 1
    return "end", acc, None


def run_program(program):
    kind, acc, line = _trace(program)
    if kind == "out":
        raise ProgramError("pc left the program")
    return kind == "end", acc, line


def find_fix(program):
    swap = {"jump": "next", "next": "jump"}
    for i, (op, value) in enumerate(program):
        if op not in swap:
            continue
        patched = program[:i] + [(swap[op], value)] + program[i + 1:]
        kind, acc, _ = _trace(patched)
        if kind == "end":
            return i, acc
    raise ValueError("no single flip terminates")


_TOKEN = re.compile(r"[+-][0-9]+\Z")


def parse_program(text):
    program = []
    for number, raw in enumerate(text.split("\n"), 1):
        body = raw.partition("#")[0]
        words = body.split()
        if not words:
            continue
        if len(words) != 2:
            raise ParseError(number, "expected two tokens")
        op, value = words
        if op not in ("plus", "next", "jump") or not _TOKEN.match(value) or not value[1:].isascii():
            raise ParseError(number, "bad instruction")
        program.append((op, int(value)))
    return program
