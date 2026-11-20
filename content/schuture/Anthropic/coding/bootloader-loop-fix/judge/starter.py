class ProgramError(Exception):
    """Raised when a jump would move pc outside 0 .. len(program)."""


def run_program(program: list[tuple[str, int]]) -> tuple[bool, int, int | None]:
    """Run the program from pc = 0, acc = 0 and return (terminated, acc, loop_line)."""
    raise NotImplementedError


def find_fix(program: list[tuple[str, int]]) -> tuple[int, int]:
    """Return the line whose operator flip makes the program terminate, and acc at termination."""
    raise NotImplementedError


class ParseError(ValueError):
    """line_number is 1-based and counts every line of the input text -- not the 0-based index
    the matching instruction gets in the returned program."""
    line_number: int


def parse_program(text: str) -> list[tuple[str, int]]:
    """Turn the program text into a list of (op, value), skipping blank and comment lines."""
    raise NotImplementedError
