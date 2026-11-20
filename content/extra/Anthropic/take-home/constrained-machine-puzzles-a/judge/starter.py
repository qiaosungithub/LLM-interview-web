# One program per puzzle, as text in the checker's syntax: instructions separated by newlines or ';',
# '#' starts a comment, an optional guard '+' (run only if F=1) or '-' (only if F=0) in front, 1..12 instructions.
# The judge runs each program with the checker's parse() and run(). To try one in 自定义运行:
#     from ref import parse, run
#     print(run(parse(p1), [5, 0, 3, 0, -2, 7]))

# Part 1: output every number after the first 0, in order; the first 0 itself is not output.
p1 = """
"""

# Part 2: at each 0, output the sum of the numbers since the previous 0 (or the start).
p2 = """
"""

# Part 3: each 0 toggles playing / muted (start: playing); output the other numbers only while playing.
p3 = """
"""

# Part 4: for each x, output x, x-1, ..., 1 (nothing for x = 0).
p4 = """
"""

# Part 5: for each x, output 1, 2, ..., x (nothing for x = 0).
p5 = """
"""
