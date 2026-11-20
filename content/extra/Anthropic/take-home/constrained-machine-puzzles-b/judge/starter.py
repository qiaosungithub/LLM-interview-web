# One program per puzzle, as text in the checker's syntax: instructions separated by newlines or ';' (a ';'
# splits even inside a comment), '#' starts a comment, 1..12 instructions; JMP k / JZ k go to instruction k,
# and k = program length halts. The judge runs each program with the checker's parse() and run().
# To try one in 自定义运行:
#     from ref import parse, run
#     print(run(parse(q1), [3, 8]))

# Part 1: input a b (each 0..99); output one number, max(a, b).
q1 = """
"""

# Part 2: input a b (each 0..99); output one number, |a - b|.
q2 = """
"""

# Part 3: input 1 to 15 numbers, each 1..99; output one number, how many numbers the input has.
q3 = """
"""

# Part 4: input a b (each 0..98); output 1 if a = b, otherwise 0.
q4 = """
"""
