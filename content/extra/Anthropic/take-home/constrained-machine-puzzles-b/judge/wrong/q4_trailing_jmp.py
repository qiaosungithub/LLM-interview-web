# expect: WA
# Wrong: Part 4 is correct but has 13 instructions (limit 12): the final JMP 13 is redundant, running off the end already halts.

q1 = "DUP; SUB; ADD"

q2 = "DUP; DUP; ROT; SUB; SUB; ADD"

q3 = "ZERO; JZ 3; JMP 0; INC; ADD; JMP 3"

q4 = """
DUP       # 0: a b ->  b a a
DUP       # 1:     ->  a a b b
DUP       # 2:     ->  a b b a a
SUB       # 3:     ->  b a a (a ∸ b)
SUB       # 4:     ->  a (a ∸ b) (b ∸ a)
DROP      # 5:     ->  (a ∸ b) (b ∸ a)
ADD       # 6:     ->  |a - b|
JZ 10     # 7: equal?
ZERO      # 8: not equal: output 0
JMP 13    # 9: halt
ZERO      # 10: equal: output 1
INC       # 11
JMP 13    # 12: halt
"""
