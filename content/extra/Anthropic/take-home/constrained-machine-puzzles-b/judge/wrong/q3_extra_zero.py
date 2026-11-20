# expect: WA
# Wrong: Part 3 keeps one 0 too many while replacing the inputs, so 15 inputs overflow the 16-value queue. Passes every example.

q1 = "DUP; SUB; ADD"

q2 = "DUP; DUP; ROT; SUB; SUB; ADD"

q3 = """
ZERO      # 0: an end marker
ZERO      # 1: one 0 per input (the first lap pushes it before any input is popped)
JZ 4      # 2: pop the front: a 0 means every input is used up
JMP 1     # 3
DROP      # 4: k + 1 zeros are left: drop one
INC       # 5: count them as in the reference
ADD       # 6
JMP 5     # 7: until ADD starves
"""

q4 = "INC; ROT; SUB; DUP; JZ 1; DEC; ZERO; JZ 0"
