# The page's reference programs (参考解答), one per puzzle, copied from its code blocks: 3 + 6 + 6 + 8 = 23.

q1 = """
DUP       # a b      ->  b a a
SUB       # b a a    ->  a (b ∸ a)
ADD       # a (b ∸ a) -> a + (b ∸ a) = max(a, b)
"""

q2 = """
DUP       # a b      ->  b a a
DUP       # b a a    ->  a a b b
ROT       # a a b b  ->  a b b a
SUB       # a b b a  ->  b a (a ∸ b)
SUB       # b a (a ∸ b) -> (a ∸ b) (b ∸ a)
ADD       # -> (a ∸ b) + (b ∸ a) = |a − b|
"""

q3 = """
ZERO      # phase 1: append a 0 ...
JZ 3      # ... and pop the front: a 0 at the front means every input is used up
JMP 0
INC       # phase 2: turn one 0 into a 1 at the back
ADD       # merge the two front values
JMP 3     # until ADD starves
"""

q4 = """
INC       # a b          ->  b (a+1)
ROT       # b (a+1)      ->  (a+1) b
SUB       #              ->  d = (a+1) ∸ b:  0 if a < b,  1 if a = b,  >= 2 if a > b
DUP       # d            ->  d d
JZ 1      # pops d. d = 0: jump to ROT with queue "0", ROT, then SUB starves: output 0
DEC       # d - 1
ZERO      # (d-1) 0
JZ 0      # pops d-1. a = b: jump to INC with queue "0": INC gives 1, ROT, SUB starves: output 1
          # a > b: fall off the end with queue "0": output 0
"""
