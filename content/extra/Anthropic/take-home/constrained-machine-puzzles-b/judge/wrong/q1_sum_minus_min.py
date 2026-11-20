# expect: WA
# Wrong: Part 1 computes max(a, b) as (a + b) ∸ min(a, b), forgetting that ADD saturates at 99. Passes every example.

q1 = """
DUP       # a b        ->  b a a
DUP       # b a a      ->  a a b b
DUP       # a a b b    ->  a b b a a
SUB       #            ->  b a a (a ∸ b)
ADD       #            ->  a (a ∸ b) (b + a)
SUB       #            ->  (a + b) min(a, b),  min = a ∸ (a ∸ b)
SUB       #            ->  (a + b) ∸ min(a, b): wrong once a + b > 99
"""

q2 = "DUP; DUP; ROT; SUB; SUB; ADD"

q3 = "ZERO; JZ 3; JMP 0; INC; ADD; JMP 3"

q4 = "INC; ROT; SUB; DUP; JZ 1; DEC; ZERO; JZ 0"
