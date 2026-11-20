# expect: WA
# Wrong: Part 4 adds 1 before the decrement-then-test loop; INC saturates at 99, so x = 99 counts down from 98. Passes every example.

p1 = "IN; TZ; -JMP 0; IN; OUT; JMP 3"

p2 = "IN; TZ; +JMP 6; ADD; SAV; JMP 0; SWP; OUT"

p3 = "IN; TZ; -OUT; -JMP 0; IN; TZ; -JMP 4"

p4 = """
-IN       # new x only when the previous count is finished
-INC      # x + 1, so the loop can decrement before testing
NEG       # acc - 1
INC
NEG
TP        # F = (acc > 0)
+OUT
"""

p5 = "IN; NEG; TZ; +SWP; +JMP 0; INC; SWP; INC; OUT; SWP; JMP 2"
