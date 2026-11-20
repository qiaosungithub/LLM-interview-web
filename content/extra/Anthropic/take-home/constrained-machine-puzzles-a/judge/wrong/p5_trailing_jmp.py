# expect: WA
# Wrong: Part 5 is correct but has 13 instructions (limit 12): the trailing JMP 0 is redundant because the program is a ring.

p1 = """
IN        # read x
+OUT      # after the first 0: echo
-TZ       # before the first 0: F = (x == 0)
"""

p2 = """
IN        # read x
TZ        # F = (x == 0)
-ADD      # x != 0: acc = x + running sum
SWP       # x != 0: bak = new sum   |  x == 0: acc = sum, bak = 0
+OUT      # x == 0: output the sum
"""

p3 = """
-IN       # playing: read x
-TZ       # playing: F = (x == 0)
+IN       # muted: read y
-OUT      # playing: echo x
+TP       # muted: F = (y > 0)
"""

p4 = """
-IN       # only when the previous count is finished
TP        # F = (acc > 0)
+OUT
NEG       # acc - 1 written as -(-acc + 1)
INC
NEG
"""

p5 = """
IN        # 0: read x
NEG       # 1: remaining stored as -x
TZ        # 2
+JMP 0    # 3: x = 0: nothing to count
SWP       # 4: acc = counter, bak = -remaining
INC       # 5
OUT       # 6
SWP       # 7: acc = -remaining, bak = counter
INC       # 8: one fewer remaining
TZ        # 9
-JMP 4    # 10
SAV       # 11: counter back to 0
JMP 0     # 12
"""
