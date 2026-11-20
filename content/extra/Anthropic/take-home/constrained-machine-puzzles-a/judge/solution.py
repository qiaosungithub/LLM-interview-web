# The page's reference programs (参考解答), one per puzzle, copied from its code blocks: 3 + 5 + 5 + 6 + 7 = 26.

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
-TZ       # playing: F = (x == 0), i.e. a 0 switches to muted
+IN       # muted: read y (also right after the switch)
-OUT      # playing: echo x
+TP       # muted: F = (y > 0), i.e. a 0 switches back to playing
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
TZ        # F = 1 exactly when the remaining count just reached 0
SWP       # every lap: switch which register INC works on
INC
+IN       # remaining is 0: read the next x ...
+NEG      # ... and store it as -x
TP        # positive = counter, negative (or 0) = remaining
+OUT      # the counter just went up: output it
"""
