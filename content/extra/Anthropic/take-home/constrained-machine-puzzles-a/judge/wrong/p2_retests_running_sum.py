# expect: WA
# Wrong: Part 2 re-tests the running sum after ADD, so a group whose partial sum returns to 0 is cut there. Passes every example.

p1 = """
IN        # read x
+OUT      # after the first 0: echo
-TZ       # before the first 0: F = (x == 0)
"""

p2 = """
IN        # read x
TZ        # F = (x == 0)
-ADD      # x != 0: acc = x + running sum
TZ        # refresh the flag on the new sum (the bug)
SWP
+OUT
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
TZ        # remaining just reached 0?
SWP       # INC works on the other register every lap
INC
+IN
+NEG      # store the new x as -x
TP        # positive = counter
+OUT
"""
