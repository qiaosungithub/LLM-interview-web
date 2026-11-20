# expect: WA
# Wrong: Part 2 is the reference ring written from TZ; the first TZ sees the initial acc = 0 and prints a 0 before reading.

p1 = """
IN        # read x
+OUT      # after the first 0: echo
-TZ       # before the first 0: F = (x == 0)
"""

p2 = """
TZ        # F = (x == 0)
-ADD      # x != 0: acc = x + running sum
SWP       # x != 0: bak = new sum   |  x == 0: acc = sum, bak = 0
+OUT      # x == 0: output the sum
IN        # read the next x
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
