# expect: WA
# Part 1: total variation / 2 with true division: a float, so heights near 10**18 lose precision.
def min_horizontal_strokes(h: list[int]) -> int:
    # count run *ends*: plank i ends a run in rows h[i+1] <= y < h[i], with h[n] = 0
    padded = [0] + list(h) + [0]
    # every rise is matched by a fall, so the rises are half the total variation
    return sum(abs(b - a) for a, b in zip(padded, padded[1:])) / 2


class Fence:
    def __init__(self, h: list[int]):
        self.h = list(h)
        self.n = len(self.h)
        self.total = min_horizontal_strokes(self.h)

    def _ends(self, i):
        # run-end contribution of plank i
        if i < 0 or i >= self.n:
            return 0
        right = self.h[i + 1] if i + 1 < self.n else 0
        return max(0, self.h[i] - right)

    def set_height(self, i: int, new_height: int) -> None:
        before = self._ends(i - 1) + self._ends(i)
        self.h[i] = new_height
        self.total += self._ends(i - 1) + self._ends(i) - before

    def strokes(self) -> int:
        return self.total


def min_strokes(h: list[int]) -> int:
    n = len(h)
    if n == 0:
        return 0
    # Cartesian tree on minima (leftmost minimum at the root) via a monotonic stack
    left = [-1] * n
    right = [-1] * n
    stack = []
    for i in range(n):
        last = -1
        while stack and h[stack[-1]] > h[i]:
            last = stack.pop()
        left[i] = last
        if stack:
            right[stack[-1]] = i
        stack.append(i)
    root = stack[0]

    def solve(node, lo, hi, base):
        # cells of planks lo..hi at rows >= base; node is the position of a minimum of h[lo..hi]
        if node == -1:
            return 0
        m = h[node]
        inner = solve(left[node], lo, node - 1, max(m, base)) + solve(right[node], node + 1, hi, max(m, base))
        if m <= base:
            return inner
        return min(hi - lo + 1, (m - base) + inner)

    return solve(root, 0, n - 1, 0)
