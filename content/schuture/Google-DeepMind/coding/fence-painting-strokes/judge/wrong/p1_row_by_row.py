# expect: TLE
# Part 1 counts the runs of every row one by one: right, but its time grows with the heights (10**9 rows).

def min_horizontal_strokes(h: list[int]) -> int:
    n = len(h)
    total = 0
    for y in range(max(h, default=0)):
        i = 0
        while i < n:
            if h[i] <= y:
                i += 1
                continue
            total += 1
            while i < n and h[i] > y:
                i += 1
    return total


class Fence:
    def __init__(self, h: list[int]) -> None:
        self._h = list(h)
        self._term = [0] * len(self._h)
        self._total = 0
        previous = 0
        for i, height in enumerate(self._h):
            self._term[i] = max(0, height - previous)
            self._total += self._term[i]
            previous = height

    def set_height(self, i: int, new_height: int) -> None:
        n = len(self._h)
        left = self._h[i - 1] if i > 0 else 0        # NOTE: i == 0 means the virtual h[-1] == 0
        new_term_i = max(0, new_height - left)
        self._total += new_term_i - self._term[i]
        self._term[i] = new_term_i
        if i + 1 < n:                                 # NOTE: no term i + 1 to fix when i is the last plank
            new_term_next = max(0, self._h[i + 1] - new_height)
            self._total += new_term_next - self._term[i + 1]
            self._term[i + 1] = new_term_next
        self._h[i] = new_height

    def strokes(self) -> int:
        return self._total


def _runs_above(h: list[int], lo: int, hi: int, floor: int) -> list[tuple[int, int]]:
    """Maximal contiguous sub-ranges of [lo, hi] where every plank's height exceeds floor."""
    runs = []
    i = lo
    while i <= hi:
        if h[i] <= floor:
            i += 1
            continue
        j = i
        while j <= hi and h[j] > floor:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _solve_range(h: list[int], l: int, r: int, base: int) -> int:
    """Minimum strokes to paint cells (i, y) with l <= i <= r and base <= y < h[i]."""
    answer: dict[tuple[int, int], int] = {}
    info: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}
    stack = [(l, r, base, False)]   # NOTE: explicit stack -- a sorted array needs depth O(n), past Python's limit
    while stack:
        lo, hi, b, expanded = stack.pop()
        if not expanded:
            m = min(h[lo:hi + 1])
            children = _runs_above(h, lo, hi, m)
            info[(lo, hi)] = (m, children)
            stack.append((lo, hi, b, True))
            for cl, cr in children:
                stack.append((cl, cr, m, False))
        else:
            m, children = info[(lo, hi)]
            horizontal = (m - b) + sum(answer[child] for child in children)
            vertical = hi - lo + 1
            answer[(lo, hi)] = min(vertical, horizontal)   # NOTE: (lo, hi) is a safe key alone -- ranges never collide
    return answer[(l, r)]


def min_strokes(h: list[int]) -> int:
    n = len(h)
    return sum(_solve_range(h, lo, hi, 0) for lo, hi in _runs_above(h, 0, n - 1, 0))
