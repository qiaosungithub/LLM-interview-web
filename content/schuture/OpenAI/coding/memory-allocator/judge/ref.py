"""Helpers for tests.py: an oracle that stores only the allocated blocks, and random operation sequences.

The free space is the complement of the allocated blocks, so the maximal free blocks are exactly the gaps between
consecutive allocated blocks; first fit is the lowest such gap that is large enough. Nothing is ever merged.
"""
import bisect


class Oracle:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError
        self.capacity = capacity
        self.starts = []            # sorted starts of allocated blocks
        self.size = {}              # start -> size

    def allocate(self, size):
        if size <= 0:
            raise ValueError
        prev_end = 0
        for s in self.starts + [self.capacity]:
            if s - prev_end >= size:
                bisect.insort(self.starts, prev_end)
                self.size[prev_end] = size
                return prev_end
            if s < self.capacity:
                prev_end = s + self.size[s]
        raise MemoryError

    def free(self, address, size):
        if size <= 0 or self.size.get(address) != size:
            raise ValueError
        del self.size[address]
        self.starts.pop(bisect.bisect_left(self.starts, address))

    def gaps(self):
        out, prev_end = [], 0
        for s in self.starts + [self.capacity]:
            if s > prev_end:
                out.append((prev_end, s - prev_end))
            if s < self.capacity:
                prev_end = s + self.size[s]
        return out


def outcome(obj, method, args):
    """("ok", value) or (exception class name, None) for the two exceptions the statement names."""
    try:
        return "ok", getattr(obj, method)(*args)
    except (ValueError, MemoryError) as e:
        return ("ValueError" if isinstance(e, ValueError) else "MemoryError"), None


def random_ops(r, capacity, length, bad_rate=0.15):
    """A random sequence of (method, args), valid and invalid calls mixed, planned against an Oracle."""
    model = Oracle(capacity)
    ops = []
    for _ in range(length):
        live = list(model.size.items())
        x = r.random()
        if live and x < 0.45:
            addr, size = r.choice(live)
            if r.random() < bad_rate:
                kind = r.randrange(5)
                if kind == 0:
                    size = size + r.choice([-1, 1]) if size > 1 else size + 1    # wrong size
                elif kind == 1:
                    addr = addr + r.randint(1, max(1, size - 1)) if size > 1 else addr + 1   # interior / other
                elif kind == 2:
                    size = r.choice([0, -size])                                   # non-positive size
                elif kind == 3:
                    addr = r.randint(-2, capacity + 2)                            # arbitrary address
                # kind 4: a valid free, then the same free again (double free)
                ops.append(("free", (addr, size)))
                outcome(model, "free", (addr, size))
                if kind == 4:
                    ops.append(("free", (addr, size)))
                continue
            ops.append(("free", (addr, size)))
            outcome(model, "free", (addr, size))
        else:
            if r.random() < 0.04:
                size = r.choice([0, -1, -capacity])
            elif r.random() < 0.6:
                size = r.randint(1, max(1, capacity // 6))
            else:
                size = r.randint(1, capacity + 1)
            ops.append(("allocate", (size,)))
            outcome(model, "allocate", (size,))
    return ops
