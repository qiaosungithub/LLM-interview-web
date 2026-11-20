# expect: WA
# Part 1 free checks only that the address was allocated; a wrong size frees the recorded block.
import random


class MemoryAllocator:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.gaps = [[0, capacity]]        # NOTE: not self.free -- would shadow the free() method below
        self.allocated = {}                # address -> size, for validating free()

    def allocate(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        for gap in self.gaps:              # NOTE: linear scan; the first block with enough room wins
            if gap[1] >= size:
                start = gap[0]
                if gap[1] == size:
                    self.gaps.remove(gap)
                else:
                    gap[0] += size
                    gap[1] -= size
                self.allocated[start] = size
                return start
        raise MemoryError(f"no free block of size >= {size}")

    def free(self, address, size):
        if size <= 0 or address not in self.allocated:
            raise ValueError(f"invalid free({address}, {size})")
        size = self.allocated.pop(address)     # trusts the recorded size and ignores a wrong one
        i = 0
        while i < len(self.gaps) and self.gaps[i][0] < address:
            i += 1
        # NOTE: read both flags first -- merging into gaps[i - 1] must not disturb index i below
        merge_left = i > 0 and self.gaps[i - 1][0] + self.gaps[i - 1][1] == address
        merge_right = i < len(self.gaps) and address + size == self.gaps[i][0]
        if merge_left and merge_right:
            self.gaps[i - 1][1] += size + self.gaps[i][1]
            del self.gaps[i]
        elif merge_left:
            self.gaps[i - 1][1] += size
        elif merge_right:
            self.gaps[i][0] = address
            self.gaps[i][1] += size
        else:
            self.gaps.insert(i, [address, size])


class _Node:
    __slots__ = ("start", "size", "priority", "left", "right", "max_size")

    def __init__(self, start, size):
        self.start, self.size, self.max_size = start, size, size
        self.priority = random.random()        # NOTE: random priorities -> balanced in expectation
        self.left = self.right = None

def _pull(node):                                # recompute the augmentation from the two children
    node.max_size = max(node.size, node.left.max_size if node.left else 0,
                         node.right.max_size if node.right else 0)

def _merge(a, b):                               # every key in a is < every key in b; max-heap on priority
    if a is None or b is None:
        return a or b
    if a.priority > b.priority:
        a.right = _merge(a.right, b)
        _pull(a)
        return a
    b.left = _merge(a, b.left)
    _pull(b)
    return b

def _split(node, key):                          # -> (start <= key, start > key), same relative order
    if node is None:
        return None, None
    if node.start <= key:
        lo, hi = _split(node.right, key)
        node.right = lo
        _pull(node)
        return node, hi
    lo, hi = _split(node.left, key)
    node.left = hi
    _pull(node)
    return lo, node

def _insert(root, start, size):
    lo, hi = _split(root, start)
    return _merge(_merge(lo, _Node(start, size)), hi)

def _delete(root, start):
    lo, hi = _split(root, start)                # lo: start <= key, hi: start > key
    lo2, _mid = _split(lo, start - 1)           # NOTE: unique keys -> isolates exactly one node
    return _merge(lo2, hi)

def _first_fit(node, size):                     # leftmost node with size >= `size`, else None
    if node is None or node.max_size < size:
        return None
    if node.left and node.left.max_size >= size:    # NOTE: left subtree only has smaller addresses
        return _first_fit(node.left, size)
    return node if node.size >= size else _first_fit(node.right, size)

def _neighbours(node, key):                     # (predecessor, successor) of `key`, one walk down
    pred = succ = None
    while node:
        if node.start < key:
            pred, node = node, node.right
        else:
            succ, node = node, node.left
    return pred, succ


class MemoryAllocatorLog:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._free = _Node(0, capacity)         # treap root; None once memory is exhausted
        self.allocated = {}                     # address -> size, for validating free()

    def allocate(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        node = _first_fit(self._free, size)
        if node is None:
            raise MemoryError(f"no free block of size >= {size}")
        start = node.start
        self._free = _delete(self._free, start)
        if node.size > size:
            self._free = _insert(self._free, start + size, node.size - size)
        self.allocated[start] = size
        return start

    def free(self, address, size):
        if size <= 0 or self.allocated.get(address) != size:
            raise ValueError(f"invalid free({address}, {size})")
        del self.allocated[address]
        pred, succ = _neighbours(self._free, address)
        start, total = address, size
        if pred is not None and pred.start + pred.size == address:
            start, total = pred.start, pred.size + total
            self._free = _delete(self._free, pred.start)
        if succ is not None and address + size == succ.start:
            total += succ.size
            self._free = _delete(self._free, succ.start)
        self._free = _insert(self._free, start, total)
