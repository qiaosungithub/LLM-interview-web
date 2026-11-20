# expect: TLE
# Part 2 finds neighbours with bisect but still scans every free block in allocate: O(n) per call.
import bisect


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
        if size <= 0 or self.allocated.get(address) != size:
            raise ValueError(f"invalid free({address}, {size})")
        del self.allocated[address]
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


class MemoryAllocatorLog:
    """Free blocks as two parallel lists sorted by start: free uses bisect, but allocate still scans."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.starts = [0]
        self.sizes = [capacity]
        self.allocated = {}

    def allocate(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        sizes = self.sizes
        for i in range(len(sizes)):            # O(n) per call
            if sizes[i] >= size:
                start = self.starts[i]
                if sizes[i] == size:
                    del self.starts[i], sizes[i]
                else:
                    self.starts[i] += size
                    sizes[i] -= size
                self.allocated[start] = size
                return start
        raise MemoryError(f"no free block of size >= {size}")

    def free(self, address, size):
        if size <= 0 or self.allocated.get(address) != size:
            raise ValueError(f"invalid free({address}, {size})")
        del self.allocated[address]
        starts, sizes = self.starts, self.sizes
        i = bisect.bisect_left(starts, address)
        merge_left = i > 0 and starts[i - 1] + sizes[i - 1] == address
        merge_right = i < len(starts) and address + size == starts[i]
        if merge_left and merge_right:
            sizes[i - 1] += size + sizes[i]
            del starts[i], sizes[i]
        elif merge_left:
            sizes[i - 1] += size
        elif merge_right:
            starts[i] = address
            sizes[i] += size
        else:
            starts.insert(i, address)
            sizes.insert(i, size)
