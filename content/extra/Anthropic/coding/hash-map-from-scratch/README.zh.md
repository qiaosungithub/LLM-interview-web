# 从零实现哈希表

## 题目

你刚用 `dict` 把一道题解完，面试官接着说：现在不许用 `dict` 和 `set`，自己实现一个哈希表（hash map）。
可以用 `list`、内置的 `hash()` 和 `==`。冲突处理规定用开放寻址（open addressing）加线性探测
（linear probing）。

为了让例子能手算，表的布局规定如下：

- 表是一个长度为 `capacity` 的槽（slot）数组。`capacity` 始终是 2 的幂，初始为 8。
- 键 `key` 的起始槽是 `hash(key) & (capacity - 1)`。这个槽被别的键占了，就看下一个槽，到末尾后绕回
  槽 0，直到找到这个键或找到空槽。
- 两个键 `a`、`b` 算同一个键，当且仅当 `a is b or a == b`（和 `dict` 一样）。
- 扩容规则：插入一个**新**键之前，如果 `3 * (已占用的槽数 + 1) > 2 * capacity`，先把容量翻倍，
  把所有键按旧表槽 0、1、2、…的顺序重新放进新表，然后再插入新键。也就是占用率不超过 2/3：
  8 个槽最多放 5 个键，放第 6 个之前先扩到 16。

### Part 1 —— 插入、查找、扩容

```py
class HashMap:
    def __init__(self) -> None: ...

    def put(self, key, value) -> None:
        """Insert key, or overwrite its value if it is already present."""

    def get(self, key, default=None):
        """The value stored for key, or default if key is absent."""

    def __contains__(self, key) -> bool: ...

    def __len__(self) -> int:
        """Number of keys stored."""
```

```text
m = HashMap()
m.put("a", 1); m.put("b", 2); m.put("a", 3)
len(m)            -> 2
m.get("a")        -> 3
m.get("zzz")      -> None
m.get("zzz", 0)   -> 0
"b" in m          -> True

m.put(1, "int"); m.put(1.0, "float")     # 1 == 1.0 且 hash(1) == hash(1.0)：是同一个键
len(m)            -> 3
m.get(1)          -> "float"
m.get(True)       -> "float"             # True == 1，同理
```

同时回答：`put` 和 `get` 的代价，期望是多少、最坏是多少？扩容一次是 $O(n)$，为什么说 `put` 仍然是
均摊（amortised）$O(1)$？

### Part 2 —— 删除

```py
    def delete(self, key) -> bool:
        """Remove key if it is present. Returns whether it was present."""
```

先回答：为什么不能直接把被删的那个槽清空？给一个三四步操作的反例。然后把删除实现正确，墓碑
（tombstone）和后移删除（backward-shift deletion）任选一种。要求：

1. 任意操作序列之后，结果和 `dict` 一致。
2. 表不能因为反复“插入一个新键、删除一个旧键”而越来越大：如果表里同时存在的键数从未超过 $k$，
   容量始终不超过 $\max(8, 4k)$。
3. 任何操作都不能因为“表里一个空槽都没有了”而死循环。

有了删除之后，Part 1 的扩容规则可以按需要调整，只要满足上面三条，并且一个键都没删过时仍和 Part 1 的规则一致。

```text
m = HashMap()
for k in (0, 8, 16): m.put(k, str(k))    # 小整数的 hash 就是它自己，这三个键的起始槽都是 0
m.delete(8)       -> True
m.get(16)         -> "16"
m.delete(8)       -> False
len(m)            -> 2
m.put(8, "again")
m.get(8)          -> "again"
len(m)            -> 3
```

### Part 3 —— 遍历

```py
    def items(self):
        """Iterates over (key, value) for every stored key, in slot order."""

    def keys(self): ...

    def __iter__(self):
        """Same as keys()."""
```

遍历顺序规定为槽的顺序，从槽 0 到槽 `capacity - 1`，每个键恰好出现一次。遍历过程中如果有键被加入
或删除，之后第一次向迭代器取下一个元素时必须抛 `RuntimeError`；只是覆盖已有键的值不算。

```text
m = HashMap()
for k in (9, 1, 2): m.put(k, None)
list(m)           -> [9, 1, 2]           # 9 的起始槽是 9 & 7 = 1；1 的起始槽也是 1，被占，落到槽 2；2 落到槽 3
for k in (3, 4): m.put(k, None)
list(m)           -> [9, 1, 2, 3, 4]
m.put(5, None)                           # 第 6 个键：先扩到 16，再插入
list(m)           -> [1, 2, 3, 4, 5, 9]
```

同时回答：这个顺序和插入顺序是什么关系？一次完整遍历的代价是多少？（有删除之后，具体顺序取决于你选的
删除方案，题目不作规定。）

### Part 4 —— 自定义哈希与相等

有些键没法直接用内置的 `hash`：`list` 不可哈希；或者想让字符串键不区分大小写。让构造函数接受两个
可选参数：

```py
class HashMap:
    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        """hash_fn(key) -> int replaces hash(key); eq_fn(a, b) -> bool replaces a == b.
        The caller guarantees: eq_fn(a, b) implies hash_fn(a) == hash_fn(b)."""
```

要求：`put`、`get`、`delete`、`in` 每次调用，对传入的键恰好调用一次 `hash_fn`；扩容时一次都不调用
（对一个大数组算哈希可能很贵）。

```text
m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
m.put("Key", 1); m.put("KEY", 2)
len(m)            -> 1
m.get("key")      -> 2
list(m)           -> ["Key"]             # 覆盖只换值，保留最先存入的那个键对象

m = HashMap(hash_fn=lambda xs: hash(tuple(xs)))
m.put([1, 2], "a")
m.get([1, 2])     -> "a"
[2, 1] in m       -> False
```

再回答两个问题：

- 用 `list` 作键，放进去之后又被原地修改了，会发生什么？
- 键是 `i << 40`（`i = 0, 1, …, 1999`）时，默认的 `hash` 会让这张表变成什么样？写一个 `hash_fn` 修好它。

## 参考解答

<details>
<summary>展开参考解答</summary>

下面用继承把四个部分串起来，每个部分只写新增的代码：`LinearProbingMap`（Part 1）→ `TombstoneMap`
（Part 2）→ `IterableMap`（Part 3）→ `HashMap`（Part 4）。面试时直接在同一个类上一步步改就行。

### Part 1

用三个等长的列表存表：键、值、以及每个键的哈希值。空槽用一个专门的哨兵对象 `_EMPTY` 标记，不能用
`None`，因为 `None` 本身可以是键。

所有操作都归结为一个内部函数 `_find(key, h)`：从起始槽开始往后走，遇到这个键就返回“找到了，在槽 i”；
遇到空槽就返回“没有，它该放在槽 i”。它成立靠一条不变量：

> 从一个键的起始槽到它实际所在的槽，中间没有空槽。

所以查找时一旦碰到空槽就可以断定键不存在。插入总是放在探测路径上的第一个空槽，不变量保持。只要表里
至少有一个空槽，`_find` 就一定会停；扩容规则保证占用率不超过 2/3，所以空槽总是有的。

把哈希值存下来有两个好处：扩容时不用重新算哈希；探测时先比哈希值（整数比较，很便宜），哈希值相同
才调用 `==`。扩容时往新表里放键，以及扩容之后放那个新键，都已经知道键不在表里，只要找第一个空槽
（`_free_slot`），一次键比较都不用做。

```python
_EMPTY = object()   # a slot that has never held a key: every probe sequence stops here


class LinearProbingMap:
    """Part 1: open addressing with linear probing. Grows by doubling; no deletion yet."""

    def __init__(self) -> None:
        self._allocate(8)

    def _allocate(self, capacity: int) -> None:
        self._capacity = capacity                 # always a power of two
        self._hashes = [0] * capacity             # hash of the key in each slot (unused if no key)
        self._keys = [_EMPTY] * capacity
        self._values = [None] * capacity
        self._size = 0                            # number of keys stored

    def _hash(self, key) -> int:
        return hash(key)

    def _same(self, a, b) -> bool:
        return a is b or a == b

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is the
        empty slot that ended the probe sequence -- which is where key would be inserted."""
        mask = self._capacity - 1
        i = h & mask                              # NOTE: for a negative h, & still lands in [0, mask]
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return i, False
            if self._hashes[i] == h and self._same(k, key):   # compare the stored hash first
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        """(hash, key, value) of every stored key, in slot order."""
        for i in range(self._capacity):
            if self._keys[i] is not _EMPTY:
                yield self._hashes[i], self._keys[i], self._values[i]

    def _free_slot(self, h: int) -> int:
        """The first empty slot on the probe sequence of hash h, for a key known to be absent."""
        mask = self._capacity - 1
        i = h & mask
        while self._keys[i] is not _EMPTY:
            i = (i + 1) & mask
        return i

    def _rebuild(self, capacity: int) -> None:
        entries = list(self._entries())
        self._allocate(capacity)
        for h, key, value in entries:
            i = self._free_slot(h)                # the keys are distinct: no comparison needed
            self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size = len(entries)

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value               # overwrite: the stored key object is kept
            return
        if 3 * (self._size + 1) > 2 * self._capacity:
            self._rebuild(2 * self._capacity)
            i = self._free_slot(h)                # NOTE: the slot found before the rebuild is stale
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def get(self, key, default=None):
        i, found = self._find(key, self._hash(key))
        return self._values[i] if found else default

    def __contains__(self, key) -> bool:
        return self._find(key, self._hash(key))[1]

    def __len__(self) -> int:
        return self._size
```

**代价。** 一次操作的代价是探测的槽数。设占用率为 $\alpha$。

- 期望：如果哈希值表现得像均匀随机，线性探测的经典结论（Knuth）是：查找一个存在的键平均探测约
  $\frac{1}{2}\left(1 + \frac{1}{1-\alpha}\right)$ 个槽，查找不存在的键或插入约
  $\frac{1}{2}\left(1 + \frac{1}{(1-\alpha)^2}\right)$ 个槽。$\alpha = 2/3$ 时分别是 2 和 5，所以是期望
  $O(1)$。占用率上限不能定得太高也是这个原因：$\alpha = 0.9$ 时后一个式子是 50.5。这些是大表上的
  平均值；单独一张表实测会上下浮动，表越小浮动越大。
- 最坏：$O(n)$。如果所有键的哈希值都撞在一起，它们连成一段，每次查找都要走完这一段。$n$ 个哈希值
  完全相同的键，光是依次插入就要做恰好 $n(n-1)/2$ 次键比较。这由哈希函数和键决定，表自己防不了
  （Part 4 会看到一个例子）。

所以准确的说法是“期望 $O(1)$，最坏 $O(n)$”，不是“$O(1)$”。

**扩容为什么是均摊 $O(1)$。** 只插入时，容量依次是 8、16、32、…。从容量 $C$ 扩到 $2C$ 要扫描 $C$ 个
旧槽、分配 $2C$ 个新槽，共 $3C$。所有扩容加起来是 $3(8 + 16 + \dots + C_{\text{最终}}/2) < 3C_{\text{最终}}$。
而最后一次扩容发生时键数已经超过旧容量的 2/3，即 $n > C_{\text{最终}}/3$，所以总量小于 $9n$：
摊到每次插入上是常数。重新放入每个键时还要探测，新表占用率不到 1/2，这部分是期望 $O(1)$。
合起来，`put` 是**均摊期望** $O(1)$；单次 `put` 最坏仍是 $O(n)$（正好赶上扩容的那一次）。

### Part 2

**为什么不能直接清空。** 看题目里的例子：0、8、16 的起始槽都是 0，依次落在槽 0、1、2。如果删除 8 时
把槽 1 直接标成空，再查 16：从槽 0 开始，槽 0 是键 0，不是；槽 1 是空的，于是断定 16 不存在。但 16
明明在槽 2。原因是清空破坏了 Part 1 的不变量：16 的探测路径中间出现了空槽。之后如果再 `put(16, …)`，
还会在槽 1 再放一个 16，表里同一个键出现两次。

**墓碑。** 被删的槽不标成空，而是放一个墓碑标记 `_TOMB`。查找时把墓碑当作“有东西，继续走”；插入新键
时可以复用路径上遇到的第一个墓碑。不变量改成：从起始槽到键所在的槽之间没有 `_EMPTY`（可以有墓碑）。

墓碑带来两个必须处理的问题，正对应题目的要求 3 和要求 2：

- 墓碑也挡路。如果负载规则只数存活的键，反复增删之后表里可能全是键和墓碑，一个 `_EMPTY` 都没有，
  查找不存在的键就会死循环。所以**负载规则里墓碑也算占用**：`3 * (键数 + 墓碑数 + 1) > 2 * capacity`
  时重建。复用墓碑的插入不消耗 `_EMPTY`，不用检查。
- 重建会扔掉所有墓碑，所以新容量应该**只按存活的键数定**：取不小于 `2 * (键数 + 1)` 的最小的 2 的幂
  （至少 8）。没有墓碑时，这正好是翻倍，和 Part 1 一致；墓碑很多时，容量不变甚至变小，
  相当于原地清理了一遍。

```python
_TOMB = object()    # a slot whose key was deleted: probe sequences walk past it


class TombstoneMap(LinearProbingMap):
    """Part 2: deletion with tombstones."""

    def _allocate(self, capacity: int) -> None:
        super()._allocate(capacity)
        self._tombs = 0                           # slots holding _TOMB

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is where key
        should be inserted: the first tombstone passed, or else the empty slot that ended the search."""
        mask = self._capacity - 1
        i = h & mask
        first_tomb = -1
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return (i if first_tomb < 0 else first_tomb), False
            if k is _TOMB:
                if first_tomb < 0:
                    first_tomb = i                # NOTE: keep walking -- the key may still be further on
            elif self._hashes[i] == h and self._same(k, key):
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        for i in range(self._capacity):
            k = self._keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield self._hashes[i], k, self._values[i]

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value
            return
        if self._keys[i] is _TOMB:
            self._tombs -= 1                      # reusing a tombstone uses up no empty slot
        elif 3 * (self._size + self._tombs + 1) > 2 * self._capacity:   # NOTE: tombstones count
            capacity = 8
            while capacity < 2 * (self._size + 1):                      # NOTE: sized by live keys only
                capacity *= 2
            self._rebuild(capacity)               # drops every tombstone
            i = self._free_slot(h)
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        self._keys[i] = _TOMB                     # NOTE: not _EMPTY -- other keys may sit beyond this slot
        self._values[i] = None                    # drop the reference to the value
        self._size -= 1
        self._tombs += 1
        return True
```

**这样定容量为什么够。**

- 容量上界（要求 2）：重建时新容量是不小于 `2 * (键数 + 1)` 的最小的 2 的幂。同时存在的键数不超过 $k$
  时，`键数 + 1` 不超过 $k$，而不小于 $2k$ 的最小的 2 的幂小于 $4k$。所以容量不超过 $\max(8, 4k)$。
- 均摊：把一次重建的开销记为“旧容量 + 新容量”个槽。新表建好并放入触发它的那个键之后，占用不超过
  一半；要到占用超过 2/3 才会再次重建。占用数只在“新键放进 `_EMPTY`”时增加（删除是键变墓碑，
  复用墓碑是墓碑变键，都不改变占用数）。所以容量为 $C$ 的表在被换掉之前，至少吸收了
  $\lfloor 2C/3 \rfloor - C/2 \ge C/8$ 次这样的插入。换掉它的那次重建开销不超过 $C + 2C = 3C$。
  每次插入新键因此最多分摊到 $3C / (C/8) = 24$ 个槽。和 Part 1 一样，重新放入时的探测是期望 $O(1)$，
  所以 `put` 仍是均摊期望 $O(1)$；`delete` 和 `get` 不触发重建，是期望 $O(1)$。
- 如果新容量按“刚好不超过 2/3”来定，会出问题：键数卡在边界上时，每删一个再插一个就可能又触发一次
  重建，均摊就不成立了。留出从 1/2 到 2/3 这一段余量，才有上面的 $C/8$。

有墓碑时，查找要走过的是键和墓碑的总和，所以探测长度取决于“键数 + 墓碑数”的占用率。负载规则限制的
正是这个量。

**另一种做法：后移删除。** 不留墓碑。删掉一个键之后，往后看这一段连续的非空槽：如果某个键搬到空出来
的洞里之后，仍然在“起始槽到所在槽之间没有空槽”的范围内，就把它搬过来，洞移到它原来的位置，继续往后，
直到遇到空槽。判断条件是：槽 `j` 里的键可以搬到洞 `i`，当且仅当它的起始槽不在环形区间 `(i, j]` 里。
这样表里永远没有墓碑，Part 1 的 `put` 和负载规则一行都不用改；代价是删除时要搬动元素，
而且那个环形区间的条件很容易写错。

```python
class ShiftDeleteMap(LinearProbingMap):
    """Part 2, the alternative: backward-shift deletion. No tombstones, so put() is unchanged."""

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        mask = self._capacity - 1
        j = i                                     # i is the hole; j scans the rest of the run
        while True:
            j = (j + 1) & mask
            if self._keys[j] is _EMPTY:
                break
            home = self._hashes[j] & mask
            # distances measured backwards from j, modulo capacity: home is outside (i, j]
            # exactly when home is at least as far back as the hole is
            if ((j - home) & mask) >= ((j - i) & mask):
                self._hashes[i], self._keys[i], self._values[i] = \
                    self._hashes[j], self._keys[j], self._values[j]
                i = j                             # the hole moves to j
        self._keys[i] = _EMPTY
        self._values[i] = None
        self._size -= 1
        return True
```

它也满足要求 2：容量只在 `3 * (键数 + 1) > 2 * capacity` 时翻倍，此时 `capacity < 1.5k`，翻倍后小于
$3k$。怎么选：删除很少或者想要最简单的 `put`，用后移；删除很多且希望 `delete` 本身尽量便宜，用墓碑，
让重建批量清理。

### Part 3

按槽的顺序走一遍，跳过空槽和墓碑。为了发现“遍历时表被改了”，维护一个版本号：键被加入或删除时加一。
迭代器创建时记下版本号，每次被要求取下一个元素时核对。覆盖已有键的值不改版本号，也不会触发重建，
所以是允许的。

```python
class IterableMap(TombstoneMap):
    """Part 3: iteration in slot order; adding or removing a key during iteration is detected."""

    def __init__(self) -> None:
        super().__init__()
        self._version = 0                         # bumped whenever a key is added or removed

    def put(self, key, value) -> None:
        before = self._size
        super().put(key, value)
        if self._size != before:
            self._version += 1

    def delete(self, key) -> bool:
        removed = super().delete(key)
        if removed:
            self._version += 1
        return removed

    def items(self):
        return self._walk(self._version, self._keys, self._values)

    def _walk(self, version: int, keys: list, values: list):
        if self._version != version:
            raise RuntimeError("HashMap changed size during iteration")
        for i in range(len(keys)):
            k = keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield k, values[i]
                if self._version != version:      # checked when the caller asks for the next item
                    raise RuntimeError("HashMap changed size during iteration")

    def keys(self):
        return (k for k, _ in self.items())

    def __iter__(self):
        return self.keys()
```

**顺序和插入顺序的关系：没有关系。** 顺序由哈希值的低位和当前容量决定。例子里 9 先插入，容量为 8 时
它在槽 1，排在最前；扩到 16 之后它在槽 9，排到了最后。所以一次扩容就可能把顺序整个打乱。键是字符串时，
Python 每次启动用的哈希种子不同，同一段程序两次运行的遍历顺序都可能不一样。调用方不能依赖这个顺序。

**代价：** 一次完整遍历是 $O(\text{capacity})$，不是 $O(n)$：空槽和墓碑也要看一眼。按 Part 2 的规则，
容量不超过 $\max(8, 4k)$，其中 $k$ 是历史上同时存在的键数的最大值，而不是当前的键数。先放一百万个键
再删到只剩十个，遍历仍然要扫过整张大表。

### Part 4

把哈希和相等判断收进 `_hash` 和 `_same` 两个方法，前面的代码都只通过它们使用键，所以这里只要换掉
这两个方法。

```python
class HashMap(IterableMap):
    """Part 4: caller-supplied hash and equality."""

    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        self._hash_fn = hash if hash_fn is None else hash_fn
        self._eq_fn = eq_fn
        super().__init__()

    def _hash(self, key) -> int:
        return self._hash_fn(key)

    def _same(self, a, b) -> bool:
        if a is b:
            return True
        return a == b if self._eq_fn is None else self._eq_fn(a, b)
```

“每次操作恰好调用一次 `hash_fn`、扩容时不调用”是 Part 1 就定下的结构带来的：每个公开方法开头算一次
哈希，之后传给 `_find`；重建用的是存下来的哈希值。契约“`eq_fn(a, b)` 为真则 `hash_fn(a) == hash_fn(b)`”
必须由调用方保证：相等的两个键如果哈希不同，会从不同的起始槽出发，互相找不到，表里就会出现“重复”的键。

**键被原地修改。** 表里存的是当初算出的哈希值和这个对象的引用。`list` 被改了之后：用新内容去查，
算出的哈希和存下来的不一样，找不到；用旧内容去查，哈希对得上，但 `==` 不成立，也找不到。这个条目还在
表里，`len` 和遍历都能看到它，只是再也查不到、删不掉。这就是 Python 不让 `list` 作 `dict` 键的原因；
自己提供 `hash_fn` 就要自己保证键放进去之后不变。

**全部撞在一起的键。** 小整数的 `hash` 就是它自己，而起始槽只取哈希值的低位。`i << 40` 的低 40 位全是
0，所以 2000 个键的起始槽全是 0，连成一段：第 $j$ 个键要探测 $j$ 个槽，平均 1000.5 个，整张表退化成
线性查找。修法是先把哈希值的各位搅匀，让高位也影响低位，比如用 splitmix64 的收尾混合函数：

```python
_MASK64 = (1 << 64) - 1


def mix64(key) -> int:
    """hash(key) pushed through the splitmix64 finaliser, so that high input bits reach the low bits."""
    x = hash(key) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)
```

`HashMap(hash_fn=mix64)` 放同样的 2000 个键，平均探测回到 1.5 个左右。注意这只能对付“碰巧有规律”的键。
`mix64` 是公开的固定函数，知道它的人照样能构造出一批全部相撞的键；要防恶意输入，哈希里必须混进一个
对方不知道的随机种子。

### 追问

**问：扩容那一次 `put` 是 $O(n)$，能不能不卡这一下？** 可以做渐进式扩容（incremental resize）：同时保留
旧表和新表，之后每次操作顺手从旧表搬固定个数的槽到新表；查找两张表都要看，插入只进新表；旧表搬空后
丢掉。搬迁速度要保证旧表搬完之前新表不会又满。这样单次操作的搬迁量是常数，但代码复杂得多，查找也慢了
一截。还要说清一点：在 Python 里光是分配新表的 `[_EMPTY] * capacity` 就是 $O(\text{capacity})$，
所以除非新表也分块按需分配，否则单次操作的最坏代价并没有真的降到 $O(1)$。

**问：CPython 的 `dict` 也是开放寻址，它是怎么保持插入顺序的？** 它把数据分成两层：一个紧凑的条目数组，
按插入顺序追加（哈希值、键、值）；一个稀疏的索引表，槽里只存条目数组的下标。探测在索引表上做，
遍历直接走条目数组，所以顺序就是插入顺序。删除时在条目数组里留一个空位，重建时再压紧。另外它的探测
序列不是线性的，而是用哈希值的高位做扰动，减轻下面说的聚集。

**问：线性探测和链地址法（chaining）怎么选？** 线性探测的数据是连续存放的，探测就是顺序读内存，
对缓存友好，也没有每个节点一次的分配；代价是占用率必须明显小于 1，删除麻烦，而且有一次聚集
（primary clustering）：连续的一段越长，新键越容易落进去把它变得更长，所以对哈希函数的质量更敏感。
链地址法删除简单，占用率可以超过 1，哈希差一点时退化得也平缓一些；代价是指针跳转和额外的内存。

**问：把表删空之后，容量会缩回去吗？** 上面的实现里 `delete` 从不重建，所以只删不插时容量不变，
墓碑也一直留着；要等下一次需要新空槽的 `put` 触发重建，才会按存活键数缩小。想让删除也能及时缩容，
可以在 `delete` 里加一条：键数低于容量的 1/8 时重建。缩容和扩容的门槛之间必须留出间隔，
否则在边界上交替插入和删除，每次操作都会触发重建。

**问：为什么容量取 2 的幂？** 这样取模可以写成一次按位与，扩容时也只是多看一位。代价是只用到哈希值
的低位，低位分布不好就会出现 Part 4 里的情况。另一种选择是容量取素数、用 `%`，所有位都会参与，
但取模更慢。用 2 的幂时，把混合做在哈希函数里是更常见的做法。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
import random

# ---- Part 1 examples ----
for cls in (LinearProbingMap, TombstoneMap, ShiftDeleteMap, IterableMap, HashMap):
    m = cls()
    m.put("a", 1); m.put("b", 2); m.put("a", 3)
    assert len(m) == 2 and m.get("a") == 3
    assert m.get("zzz") is None and m.get("zzz", 0) == 0
    assert "b" in m and "zzz" not in m
    m.put(1, "int"); m.put(1.0, "float")
    assert len(m) == 3 and m.get(1) == "float" and m.get(True) == "float"
    # 8 slots hold at most 5 keys; the 6th distinct key doubles the table first
    m = cls()
    for k in range(5):
        m.put(k, k)
    assert m._capacity == 8
    m.put(0, "overwrite")                            # not a new key: no growth
    assert m._capacity == 8
    m.put(5, 5)
    assert m._capacity == 16 and len(m) == 6
    # None is an ordinary key and an ordinary value
    m.put(None, None)
    assert None in m and m.get(None, "default") is None and len(m) == 7

# ---- Part 2 example, and the naive deletion it rules out ----
for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    m = cls()
    for k in (0, 8, 16):
        m.put(k, str(k))
    assert m._keys[:3] == [0, 8, 16]                 # all three start at slot 0
    assert m.delete(8) is True
    assert m.get(16) == "16"
    assert m.delete(8) is False
    assert len(m) == 2
    m.put(8, "again")
    assert m.get(8) == "again" and len(m) == 3


class NaiveDeleteMap(LinearProbingMap):
    """What Part 2 warns against: deletion that just empties the slot."""

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        self._keys[i] = _EMPTY
        self._values[i] = None
        self._size -= 1
        return True


m = NaiveDeleteMap()
for k in (0, 8, 16):
    m.put(k, str(k))
assert m.delete(8)
assert m.get(16) is None                             # lost: the probe stops at the emptied slot 1
m.put(16, "dup")
assert [k for k in m._keys if k is not _EMPTY].count(16) == 2   # and now the key is stored twice


# ---- random operation sequences against dict, invariants checked after every operation ----
def check_invariants(m) -> None:
    cap = m._capacity
    assert cap >= 8 and cap & (cap - 1) == 0
    live = sum(1 for k in m._keys if k is not _EMPTY and k is not _TOMB)
    tombs = sum(1 for k in m._keys if k is _TOMB)
    assert live == m._size == len(m)
    assert tombs == getattr(m, "_tombs", 0)
    assert 3 * (live + tombs) <= 2 * cap             # so at least one slot is always empty
    for i, k in enumerate(m._keys):
        if k is not _EMPTY and k is not _TOMB:
            h = m._hashes[i]
            assert h == m._hash(k)
            j = h & (cap - 1)
            while j != i:                            # no empty slot between home and actual slot
                assert m._keys[j] is not _EMPTY
                j = (j + 1) & (cap - 1)


def run_against_dict(cls, rng, steps, pool, check=True):
    """Returns the number of answers that differ from dict's."""
    m, model, wrong, peak = cls(), {}, 0, 0
    for _ in range(steps):
        op, key = rng.random(), rng.choice(pool)
        if op < 0.45:
            value = rng.randint(0, 999)
            m.put(key, value)
            model[key] = value
        elif op < 0.75:
            wrong += m.delete(key) != (key in model)
            model.pop(key, None)
        else:
            wrong += m.get(key, "missing") != model.get(key, "missing")
            wrong += (key in m) != (key in model)
        wrong += len(m) != len(model)
        peak = max(peak, len(model))
        if check:
            check_invariants(m)
            assert m._capacity <= max(8, 4 * peak)   # requirement 2 of Part 2
    wrong += sum(m.get(k, "missing") != v for k, v in model.items())
    return wrong


def random_pool(rng, size):
    makers = (
        lambda: rng.randint(-50, 50),
        lambda: rng.randint(0, 20) * 64,             # many keys sharing a home slot
        lambda: str(rng.randint(0, 30)),
        lambda: (rng.randint(0, 3), "t"),
        lambda: float(rng.randint(0, 5)),            # equal to some of the ints
        lambda: None,
        lambda: True,
    )
    return [rng.choice(makers)() for _ in range(size)]


rng = random.Random(20261117)
for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    for trial in range(120):
        pool = random_pool(rng, rng.choice([3, 10, 40, 150]))
        assert run_against_dict(cls, rng, 300, pool) == 0, cls.__name__

# the same workload exposes the naive deletion: most runs give wrong answers
broken_runs = sum(
    run_against_dict(NaiveDeleteMap, rng, 300, random_pool(rng, 40), check=False) > 0 for _ in range(50))
assert broken_runs >= 40, broken_runs


# ---- requirement 3: tombstones must count towards the load ----
class TombstoneBlindMap(TombstoneMap):
    """Load rule that counts live keys only."""

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value
            return
        if self._keys[i] is _TOMB:
            self._tombs -= 1
        elif 3 * (self._size + 1) > 2 * self._capacity:
            self._rebuild(2 * self._capacity)
            i = self._free_slot(h)
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1


def churn_until_full(cls, steps):
    """Keep 3 keys alive: insert a fresh key, delete the oldest. Returns the step at which no
    empty slot is left (the next lookup of an absent key would never return), or None."""
    m = cls()
    for k in range(3):
        m.put(k, k)
    for step in range(3, steps):
        m.put(step, step)
        assert m.delete(step - 3)
        if not any(k is _EMPTY for k in m._keys):
            return step
    return None


assert churn_until_full(TombstoneBlindMap, 1000) is not None
assert churn_until_full(TombstoneMap, 1000) is None
assert churn_until_full(ShiftDeleteMap, 1000) is None


# ---- delete-heavy workloads: bounded capacity, bounded rebuild work ----
class Tally:
    """Mixin that records the cost of every rebuild as old capacity + new capacity."""
    rebuild_slots = 0

    def _rebuild(self, capacity: int) -> None:
        type(self).rebuild_slots += self._capacity + capacity
        super()._rebuild(capacity)


for base in (TombstoneMap, ShiftDeleteMap):
    for window in (1, 5, 10, 100, 1000):
        cls = type("Tallied" + base.__name__, (Tally, base), {})
        m = cls()
        new_keys = 0
        for step in range(30 * window + 200):        # a sliding window: `window` to `window + 1` live keys
            m.put(step, step)
            new_keys += 1
            if step >= window:
                assert m.delete(step - window)
            assert m._capacity <= max(8, 4 * (window + 1))
        assert len(m) == window
        assert all(m.get(k) == k for k in range(step - window + 1, step + 1))
        assert step - window not in m
        assert cls.rebuild_slots <= 24 * new_keys, (base.__name__, window, cls.rebuild_slots, new_keys)

# pure growth: total rebuild work is below 9n, as computed in Part 1
for n in (6, 100, 1000, 5461, 5462, 20000):
    cls = type("TalliedGrow", (Tally, LinearProbingMap), {})
    m = cls()
    for k in range(n):
        m.put(k, k)
    assert cls.rebuild_slots < 9 * n, (n, cls.rebuild_slots)
    assert 3 * n <= 2 * m._capacity and (m._capacity == 8 or 3 * n > m._capacity)

# fill a table, delete everything, refill: tombstones are reused or cleaned up, nothing is lost
m = TombstoneMap()
for k in range(1000):
    m.put(k, k)
grown = m._capacity
for k in range(1000):
    assert m.delete(k)
assert len(m) == 0 and m._capacity == grown and m._tombs == 1000   # deletes alone never rebuild
assert all(k not in m for k in range(0, 2000, 7))                  # misses still terminate
m.put("x", 1)
m.put(123456, 2)
for k in range(5000, 5700):
    m.put(k, k)
assert len(m) == 702 and m.get("x") == 1 and m._capacity <= grown
check_invariants(m)


# ---- probe lengths ----
def probe_count(m, key) -> int:
    """Slots that a lookup of key inspects (the walk of _find, written out a second time)."""
    h = m._hash(key)
    mask = m._capacity - 1
    i, count = h & mask, 1
    while True:
        k = m._keys[i]
        if k is _EMPTY:
            return count
        if k is not _TOMB and m._hashes[i] == h and m._same(k, key):
            return count
        i = (i + 1) & mask
        count += 1


def mean_probes(m, keys) -> float:
    return sum(probe_count(m, k) for k in keys) / len(keys)


def knuth_hit(load):
    return 0.5 * (1 + 1 / (1 - load))


def knuth_miss(load):
    return 0.5 * (1 + 1 / (1 - load) ** 2)


assert abs(knuth_hit(2 / 3) - 2.0) < 1e-9 and abs(knuth_miss(2 / 3) - 5.0) < 1e-9
assert abs(knuth_miss(0.9) - 50.5) < 1e-9

def miss_probes_by_home(m) -> list:
    """For every possible home slot, the slots inspected by a lookup that fails from there."""
    cap = m._capacity
    cost, run = [0] * cap, None
    for i in range(2 * cap - 1, -1, -1):             # backwards and twice around, for the wrap-around
        if m._keys[i % cap] is _EMPTY:
            run = 0
        elif run is not None:
            run += 1
        if i < cap:
            cost[i] = run + 1
    return cost


capacity = 65536
full = (2 * capacity) // 3                           # the most keys this capacity ever holds
sample = rng.sample(range(1 << 60), full + 2000)     # random ints: hash(k) == k, uniform low bits
keys, absent = sample[:full], sample[full:]
m = HashMap()
for k in keys[:capacity // 2]:
    m.put(k, None)
assert m._capacity == capacity                       # load exactly 1/2
cost = miss_probes_by_home(m)
hit, miss = mean_probes(m, keys[:capacity // 2]), sum(cost) / capacity
assert abs(hit - knuth_hit(0.5)) < 0.05 and abs(miss - knuth_miss(0.5)) < 0.15, (hit, miss)
for k in keys[capacity // 2:]:
    m.put(k, None)
assert m._capacity == capacity and len(m) == full    # load 2/3: the highest the rule allows
cost = miss_probes_by_home(m)
assert all(probe_count(m, k) == cost[k & (capacity - 1)] for k in absent)
hit, miss = mean_probes(m, keys), sum(cost) / capacity
assert abs(hit - 2.0) < 0.1 and abs(miss - 5.0) < 0.4, (hit, miss)
print(f"capacity {capacity}, load 2/3: mean probes hit {hit:.2f} (theory 2), miss {miss:.2f} (theory 5)")

# keys that all collide under the default hash, and the fix from Part 4
colliding = [i << 40 for i in range(2000)]
assert all(hash(k) == k for k in colliding)
m = HashMap()
for k in colliding:
    m.put(k, None)
assert mean_probes(m, colliding) == 1000.5           # one run: the j-th key costs j probes
m = HashMap(hash_fn=mix64)
for k in colliding:
    m.put(k, None)
assert all(k in m for k in colliding) and len(m) == 2000
fixed = mean_probes(m, colliding)
assert fixed < 1.8, fixed
print(f"keys i << 40: mean hit probes 1000.5 with hash(), {fixed:.2f} with mix64")


# a hash function that maps everything to one value: n distinct keys cost exactly n(n-1)/2 comparisons
class Key:
    eq_calls = 0

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return 42

    def __eq__(self, other):
        Key.eq_calls += 1
        return isinstance(other, Key) and self.name == other.name


for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    n = 300
    keys = [Key(i) for i in range(n)]
    Key.eq_calls = 0
    m = cls()
    for i, k in enumerate(keys):
        m.put(k, i)
    assert Key.eq_calls == n * (n - 1) // 2          # the i-th insert is compared with all i earlier keys
    assert all(m.get(Key(i)) == i for i in range(n)) and Key("nope") not in m
    for i in range(0, n, 2):
        assert m.delete(Key(i))
    assert all((Key(i) in m) == (i % 2 == 1) for i in range(n)) and len(m) == n // 2
    check_invariants(m)


# ---- Part 3 ----
m = HashMap()
for k in (9, 1, 2):
    m.put(k, None)
assert list(m) == [9, 1, 2] and m._keys[1:4] == [9, 1, 2]
for k in (3, 4):
    m.put(k, None)
assert list(m) == [9, 1, 2, 3, 4] and m._capacity == 8
m.put(5, None)
assert list(m) == [1, 2, 3, 4, 5, 9] and m._capacity == 16
assert list(m.keys()) == list(m) and list(m.items()) == [(k, None) for k in list(m)]
assert list(HashMap()) == []

def expect_runtime_error(iterator) -> None:
    try:
        next(iterator)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


for change in (lambda m: m.put("new", 0), lambda m: m.delete(2)):
    for consumed in (0, 1, 5):                       # before the first item, in the middle, after the last
        for view in (iter, HashMap.keys, HashMap.items):
            m = HashMap()
            for k in range(5):
                m.put(k, k)
            it = view(m)
            for _ in range(consumed):
                next(it)
            change(m)
            expect_runtime_error(it)

m = HashMap()
for k in range(5):
    m.put(k, k)
for k, v in m.items():
    m.put(k, v * 10)                                 # overwriting values while iterating is allowed
assert sorted(m.items()) == [(k, k * 10) for k in range(5)]

for trial in range(200):                             # after any history: every key exactly once
    m, model = HashMap(), {}
    pool = random_pool(rng, 30)
    for _ in range(200):
        key = rng.choice(pool)
        if rng.random() < 0.6:
            m.put(key, 1)
            model[key] = 1
        else:
            m.delete(key)
            model.pop(key, None)
    seen = list(m)
    assert len(seen) == len(model) and all(k in model for k in seen)
    assert all(sum(1 for s in seen if s is k or s == k) == 1 for k in seen)


# ---- Part 4 ----
m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
m.put("Key", 1); m.put("KEY", 2)
assert len(m) == 1 and m.get("key") == 2 and list(m) == ["Key"]

m = HashMap(hash_fn=lambda xs: hash(tuple(xs)))
m.put([1, 2], "a")
assert m.get([1, 2]) == "a" and [1, 2] in m and [2, 1] not in m

# overwriting keeps the key object that was stored first
m = HashMap()
m.put(1, "a"); m.put(1.0, "b"); m.put(True, "c")
assert list(m.items()) == [(1, "c")] and type(list(m)[0]) is int

# NaN behaves as in dict: the same object is the same key, another NaN is not
nan = float("nan")
m = HashMap()
m.put(nan, 1)
assert m.get(nan) == 1 and float("nan") not in m and {nan: 1}.get(nan) == 1

# hash_fn is called once per public operation and never during a rebuild
calls = []


def counting_hash(key):
    calls.append(key)
    return hash(key)


m = HashMap(hash_fn=counting_hash)
for k in range(100):                                 # several rebuilds happen on the way
    m.put(k, k)
assert m._capacity == 256 and len(calls) == 100
m.put(7, "x"); m.get(7); m.get(-1); 7 in m; m.delete(7); m.delete(7)
assert len(calls) == 106

# a key mutated after insertion can no longer be found, but is still stored
xs = [1, 2]
m = HashMap(hash_fn=lambda v: hash(tuple(v)))
m.put(xs, "a")
xs.append(3)
assert [1, 2, 3] not in m and xs not in m and [1, 2] not in m
assert len(m) == 1 and list(m) == [xs] and m.delete(xs) is False

# random operations with list keys under a custom hash, against a dict keyed by tuples
for trial in range(60):
    m, model = HashMap(hash_fn=lambda v: hash(tuple(v)) & 7), {}   # only 8 hash values: long runs
    for _ in range(300):
        key = [rng.randint(0, 3) for _ in range(rng.randint(0, 3))]
        op = rng.random()
        if op < 0.45:
            m.put(key, op)
            model[tuple(key)] = op
        elif op < 0.75:
            assert m.delete(key) == (tuple(key) in model)
            model.pop(tuple(key), None)
        else:
            assert m.get(key, "missing") == model.get(tuple(key), "missing")
        check_invariants(m)
    assert sorted(map(tuple, m)) == sorted(model)

print("all checks passed")
```

</details>

</details>
