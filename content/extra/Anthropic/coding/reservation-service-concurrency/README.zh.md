# 带并发约束的预订服务

## 题目

实现一个放在内存里的预订服务（reservation service）。服务管理一组固定的*资源*（resource，比如会议室、GPU），
资源名是字符串，在构造时一次给定，之后不增不减。时间用整数表示（可以理解成“第几分钟”）。一次预订占用某个
资源上的一个**左闭右开**区间 `[start, end)`。

下面的规则对三个部分都适用：

- 区间 `[a, b)` 和 `[c, d)` **重叠**，当且仅当 `a < d` 且 `c < b`。首尾相接不算重叠：`[9, 11)` 和
  `[11, 12)` 可以同时存在。
- 同一个资源上，任意两条有效的预订不能重叠。不同资源互不影响。
- `0 <= start < end <= 10**9`。`start >= end` 时抛 `ValueError`；资源名不存在时抛 `KeyError`。抛异常的调用
  不能改变任何状态。两种错误同时出现时，抛哪一种都可以。
- 预订成功时返回一个预订编号（booking id）。编号从 1 开始，每成功一次加 1；失败的调用不消耗编号。
- `owner` 只是一个标签，不参与任何判断；任何人都可以凭编号取消一条预订。

### Part 1 —— 单线程版本

```py
class ReservationService:
    def __init__(self, resources: list[str]): ...

    def reserve(self, resource: str, start: int, end: int, owner: str) -> int | None:
        """Book [start, end) on `resource`. Returns the new booking id,
        or None if it overlaps a booking that is already there."""

    def cancel(self, booking_id: int) -> bool:
        """True if the booking existed and is now cancelled.
        False if the id was never issued or is already cancelled."""

    def query(self, resource: str, start: int, end: int) -> list[tuple[int, int, int, str]]:
        """The bookings on `resource` that overlap [start, end),
        as (start, end, booking_id, owner), sorted by start."""
```

一个资源上可能有上万条预订，所以 `reserve` 判断“有没有冲突”时不能把这个资源的预订全部扫一遍，要求只做
$O(\log n)$ 次比较（$n$ 是该资源上的有效预订数）。

```text
svc = ReservationService(["room-a", "room-b"])
svc.reserve("room-a", 9, 11, "ann")     -> 1
svc.reserve("room-a", 11, 12, "bob")    -> 2       # [9,11) 和 [11,12) 只是首尾相接
svc.reserve("room-a", 10, 13, "cat")    -> None    # 和 1 号、2 号都重叠
svc.reserve("room-b", 10, 13, "cat")    -> 3       # 另一个资源；上一次失败没有消耗编号
svc.query("room-a", 0, 24)              -> [(9, 11, 1, "ann"), (11, 12, 2, "bob")]
svc.query("room-a", 11, 12)             -> [(11, 12, 2, "bob")]
svc.query("room-a", 12, 24)             -> []
svc.cancel(1)                           -> True
svc.cancel(1)                           -> False   # 已经取消过
svc.cancel(99)                          -> False   # 从未发出过的编号
svc.reserve("room-a", 10, 11, "cat")    -> 4
svc.query("room-a", 0, 24)              -> [(10, 11, 4, "cat"), (11, 12, 2, "bob")]
svc.reserve("room-c", 1, 2, "dan")      -> KeyError
svc.reserve("room-a", 5, 5, "dan")      -> ValueError
```

### Part 2 —— 多个线程同时调用

现在很多线程会同时调用同一个对象的 `reserve`、`cancel`、`query`。要求整个服务的表现，像是这些调用按某个先后
顺序一个一个执行的；并且一个调用返回之后才开始的调用，必须排在它后面，看到它的全部效果（这叫线性一致，
linearizable）。具体来说：

1. **不重复预订。** 任何时刻，同一资源上的有效预订两两不重叠。几个线程同时抢同一个时段，恰好一个拿到编号，
   其余得到 `None`。
2. **编号不重复、不跳号。** 一共成功了 $K$ 次，发出去的编号恰好是 $1..K$。
3. **取消恰好生效一次。** 多个线程同时 `cancel` 同一个编号，恰好一个返回 `True`。如果某个线程的 `cancel`
   因为“别人已经取消了”而返回 `False`，那么它返回之后开始的 `reserve` 必须看到这个时段已经空出来。
4. **查询是快照。** `query` 返回的是某一时刻真实存在过的状态，返回的列表不受之后的操作影响。
5. **不同资源互不等待。** 不能用一把全局锁把整个 `reserve` 包起来：以后临界区（critical section）里可能有
   慢操作，比如写数据库，订 A 会议室的人不应该排在订 B 会议室的人后面。只保护编号计数器这类极短操作的小锁
   不受这条限制。
6. 抛异常时不能留下没释放的锁。

真实面试里没有现成的测试。请自己写一个测试，证明 Part 1 的代码直接拿到多线程里用，确实会把同一时段订给
两个人。要求**每次运行都能复现**，不能靠运气；再用同一个测试说明新版本没有这个问题。

### Part 3 —— 一次预订多个资源

一场会需要一间会议室加两块 GPU，缺一样就开不成。增加：

```py
    def reserve_all(self, requests: list[tuple[str, int, int]], owner: str) -> int | None:
        """Book every (resource, start, end) in `requests`, or none of them.
        One booking id covers all of them."""
```

- **全有或全无。** 只要有一段和已有预订冲突，就返回 `None`，一段也不订。别的线程不能看到“只订了一半”的
  中间状态，也不能被它挡住。
- 同一个请求里可以多次出现同一个资源（比如同一间会议室的上午和下午）。如果请求里的两段自己就互相重叠，
  整个请求失败，返回 `None`。
- `requests` 有 1 到 16 段；为空时抛 `ValueError`。任何一段的资源名或区间不合法，按前面的规则抛异常，
  一段也不订。
- 整个请求只占一个编号。`query` 里它的每一段都带这个编号；`cancel(编号)` 一次取消全部段，仍然恰好生效一次。
- `reserve(r, s, e, owner)` 的行为和 `reserve_all([(r, s, e)], owner)` 完全一样。
- **不能死锁（deadlock）。** 一个线程调用 `reserve_all([("a", …), ("b", …)], …)`，另一个同时调用
  `reserve_all([("b", …), ("a", …)], …)`，两个调用都必须返回。
- Part 2 的要求全部继续成立。其中第 5 条变成：两个请求涉及的资源没有交集时，不能互相等待。

```text
svc = ReservationService(["gpu-0", "gpu-1", "room-a"])
svc.reserve("gpu-1", 5, 8, "ann")                               -> 1
svc.reserve_all([("gpu-0", 0, 10), ("gpu-1", 0, 5)], "bob")     -> 2      # gpu-1 上 [0,5) 与 [5,8) 相接
svc.reserve_all([("room-a", 0, 4), ("gpu-1", 7, 9)], "cat")     -> None   # gpu-1 上与 1 号冲突
svc.query("room-a", 0, 100)                                     -> []     # 所以 room-a 也没有订
svc.reserve_all([("room-a", 0, 4), ("room-a", 3, 6)], "cat")    -> None   # 自己的两段互相重叠
svc.reserve_all([("room-a", 0, 4), ("room-a", 4, 6)], "cat")    -> 3
svc.query("room-a", 0, 100)                                     -> [(0, 4, 3, "cat"), (4, 6, 3, "cat")]
svc.cancel(2)                                                   -> True
svc.query("gpu-0", 0, 100)                                      -> []
svc.query("gpu-1", 0, 100)                                      -> [(5, 8, 1, "ann")]
```

## 参考解答

<details>
<summary>展开参考解答</summary>

三个部分写成三个类：`ReservationService`（Part 1）、`ConcurrentReservationService`（Part 2）、
`AtomicReservationService`（Part 3）。分开写是为了能把 Part 1 的类原样拿到多线程里，演示它怎么出错。面试时
在同一个类上逐步修改即可。

### Part 1

每个资源维护一个按 `start` 排序的列表，元素是 `(start, end, booking_id, owner)`。列表里的区间两两不重叠，
所以按 `start` 排好序的同时，`end` 也是递增的。两个操作都靠这一点：

- **判断 `[start, end)` 是否空闲。** 可能和它重叠的区间必须在 `end` 之前开始，这些区间是列表的一个前缀
  `cal[:i]`，`i` 用二分找到。前缀里最后一个区间的 `end` 最大。它的 `end <= start`，前缀里就没有区间伸进
  `[start, end)`，空闲；否则它自己就和 `[start, end)` 重叠。所以只需要看 `cal[i-1]` 这一个区间。
- **查询与 `[start, end)` 重叠的区间。** “在 `end` 之前开始”是一个前缀，“在 `start` 之后结束”是一个后缀，
  两次二分取交集，得到一段连续的切片。

二分时用了元组比较的一个小技巧：`(end,)` 比任何以 `end` 开头的更长元组都小，所以
`bisect_left(cal, (end,))` 恰好是“第一个 `start >= end` 的位置”。

复杂度：冲突判断 $O(\log n)$ 次比较。插入和删除用二分定位，但 Python 列表要搬动后面的元素，最坏 $O(n)$
（一次内存搬移，$n$ 是几万时仍然很快；想要严格的 $O(\log n)$ 需要平衡树或 `sortedcontainers` 这类结构）。
查询 $O(\log n + k)$，$k$ 是返回的条数。

代码故意把“检查”和“写入”分成两步（`_commit`、`_uncommit`）。Part 2 要保护的正是这两步之间的空隙，测试也
从这里下手。

```python
import bisect
import threading
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Booking:
    id: int
    owner: str
    legs: tuple                  # ((resource, start, end), ...)
    active: bool = True          # guarded by the locks of the resources in legs


def check_interval(start, end):
    if start >= end:
        raise ValueError(f"empty interval [{start}, {end})")


def is_free(cal, start, end):
    """cal: sorted, pairwise disjoint [(start, end, booking_id, owner)]. True iff [start, end) hits none."""
    i = bisect.bisect_left(cal, (end,))          # cal[:i] are exactly the intervals that start before `end`
    return i == 0 or cal[i - 1][1] <= start      # disjoint + sorted: the last of them also ends last


def overlapping(cal, start, end):
    """The intervals of cal that overlap [start, end), as a new list (still sorted by start)."""
    lo = bisect.bisect_right(cal, start, key=lambda item: item[1])   # first interval that ends after `start`
    hi = bisect.bisect_left(cal, (end,))                              # first interval that starts at/after `end`
    return cal[lo:hi]


def remove_leg(cal, start, end, booking_id):
    i = bisect.bisect_left(cal, (start, end, booking_id))
    assert cal[i][:3] == (start, end, booking_id)
    del cal[i]


class ReservationService:
    """Part 1: correct for one caller at a time. No locks."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}     # resource -> sorted [(start, end, booking_id, owner)]
        self._bookings = {}                              # booking_id -> Booking, active ones only
        self._next_id = 1

    def reserve(self, resource, start, end, owner):
        cal = self._cal[resource]                        # KeyError for an unknown resource
        check_interval(start, end)
        if not is_free(cal, start, end):                 # check ...
            return None
        return self._commit(owner, ((resource, start, end),))    # ... then act

    def _commit(self, owner, legs):
        booking = Booking(self._next_id, owner, legs)
        self._next_id += 1
        self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        booking = self._bookings.get(booking_id)
        if booking is None:                              # check ...
            return False
        self._uncommit(booking)                          # ... then act
        return True

    def _uncommit(self, booking):
        del self._bookings[booking.id]
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)

    def query(self, resource, start, end):
        cal = self._cal[resource]
        check_interval(start, end)
        return overlapping(cal, start, end)
```

### Part 2

**为什么 Part 1 的代码在多线程下是错的。** `reserve` 是“先检查，再写入”（check-then-act）。线程 A 检查完
发现空闲，还没写入，线程 B 也来检查，同样发现空闲，于是两人都写入，同一时段订给了两个人。`cancel` 同理：
两个线程都看到“预订还在”，都去删。`self._next_id += 1` 是“读、加、写”三步，也可能两个线程读到同一个值。
CPython 的全局解释器锁（GIL）只保证单条字节码不被打断，不保证一连串操作不被打断，所以帮不上忙。只是这个
空隙通常只有几微秒，随手写的压力测试多半碰不到它，这正是这类 bug 危险的地方。

**哪些东西必须是原子的，用哪把锁保护：**

| 共享状态 | 由谁保护 |
| --- | --- |
| 某个资源的预订列表 `_cal[r]`；“检查空闲 + 插入”整体 | 该资源的锁 `_locks[r]` |
| `Booking.active`；“判断还有效 + 删除”整体 | 这条预订涉及的资源的锁 |
| 编号计数器 `_next_id`、编号索引 `_bookings` | 小锁 `_meta` |

- **每个资源一把锁。** 不同资源的操作拿的是不同的锁，互不等待。所有锁在构造函数里一次建好：如果改成
  “用到时再建”，建锁这一步本身又是一次先检查再写入。
- **`_meta` 永远最后拿。** 只会在已经持有资源锁的情况下去拿 `_meta`，拿着 `_meta` 时绝不再去拿别的锁，
  而且只做几条赋值就放开。加锁顺序固定，就不会死锁。
- **`cancel` 要“先查、再锁、再确认”。** 只有编号时并不知道该锁哪个资源，所以先在 `_meta` 下查到这条预订，
  放开 `_meta`，再去拿资源锁。查和锁之间有空隙，别的线程可能已经把它取消了，所以拿到资源锁之后必须再看
  一眼 `booking.active`。
- **为什么不在 `_meta` 下直接 `pop` 来决定谁取消成功？** 那样“恰好一个 `True`”也成立，但会出现这样的顺序：
  线程 A `pop` 成功，还没来得及从列表里删掉区间；线程 B 的 `cancel` 得到 `False` 并返回；B 接着 `reserve`
  同一时段，却得到 `None`，因为区间还在。B 刚被告知“已经取消了”，时段却订不上，违反了第 3 条。把“是否
  还有效”的判断和区间的删除放进同一把资源锁里，就没有这个问题。
- **`query` 也要拿资源锁**，否则可能在列表被改到一半时做二分。返回的是切片，也就是一份新列表，里面是
  不可变的元组，所以放开锁之后别人怎么改都不影响它。
- 参数检查放在加锁之前；加锁一律用 `with`，抛异常时锁也会放开。

**怎么测。** 竞态的空隙太窄，就在测试里把它撑开。写一个子类，覆盖 `_commit`，在真正写入之前先停一下。停的
方式是 `threading.Barrier(8)`：8 个线程都通过了检查、都停在这里之后，才一起放行。这样不是“有可能”出错，
而是每次必然 8 个线程都拿到编号。对加了锁的版本做同样的事：同一时刻只有一个线程能进入临界区，8 个人永远
凑不齐，第一个线程等到超时后继续，其余 7 个随后看到时段已被占用，得到 `None`。下面的验证代码里就是这样做
的；第 5 条（不同资源互不等待）也用同一个办法测：让 6 个线程各订一个不同的资源，并要求它们同时停在各自的
临界区里。

```python
class ConcurrentReservationService:
    """Part 2: one lock per resource; a small inner lock for the id counter and the id index."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}
        self._locks = {name: threading.Lock() for name in self._cal}    # all created here, never lazily
        self._bookings = {}
        self._next_id = 1
        self._meta = threading.Lock()        # guards _bookings and _next_id; always the innermost lock

    def reserve(self, resource, start, end, owner):
        lock = self._locks[resource]                     # KeyError before any lock is taken
        check_interval(start, end)
        with lock:                                       # check and insert are one atomic step
            if not is_free(self._cal[resource], start, end):
                return None
            return self._commit(owner, ((resource, start, end),))

    def _commit(self, owner, legs):
        """Caller holds the lock of every resource in legs."""
        with self._meta:
            booking = Booking(self._next_id, owner, legs)
            self._next_id += 1
            self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locks[booking.legs[0][0]]:
            if not booking.active:                       # another cancel got in between the lookup and the lock
                return False
            self._uncommit(booking)
        return True

    def _uncommit(self, booking):
        """Caller holds the lock of every resource in booking.legs."""
        booking.active = False
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)
        with self._meta:
            del self._bookings[booking.id]

    def query(self, resource, start, end):
        lock = self._locks[resource]
        check_interval(start, end)
        with lock:
            return overlapping(self._cal[resource], start, end)      # a slice: a private copy of immutable tuples
```

### Part 3

**先想清楚会怎么死锁。** 一个请求涉及几个资源，就要同时持有几把锁，才能保证“检查全部 + 写入全部”是原子的。
如果按请求里写的顺序加锁，线程 1 拿了 `a` 等 `b`，线程 2 拿了 `b` 等 `a`，谁也不会放手。

**办法：所有人按同一个全局顺序加锁。** 这里按资源名排序。这样每个线程等的锁，都排在它已经持有的所有锁
后面。死锁需要一圈线程首尾相等，沿着这一圈走，锁的名字要一路变大最后又回到起点，这不可能。

三个细节：

- **去重。** 同一个资源可以在请求里出现两次，而 `threading.Lock` 不可重入，同一个线程拿第二次会把自己
  卡死。所以是 `sorted(set(...))`。
- **先全部检查，再全部写入。** 拿齐所有锁之后，先确认每一段都空闲，再确认请求内部的各段互不重叠（把各段
  排序后只比较相邻两段），全部通过才开始写。失败时还什么都没写，不需要回滚。另一种写法是边检查边插入、
  失败再撤销；只要全程持有所有锁，别人同样看不到中间状态，但多了一条容易写错的撤销路径。
- **释放。** `try/finally` 里只释放已经拿到的那些锁。释放顺序不影响正确性。

`cancel` 也要拿这条预订涉及的全部资源锁，按同样的顺序。`reserve` 和 Part 2 一样，只涉及一把锁。

一个 $m$ 段的请求：排序 $O(m \log m)$，检查 $O(m \log n)$ 次比较，插入每段最坏 $O(n)$。两个请求涉及的资源
没有交集时，拿的锁完全不同，互不等待。

```python
class AtomicReservationService(ConcurrentReservationService):
    """Part 3: all-or-nothing bookings across several resources, locks taken in one global order."""

    def _lock_order(self, resources):
        return sorted(set(resources))        # set(): Lock is not reentrant; sorted(): the same order for everyone

    @contextmanager
    def _locked(self, resources):
        held = []
        try:
            for name in self._lock_order(resources):
                self._locks[name].acquire()
                held.append(name)
            yield
        finally:
            for name in reversed(held):
                self._locks[name].release()

    def reserve_all(self, requests, owner):
        legs = tuple((resource, start, end) for resource, start, end in requests)
        if not legs:
            raise ValueError("empty request")
        for resource, start, end in legs:                # validate everything before taking any lock
            self._locks[resource]
            check_interval(start, end)
        with self._locked([resource for resource, _, _ in legs]):
            if not self._all_free(legs):
                return None                              # nothing has been written yet: no rollback needed
            return self._commit(owner, legs)

    def _all_free(self, legs):
        if not all(is_free(self._cal[resource], start, end) for resource, start, end in legs):
            return False
        ordered = sorted(legs)                           # legs of one request must not hit each other either
        return all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(ordered, ordered[1:]))

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locked([resource for resource, _, _ in booking.legs]):
            if not booking.active:
                return False
            self._uncommit(booking)
        return True
```

### 追问

**有 GIL，为什么还要加锁？** GIL 让“同一时刻只有一个线程在执行字节码”，但解释器会在字节码之间切换线程
（默认每 5 毫秒要求当前线程让出一次）。检查和写入之间隔着几十条字节码和好几次函数调用，切换落在中间就
出错。`list.append`、`dict.pop` 这类单个操作在 CPython 里碰巧是原子的，但依赖它既不可移植，也经不起代码
改动；在没有 GIL 的自由线程版 Python（3.13 起可选）里，连这点“碰巧”也要重新审视。

**只用一把全局锁行不行？** 行，而且最不容易写错；Part 3 也不用考虑加锁顺序。代价是所有操作排成一队。纯内存
操作的临界区只有几微秒，加上 GIL 本来就让 CPU 计算串行，这时拆成每个资源一把锁几乎换不来吞吐。拆锁真正
有用的场合是临界区里有 I/O（写日志、写数据库），或者换成真正并行的运行时。面试时可以先说清楚这个权衡，
再按题目要求拆。读多写少时还可以让 `query` 不加锁：写入方每次生成一个新的不可变列表再整体替换引用，读的人
拿到哪一版就用哪一版。

**怎么测并发 bug？测试通过说明了什么？** 两类测试配合用。一类是确定性的：在怀疑有空隙的地方插入暂停点，用
`Barrier` 强制出最坏的交错，就像上面那样，每次必现。另一类是压力测试：很多线程做随机操作，结束后检查
不变量（同一资源上无重叠、编号恰好是 $1..K$、每个编号最多被成功取消一次、最终状态恰好等于“发出去的减去
取消掉的”）；还可以记录每次调用的先后，检查两条重叠的预订有没有同时存在过。压力测试时在临界区里加一个很短
的 `sleep`，或者调用 `sys.setswitchinterval(1e-6)`，线程切换会密集得多。但测试通过只说明“试过的这些交错
没出错”。所以还要靠上面那张“每个共享状态由哪把锁保护”的表来论证，测试只是用来抓论证里的漏洞。

**服务跑成多个进程、多台机器怎么办？** 进程内的锁只在一个进程里有效，不变量必须交给大家共用的存储来守。
常见做法：

- *数据库约束。* PostgreSQL 的排斥约束（exclusion constraint）可以直接表达“同一资源的时间段不重叠”：
  `EXCLUDE USING gist (resource WITH =, during WITH &&)`，其中 `during` 是范围类型（如 `tstzrange`，
  默认就是左闭右开），对普通标量列用 `=` 需要 `btree_gist` 扩展。违反约束的插入会报错，应用把它翻译成
  “已被占用”。没有这种约束的数据库，可以把时间切成固定长度的格子，对 `(resource, slot)` 建唯一索引。
- *悲观锁。* 在一个事务里先用 `SELECT … FOR UPDATE` 锁住这个资源在资源表里的那一行，再查冲突，再插入。
  锁资源行最稳妥：在 PostgreSQL 里，只锁查到的已有预订行，挡不住别人插入一条新的。一次订多个资源时按固定
  顺序锁行，道理和 Part 3 相同；数据库发现死锁时会回滚其中一个事务，应用要准备重试。
- *乐观并发（optimistic concurrency）。* 资源行上放一个版本号。读出版本 `v`，查冲突，然后在同一个事务里
  插入预订并执行 `UPDATE … SET version = v + 1 WHERE id = ? AND version = v`。受影响行数是 0，说明有人
  抢先了，回滚后重读重试。冲突少时很快，热门资源上会反复重试。
- 只是“先 `SELECT` 查冲突，再 `INSERT`”，在默认隔离级别下是错的，和 Part 1 的竞态一模一样。PostgreSQL 的
  `SERIALIZABLE` 级别能发现这种冲突，让其中一个事务以序列化失败（SQLSTATE 40001）结束，应用必须重试；
  它的 `REPEATABLE READ`（快照隔离）挡不住这种情况。
- 用 Redis 之类做分布式锁也可以，但这种锁带租约：持有者卡顿一下，锁可能已经过期而它自己不知道。所以写入
  那一端仍然要有数据库约束，或者检查一个只增不减的令牌（fencing token）。

**客户端超时后重试，会不会出问题？** 会。请求其实已经成功，只是回包丢了。客户端重试时发现时段被占用（被它
自己上一次占的），得到 `None`，以为没订上，于是留下一条没人认领的预订。解决办法是幂等键（idempotency
key）：客户端为每个逻辑请求生成一个唯一的键，重试时带同一个键；服务端把“键 → 结果”和预订本身放在同一个
事务里写入，键上有唯一约束。再次见到同一个键时不重新执行，直接返回存下的结果。还要处理三件事：同一个键的
两个请求同时到达（唯一约束让后到的等待或失败，然后读已有结果）；同一个键但参数不同（拒绝）；键保存多久。

**如果要“先占座，N 秒内确认，否则自动释放”呢？** 给预订加状态和过期时间：`hold(...)` 生成一条带
`expires_at` 的临时预订，`confirm(id)` 把它变成正式的。关键是所有判断都在资源锁里做：判断空闲时，把已经
过期的临时预订当作不存在，顺手删掉；`confirm` 在同一把锁里检查“还没过期”再改状态。这样就不会出现
“`confirm` 返回成功，时段却已经给了别人”。过期用单调时钟（`time.monotonic`）判断，并且把时钟做成可以
注入的参数，测试时用假时钟，不用真的 `sleep`。放到数据库里，通常是状态列加 `expires_at`，由清理任务把
过期的记录标成已释放，排斥约束只对未释放的行生效（约束后面加 `WHERE` 条件）。

<details>
<summary>验证代码（可运行）</summary>

```python
import itertools
import random
import threading
import time

HORIZON = 10 ** 9


def expect_raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {exc.__name__}")


def run_threads(targets, timeout=60.0):
    """Start one daemon thread per callable, wait for all; fail loudly instead of hanging."""
    threads = [threading.Thread(target=t, daemon=True) for t in targets]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "threads are stuck"


# ---------- 1. the statement's examples (one caller) ----------
for cls in (ReservationService, ConcurrentReservationService, AtomicReservationService):
    svc = cls(["room-a", "room-b"])
    assert svc.reserve("room-a", 9, 11, "ann") == 1
    assert svc.reserve("room-a", 11, 12, "bob") == 2            # touching is not overlapping
    assert svc.reserve("room-a", 10, 13, "cat") is None
    assert svc.reserve("room-b", 10, 13, "cat") == 3            # the failed call did not use up an id
    assert svc.query("room-a", 0, 24) == [(9, 11, 1, "ann"), (11, 12, 2, "bob")]
    assert svc.query("room-a", 11, 12) == [(11, 12, 2, "bob")]
    assert svc.query("room-a", 12, 24) == []
    assert svc.cancel(1) is True
    assert svc.cancel(1) is False
    assert svc.cancel(99) is False
    assert svc.reserve("room-a", 10, 11, "cat") == 4
    assert svc.query("room-a", 0, 24) == [(10, 11, 4, "cat"), (11, 12, 2, "bob")]
    expect_raises(KeyError, svc.reserve, "room-c", 1, 2, "dan")
    expect_raises(ValueError, svc.reserve, "room-a", 5, 5, "dan")
    expect_raises(ValueError, svc.query, "room-a", 7, 3)
    expect_raises(KeyError, svc.query, "room-c", 1, 2)
    assert svc.reserve("room-a", 12, 13, "dan") == 5            # errors used no id and left no lock held

svc = AtomicReservationService(["gpu-0", "gpu-1", "room-a"])
assert svc.reserve("gpu-1", 5, 8, "ann") == 1
assert svc.reserve_all([("gpu-0", 0, 10), ("gpu-1", 0, 5)], "bob") == 2
assert svc.reserve_all([("room-a", 0, 4), ("gpu-1", 7, 9)], "cat") is None
assert svc.query("room-a", 0, 100) == []                        # all or nothing
assert svc.reserve_all([("room-a", 0, 4), ("room-a", 3, 6)], "cat") is None
assert svc.reserve_all([("room-a", 0, 4), ("room-a", 4, 6)], "cat") == 3
assert svc.query("room-a", 0, 100) == [(0, 4, 3, "cat"), (4, 6, 3, "cat")]
assert svc.cancel(2) is True
assert svc.query("gpu-0", 0, 100) == []
assert svc.query("gpu-1", 0, 100) == [(5, 8, 1, "ann")]
assert svc.cancel(3) is True and svc.query("room-a", 0, 100) == []
expect_raises(ValueError, svc.reserve_all, [], "dan")
expect_raises(KeyError, svc.reserve_all, [("gpu-0", 20, 30), ("nope", 0, 1)], "dan")
expect_raises(ValueError, svc.reserve_all, [("gpu-0", 20, 30), ("gpu-1", 9, 9)], "dan")
assert svc.query("gpu-0", 0, 100) == []                         # the failed calls booked nothing
assert svc.reserve_all([("gpu-0", 20, 30)], "dan") == 4         # ... and left every lock free


# ---------- 2. one caller, random operations, against a model that only scans lists ----------
class ScanModel:
    def __init__(self, resources):
        self.resources = set(resources)
        self.active = {}                       # id -> (owner, legs)
        self.issued = 0

    def reserve_all(self, requests, owner):
        legs = list(requests)
        taken = [leg for _, booked in self.active.values() for leg in booked]
        for i, (r, s, e) in enumerate(legs):
            if any(r == r2 and s < e2 and s2 < e for r2, s2, e2 in taken + legs[:i]):
                return None
        self.issued += 1
        self.active[self.issued] = (owner, legs)
        return self.issued

    def cancel(self, booking_id):
        return self.active.pop(booking_id, None) is not None

    def query(self, resource, start, end):
        return sorted((s, e, bid, owner) for bid, (owner, legs) in self.active.items()
                      for r, s, e in legs if r == resource and s < end and start < e)


def random_leg(rng, names, span=30):
    start = rng.randrange(span)
    return rng.choice(names), start, start + rng.randint(1, 6)


for cls in (ReservationService, ConcurrentReservationService, AtomicReservationService):
    rng = random.Random(2026)
    for _ in range(150):
        names = ["r%d" % i for i in range(rng.randint(1, 3))]
        svc, model = cls(names), ScanModel(names)
        for _ in range(80):
            p = rng.random()
            if p < 0.45:
                leg = random_leg(rng, names)
                assert svc.reserve(*leg, "u") == model.reserve_all([leg], "u")
            elif p < 0.60 and cls is AtomicReservationService:
                legs = [random_leg(rng, names) for _ in range(rng.randint(1, 3))]
                assert svc.reserve_all(legs, "v") == model.reserve_all(legs, "v")
            elif p < 0.80:
                bid = rng.randint(1, model.issued + 2)
                assert svc.cancel(bid) == model.cancel(bid)
            else:
                r, s, e = random_leg(rng, names)
                assert svc.query(r, s, e) == model.query(r, s, e)
        for r in names:
            assert svc.query(r, 0, HORIZON) == model.query(r, 0, HORIZON)
        assert sorted(svc._bookings) == sorted(model.active)


# ---------- 3. the race, made deterministic ----------
def paused(cls, pause):
    """cls with pause() called in the gap between a check and the write that follows it."""
    class Paused(cls):
        def _commit(self, owner, legs):          # reserve: after "is it free?", before the insert
            pause()
            return super()._commit(owner, legs)

        def _uncommit(self, booking):            # cancel: after "is it still there?", before the removal
            pause()
            return super()._uncommit(booking)
    return Paused


def meeting_point(n, wait):
    """pause() that blocks until n threads are in it together; gives up after `wait` seconds.
    Returns (pause, gave_up) where gave_up collects one item per thread that gave up."""
    gate, gave_up = threading.Barrier(n), []

    def pause():
        try:
            gate.wait(timeout=wait)
        except threading.BrokenBarrierError:
            gave_up.append(1)
    return pause, gave_up


def same_slot_race(cls, wait, n=8):
    """n threads book the same slot; each is held after its check until all n have passed theirs."""
    pause, _ = meeting_point(n, wait)
    svc = paused(cls, pause)(["room"])
    results = []                                 # list.append is atomic in CPython; good enough for a harness
    run_threads([lambda i=i: results.append(svc.reserve("room", 10, 20, f"user{i}")) for i in range(n)])
    return results, svc.query("room", 0, HORIZON)


# Part 1 code called from 8 threads. Nothing stops all 8 from passing the check, so the gate fills at once
# (the 30 s is only a safety net) and every caller is told the slot is theirs.
results, cal = same_slot_race(ReservationService, wait=30.0)
assert results.count(None) == 0 and len(cal) == 8
assert all(item[:2] == (10, 20) for item in cal)
print("no locks:", len(cal), "bookings of room [10, 20) at the same time")

# With the lock only one thread can be between check and insert, so the gate can never fill: the first
# thread gives up after 0.3 s, books, and the other seven then see the slot taken.
for cls in (ConcurrentReservationService, AtomicReservationService):
    results, cal = same_slot_race(cls, wait=0.3)
    assert sorted(results, key=str) == [1] + [None] * 7 and len(cal) == 1


# ---------- 4. different resources really do not wait for each other ----------
def together_in_critical_section(cls, wait, n=6):
    """n threads book n different resources; each pauses inside its critical section until all n are inside.
    Returns how many gave up waiting (0 means all n critical sections were open at the same moment)."""
    pause, gave_up = meeting_point(n, wait)
    svc = paused(cls, pause)([f"r{i}" for i in range(n)])
    results = []
    run_threads([lambda i=i: results.append(svc.reserve(f"r{i}", 0, 1, "u")) for i in range(n)])
    assert sorted(results) == list(range(1, n + 1))
    return len(gave_up)


class GlobalLockService(ConcurrentReservationService):
    """Negative control: every resource shares one lock."""
    def __init__(self, resources):
        super().__init__(resources)
        one = threading.Lock()
        self._locks = {name: one for name in self._cal}


assert together_in_critical_section(ConcurrentReservationService, wait=30.0) == 0
assert together_in_critical_section(AtomicReservationService, wait=30.0) == 0
assert together_in_critical_section(GlobalLockService, wait=0.3) == 6      # the test can tell the difference


# ---------- 5. one booking, many cancels: exactly one True ----------
class NoLock:
    def acquire(self):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class UnsynchronisedService(AtomicReservationService):
    """The Part 3 reference with every lock replaced by a no-op: the deliberately broken version."""
    def __init__(self, resources):
        super().__init__(resources)
        self._locks = {name: NoLock() for name in self._cal}
        self._meta = NoLock()


def cancel_race(cls, wait, n=8):
    """n threads cancel the same booking; each is held after its "still there?" check until all n passed."""
    pause, _ = meeting_point(n, wait)
    svc = cls(["a", "b"])                        # book first, without any pause ...
    bid = svc.reserve("a", 0, 5, "u")
    svc.__class__ = paused(cls, pause)           # ... then switch the pauses on for the cancels
    answers = []

    def cancel_once():
        try:
            answers.append(svc.cancel(bid))
        except Exception as exc:
            answers.append(type(exc).__name__)

    run_threads([cancel_once] * n)
    return answers


for cls in (ReservationService, UnsynchronisedService):         # no locks: all 8 pass the check, then all 8 act
    answers = cancel_race(cls, wait=30.0)
    assert answers.count(True) + answers.count(False) < 8 or answers.count(True) > 1, answers
    print("no locks, 8 cancels of one booking:", sorted(answers, key=str))
for cls in (ConcurrentReservationService, AtomicReservationService):
    assert sorted(cancel_race(cls, wait=0.3)) == [False] * 7 + [True]

for cls in (ConcurrentReservationService, AtomicReservationService):       # and without any pause, many times
    for _ in range(30):
        svc = cls(["a", "b"])
        bid = svc.reserve("a", 0, 5, "u")
        if cls is AtomicReservationService:
            bid = svc.reserve_all([("a", 5, 9), ("b", 0, 9)], "u")
        start, answers = threading.Barrier(8), []

        def cancel_once():
            start.wait()
            answers.append(svc.cancel(bid))

        run_threads([cancel_once] * 8)
        assert sorted(answers) == [False] * 7 + [True]
        assert svc.query("b", 0, 99) == [] and bid not in svc._bookings


# ---------- 6. two callers, two resources, opposite order ----------
class ProbeLock:
    """Stands in for a Lock. Gives up after `timeout` seconds instead of waiting forever, and
    calls after_acquire() each time it has been taken."""
    def __init__(self, timeout, after_acquire):
        self._lock, self._timeout, self._after = threading.Lock(), timeout, after_acquire

    def acquire(self):
        if not self._lock.acquire(timeout=self._timeout):
            raise TimeoutError("still waiting for a lock")
        self._after()

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


class RequestOrderService(AtomicReservationService):
    """The tempting version: take the locks in the order the request lists them."""
    def _lock_order(self, resources):
        return list(dict.fromkeys(resources))


def opposite_orders(cls, lock_timeout, gate_timeout):
    """Thread 1 books (a, b), thread 2 books (b, a), different hours. After its first lock each thread
    waits at a gate until the other one also holds a lock (or until gate_timeout)."""
    gate, first = threading.Barrier(2), threading.local()

    def after_acquire():
        if getattr(first, "done", False):
            return
        first.done = True
        try:
            gate.wait(timeout=gate_timeout)
        except threading.BrokenBarrierError:
            pass

    svc = cls(["a", "b"])
    svc._locks = {name: ProbeLock(lock_timeout, after_acquire) for name in svc._locks}
    outcome = []

    def book(legs, owner):
        try:
            outcome.append(svc.reserve_all(legs, owner))
        except TimeoutError:
            outcome.append("deadlock")

    run_threads([lambda: book([("a", 0, 1), ("b", 0, 1)], "t1"), lambda: book([("b", 5, 6), ("a", 5, 6)], "t2")])
    return outcome


# With the gate both threads hold their first lock before either asks for its second one, so neither
# request can be granted until somebody gives up: the timeout below is not a guess about speed.
outcome = opposite_orders(RequestOrderService, lock_timeout=1.0, gate_timeout=10.0)
assert "deadlock" in outcome, outcome
print("request-order locking:", outcome)
# Sorted order: both want "a" first, so they can never each hold one lock. The gate times out, nobody is stuck.
outcome = opposite_orders(AtomicReservationService, lock_timeout=30.0, gate_timeout=0.2)
assert sorted(outcome) == [1, 2], outcome


# ---------- 7. the storm ----------
class Ticker:
    """A logical clock shared by all threads."""
    def __init__(self):
        self._lock, self._now = threading.Lock(), 0

    def tick(self):
        with self._lock:
            self._now += 1
            return self._now


def storm(svc, n_threads=12, n_ops=250, seed=0, span=120):
    """Random reserve / reserve_all / cancel / query from n_threads threads. Returns a dict of problem counts."""
    names = sorted(svc._cal)
    ticker, pool_lock = Ticker(), threading.Lock()
    pool = []                                    # ids anyone may try to cancel
    logs = [dict(issued=[], cancelled=[], errors=[], bad_snapshots=0) for _ in range(n_threads)]

    def worker(k):
        rng, log = random.Random(seed * 1000 + k), logs[k]
        for _ in range(n_ops):
            p = rng.random()
            try:
                if p < 0.40:
                    legs = [random_leg(rng, names, span)]
                    bid = svc.reserve(*legs[0], f"t{k}")
                elif p < 0.60:
                    legs = [random_leg(rng, names, span) for _ in range(rng.randint(2, 3))]
                    bid = svc.reserve_all(legs, f"t{k}")
                elif p < 0.90:
                    with pool_lock:
                        victim = rng.choice(pool) if pool else 0
                    before = ticker.tick()                       # taken before the call starts
                    if svc.cancel(victim):
                        log["cancelled"].append((victim, before))
                    continue
                else:
                    r, s, e = random_leg(rng, names, span)
                    snap = svc.query(r, 0, HORIZON)
                    if any(a[1] > b[0] for a, b in zip(snap, snap[1:])) or snap != sorted(snap):
                        log["bad_snapshots"] += 1
                    continue
                if bid is not None:
                    log["issued"].append((bid, legs, ticker.tick()))     # tick taken after the call returned
                    with pool_lock:
                        pool.append(bid)
            except Exception as exc:                             # only the broken version gets here
                log["errors"].append(repr(exc))

    run_threads([lambda k=k: worker(k) for k in range(n_threads)], timeout=120.0)

    issued = [x for log in logs for x in log["issued"]]
    cancelled = [x for log in logs for x in log["cancelled"]]
    ids = sorted(bid for bid, _, _ in issued)
    cancelled_ids = [bid for bid, _ in cancelled]
    problems = dict(
        errors=sum(len(log["errors"]) for log in logs),
        bad_snapshots=sum(log["bad_snapshots"] for log in logs),
        ids_not_1_to_k=int(ids != list(range(1, len(ids) + 1))),
        cancelled_twice=len(cancelled_ids) - len(set(cancelled_ids)),
    )
    # (a) what is left must be exactly: everything issued and not cancelled
    gone = set(cancelled_ids)
    expected = {name: [] for name in names}
    for bid, legs, _ in issued:
        if bid not in gone:
            for r, s, e in legs:
                expected[r].append((s, e, bid))
    final = {name: [item[:3] for item in svc.query(name, 0, HORIZON)] for name in names}
    problems["final_state_wrong"] = sum(final[name] != sorted(expected[name]) for name in names)
    problems["final_overlaps"] = sum(1 for name in names for a, b in itertools.combinations(final[name], 2)
                                     if a[0] < b[1] and b[0] < a[1])
    problems["index_wrong"] = int(sorted(svc._bookings) != sorted(set(ids) - gone))
    # (b) double bookings that were cancelled again before the end: booking X certainly existed from the
    # tick after its reserve returned until the tick before its successful cancel started. Two bookings
    # that overlap on a resource must never have certainly existed at the same tick.
    end_tick = dict(cancelled)
    alive = [(r, s, e, born, end_tick.get(bid, float("inf"))) for bid, legs, born in issued for r, s, e in legs]
    by_resource = {name: [x for x in alive if x[0] == name] for name in names}
    problems["double_booked_pairs"] = sum(
        1 for name in names for a, b in itertools.combinations(by_resource[name], 2)
        if a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4])
    return problems, len(ids), len(cancelled_ids)


def yield_a_little():
    time.sleep(0.0002)       # lets other threads run in the middle of the critical section


RESOURCES = ["r0", "r1", "r2", "r3"]
for seed in range(3):
    problems, n_issued, n_cancelled = storm(paused(AtomicReservationService, yield_a_little)(RESOURCES), seed=seed)
    assert not any(problems.values()), problems
    assert n_issued > 200 and n_cancelled > 100            # the storm really did book and cancel a lot
    print("storm, reference:", n_issued, "bookings,", n_cancelled, "cancels, no problems")

problems, n_issued, n_cancelled = storm(paused(UnsynchronisedService, yield_a_little)(RESOURCES), seed=0)
assert problems["double_booked_pairs"] > 0, problems       # the same storm catches the version without locks
print("storm, no locks:", problems)
print("all checks passed")
```

</details>

</details>
