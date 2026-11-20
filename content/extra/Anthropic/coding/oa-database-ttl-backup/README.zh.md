# 在线测评：内存数据库（TTL、条件写入、备份与历史读取）

## 题目

按顺序实现下面四关：只有前一关的测试全部通过，下一关的测试才会运行；每一关都在同一个
`InMemoryDatabase` 类上继续加方法，不是另起一个类。

这是“内存数据库”这类题的另一套关卡，和 [在线测评：内存数据库](#/anthropic/coding/oa-in-memory-database)
不是同一套。主要差别：第三关的时间戳是**最后**一个参数，第四关的时间戳是**第一个**参数；时间戳严格递增；
多了条件写入（compare-and-set）和条件删除（compare-and-delete）；`backup` 返回一个计数；第四关多了读取
过去某一时刻的值的 `get_when`。两页的规则不要混用，做这一页时只看这一页的题面。

通用约定：

- 数据库把 `key` 映射到一条*记录*（record），一条记录把 `field`（字段名）映射到 `value`。一个字段都没有的
  记录和不存在的记录没有区别。
- `key`、`field`、`value`、`expected_value`、`new_value`、`prefix` 都是非空字符串，区分大小写，不含 `(`、
  `)`、`,` 这三个字符。
- 所有方法都返回字符串。没有内容可返回时是空字符串 `""`；表示成功与否用小写的 `"true"` / `"false"`，
  不是布尔值。

### Level 1 —— 基本操作与条件写入

```py
class InMemoryDatabase:
    def set(self, key: str, field: str, value: str) -> str:
        """Sets field of record key to value, creating the record or the field if needed, and
        overwriting the old value if there is one. Returns ""."""

    def get(self, key: str, field: str) -> str:
        """Returns the value of the field, or "" if the record or the field does not exist."""

    def delete(self, key: str, field: str) -> str:
        """Removes the field. Returns "true" if it existed, "false" otherwise."""

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        """If the field exists and its value equals expected_value, changes the value to
        new_value and returns "true". Otherwise changes nothing and returns "false"; in
        particular it never creates a field."""

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        """If the field exists and its value equals expected_value, removes the field and
        returns "true". Otherwise changes nothing and returns "false"."""
```

示例：

```text
set("u1", "name", "Ann")                        # -> ""
set("u1", "city", "Rome")                       # -> ""
get("u1", "name")                               # -> "Ann"
get("u1", "age")                                # -> ""        没有这个字段
get("u2", "name")                               # -> ""        没有这条记录
compare_and_set("u1", "name", "Bob", "Cat")     # -> "false"   当前值是 "Ann"，不是 "Bob"
compare_and_set("u1", "name", "Ann", "Cat")     # -> "true"
get("u1", "name")                               # -> "Cat"
compare_and_set("u1", "age", "30", "31")        # -> "false"   字段不存在；条件写入不会创建字段
compare_and_delete("u1", "city", "Paris")       # -> "false"
compare_and_delete("u1", "city", "Rome")        # -> "true"
get("u1", "city")                               # -> ""
set("u1", "name", "Dan")                        # -> ""        覆盖
delete("u1", "name")                            # -> "true"
delete("u1", "name")                            # -> "false"   已经删掉了
delete("u9", "name")                            # -> "false"
```

### Level 2 —— 扫描

```py
class InMemoryDatabase:
    def scan(self, key: str) -> str:
        """Returns all fields of the record as "<field1>(<value1>), <field2>(<value2>), ...",
        sorted by field name ascending (ordinary string comparison, so uppercase letters come
        before lowercase ones) and joined by a comma and one space. Returns "" if the record
        does not exist or has no fields."""

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        """The same as scan, but only the fields whose name starts with prefix."""
```

示例：

```text
set("u1", "name", "Ann"); set("u1", "nick", "an"); set("u1", "age", "30"); set("u1", "Zip", "00100")
scan("u1")                                      # -> "Zip(00100), age(30), name(Ann), nick(an)"   大写字母排在小写前面
scan_by_prefix("u1", "n")                       # -> "name(Ann), nick(an)"
scan_by_prefix("u1", "na")                      # -> "name(Ann)"
scan_by_prefix("u1", "name")                    # -> "name(Ann)"    字段名本身也算以它开头
scan_by_prefix("u1", "z")                       # -> ""             区分大小写：没有以小写 z 开头的字段
scan("u2")                                      # -> ""
delete("u1", "age")                             # -> "true"
scan("u1")                                      # -> "Zip(00100), name(Ann), nick(an)"
```

### Level 3 —— 时间戳与 TTL

前两关的每个方法都多一个带时间戳的版本，名字加 `_at`。`timestamp` 是**最后**一个参数；带 TTL（time to
live，存活时间）的两个方法里，`ttl` 排在 `timestamp` 之后。测试可能用关键字传 `timestamp=`、`ttl=`、
`prefix=`，所以这三个参数名要和下面一致。

```py
class InMemoryDatabase:
    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str: ...
    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str: ...
    def get_at(self, key: str, field: str, timestamp: int) -> str: ...
    def delete_at(self, key: str, field: str, timestamp: int) -> str: ...
    def scan_at(self, key: str, timestamp: int) -> str: ...
    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str: ...
    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str: ...
    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str: ...
    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str: ...
```

返回值的约定和前两关对应的方法相同。规则：

- `timestamp` 和 `ttl` 都是正整数。同一个实例上，带时间戳的调用（包括第四关的方法）的 `timestamp`
  严格递增，不会重复。
- `set_at_with_ttl` 写入的字段在 `[timestamp, timestamp + ttl)` 这段时间内有效：到 `timestamp + ttl` 这一刻
  它已经过期。`set_at` 写入的字段永不过期。
- 覆盖已有字段时，旧的过期时间作废，按这一次写入重新算：用 `set_at` 覆盖，字段变成永不过期；用
  `set_at_with_ttl` 覆盖，从这一次的 `timestamp` 重新计时。
- 对带时间戳的方法来说，过期的字段就是不存在：`get_at` 返回 `""`，`delete_at` 返回 `"false"`，两个扫描
  跳过它，条件写入和条件删除返回 `"false"`。
- 条件写入成功时，写的那一步和对应的 set 完全一样：`compare_and_set_at` 相当于 `set_at`（新值永不过期，
  哪怕旧值带着 TTL），`compare_and_set_at_with_ttl` 相当于 `set_at_with_ttl`。
- 前两关的方法必须继续可用。一个测试里**写**数据只用一套方法：要么全是不带时间戳的，要么全是带时间戳的。
  不带时间戳的读取（`get`、`scan`、`scan_by_prefix`）可以出现在带时间戳的写入之后；这时它们不看过期时间：
  数据库里现有的字段，即没有被覆盖、没有被成功删除、也没有被第四关的 `restore` 换掉的字段，过期了也照常
  返回。

示例：

```text
set_at("u1", "name", "Ann", 1)                                   # -> ""
set_at_with_ttl("u1", "token", "t1", 2, 10)                      # -> ""        有效时间 [2, 12)
get_at("u1", "token", 11)                                        # -> "t1"
get_at("u1", "token", 12)                                        # -> ""        恰好到 12：已经过期
delete_at("u1", "token", 13)                                     # -> "false"   过期的字段当作不存在
set_at_with_ttl("u1", "token", "t2", 14, 5)                      # -> ""        [14, 19)
set_at_with_ttl("u1", "token", "t3", 16, 10)                     # -> ""        覆盖并重新计时：[16, 26)
scan_at("u1", 19)                                                # -> "name(Ann), token(t3)"
compare_and_set_at("u1", "token", "t3", "t4", 20)                # -> "true"    和 set_at 一样写入：t4 永不过期
get_at("u1", "token", 30)                                        # -> "t4"
compare_and_set_at_with_ttl("u1", "name", "Ann", "Bob", 31, 4)   # -> "true"    name 现在的有效时间是 [31, 35)
scan_by_prefix_at("u1", "n", 34)                                 # -> "name(Bob)"
compare_and_delete_at("u1", "name", "Bob", 35)                   # -> "false"   时刻 35 它已经过期
scan_at("u1", 36)                                                # -> "token(t4)"
compare_and_set_at("u1", "name", "Bob", "Cat", 37)               # -> "false"   条件写入不能让过期的字段复活
get("u1", "name")                                                # -> "Bob"     不带时间戳的读取不看过期时间
```

### Level 4 —— 备份、恢复与历史读取

这一关的三个方法里，`timestamp` 是**第一个**参数。它们都属于“带时间戳”的那一套。

```py
class InMemoryDatabase:
    def backup(self, timestamp: int) -> str:
        """Saves the state of the database at timestamp: every field that has not expired,
        and, for each field with a TTL, its remaining lifetime (its expiry time minus
        timestamp). Returns the number of records that have at least one unexpired field at
        timestamp, as a string, e.g. "2"."""

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        """Replaces the whole database with the latest backup whose timestamp is less than or
        equal to timestamp_to_restore (such a backup is guaranteed to exist;
        timestamp_to_restore < timestamp). A restored field that had a remaining lifetime r in
        the backup now expires at timestamp + r; a field that had no TTL still has none.
        Returns ""."""

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        """Returns what get_at(key, field, at_timestamp) would have returned at the moment
        at_timestamp, i.e. the value the field had then, or "" if it did not exist or had
        expired by then. 1 <= at_timestamp <= timestamp."""
```

规则：

- 备份不会被删除，也不受之后操作的影响；同一份备份可以被恢复多次。`restore` 本身不产生新的备份。
- `restore` 之后，数据库里恰好是那份备份里的字段：备份之后才写入的字段消失，备份之后被删除或过期的字段
  回来。
- `get_when` 看的是 `at_timestamp` 那一刻的数据库：只有时间戳不超过 `at_timestamp` 的调用起作用（包括那
  之前的 `restore`），并且按 `at_timestamp` 判断过期。之后发生的写入、删除和 `restore` 都不改变过去：
  `restore` 改变的是从它自己的 `timestamp` 起的状态。
- `at_timestamp` 恰好等于某次调用的时间戳时，那次调用已经生效。

示例：

```text
set_at("a", "x", "1", 1)                        # -> ""
set_at_with_ttl("a", "y", "2", 2, 10)           # -> ""       有效时间 [2, 12)
set_at_with_ttl("b", "z", "3", 3, 4)            # -> ""       有效时间 [3, 7)
backup(5)                                       # -> "2"      a 和 b 都有未过期的字段；此刻 y 还剩 7，z 还剩 2
delete_at("a", "x", 6)                          # -> "true"
backup(8)                                       # -> "1"      b 唯一的字段在 7 过期了；此刻 y 还剩 4
set_at("c", "w", "9", 9)                        # -> ""
restore(20, 7)                                  # -> ""       不超过 7 的最新备份是时刻 5 的那份
scan_at("a", 21)                                # -> "x(1), y(2)"   y 的新过期时刻是 20 + 7 = 27
get_at("b", "z", 22)                            # -> ""       z 的新过期时刻是 20 + 2 = 22
get_at("c", "w", 23)                            # -> ""       c 是备份之后才写的，恢复后不存在
get_at("a", "y", 26)                            # -> "2"
get_at("a", "y", 27)                            # -> ""
backup(28)                                      # -> "1"      只剩 a（字段 x）
get_when(30, "a", "x", 5)                       # -> "1"
get_when(31, "a", "x", 6)                       # -> ""       时刻 6 的删除已经生效
get_when(32, "a", "x", 19)                      # -> ""
get_when(33, "a", "x", 20)                      # -> "1"      时刻 20 的 restore 把它带了回来
get_when(34, "a", "y", 11)                      # -> "2"
get_when(35, "a", "y", 12)                      # -> ""       原来的有效时间到 12 为止
get_when(36, "a", "y", 26)                      # -> "2"      恢复之后的有效时间是 [20, 27)
get_when(37, "b", "z", 21)                      # -> "3"      恢复之后的有效时间是 [20, 22)
get_when(38, "c", "w", 10)                      # -> "9"      时刻 10 它还在；restore 不改写过去
get_when(39, "c", "w", 20)                      # -> ""
get_when(40, "b", "z", 2)                       # -> ""       那时还没写入
restore(41, 8)                                  # -> ""       时刻 8 的备份里只有 a 的 y，当时还剩 4
scan_at("a", 44)                                # -> "y(2)"
scan_at("a", 45)                                # -> ""       41 + 4 = 45
```

## 参考解答

<details>
<summary>展开参考解答</summary>

值得事先确认的三点，因为合理的设计在这三点上可能不同：TTL 出现之后，前两关不带时间戳的方法怎么
处理过期（这里：不看过期时间）；条件写入成功后，旧值的 TTL 是作废还是保留（这里：作废，写的那一步和
对应的 set 完全一样）；`restore` 之后去问 `restore` 之前的时刻，看到的是当时真实的值还是备份里的值
（这里：当时真实的值）。真实测评时这些都以平台给的例子为准；例子里没有的，就选改动最小的那种写法。

### 时间分配

90 分钟四关，分量前轻后重。一个可行的预算，剩下的留作机动：

| 关 | 预算 | 主要工作 |
| --- | --- | --- |
| Level 1 | 10 分钟以内 | 五个方法，各两到四行 |
| Level 2 | 5 到 10 分钟 | 一个带过滤的排序拼接 |
| Level 3 | 25 分钟左右 | 存储里多存一个过期时刻；九个 `_at` 方法；旧方法改成转发 |
| Level 4 | 30 分钟左右 | 备份和恢复；给写入和删除加一份历史记录 |

几条和具体题目无关的做法：

- 每一关动手之前，把这一关的签名和例子从头读到尾，尤其是返回值和参数顺序。这一套里返回的全是字符串
  （`"true"` 不是 `True`，备份计数是 `"2"` 不是 `2`），而且第三关时间戳在最后、第四关时间戳在最前。
- 写完一个方法就跑一次平台的测试，不要攒着。
- 让新方法尽量由旧方法拼出来。条件写入就是“读一次，比较，再调 set”；这样到了第三关，带 TTL 的规则只需
  要在读和写两个地方实现一次，条件写入自动跟着对。
- 后面的关卡事先看不到，所以不要提前设计；但每加一关，先想清楚“改动数据的入口有几个”。这里从第三关起
  所有写入都经过 `_write`，所有删除都经过 `_erase`，第四关的历史记录就只需要在这两处各加一行。
- 最后一关没写完也有分：[面试流程](#/anthropic/process) 里记了候选人以 750/1000、800/1000 进入下一轮的
  例子。第四关先写 `backup` 和 `restore`，`get_when` 放到最后：它要求先改 `_write` 和 `_erase`，改坏了会连累
  前面已经通过的测试。
- 不追求最优复杂度。扫描时现排序、`get_when` 从后往前线性找，写起来最快；题面没有提性能要求。

### Level 1

整个数据库是 `dict[str, dict[str, str]]`：外层按 `key`，内层按字段名。`get` 连用两次 `.get`，一行同时处理
“记录不存在”和“字段不存在”。

条件写入和条件删除都先调 `get`。题目保证值是非空字符串，所以字段不存在时 `get` 返回的 `""` 不会等于任何
`expected_value`——“字段不存在”和“值不相等”用同一个比较就都挡住了，不需要单独判断。

```python
class InMemoryDatabase:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}          # key -> {field: value}

    def set(self, key: str, field: str, value: str) -> str:
        self._data.setdefault(key, {})[field] = value
        return ""

    def get(self, key: str, field: str) -> str:
        return self._data.get(key, {}).get(field, "")

    def delete(self, key: str, field: str) -> str:
        if field not in self._data.get(key, {}):
            return "false"
        del self._data[key][field]
        return "true"

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        if self.get(key, field) != expected_value:        # NOTE: a missing field reads as "", never equal
            return "false"
        self.set(key, field, new_value)
        return "true"

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        if self.get(key, field) != expected_value:
            return "false"
        return self.delete(key, field)
```

五个方法都是 $O(1)$（字符串的哈希和比较按常数算）。删光字段后留下的空记录不用清理：空记录的 `get` 和
`scan` 结果与不存在的记录相同。

### Level 2

`scan` 就是前缀为空串的 `scan_by_prefix`，只写一份。字段名先排序再过滤，拼接用 `", ".join`，这样首尾不会
多出分隔符，空记录自然得到 `""`。

```python
class InMemoryDatabase(InMemoryDatabase):
    def scan(self, key: str) -> str:
        return self.scan_by_prefix(key, "")

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field]})" for field in sorted(record) if field.startswith(prefix))
```

把类重新打开、定义成它自己的子类，只是这一页用来逐关展示代码的写法；考试时直接在原来的类里加方法。
一条有 $m$ 个字段的记录，扫描一次是 $O(m \log m)$。

### Level 3

存储只改一处：内层字典的值从 `value` 变成 `(value, expires_at)`。永不过期的字段把 `expires_at` 记成
`math.inf`，这样“是否有效”永远是同一个比较 `timestamp < expires_at`，不需要为“没有 TTL”写分支；
`timestamp + math.inf` 仍是 `math.inf`，所以 `_write` 里算过期时刻也不用分支。

过期用*惰性*判断（lazy expiration）：不去主动删除过期的字段，只在读的时候比较。有效时间是左闭右开的
`[timestamp, timestamp + ttl)`，所以比较是严格的 `<`：`timestamp` 等于 `expires_at` 时已经过期。

所有规则只落在三处：

- `get_at` 是唯一判断“这个字段此刻在不在”的单点读取；`delete_at`、三个条件方法都先调它，所以“过期等于
  不存在”对它们自动成立。
- `scan_by_prefix_at` 自己过滤一次过期。
- `_write` 是唯一的写入口，每次都整体换掉 `(value, expires_at)`，所以“覆盖时旧的过期时间作废”自动成立。

前两关不带时间戳的方法改成一行转发：相当于在时刻 $0$ 调用对应的 `_at` 方法。真实的时间戳都是正整数，
过期时刻至少是 $2$，所以在时刻 $0$ 看，没有任何字段过期——这正是题目要求的“不看过期时间”；时刻 $0$ 的
`set` 写入的过期时刻是 `math.inf`，即永不过期。

```python
import math


class InMemoryDatabase(InMemoryDatabase):
    # self._data now maps key -> {field: (value, expires_at)}; expires_at is math.inf without a TTL

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        self._data.setdefault(key, {})[field] = (value, timestamp + ttl)

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        del self._data[key][field]

    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str:
        self._write(key, field, value, timestamp)
        return ""

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str:
        self._write(key, field, value, timestamp, ttl)
        return ""

    def get_at(self, key: str, field: str, timestamp: int) -> str:
        value, expires_at = self._data.get(key, {}).get(field, ("", math.inf))
        return value if timestamp < expires_at else ""    # NOTE: strict -- at expires_at the field is already gone

    def delete_at(self, key: str, field: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) == "":
            return "false"
        self._erase(key, field, timestamp)
        return "true"

    def scan_at(self, key: str, timestamp: int) -> str:
        return self.scan_by_prefix_at(key, "", timestamp)

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field][0]})" for field in sorted(record)
                         if field.startswith(prefix) and timestamp < record[field][1])

    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp)     # NOTE: exactly a set_at -- the old TTL is dropped
        return "true"

    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp, ttl)
        return "true"

    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._erase(key, field, timestamp)
        return "true"

    # Levels 1-2: the same operations at time 0, before every real timestamp, when nothing has expired yet
    def set(self, key: str, field: str, value: str) -> str:
        return self.set_at(key, field, value, 0)

    def get(self, key: str, field: str) -> str:
        return self.get_at(key, field, 0)

    def delete(self, key: str, field: str) -> str:
        return self.delete_at(key, field, 0)

    def scan(self, key: str) -> str:
        return self.scan_at(key, 0)

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        return self.scan_by_prefix_at(key, prefix, 0)

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        return self.compare_and_set_at(key, field, expected_value, new_value, 0)

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        return self.compare_and_delete_at(key, field, expected_value, 0)
```

考试时这一关的实际操作是：把第一、二关的七个方法逐个改名加 `_at`、加上 `timestamp` 参数、把读写换成
上面的样子，再补七个一行的转发。`_erase` 的 `timestamp` 参数这一关用不上，是第四关加历史记录时才需要的；
考试时到第四关再加也来得及。

单点读写仍是 $O(1)$，扫描仍是 $O(m \log m)$，其中 $m$ 包含还留在字典里的过期字段。惰性过期的代价是
过期字段一直占着内存，直到被覆盖或被 `restore` 换掉；见“追问”。

### Level 4

**备份。** 遍历全部数据，只留此刻未过期的字段，把每个字段的 `expires_at` 换成*剩余寿命*
`expires_at - timestamp`（永不过期的字段是 `math.inf - timestamp`，仍是 `math.inf`）。只收有至少一个有效
字段的记录，所以要返回的计数就是这份快照的长度。快照是新建的字典，里面的元组不可变，之后的写入不会
影响它。

**恢复。** 从新到旧找第一份时间戳不超过 `timestamp_to_restore` 的备份。然后把现在的字段全部经 `_erase`
删掉，再把快照里的字段逐个经 `_write` 写回去，`ttl` 传剩余寿命——`_write` 算出的过期时刻正好是
`timestamp + 剩余寿命`。不直接把 `self._data` 整体换掉，是因为下面的历史记录要知道每个字段在这一刻
发生了什么。

**历史读取。** 给每个 `(key, field)` 记一份只追加的版本列表：每次写入追加 `(timestamp, value, expires_at)`，
每次删除追加 `(timestamp, "", math.inf)`。`_write` 和 `_erase` 是仅有的两个改动入口，所以只要在这两处各加
一行，包括 `restore` 在内的所有改动都会被记下来。`get_when` 从后往前找第一条时间戳不超过 `at_timestamp`
的版本，那就是当时生效的版本；再用 `at_timestamp` 判断它是否已经过期。删除记成值为 `""` 的版本，读出来
自然是 `""`。

`restore` 会在同一个时间戳给一个字段先记一条删除、再记一条写入；从后往前找，先碰到的是后记的那条写入，
结果正确。

```python
class InMemoryDatabase(InMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self._backups: list = []      # (timestamp, {key: {field: (value, remaining)}}), oldest first
        self._history: dict = {}      # (key, field) -> [(timestamp, value, expires_at)]; value "" = deleted

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        super()._write(key, field, value, timestamp, ttl)
        self._history.setdefault((key, field), []).append((timestamp, value, timestamp + ttl))

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        super()._erase(key, field, timestamp)
        self._history[(key, field)].append((timestamp, "", math.inf))

    def backup(self, timestamp: int) -> str:
        snapshot = {}
        for key, record in self._data.items():
            alive = {field: (value, expires_at - timestamp)
                     for field, (value, expires_at) in record.items() if timestamp < expires_at}
            if alive:                                     # NOTE: a record whose fields have all expired is not counted
                snapshot[key] = alive
        self._backups.append((timestamp, snapshot))
        return str(len(snapshot))

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        snapshot = next(saved for when, saved in reversed(self._backups) if when <= timestamp_to_restore)
        for key, record in self._data.items():
            for field in list(record):                    # list(): _erase removes from record while we loop
                self._erase(key, field, timestamp)
        for key, record in snapshot.items():
            for field, (value, remaining) in record.items():
                self._write(key, field, value, timestamp, remaining)     # NOTE: expires at timestamp + remaining
        return ""

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        for when, value, expires_at in reversed(self._history.get((key, field), [])):
            if when <= at_timestamp:
                return value if at_timestamp < expires_at else ""
        return ""
```

设数据库里现有 $F$ 个字段（含未清理的过期字段）。`backup` 是 $O(F)$，每份备份占 $O(F)$ 空间。`restore`
是 $O(B + F + S)$：$B$ 是备份份数，$S$ 是那份快照里的字段数。`get_when` 是 $O(v)$，$v$ 是这个字段的版本
数。历史记录每次写入或删除增加一条，每次 `restore` 增加 $O(F + S)$ 条。

### 追问

- **`get_when` 能不能更快？** 版本列表按时间戳不减排列，可以二分：Python 3.10 起可以写
  `bisect_right(versions, at_timestamp, key=lambda v: v[0]) - 1`；更早的版本就另存一个只有时间戳的平行列表。
  同一时间戳有两条时 `bisect_right` 取到的也是后一条。这样是 $O(\log v)$。
- **如果规则是“条件写入保留旧值的 TTL”，改哪里？** 只改 `compare_and_set_at` 的写入那一行：先取出旧的
  `expires_at`，写入时传 `ttl = expires_at - timestamp`。因为过期逻辑只在 `get_at` 和 `_write` 两处，别的
  方法都不用动。考试时看例子里有没有“对带 TTL 的字段做条件写入，之后再读”的一行，就能分辨是哪种规则。
- **惰性过期会让过期字段一直占内存，怎么清理？** 加一个按过期时刻排序的最小堆，元素是
  `(expires_at, key, field)`。每次带时间戳的调用开始时，弹出所有到期的元素；弹出时要核对字典里这个字段的
  过期时刻是否仍等于堆里记的值，因为它可能已经被覆盖或删除。这里没有这样做：题目里不带时间戳的读取要
  能看到“过期”的字段，主动清理反而会破坏这条规则。
- **备份很频繁、数据很大时，每次全量复制太贵，怎么办？** 这一页的实现其实已经有了替代品：历史记录。备份
  可以只记下一个时间戳；恢复时对每个字段从历史记录里读出它在备份那一刻的版本，用
  `expires_at - 备份时间戳` 得到剩余寿命，再写回去。代价是恢复变慢，并且要有“这条记录下有哪些字段”的
  索引。
- **`restore` 之后问更早的时刻，为什么不返回备份里的值？** 两种定义都说得通。这里的定义是“当时调用
  `get_at` 会得到什么”，所以历史只增不改，实现上就是只追加的列表。如果题目要的是另一种——恢复之后，
  历史也回到备份时的样子——那就要在备份里把历史记录一起存下来，恢复时整体换回去。
- **如果时间戳只是不减而不是严格递增？** 同一时间戳的多次调用按调用顺序生效，版本列表里时间戳会重复，
  从后往前找仍然取到最后生效的那条。要另外确定的是同一时间戳上备份两次怎么算（后一份覆盖前一份，还是
  都留着），这要看题面。

<details>
<summary>验证代码（可运行）</summary>

```python
import random
from collections import Counter

# The page builds the class level by level; pick the intermediate versions back out so that the code shown
# for Levels 1-2 (later overridden by one-line forwards) is tested as written.
_final, _level3, _level2, _level1 = InMemoryDatabase.__mro__[:4]
assert _level1.get is not _final.get and "compare_and_set" in vars(_level1) and "scan" in vars(_level2)
assert "set_at" in vars(_level3) and "backup" in vars(_final)

# ---- Level 1: the statement's example, on every version of the class ----
for cls in (_level1, _level2, _level3, _final):
    db = cls()
    assert db.set("u1", "name", "Ann") == ""
    assert db.set("u1", "city", "Rome") == ""
    assert db.get("u1", "name") == "Ann"
    assert db.get("u1", "age") == ""
    assert db.get("u2", "name") == ""
    assert db.compare_and_set("u1", "name", "Bob", "Cat") == "false"
    assert db.compare_and_set("u1", "name", "Ann", "Cat") == "true"
    assert db.get("u1", "name") == "Cat"
    assert db.compare_and_set("u1", "age", "30", "31") == "false"
    assert db.compare_and_delete("u1", "city", "Paris") == "false"
    assert db.compare_and_delete("u1", "city", "Rome") == "true"
    assert db.get("u1", "city") == ""
    assert db.set("u1", "name", "Dan") == ""
    assert db.delete("u1", "name") == "true"
    assert db.delete("u1", "name") == "false"
    assert db.delete("u9", "name") == "false"

# ---- Level 2: the statement's example ----
for cls in (_level2, _level3, _final):
    db = cls()
    for field, value in (("name", "Ann"), ("nick", "an"), ("age", "30"), ("Zip", "00100")):
        assert db.set("u1", field, value) == ""
    assert db.scan("u1") == "Zip(00100), age(30), name(Ann), nick(an)"
    assert db.scan_by_prefix("u1", "n") == "name(Ann), nick(an)"
    assert db.scan_by_prefix("u1", "na") == "name(Ann)"
    assert db.scan_by_prefix("u1", "name") == "name(Ann)"
    assert db.scan_by_prefix("u1", "z") == ""
    assert db.scan("u2") == ""
    assert db.delete("u1", "age") == "true"
    assert db.scan("u1") == "Zip(00100), name(Ann), nick(an)"

# ---- Level 3: the statement's example ----
for cls in (_level3, _final):
    db = cls()
    assert db.set_at("u1", "name", "Ann", 1) == ""
    assert db.set_at_with_ttl("u1", "token", "t1", 2, 10) == ""
    assert db.get_at("u1", "token", 11) == "t1"
    assert db.get_at("u1", "token", 12) == ""
    assert db.delete_at("u1", "token", 13) == "false"
    assert db.set_at_with_ttl("u1", "token", "t2", 14, 5) == ""
    assert db.set_at_with_ttl("u1", "token", "t3", 16, 10) == ""
    assert db.scan_at("u1", 19) == "name(Ann), token(t3)"
    assert db.compare_and_set_at("u1", "token", "t3", "t4", 20) == "true"
    assert db.get_at("u1", "token", 30) == "t4"
    assert db.compare_and_set_at_with_ttl("u1", "name", "Ann", "Bob", 31, 4) == "true"
    assert db.scan_by_prefix_at("u1", "n", 34) == "name(Bob)"
    assert db.compare_and_delete_at("u1", "name", "Bob", 35) == "false"
    assert db.scan_at("u1", 36) == "token(t4)"
    assert db.compare_and_set_at("u1", "name", "Bob", "Cat", 37) == "false"
    assert db.get("u1", "name") == "Bob"

# ---- Level 4: the statement's example ----
db = InMemoryDatabase()
assert db.set_at("a", "x", "1", 1) == ""
assert db.set_at_with_ttl("a", "y", "2", 2, 10) == ""
assert db.set_at_with_ttl("b", "z", "3", 3, 4) == ""
assert db.backup(5) == "2"
assert db.delete_at("a", "x", 6) == "true"
assert db.backup(8) == "1"
assert db.set_at("c", "w", "9", 9) == ""
assert db.restore(20, 7) == ""
assert db.scan_at("a", 21) == "x(1), y(2)"
assert db.get_at("b", "z", 22) == ""
assert db.get_at("c", "w", 23) == ""
assert db.get_at("a", "y", 26) == "2"
assert db.get_at("a", "y", 27) == ""
assert db.backup(28) == "1"
assert db.get_when(30, "a", "x", 5) == "1"
assert db.get_when(31, "a", "x", 6) == ""
assert db.get_when(32, "a", "x", 19) == ""
assert db.get_when(33, "a", "x", 20) == "1"
assert db.get_when(34, "a", "y", 11) == "2"
assert db.get_when(35, "a", "y", 12) == ""
assert db.get_when(36, "a", "y", 26) == "2"
assert db.get_when(37, "b", "z", 21) == "3"
assert db.get_when(38, "c", "w", 10) == "9"
assert db.get_when(39, "c", "w", 20) == ""
assert db.get_when(40, "b", "z", 2) == ""
assert db.restore(41, 8) == ""
assert db.scan_at("a", 44) == "y(2)"
assert db.scan_at("a", 45) == ""

# ---- the practice set's own example tables, transcribed. Its Level 3-4 tables write timestamps and ttl
#      as strings ("1", "9"); its tests pass integers, and so do these. ----
for cls in (_level1, _final):                                 # its Level 1 table
    db = cls()
    assert db.set("A", "B", "E") == ""
    assert db.set("A", "C", "F") == ""
    assert db.get("A", "B") == "E"
    assert db.get("A", "D") == ""
    assert db.delete("A", "B") == "true"
    assert db.delete("A", "D") == "false"

for cls in (_level2, _final):                                 # its Level 2 table
    db = cls()
    assert db.set("A", "BC", "E") == ""
    assert db.set("A", "BD", "F") == ""
    assert db.set("A", "C", "G") == ""
    assert db.scan_by_prefix("A", "B") == "BC(E), BD(F)"
    assert db.scan("A") == "BC(E), BD(F), C(G)"
    assert db.scan_by_prefix("B", "B") == ""

db = InMemoryDatabase()                                       # its first Level 3 table
assert db.set_at_with_ttl("A", "BC", "E", 1, 9) == ""
assert db.set_at_with_ttl("A", "BC", "E", 5, 10) == ""
assert db.set_at("A", "BD", "F", 6) == ""
assert db.scan_by_prefix_at("A", "B", 14) == "BC(E), BD(F)"
assert db.scan_by_prefix_at("A", "B", 15) == "BD(F)"

db = InMemoryDatabase()                                       # its second Level 3 table
assert db.set_at("A", "B", "C", 1) == ""
assert db.set_at_with_ttl("X", "Y", "Z", 2, 15) == ""
assert db.get_at("X", "Y", 3) == "Z"
assert db.set_at_with_ttl("A", "D", "E", 4, 10) == ""
assert db.scan_at("A", 13) == "B(C), D(E)"
assert db.scan_at("X", 16) == "Y(Z)"
assert db.scan_at("X", 17) == ""
assert db.delete_at("X", "Y", 20) == "false"

db = InMemoryDatabase()                                       # its Level 4 table
assert db.set_at_with_ttl("A", "B", "C", 1, 10) == ""
assert db.backup(3) == "1"
assert db.set_at("A", "D", "E", 4) == ""
assert db.backup(5) == "1"
assert db.delete_at("A", "B", 8) == "true"
assert db.backup(9) == "1"
assert db.restore(10, 7) == ""                                # the backup at 5; B had 6 left, so it now ends at 16
assert db.set_at("B", "C", "D", 11) == ""
assert db.backup(12) == "2"
assert db.scan_at("A", 15) == "B(C), D(E)"
assert db.scan_at("A", 16) == "D(E)"
assert db.scan_at("B", 17) == "C(D)"

# ---- the practice set's unit tests, transcribed (keyword arguments as in the source) ----
for cls in (_level1, _final):
    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.set("user1", "age", "30") == ""
    assert db.get("user1", "name") == "Alice"
    assert db.get("user1", "age") == "30"

    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.set("user1", "name", "Bob") == ""
    assert db.get("user1", "name") == "Bob"

    db = cls()
    assert db.get("user1", "field") == ""
    assert db.set("user1", "name", "Alice") == ""
    assert db.get("user1", "non_existent") == ""

    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.delete("user1", "name") == "true"
    assert db.get("user1", "name") == ""
    assert db.delete("user1", "name") == "false"
    assert db.delete("non_existent", "field") == "false"

for cls in (_level2, _final):
    db = cls()
    for field, value in (("name", "Alice"), ("age", "30"), ("city", "NY"), ("abc", "123")):
        assert db.set("user1", field, value) == ""
    assert db.scan("user1") == "abc(123), age(30), city(NY), name(Alice)"
    assert db.scan("non_existent") == ""
    assert db.scan_by_prefix("user1", "a") == "abc(123), age(30)"
    assert db.scan_by_prefix("user1", "n") == "name(Alice)"
    assert db.scan_by_prefix("user1", "xyz") == ""

db = InMemoryDatabase()
assert db.set_at("user1", "name", "Alice", timestamp=100) == ""
assert db.set_at("user1", "age", "30", timestamp=101) == ""
assert db.get_at("user1", "name", timestamp=102) == "Alice"
assert db.get_at("user1", "age", timestamp=103) == "30"

db = InMemoryDatabase()
assert db.get_at("user2", "name", timestamp=100) == ""
assert db.get_at("user1", "non_existent", timestamp=101) == ""

db = InMemoryDatabase()
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=105) == "Alice"
assert db.get_at("user1", "name", timestamp=110) == ""
assert db.get_at("user1", "name", timestamp=115) == ""

db = InMemoryDatabase()                                       # set_at over a TTL field: no expiry any more
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.set_at("user1", "name", "Bob", timestamp=105) == ""
assert db.get_at("user1", "name", timestamp=110) == "Bob"
assert db.get_at("user1", "name", timestamp=140) == "Bob"

db = InMemoryDatabase()                                       # set_at_with_ttl over a TTL field: [106, 116)
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=105) == "Alice"
assert db.set_at_with_ttl("user1", "name", "Bob", timestamp=106, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=110) == "Bob"
assert db.get_at("user1", "name", timestamp=117) == ""


def _user1(cls, with_nationality=False):
    db = cls()
    assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""     # [100, 110)
    assert db.set_at_with_ttl("user1", "age", "30", timestamp=101, ttl=5) == ""          # [101, 106)
    assert db.set_at_with_ttl("user1", "city", "NY", timestamp=102, ttl=15) == ""        # [102, 117)
    if with_nationality:
        assert db.set_at_with_ttl("user1", "nationality", "free_country", timestamp=103, ttl=5) == ""   # [103, 108)
    return db


for cls in (_level3, _final):
    db = _user1(cls)
    assert db.scan_at("user1", timestamp=105) == "age(30), city(NY), name(Alice)"
    assert db.scan_at("user1", timestamp=106) == "city(NY), name(Alice)"
    assert db.scan_at("user1", timestamp=110) == "city(NY)"
    assert db.scan_at("user1", timestamp=116) == "city(NY)"
    assert db.scan_at("user1", timestamp=117) == ""

    db = _user1(cls, with_nationality=True)
    assert db.scan_by_prefix_at("user1", prefix="a", timestamp=105) == "age(30)"
    assert db.scan_by_prefix_at("user1", prefix="a", timestamp=106) == ""
    assert db.scan_by_prefix_at("user1", prefix="n", timestamp=107) == "name(Alice), nationality(free_country)"
    assert db.scan_by_prefix_at("user1", prefix="n", timestamp=109) == "name(Alice)"

    # Three source tests call the plain reads right after timestamped writes, although the source's own
    # Level 3 text says a test never mixes the two families. They are what fixes the rule "a plain read
    # does not look at expiry".
    db = _user1(cls)
    assert db.get("user1", "name") == "Alice" and db.get("user1", "age") == "30" and db.get("user1", "city") == "NY"
    db = _user1(cls)
    assert db.scan("user1") == "age(30), city(NY), name(Alice)"
    db = _user1(cls, with_nationality=True)
    assert db.scan_by_prefix("user1", prefix="a") == "age(30)"
    assert db.scan_by_prefix("user1", prefix="n") == "name(Alice), nationality(free_country)"

db = InMemoryDatabase()
assert db.set_at_with_ttl("A", "B", "C", timestamp=1, ttl=10) == ""
assert db.backup(3) == "1"

db = InMemoryDatabase()
db.set_at_with_ttl("A", "B", "C", 1, 10)                      # expires at 11
assert db.backup(12) == "0"

# ---- edge cases beyond the source ----
db = InMemoryDatabase()                                       # every kind of call exactly at the expiry time
db.set_at_with_ttl("k", "f", "v", 10, 5)                      # [10, 15)
assert db.get_at("k", "f", 14) == "v"
db.set_at_with_ttl("k", "g", "v", 15, 1)                      # [15, 16)
assert db.scan_at("k", 16) == ""
db.set_at_with_ttl("k", "f", "v", 20, 5)                      # [20, 25)
assert db.compare_and_set_at("k", "f", "v", "w", 25) == "false"
db.set_at_with_ttl("k", "f", "v", 30, 5)                      # [30, 35)
assert db.compare_and_set_at_with_ttl("k", "f", "v", "w", 35, 9) == "false"
db.set_at_with_ttl("k", "f", "v", 40, 5)                      # [40, 45)
assert db.compare_and_delete_at("k", "f", "v", 45) == "false"
db.set_at_with_ttl("k", "f", "v", 50, 5)                      # [50, 55)
assert db.delete_at("k", "f", 55) == "false"
db.set_at_with_ttl("k", "f", "v", 60, 5)                      # [60, 65)
assert db.backup(65) == "0"
assert db.get_when(66, "k", "f", 64) == "v" and db.get_when(67, "k", "f", 65) == ""
assert [db.get_when(68, "k", "f", t) for t in (9, 10, 14, 15, 19, 20, 24, 25)] == ["", "v", "v", "", "", "v", "v", ""]

db = InMemoryDatabase()                                       # a conditional write resets the lifetime either way
db.set_at_with_ttl("k", "f", "v", 1, 10)                      # [1, 11)
assert db.compare_and_set_at_with_ttl("k", "f", "v", "w", 5, 3) == "true"     # now [5, 8)
assert db.get_at("k", "f", 7) == "w" and db.get_at("k", "f", 8) == ""
db.set_at_with_ttl("k", "f", "v", 9, 2)                       # [9, 11)
assert db.compare_and_set_at("k", "f", "v", "w", 10) == "true"                # permanent
assert db.get_at("k", "f", 1000) == "w"
assert db.compare_and_set_at("k", "f", "w", "w", 1001) == "true"              # same value: still succeeds
assert db.compare_and_delete_at("k", "f", "v", 1002) == "false"
assert db.compare_and_delete_at("k", "f", "w", 1003) == "true"
assert db.compare_and_delete_at("k", "f", "w", 1004) == "false"
assert db.compare_and_set_at("nokey", "f", "w", "x", 1005) == "false" and db.scan_at("nokey", 1006) == ""

db = InMemoryDatabase()                                       # backup counts records, and only live ones
db.set_at("a", "p", "1", 1)
db.set_at_with_ttl("a", "q", "2", 2, 3)                       # [2, 5)
db.set_at_with_ttl("b", "r", "3", 3, 3)                       # [3, 6)
db.set_at("c", "s", "4", 4)
assert db.backup(5) == "3"                                    # a (one live field of two), b, c
assert db.backup(6) == "2"                                    # b has expired
assert db.delete_at("c", "s", 7) == "true"
assert db.backup(8) == "1"                                    # c is an empty record now
assert db.restore(9, 5) == ""                                 # exactly the timestamp of a backup: that backup
assert db.scan_at("a", 10) == "p(1)" and db.get_at("b", "r", 10) == ""     # r had 1 left at 5: [9, 10)
assert db.restore(12, 5) == ""                                # the same backup again: lifetimes restart again
assert db.get_at("b", "r", 12) == "3" and db.get_at("b", "r", 13) == ""
assert db.backup(14) == "2"                                   # a and c
assert db.restore(15, 7) == ""                                # the backup at 6, not the one at 8
assert db.scan_at("c", 16) == "s(4)" and db.get_at("b", "r", 16) == ""
assert db.restore(17, 16) == ""                               # a backup made after an earlier restore
assert db.scan_at("a", 18) == "p(1)" and db.scan_at("c", 18) == "s(4)"
assert [db.get_when(19, "b", "r", t) for t in (2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15)] == \
    ["", "3", "3", "", "", "3", "", "", "3", "", ""]
assert [db.get_when(20, "c", "s", t) for t in (3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17)] == \
    ["", "4", "4", "", "", "4", "4", "4", "4", "4", "4"]

db = InMemoryDatabase()                                       # the plain methods alone, on the finished class
assert db.set("k", "f", "v") == "" and db.compare_and_set("k", "f", "v", "w") == "true"
assert db.scan("k") == "f(w)" and db.compare_and_delete("k", "f", "v") == "false"
assert db.compare_and_delete("k", "f", "w") == "true" and db.scan("k") == "" and db.delete("k", "f") == "false"


# ---- an independent model: a flat list of rows, None for "never expires", and get_when answered by
#      replaying every earlier call into an empty database ----
class _SlowDatabase:
    CHANGES = ("set_at", "set_at_with_ttl", "delete_at", "compare_and_set_at", "compare_and_set_at_with_ttl",
               "compare_and_delete_at", "backup", "restore", "set", "delete", "compare_and_set",
               "compare_and_delete")

    def __init__(self):
        self.rows = []               # [key, field, value, end]; end is None for a field without TTL
        self.saved = []              # (timestamp, [(key, field, value, time left or None)])
        self.journal = []            # (when, name, args) of every call that can change something
        self.seen = Counter()        # how often the awkward situations came up

    def apply(self, when, name, *args):
        if name in self.CHANGES:
            self.journal.append((when, name, args))
        return getattr(self, name)(*args)

    def _row(self, key, field, timestamp):
        """The row of this field if it is alive at timestamp; timestamp None means "do not look at expiry"."""
        for row in self.rows:
            if row[0] == key and row[1] == field:
                self.seen["call exactly at an expiry"] += row[3] == timestamp
                if timestamp is None or row[3] is None or timestamp < row[3]:
                    return row
        return None

    def _store(self, key, field, value, end):
        self.rows = [row for row in self.rows if (row[0], row[1]) != (key, field)]
        self.rows.append([key, field, value, end])
        return ""

    def _scan(self, key, prefix, timestamp):
        found = [(row[1], row[2]) for row in self.rows
                 if row[0] == key and row[1][:len(prefix)] == prefix
                 and (timestamp is None or row[3] is None or timestamp < row[3])]
        self.seen["call exactly at an expiry"] += any(row[0] == key and row[3] == timestamp for row in self.rows)
        return ", ".join("%s(%s)" % pair for pair in sorted(found))

    def _conditional(self, key, field, expected, timestamp, then):
        row = self._row(key, field, timestamp)
        stale = [r for r in self.rows if (r[0], r[1], r[2]) == (key, field, expected)]
        self.seen["conditional call on an expired field with the expected value"] += row is None and bool(stale)
        if row is None or row[2] != expected:
            return "false"
        then(row)
        return "true"

    # timestamped methods
    def set_at(self, key, field, value, timestamp):
        return self._store(key, field, value, None)

    def set_at_with_ttl(self, key, field, value, timestamp, ttl):
        return self._store(key, field, value, timestamp + ttl)

    def get_at(self, key, field, timestamp):
        row = self._row(key, field, timestamp)
        return "" if row is None else row[2]

    def delete_at(self, key, field, timestamp):
        row = self._row(key, field, timestamp)
        if row is None:
            return "false"
        self.rows.remove(row)
        return "true"

    def scan_at(self, key, timestamp):
        return self._scan(key, "", timestamp)

    def scan_by_prefix_at(self, key, prefix, timestamp):
        return self._scan(key, prefix, timestamp)

    def compare_and_set_at(self, key, field, expected, new, timestamp):
        return self._conditional(key, field, expected, timestamp, lambda row: row.__setitem__(slice(2, 4), [new, None]))

    def compare_and_set_at_with_ttl(self, key, field, expected, new, timestamp, ttl):
        return self._conditional(key, field, expected, timestamp,
                                 lambda row: row.__setitem__(slice(2, 4), [new, timestamp + ttl]))

    def compare_and_delete_at(self, key, field, expected, timestamp):
        return self._conditional(key, field, expected, timestamp, self.rows.remove)

    # plain methods: no clock at all
    def set(self, key, field, value):
        return self._store(key, field, value, None)

    def get(self, key, field):
        row = self._row(key, field, None)
        return "" if row is None else row[2]

    def delete(self, key, field):
        row = self._row(key, field, None)
        if row is None:
            return "false"
        self.rows.remove(row)
        return "true"

    def scan(self, key):
        return self._scan(key, "", None)

    def scan_by_prefix(self, key, prefix):
        return self._scan(key, prefix, None)

    def compare_and_set(self, key, field, expected, new):
        return self._conditional(key, field, expected, None, lambda row: row.__setitem__(slice(2, 4), [new, None]))

    def compare_and_delete(self, key, field, expected):
        return self._conditional(key, field, expected, None, self.rows.remove)

    # Level 4
    def backup(self, timestamp):
        alive = [(k, f, v, None if end is None else end - timestamp)
                 for k, f, v, end in self.rows if end is None or timestamp < end]
        self.saved.append((timestamp, alive))
        return str(len({row[0] for row in alive}))

    def restore(self, timestamp, timestamp_to_restore):
        chosen = None
        for when, rows in self.saved:
            if when <= timestamp_to_restore:
                chosen = rows
        self.rows = [[k, f, v, None if left is None else timestamp + left] for k, f, v, left in chosen]
        self.seen["restored field with time left"] += sum(left is not None for _, _, _, left in chosen)
        return ""

    def get_when(self, timestamp, key, field, at_timestamp):
        replica = _SlowDatabase()
        for when, name, args in self.journal:
            if when <= at_timestamp:
                getattr(replica, name)(*args)
        restores = [when for when, name, _ in self.journal if name == "restore"]
        self.seen["get_when for a moment before a later restore"] += any(at_timestamp < when for when in restores)
        self.seen["get_when for a moment after a restore"] += any(when <= at_timestamp for when in restores)
        return replica.get_at(key, field, at_timestamp)


_PLAIN_READS = ["get", "scan", "scan_by_prefix"]
_PLAIN_WRITES = ["set", "set", "delete", "compare_and_set", "compare_and_delete"]
_TIMED = ["set_at", "set_at", "set_at_with_ttl", "set_at_with_ttl", "set_at_with_ttl", "set_at_with_ttl",
          "get_at", "get_at", "delete_at", "scan_at", "scan_by_prefix_at", "compare_and_set_at",
          "compare_and_set_at_with_ttl", "compare_and_delete_at", "backup", "restore", "restore",
          "get_when", "get_when", "get_when"]


def _random_db_run(db_class, seed, mode, steps=110):
    """Drive db_class and the model with the same random calls. Returns the model, or None on a mismatch.
    mode: "plain" (Level 1-2 methods only), "timed" (timestamped methods only), "mixed" (timed + plain reads)."""
    rng = random.Random(seed)
    fast, slow = db_class(), _SlowDatabase()
    keys, fields, values = ["k", "kk", "m"], ["a", "ab", "abc", "b", "ba", "B"], ["x", "y", "z"]
    now, marks = 0, []
    for _ in range(steps):
        key, field, value = rng.choice(keys), rng.choice(fields), rng.choice(values)
        prefix = rng.choice(["a", "ab", "b", "B", "c"])
        if mode == "plain":
            name = rng.choice(_PLAIN_READS + _PLAIN_WRITES)
        else:
            ends = [(row, row[3] + d) for row in slow.rows if row[3] is not None
                    for d in (-1, 0, 0, 1) if row[3] + d > now]
            if ends and rng.random() < 0.4:          # just before, exactly at, or just after some field expires,
                row, now = rng.choice(ends)          # usually calling on that very field
                key, field = (row[0], row[1]) if rng.random() < 0.7 else (key, field)
            else:
                now += rng.randint(1, 3)
            marks.append(now)
            name = rng.choice(_TIMED + (_PLAIN_READS if mode == "mixed" else []))
            if name == "restore" and not slow.saved:
                name = "backup"
        held = [row[2] for row in slow.rows if (row[0], row[1]) == (key, field)]
        expected = held[0] if held and rng.random() < 0.6 else rng.choice(values)
        if name in ("set", "get", "delete"):
            args = (key, field) + ((value,) if name == "set" else ())
        elif name in ("scan", "scan_by_prefix"):
            args = (key,) + ((prefix,) if name == "scan_by_prefix" else ())
        elif name == "compare_and_set":
            args = (key, field, expected, value)
        elif name == "compare_and_delete":
            args = (key, field, expected)
        elif name in ("set_at", "set_at_with_ttl"):
            args = (key, field, value, now) + ((rng.randint(1, 9),) if name == "set_at_with_ttl" else ())
        elif name in ("get_at", "delete_at"):
            args = (key, field, now)
        elif name == "scan_at":
            args = (key, now)
        elif name == "scan_by_prefix_at":
            args = (key, prefix, now)
        elif name in ("compare_and_set_at", "compare_and_set_at_with_ttl"):
            args = (key, field, expected, value, now) + ((rng.randint(1, 9),) if name.endswith("ttl") else ())
        elif name == "compare_and_delete_at":
            args = (key, field, expected, now)
        elif name == "backup":
            args = (now,)
        elif name == "restore":
            args = (now, min(now - 1, rng.choice(slow.saved)[0] + rng.choice([0, 0, 1, 2])))
        else:
            near = [t + d for t in marks for d in (-1, 0, 1) if 1 <= t + d <= now]
            args = (now, key, field, rng.choice(near))
        if getattr(fast, name)(*args) != slow.apply(now, name, *args):
            return None
        marks.extend(row[3] for row in slow.rows if row[3] is not None and name in ("set_at_with_ttl", "restore",
                                                                                    "compare_and_set_at_with_ttl"))
    return slow


for cls in (_level2, _level3, _final):
    assert all(_random_db_run(cls, seed, "plain") is not None for seed in range(100))
assert all(_random_db_run(_final, seed, "mixed") is not None for seed in range(200))
runs = [_random_db_run(_final, seed, "timed") for seed in range(500)]
assert all(run is not None for run in runs)
seen = sum((run.seen for run in runs), Counter())
# the random calls really reach the awkward situations, each of them hundreds of times
for situation in ("call exactly at an expiry", "conditional call on an expired field with the expected value",
                  "restored field with time left", "get_when for a moment before a later restore",
                  "get_when for a moment after a restore"):
    assert seen[situation] > 300, (situation, seen)


# ---- the comparison has teeth: each of these mistakes is caught by the same random calls ----
class _AliveAtExpiry(InMemoryDatabase):          # "<=" instead of "<": a field still readable at its expiry time
    def get_at(self, key, field, timestamp):
        value, expires_at = self._data.get(key, {}).get(field, ("", math.inf))
        return value if timestamp <= expires_at else ""


class _KeepsOldDeadline(InMemoryDatabase):       # restore puts back the original expiry time
    def restore(self, timestamp, timestamp_to_restore):
        made = max(when for when, _ in self._backups if when <= timestamp_to_restore)
        super().restore(timestamp, timestamp_to_restore)
        for key, record in self._data.items():
            for field, (value, expires_at) in list(record.items()):
                self._write(key, field, value, timestamp, expires_at - timestamp - (timestamp - made))
        return ""


class _StrictlyBefore(InMemoryDatabase):         # restore ignores a backup made exactly at timestamp_to_restore
    def restore(self, timestamp, timestamp_to_restore):
        if self._backups[0][0] < timestamp_to_restore:
            timestamp_to_restore -= 1
        return super().restore(timestamp, timestamp_to_restore)


class _CountsDeadRecords(InMemoryDatabase):      # backup counts records whose fields have all expired
    def backup(self, timestamp):
        super().backup(timestamp)
        return str(sum(1 for record in self._data.values() if record))


class _ConditionalKeepsTtl(InMemoryDatabase):    # a conditional write keeps the old expiry time
    def compare_and_set_at(self, key, field, expected_value, new_value, timestamp):
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp, self._data[key][field][1] - timestamp)
        return "true"


class _ConditionalRevives(InMemoryDatabase):     # a conditional write compares the stored value, expired or not
    def compare_and_set_at(self, key, field, expected_value, new_value, timestamp):
        if self._data.get(key, {}).get(field, ("",))[0] != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp)
        return "true"


class _RestoreRewritesPast(InMemoryDatabase):    # restore leaves no trace in the history
    def restore(self, timestamp, timestamp_to_restore):
        lengths = {name: len(versions) for name, versions in self._history.items()}
        super().restore(timestamp, timestamp_to_restore)
        for name, versions in self._history.items():
            del versions[lengths.get(name, 0):]
        return ""


for wrong in (_AliveAtExpiry, _KeepsOldDeadline, _StrictlyBefore, _CountsDeadRecords, _ConditionalKeepsTtl,
              _ConditionalRevives, _RestoreRewritesPast):
    assert any(_random_db_run(wrong, seed, "timed") is None for seed in range(60)), wrong.__name__
```

</details>

</details>
