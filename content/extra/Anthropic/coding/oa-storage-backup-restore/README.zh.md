# 在线测评：云存储（删除、最大文件、备份与恢复）

## 题目

实现一个放在内存里的云存储系统。系统只记录每个文件的名字和大小，不保存内容。

这道题和站内的 [在线测评：云存储系统](#/anthropic/coding/oa-cloud-storage) 是同一类题，但流传的是另一套
操作：接口是字符串命令，可以删除文件，按前缀取最大的 n 个文件，恢复备份时“能恢复几个算几个”。两页的规则
不一样，不要混用。

所有操作通过一个函数进入：

```py
def solution(queries: list[list[str]]) -> list[str]:
    """Run the queries in order on one storage that starts empty.
    Returns one string per query."""
```

每条命令是一个字符串列表：第一个元素是操作名，后面是参数。**参数全是字符串**，数字写成十进制（比如
`"10"`）。每条命令返回一个字符串：成功与否用 `"true"` / `"false"` 表示，数字转成十进制字符串，“没有结果”
用空字符串 `""` 表示。每个操作具体返回什么，下面逐个说明。

题目分四关，按顺序解锁。每一关都在同一个系统上增加操作，前面各关的操作继续有效，对应的测试也会继续跑。

对四关都适用的约定：

- 文件名、用户名、前缀都是不含空白字符的非空字符串，区分大小写。例子里的文件名长得像路径
  （`/dir/file.txt`），但系统里没有目录，名字只是一个普通字符串。
- 任意时刻，一个名字最多对应一个文件，不管文件属于谁。
- `size`、`capacity`、`n` 都是正整数，不超过 $10^9$。一次调用最多 $10^4$ 条命令。
- 输入保证合法：操作名一定是下面列出的某一个，参数个数正确，该是数字的地方一定是数字。
- 失败的命令不改变任何状态。
- 每一关的例子都从一个空系统开始。

### 第 1 关 —— 添加、查询、删除

- `ADD_FILE <name> <size>`：添加一个文件。已经有同名文件时失败。成功返回 `"true"`，失败返回 `"false"`。
- `GET_FILE_SIZE <name>`：返回文件大小；文件不存在时返回 `""`。
- `DELETE_FILE <name>`：删除文件，返回被删文件的大小；文件不存在时返回 `""`。删除之后，这个名字可以
  再次被使用。

```text
["ADD_FILE", "/dir1/dir2/file.txt", "10"]    -> "true"
["ADD_FILE", "/dir1/dir2/file.txt", "5"]     -> "false"   # 同名文件已存在，大小仍是 10
["GET_FILE_SIZE", "/dir1/dir2/file.txt"]     -> "10"
["DELETE_FILE", "/not-existing.file"]        -> ""
["DELETE_FILE", "/dir1/dir2/file.txt"]       -> "10"
["GET_FILE_SIZE", "/not-existing.file"]      -> ""
```

### 第 2 关 —— 某个前缀下最大的 n 个文件

- `GET_N_LARGEST <prefix> <n>`：在名字以 `prefix` 开头的文件里，取最大的 `n` 个，返回
  `"<name1>(<size1>), <name2>(<size2>), ..."`。

规则：

- “以 `prefix` 开头”就是普通的字符串前缀（`str.startswith`），和目录层级无关：前缀 `/dir/file` 能匹配
  `/dir/file2` 和 `/dir/file1.txt`。
- 先按大小从大到小排；大小相同时，按名字的字典序从小到大排（普通的字符串比较）。
- 各项之间用一个逗号加一个空格 `", "` 分隔。
- 匹配的文件不足 `n` 个时，全部返回。一个也没有时返回 `""`。

```text
["ADD_FILE", "/dir/file1.txt", "5"]          -> "true"
["ADD_FILE", "/dir/file2", "20"]             -> "true"
["ADD_FILE", "/dir/deeper/file3.mov", "9"]   -> "true"
["GET_N_LARGEST", "/dir", "2"]               -> "/dir/file2(20), /dir/deeper/file3.mov(9)"
["GET_N_LARGEST", "/dir/file", "3"]          -> "/dir/file2(20), /dir/file1.txt(5)"     # 只有两个匹配
["GET_N_LARGEST", "/another_dir", "2"]       -> ""
["ADD_FILE", "/big_file.mp4", "20"]          -> "true"
["GET_N_LARGEST", "/", "2"]                  -> "/big_file.mp4(20), /dir/file2(20)"     # 大小并列，按名字排
```

### 第 3 关 —— 用户与容量

所有用户共用同一个文件系统（名字仍然全局唯一），但每个用户有一个容量上限（capacity）：这个用户拥有的
文件的大小之和不能超过它。一个用户的**剩余容量** = 容量上限 − 它当前拥有的文件大小之和。

- `ADD_USER <userId> <capacity>`：创建用户。用户已存在时失败。成功返回 `"true"`，失败返回 `"false"`。
- `ADD_FILE_BY <userId> <name> <size>`：和 `ADD_FILE` 一样添加文件，但文件归 `userId` 所有。以下任一情况
  失败：用户不存在；已经有同名文件（不管属于谁）；`size` 大于用户的剩余容量。成功时返回用户添加之后的
  剩余容量，失败返回 `""`。刚好用完容量是允许的，这时返回 `"0"`。
- `MERGE_USER <userId1> <userId2>`：把 `userId2` 并入 `userId1`。`userId2` 的所有文件改归 `userId1`；
  `userId2` 的容量上限加到 `userId1` 的容量上限上（所以两人的剩余容量也相加）；然后 `userId2` 被删除。
  返回合并之后 `userId1` 的剩余容量。两个用户中有一个不存在，或者两个参数相同，则失败，返回 `""`。
  被删除的用户名以后可以再用 `ADD_USER` 创建，得到的是一个全新的空用户。

这一关还改变了前两关操作的含义：

- 第 1 关的 `ADD_FILE` 现在算是由特殊用户 `"admin"` 执行的。`"admin"` 一开始就存在，容量不限。
  输入保证 `"admin"` 不会作为 `userId` 出现在任何带 `userId` 参数的操作里（包括第 4 关的两个操作）。
- `DELETE_FILE` 可以删除任何文件，不管属于谁。删掉某个用户的文件，这个用户的剩余容量相应变大。
- `GET_FILE_SIZE` 和 `GET_N_LARGEST` 照旧能看到所有文件，不管属于谁。

```text
["ADD_USER", "user1", "200"]                       -> "true"
["ADD_USER", "user1", "100"]                       -> "false"   # 已存在，容量仍是 200
["ADD_FILE_BY", "user1", "/dir/file.med", "50"]    -> "150"
["ADD_FILE_BY", "user1", "/big.blob", "140"]       -> "10"
["ADD_FILE_BY", "user1", "/file-small", "20"]      -> ""        # 只剩 10
["ADD_FILE", "/dir/admin_file", "300"]             -> "true"    # admin，容量不限
["ADD_USER", "user2", "110"]                       -> "true"
["ADD_FILE_BY", "user2", "/dir/file.med", "45"]    -> ""        # 同名文件已存在（属于 user1）
["ADD_FILE_BY", "user2", "/new_file", "50"]        -> "60"
["MERGE_USER", "user1", "user2"]                   -> "70"      # 上限 200 + 110 = 310，已用 190 + 50 = 240
```

### 第 4 关 —— 备份与恢复

- `BACKUP_USER <userId>`：记下 `userId` 当前拥有的所有文件的名字和大小。备份存在别处，之后对文件的任何
  操作都不会改动它。每个用户只保留最新的一份备份，新的覆盖旧的。返回备份里的文件个数（可以是 `"0"`）；
  用户不存在时返回 `""`。
- `RESTORE_USER <userId>`：把 `userId` 的文件恢复成最新备份里的样子。返回成功恢复的文件个数；用户不存在
  时返回 `""`。

`RESTORE_USER` 按下面三步执行：

1. 删除 `userId` 当前拥有的全部文件。
2. 对备份里的每一个（名字，大小）：如果现在没有叫这个名字的文件，就按备份里的大小重新创建它，归 `userId`
   所有，算作恢复成功一个。如果这个名字现在已经是别人的文件（别的用户或 `"admin"`），就跳过它，别人的文件
   原样不动。
3. 返回第 2 步里创建的文件个数。

补充规则：

- 用户从来没有备份过时，按“备份是空的”处理：它的文件全部被删除，返回 `"0"`。
- 恢复不会消耗备份。同一份备份可以恢复很多次。
- 恢复不改变用户的容量上限。恢复之后，剩余容量 = 容量上限 − 恢复出来的文件大小之和。
- 一个文件如果备份时和现在都归 `userId` 所有，它在第 1 步被删掉，在第 2 步按备份里的大小重建，也计入
  返回的个数。
- `MERGE_USER` 不改动 `userId1` 的备份：那份备份里仍然只有 `userId1` 自己备份时拥有的文件。`userId2`
  的备份随 `userId2` 一起删除。

```text
["ADD_USER", "user", "100"]                        -> "true"
["ADD_FILE_BY", "user", "/dir/file1", "50"]        -> "50"
["ADD_FILE_BY", "user", "/file2.txt", "30"]        -> "20"
["RESTORE_USER", "user"]                           -> "0"      # 没有备份：两个文件都被删除
["ADD_FILE_BY", "user", "/file3.mp4", "60"]        -> "40"
["ADD_FILE_BY", "user", "/file4.txt", "10"]        -> "30"
["BACKUP_USER", "user"]                            -> "2"
["DELETE_FILE", "/file3.mp4"]                      -> "60"
["DELETE_FILE", "/file4.txt"]                      -> "10"
["ADD_FILE", "/file3.mp4", "140"]                  -> "true"   # admin 占用了这个名字
["ADD_FILE_BY", "user", "/dir/file5.new", "20"]    -> "80"
["RESTORE_USER", "user"]                           -> "1"      # 删除 file5.new；file4.txt 恢复；file3.mp4 被跳过
```

## 参考解答

<details>
<summary>展开参考解答</summary>

先定两件事，后面四关都省心：

- **字符串只在最外层处理。** 里面的类用正常的 Python 类型：数字是 `int`，失败是 `False` 或 `None`。
  `solution` 负责把参数转成数字，再把结果转成题目要求的字符串。这样每个方法都好写、好测，输出格式也只在
  一个地方出现。
- **文件的增和删各只有一个入口**（`_create`、`_remove`）。第 3 关给文件加上“属于谁”之后，只需要改这两个
  函数，不用到处补记账的代码。

下面用子类一关一关地加功能，方便分段展示。真实测评里直接在同一个类上改就行。

### Level 1

一个字典，名字 → 大小。三个操作都是一次哈希查找，平均 $O(1)$（另加对名字做哈希的时间）。

```python
class StorageL1:
    def __init__(self):
        self.size = {}                       # file name -> size

    def add_file(self, name, size):
        if name in self.size:
            return False
        self._create(name, size)
        return True

    def get_file_size(self, name):
        return self.size.get(name)           # None if there is no such file

    def delete_file(self, name):
        if name not in self.size:
            return None
        return self._remove(name)

    # The only two places that add or remove a file. Later levels extend them.
    def _create(self, name, size):
        self.size[name] = size

    def _remove(self, name):
        return self.size.pop(name)
```

### Level 2

扫一遍所有文件，留下名字以 `prefix` 开头的，排序后取前 `n` 个。排序键是 `(-size, name)`：大小取负数，
于是“大小降序、名字升序”变成一次普通的升序排序。

设文件总数是 $F$，匹配的有 $M$ 个：筛选要做 $F$ 次前缀比较，排序 $O(M \log M)$ 次比较。$10^4$ 条命令
的规模下完全够用。

```python
class StorageL2(StorageL1):
    def get_n_largest(self, prefix, n):
        matches = [(name, size) for name, size in self.size.items() if name.startswith(prefix)]
        matches.sort(key=lambda item: (-item[1], item[0]))       # size descending, then name ascending
        return matches[:n]
```

### Level 3

每个文件多记一个主人；每个用户记三样东西：容量上限 `limit`、已用字节数 `used`、拥有的文件名集合
`names`。`"admin"` 不放进这三张表：它没有上限，也不需要记账，判断时单独绕开。

要维持的不变量：对每个用户，`used` 等于 `names` 里所有文件的大小之和，并且 `used <= limit`。所有增删都
经过 `_create` 和 `_remove`，在这两处更新 `used` 和 `names`，不变量就不会被漏掉。`DELETE_FILE` 调用的
是同一个 `_remove`，所以“删文件会把空间还给主人”不需要另写代码。

`MERGE_USER`：文件改主人，`used` 相加，`limit` 相加。题面里“剩余容量也相加”和“上限相加”是一回事：
剩余 = 上限 − 已用，两项都相加，差也相加。例子里 $10 + 60 = 70$。

复杂度：`ADD_USER`、`ADD_FILE_BY` 平均 $O(1)$；`MERGE_USER` 是 $O(k)$，$k$ 是 `userId2` 拥有的文件数。

```python
ADMIN = "admin"


class StorageL3(StorageL2):
    def __init__(self):
        super().__init__()
        self.owner = {}                      # file name -> user id, or ADMIN
        self.limit = {}                      # user id -> capacity limit    (ADMIN is never a key of these three)
        self.used = {}                       # user id -> total size of the files it owns
        self.names = {}                      # user id -> set of the file names it owns

    def _create(self, name, size, owner=ADMIN):
        super()._create(name, size)
        self.owner[name] = owner
        if owner != ADMIN:
            self.names[owner].add(name)
            self.used[owner] += size

    def _remove(self, name):
        owner = self.owner.pop(name)
        size = super()._remove(name)
        if owner != ADMIN:                   # deleting a file gives its bytes back to the owner
            self.names[owner].remove(name)
            self.used[owner] -= size
        return size

    def remaining(self, user):
        return self.limit[user] - self.used[user]

    def add_user(self, user, capacity):
        if user in self.limit:
            return False
        self.limit[user], self.used[user], self.names[user] = capacity, 0, set()
        return True

    def add_file_by(self, user, name, size):
        if user not in self.limit or name in self.size or size > self.remaining(user):
            return None
        self._create(name, size, user)
        return self.remaining(user)

    def merge_user(self, target, source):
        if target == source or target not in self.limit or source not in self.limit:
            return None
        moved = self.names.pop(source)
        for name in moved:
            self.owner[name] = target
        self.names[target] |= moved
        self.used[target] += self.used.pop(source)
        self.limit[target] += self.limit.pop(source)             # limits add up, so remaining capacities add up
        return self.remaining(target)
```

### Level 4

备份是一个字典：名字 → 大小。它必须是当时状态的**副本**。如果只存了对 `names[user]` 的引用，之后的增删
会把“备份”一起改掉。

恢复照题面的三步写：先删光用户现有的文件，再逐个检查备份里的名字，没被占用的就重建。这里有两个值得
想清楚的点：

- **怎么判断“名字被别人占了”？** 第 1 步结束后，这个用户已经一个文件都没有了。所以第 2 步里只要名字
  还在 `size` 表中，它一定属于别人，不需要再比较主人。
- **恢复会不会超出容量？** 不会，所以代码里不做容量检查，只留一个断言。理由：备份那一刻，备份里的文件
  总大小不超过当时的上限；上限只会因为合并而变大，从不变小；恢复出来的是备份的一个子集，而用户原有的
  文件已经删光。所以恢复之后的已用空间不超过现在的上限。

合并时把 `userId2` 的备份一并删掉。如果漏了这一步，以后用同一个名字新建的用户会“继承”到一份不属于它
的备份。

复杂度：`BACKUP_USER` 是 $O(k)$，$k$ 是用户当前的文件数；`RESTORE_USER` 是 $O(k + b)$，$b$ 是备份里的
文件数。

```python
class StorageL4(StorageL3):
    def __init__(self):
        super().__init__()
        self.backup = {}                     # user id -> {file name: size}, the latest backup only

    def backup_user(self, user):
        if user not in self.limit:
            return None
        self.backup[user] = {name: self.size[name] for name in self.names[user]}    # a copy, not a reference
        return len(self.backup[user])

    def restore_user(self, user):
        if user not in self.limit:
            return None
        for name in list(self.names[user]):                      # 1. drop everything the user owns now
            self._remove(name)
        restored = 0
        for name, size in self.backup.get(user, {}).items():     # 2. bring back what is not taken by someone else
            if name not in self.size:
                self._create(name, size, user)
                restored += 1
        assert self.used[user] <= self.limit[user]               # cannot fail: see the argument in the text
        return restored

    def merge_user(self, target, source):
        remaining = super().merge_user(target, source)
        if remaining is not None:
            self.backup.pop(source, None)                        # the account is gone, and its backup with it
        return remaining
```

最后是字符串这一层。注意 `as_text` 里要先判断 `bool` 再处理数字：在 Python 里 `True` 也是 `int`，顺序
反了的话 `"true"` 会变成 `"True"`。

```python
def as_text(value):
    """The string protocol of the assessment, applied in one place."""
    if value is None:
        return ""
    if isinstance(value, bool):              # must come before int: True is an int in Python
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(f"{name}({size})" for name, size in value)
    return str(value)


def solution(queries):
    storage = StorageL4()
    handlers = {
        "ADD_FILE": lambda name, size: storage.add_file(name, int(size)),
        "GET_FILE_SIZE": storage.get_file_size,
        "DELETE_FILE": storage.delete_file,
        "GET_N_LARGEST": lambda prefix, n: storage.get_n_largest(prefix, int(n)),
        "ADD_USER": lambda user, capacity: storage.add_user(user, int(capacity)),
        "ADD_FILE_BY": lambda user, name, size: storage.add_file_by(user, name, int(size)),
        "MERGE_USER": storage.merge_user,
        "BACKUP_USER": storage.backup_user,
        "RESTORE_USER": storage.restore_user,
    }
    return [as_text(handlers[op](*args)) for op, *args in queries]
```

### 追问

**文件很多、`GET_N_LARGEST` 又调用得很频繁，怎么加速？** 两处可以改。一是找匹配的文件：把所有名字放进
一个有序结构，同一个前缀的名字在字典序里是连续的一段，二分找到这一段的开头，再向后走到第一个不以
`prefix` 开头的名字为止，代价是 $O(\log F + M)$ 次比较，而不是扫全部 $F$ 个。用 Python 列表维护有序
名字时，插入和删除要搬动元素，是 $O(F)$；要都快就得用平衡树，或者用字典树（trie）。二是取前 `n` 个：
`heapq.nsmallest(n, matches, key=...)` 是 $O(M \log n)$，在 $n$ 远小于 $M$ 时比整体排序省。

**如果题目改成“只要有一个文件恢复不了，整个恢复就失败，什么都不改”呢？** 站内另一页云存储就是这种
规则。写法是先检查、后动手：先确认备份里每个名字要么现在没人用，要么是自己的，全部通过才开始删和建。
本页的规则是“跳过恢复不了的”，不需要预先检查，但返回值变成了“恢复了几个”。两种规则的测试用例互不
通用，读题时要看清是哪一种。

**为什么备份只存名字和大小，不存容量上限？** 因为题面规定恢复不改变容量。站内另一页的规则相反：备份
连容量上限一起存，恢复时把上限也改回去。不管是哪种规则，写恢复时都要问一句“恢复之后会不会超出容量”，
并且能说出理由。本页的理由在 Level 4 里：上限只增不减，恢复出来的又只是备份的一个子集。

**合并之后，`userId1` 恢复备份会发生什么？** 从 `userId2` 接过来的文件全部被删。备份里只有 `userId1`
自己当时的文件，而恢复的第 1 步删的是它**现在**拥有的全部文件。这是规则的直接结果，也是最容易写错的
用例之一：有人会在合并时把 `userId2` 的备份并进 `userId1` 的备份，题面明确说了不这样做。

**四关、90 分钟，怎么安排？** 前两关各几分钟，把时间留给后两关。每过一关就把这一关的例子跑一遍；进入
第 3 关时最容易弄坏前面的东西，因为 `ADD_FILE` 和 `DELETE_FILE` 的含义变了，改完要把第 1、2 关的例子
重跑一遍。返回值的格式（`"true"`、`""`、`", "`）集中在一个函数里，改起来不会漏。

<details>
<summary>验证代码（可运行）</summary>

```python
import random

# --- 1. the example tables of the four levels (each starts from an empty storage) ---
LEVEL1 = [
    (["ADD_FILE", "/dir1/dir2/file.txt", "10"], "true"),
    (["ADD_FILE", "/dir1/dir2/file.txt", "5"], "false"),
    (["GET_FILE_SIZE", "/dir1/dir2/file.txt"], "10"),
    (["DELETE_FILE", "/not-existing.file"], ""),
    (["DELETE_FILE", "/dir1/dir2/file.txt"], "10"),
    (["GET_FILE_SIZE", "/not-existing.file"], ""),
]
LEVEL2 = [
    (["ADD_FILE", "/dir/file1.txt", "5"], "true"),
    (["ADD_FILE", "/dir/file2", "20"], "true"),
    (["ADD_FILE", "/dir/deeper/file3.mov", "9"], "true"),
    (["GET_N_LARGEST", "/dir", "2"], "/dir/file2(20), /dir/deeper/file3.mov(9)"),
    (["GET_N_LARGEST", "/dir/file", "3"], "/dir/file2(20), /dir/file1.txt(5)"),
    (["GET_N_LARGEST", "/another_dir", "2"], ""),
    (["ADD_FILE", "/big_file.mp4", "20"], "true"),
    (["GET_N_LARGEST", "/", "2"], "/big_file.mp4(20), /dir/file2(20)"),
]
LEVEL3 = [
    (["ADD_USER", "user1", "200"], "true"),
    (["ADD_USER", "user1", "100"], "false"),
    (["ADD_FILE_BY", "user1", "/dir/file.med", "50"], "150"),
    (["ADD_FILE_BY", "user1", "/big.blob", "140"], "10"),
    (["ADD_FILE_BY", "user1", "/file-small", "20"], ""),
    (["ADD_FILE", "/dir/admin_file", "300"], "true"),
    (["ADD_USER", "user2", "110"], "true"),
    (["ADD_FILE_BY", "user2", "/dir/file.med", "45"], ""),
    (["ADD_FILE_BY", "user2", "/new_file", "50"], "60"),
    (["MERGE_USER", "user1", "user2"], "70"),
]
LEVEL4 = [
    (["ADD_USER", "user", "100"], "true"),
    (["ADD_FILE_BY", "user", "/dir/file1", "50"], "50"),
    (["ADD_FILE_BY", "user", "/file2.txt", "30"], "20"),
    (["RESTORE_USER", "user"], "0"),
    (["ADD_FILE_BY", "user", "/file3.mp4", "60"], "40"),
    (["ADD_FILE_BY", "user", "/file4.txt", "10"], "30"),
    (["BACKUP_USER", "user"], "2"),
    (["DELETE_FILE", "/file3.mp4"], "60"),
    (["DELETE_FILE", "/file4.txt"], "10"),
    (["ADD_FILE", "/file3.mp4", "140"], "true"),
    (["ADD_FILE_BY", "user", "/dir/file5.new", "20"], "80"),
    (["RESTORE_USER", "user"], "1"),
]


def check_table(table, run):
    got = run([query for query, _ in table])
    want = [answer for _, answer in table]
    assert got == want, (got, want)


for table in (LEVEL1, LEVEL2, LEVEL3, LEVEL4):
    check_table(table, solution)

# What the Level 4 table leaves behind: file3 is admin's 140-byte file, file4 is back, file5 is gone.
assert solution([q for q, _ in LEVEL4] + [["GET_N_LARGEST", "/", "9"], ["ADD_FILE_BY", "user", "/x", "90"]])[-2:] == [
    "/file3.mp4(140), /file4.txt(10)", "0"]

# Without the ADD_FILE row nobody has taken "/file3.mp4", and the stated rules restore both files.
without_admin_row = [row for row in LEVEL4 if row[0][0] != "ADD_FILE"]
assert solution([q for q, _ in without_admin_row])[-1] == "2"

# --- 2. edge cases of the stated rules ---
def run(*queries):
    return solution([q.split() for q in queries])


# Level 1-2: the name is free again after a delete; n larger than the number of matches; plain string prefixes
assert run("ADD_FILE /a 5", "DELETE_FILE /a", "ADD_FILE /a 7", "GET_FILE_SIZE /a") == ["true", "5", "true", "7"]
assert run("ADD_FILE /ab 5", "ADD_FILE /a/b 5", "ADD_FILE /b 9", "GET_N_LARGEST /a 10", "GET_N_LARGEST /a 1",
           "GET_N_LARGEST /ab/ 1") == ["true", "true", "true", "/a/b(5), /ab(5)", "/a/b(5)", ""]
assert run("ADD_FILE A 1", "ADD_FILE a 1", "ADD_FILE B 1", "GET_N_LARGEST A 5", "GET_N_LARGEST a 5") == [
    "true", "true", "true", "A(1)", "a(1)"]                      # case sensitive
assert run("ADD_FILE f10 3", "ADD_FILE f9 3", "ADD_FILE f100 3", "GET_N_LARGEST f 3")[-1] == "f10(3), f100(3), f9(3)"

# Level 3: filling the capacity exactly is allowed; unknown users; deleting gives the bytes back to the owner
assert run("ADD_USER u 10", "ADD_FILE_BY u /a 10", "ADD_FILE_BY u /b 1", "ADD_FILE_BY nobody /b 1",
           "DELETE_FILE /a", "ADD_FILE_BY u /b 4", "GET_FILE_SIZE /b") == ["true", "0", "", "", "10", "6", "4"]
# a name is unique across all owners, admin included
assert run("ADD_FILE /a 1", "ADD_USER u 10", "ADD_FILE_BY u /a 1", "ADD_FILE_BY u /b 1", "ADD_FILE /b 1") == [
    "true", "true", "", "9", "false"]
# merge: failures change nothing; limits add up; the merged id is free again and starts empty
assert run("ADD_USER a 10", "ADD_USER b 20", "MERGE_USER a a", "MERGE_USER a c", "MERGE_USER c a",
           "ADD_FILE_BY a /a 7", "ADD_FILE_BY b /b 5", "MERGE_USER a b", "ADD_FILE_BY b /c 1",
           "ADD_USER b 3", "ADD_FILE_BY b /c 3", "DELETE_FILE /b", "ADD_FILE_BY a /d 23", "ADD_FILE_BY a /e 1") == [
    "true", "true", "", "", "", "3", "15", "18", "", "true", "0", "5", "0", ""]

# Level 4
# backup of an unknown user, of no files; a restore does not use the backup up; a newer backup replaces the older
assert run("BACKUP_USER u", "RESTORE_USER u", "ADD_USER u 10", "BACKUP_USER u", "ADD_FILE_BY u /a 4",
           "RESTORE_USER u", "GET_FILE_SIZE /a", "ADD_FILE_BY u /a 4", "BACKUP_USER u", "DELETE_FILE /a",
           "RESTORE_USER u", "DELETE_FILE /a", "RESTORE_USER u", "GET_FILE_SIZE /a") == [
    "", "", "true", "0", "6", "0", "", "6", "1", "4", "1", "4", "1", "4"]
# a file the user owns both at backup time and now is restored with the backed-up size, and it counts
assert run("ADD_USER u 10", "ADD_FILE_BY u /a 4", "BACKUP_USER u", "DELETE_FILE /a", "ADD_FILE_BY u /a 9",
           "RESTORE_USER u", "GET_FILE_SIZE /a", "ADD_FILE_BY u /b 6") == ["true", "6", "1", "4", "1", "1", "4", "0"]
# names taken by another user or by admin are skipped and left alone
assert run("ADD_USER u 10", "ADD_USER v 10", "ADD_FILE_BY u /a 1", "ADD_FILE_BY u /b 2", "ADD_FILE_BY u /c 3",
           "BACKUP_USER u", "DELETE_FILE /a", "DELETE_FILE /b", "ADD_FILE_BY v /a 7", "ADD_FILE /b 99",
           "RESTORE_USER u", "GET_N_LARGEST / 5", "ADD_FILE_BY u /d 7", "ADD_FILE_BY v /e 3") == [
    "true", "true", "9", "7", "4", "3", "1", "2", "3", "true", "1", "/b(99), /a(7), /c(3)", "0", "0"]
# merge keeps the target's backup as it was and throws the source's backup away
assert run("ADD_USER a 10", "ADD_USER b 10", "ADD_FILE_BY a /a 1", "ADD_FILE_BY b /b 2", "BACKUP_USER a",
           "BACKUP_USER b", "MERGE_USER a b", "RESTORE_USER a", "GET_FILE_SIZE /b", "GET_FILE_SIZE /a",
           "ADD_USER b 5", "RESTORE_USER b", "ADD_FILE_BY a /c 19") == [
    "true", "true", "9", "8", "1", "1", "17", "1", "", "1", "true", "0", "0"]
# after a merge the target can own a name that its own backup also lists: its files go first, then the backup returns
assert run("ADD_USER a 10", "ADD_USER b 10", "ADD_FILE_BY a /f 3", "BACKUP_USER a", "DELETE_FILE /f",
           "ADD_FILE_BY b /f 8", "MERGE_USER a b", "RESTORE_USER a", "GET_FILE_SIZE /f") == [
    "true", "true", "7", "1", "3", "2", "12", "1", "3"]


# --- 3. random sequences against an independent, deliberately naive model ---
class SlowStorage:
    """Flat lists, no bookkeeping: every answer is recomputed by scanning."""

    def __init__(self):
        self.rows = []          # [name, size, owner]
        self.users = []         # [user, limit]
        self.snaps = []         # [user, [(name, size), ...]]

    def row(self, name):
        hits = [r for r in self.rows if r[0] == name]
        return hits[0] if hits else None

    def user(self, user):
        hits = [u for u in self.users if u[0] == user]
        return hits[0] if hits else None

    def left(self, user):
        return self.user(user)[1] - sum(r[1] for r in self.rows if r[2] == user)

    def run(self, query):
        op, args = query[0], query[1:]
        if op == "ADD_FILE":
            if self.row(args[0]):
                return "false"
            self.rows.append([args[0], int(args[1]), "admin"])
            return "true"
        if op == "GET_FILE_SIZE":
            r = self.row(args[0])
            return str(r[1]) if r else ""
        if op == "DELETE_FILE":
            r = self.row(args[0])
            if not r:
                return ""
            self.rows.remove(r)
            return str(r[1])
        if op == "GET_N_LARGEST":
            hits = sorted(r[0] for r in self.rows if r[0][:len(args[0])] == args[0])
            hits.sort(key=lambda name: self.row(name)[1], reverse=True)          # stable: names stay ascending
            return ", ".join("%s(%d)" % (name, self.row(name)[1]) for name in hits[:int(args[1])])
        if op == "ADD_USER":
            if self.user(args[0]):
                return "false"
            self.users.append([args[0], int(args[1])])
            return "true"
        if op == "ADD_FILE_BY":
            user, name, size = args[0], args[1], int(args[2])
            if not self.user(user) or self.row(name) or self.left(user) < size:
                return ""
            self.rows.append([name, size, user])
            return str(self.left(user))
        if op == "MERGE_USER":
            a, b = self.user(args[0]), self.user(args[1])
            if not a or not b or a is b:
                return ""
            for r in self.rows:
                if r[2] == b[0]:
                    r[2] = a[0]
            a[1] += b[1]
            self.users.remove(b)
            self.snaps = [s for s in self.snaps if s[0] != b[0]]
            return str(self.left(a[0]))
        if op == "BACKUP_USER":
            if not self.user(args[0]):
                return ""
            self.snaps = [s for s in self.snaps if s[0] != args[0]]
            self.snaps.append([args[0], [(r[0], r[1]) for r in self.rows if r[2] == args[0]]])
            return str(len(self.snaps[-1][1]))
        if op == "RESTORE_USER":
            if not self.user(args[0]):
                return ""
            self.rows = [r for r in self.rows if r[2] != args[0]]
            count = 0
            for snap_user, files in self.snaps:
                if snap_user == args[0]:
                    for name, size in files:
                        if not self.row(name):
                            self.rows.append([name, size, args[0]])
                            count += 1
            assert self.left(args[0]) >= 0
            return str(count)
        raise AssertionError(op)


NAMES = ["/a", "/a/b", "/a/c", "/ab", "/b", "/b/a", "/b/ab", "/c"]
PREFIXES = ["/", "/a", "/a/", "/b", "/b/a", "/c", "/zz"]
USERS = ["u1", "u2", "u3", "u4"]
WEIGHTED_OPS = (["ADD_FILE"] * 3 + ["GET_FILE_SIZE"] + ["DELETE_FILE"] * 5 + ["GET_N_LARGEST"] * 2 + ["ADD_USER"] * 2
                + ["ADD_FILE_BY"] * 8 + ["MERGE_USER"] + ["BACKUP_USER"] * 2 + ["RESTORE_USER"] * 2)


def random_query(rng):
    op = rng.choice(WEIGHTED_OPS)
    if op == "ADD_FILE":
        return [op, rng.choice(NAMES), str(rng.randint(1, 40))]
    if op in ("GET_FILE_SIZE", "DELETE_FILE"):
        return [op, rng.choice(NAMES)]
    if op == "GET_N_LARGEST":
        return [op, rng.choice(PREFIXES), str(rng.randint(1, 5))]
    if op == "ADD_USER":
        return [op, rng.choice(USERS), str(rng.randint(30, 120))]
    if op == "ADD_FILE_BY":
        return [op, rng.choice(USERS), rng.choice(NAMES), str(rng.randint(1, 40))]
    if op == "MERGE_USER":
        return [op, rng.choice(USERS), rng.choice(USERS)]
    return [op, rng.choice(USERS)]


rng = random.Random(2026)
seen = {}                                                    # (operation, kind of answer) -> how often
skipping_restores = 0
for _ in range(2000):
    queries = [random_query(rng) for _ in range(rng.randint(1, 150))]
    slow = SlowStorage()
    want = []
    for query in queries:
        op = query[0]
        saved = [len(s[1]) for s in slow.snaps if s[0] == query[1]] if op == "RESTORE_USER" else []
        answer = slow.run(query)
        want.append(answer)
        kind = "empty" if answer == "" else "false" if answer == "false" else "zero" if answer == "0" else "value"
        seen[op, kind] = seen.get((op, kind), 0) + 1
        if saved and int(answer) < saved[0]:
            skipping_restores += 1                           # a backed-up name was taken by someone else
    got = solution(queries)
    assert got == want, (queries, got, want)

for case in [("ADD_FILE", "value"), ("ADD_FILE", "false"), ("GET_FILE_SIZE", "value"), ("GET_FILE_SIZE", "empty"),
             ("DELETE_FILE", "value"), ("DELETE_FILE", "empty"), ("GET_N_LARGEST", "value"), ("GET_N_LARGEST", "empty"),
             ("ADD_USER", "value"), ("ADD_USER", "false"), ("ADD_FILE_BY", "value"), ("ADD_FILE_BY", "empty"),
             ("ADD_FILE_BY", "zero"), ("MERGE_USER", "value"), ("MERGE_USER", "empty"), ("BACKUP_USER", "value"),
             ("BACKUP_USER", "zero"), ("BACKUP_USER", "empty"), ("RESTORE_USER", "value"), ("RESTORE_USER", "zero"),
             ("RESTORE_USER", "empty")]:
    assert seen.get(case, 0) >= 20, (case, seen.get(case, 0))    # every branch was really exercised
assert skipping_restores >= 50, skipping_restores

# --- 4. the levels are layered: an earlier level's class still passes its own table ---
def run_on(cls):
    def run_table(queries):
        storage, out = cls(), []
        for op, *args in queries:
            method = getattr(storage, op.lower())
            out.append(as_text(method(*[int(a) if a.isdigit() else a for a in args])))
        return out
    return run_table


check_table(LEVEL1, run_on(StorageL1))
check_table(LEVEL2, run_on(StorageL2))
check_table(LEVEL3, run_on(StorageL3))
print("random sequences agree with the naive model;", skipping_restores, "restores had to skip a taken name")
print("all checks passed")
```

</details>

</details>
