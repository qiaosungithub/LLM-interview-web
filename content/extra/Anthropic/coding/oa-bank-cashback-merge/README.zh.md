# 在线测评：银行系统（即时转账、返现与合并）

## 题目

按顺序实现下面四关：只有前一关的测试全部通过，下一关的测试才会运行；每一关都在同一个 `Bank` 类上继续加
方法，不是另起一个类。

这是“银行系统”这类题的另一套关卡，和 [在线测评：银行系统](#/anthropic/coding/oa-bank-system) 不是同一套：
这里的转账立即到账，第三关是付款和延迟返现，第四关的历史余额按“那一刻这个账户在不在”来回答。两页的规则
不要混用，做这一页时只看这一页的题面。

通用约定：

- 每个方法的第一个参数都是 `timestamp`：一个整数，单位是毫秒，范围是 $1$ 到 $10^9$。同一个实例上，后一次
  调用的 `timestamp` 一定大于前一次（严格递增，不会重复）。
- `account_id` 是非空字符串。`amount` 是非负整数，可以是 $0$。
- 同一时刻，一个 `account_id` 最多对应一个存在的账户。账户由 `create_account` 创建；只有第四关的
  `merge_accounts` 能让一个账户消失。
- 失败的调用（返回 `None` 或 `False`）不改变任何状态。

### Level 1 —— 开户、存款与转账

```py
class Bank:
    def create_account(self, timestamp: int, account_id: str) -> bool:
        """Opens an account with balance 0. Returns True, or False if an account with this id
        already exists."""

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Adds amount to the account's balance and returns the new balance. Returns None if the
        account does not exist."""

    def transfer(self, timestamp: int, source_account_id: str, target_account_id: str,
                 amount: int) -> int | None:
        """Moves amount from the source account to the target account immediately and returns
        the SOURCE account's new balance. Returns None if either account does not exist, if the
        two ids are the same, or if amount is greater than the source account's balance."""
```

示例：

```text
create_account(1, "alice")             # -> True
create_account(2, "alice")             # -> False   已经有这个账户
create_account(3, "bob")               # -> True
deposit(4, "carol", 500)               # -> None    没有这个账户
deposit(5, "alice", 1200)              # -> 1200
transfer(6, "alice", "bob", 1201)      # -> None    alice 只有 1200
transfer(7, "alice", "bob", 1200)      # -> 0       正好转光是允许的；返回的是转出方的新余额
transfer(8, "bob", "bob", 10)          # -> None    不能转给自己
transfer(9, "bob", "carol", 10)        # -> None    收款账户不存在
deposit(10, "bob", 0)                  # -> 1200    存 0 是合法的，结果就是当前余额
```

### Level 2 —— 支出排行

每个账户有一个*支出总额*（outgoing total）：它成功转出的金额之和；从第三关起还要加上它成功付款的金额。
收到的钱不算，第三关的返现也不算。新账户的支出总额是 $0$。

```py
class Bank:
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        """Returns the n accounts with the largest outgoing totals, largest first. Accounts with
        equal totals are ordered by account_id ascending (ordinary string comparison). Each
        entry is the string "<account_id>(<outgoing total>)", e.g. "alice(450)". If fewer than
        n accounts exist, returns all of them in the same order. n is a non-negative integer;
        n == 0, or a bank with no accounts, gives []."""
```

示例：

```text
create_account(1, "carol"); create_account(2, "bob"); create_account(3, "alice")
deposit(4, "carol", 2000)              # -> 2000
deposit(5, "bob", 3000)                # -> 3000
top_spenders(6, 3)                     # -> ["alice(0)", "bob(0)", "carol(0)"]        都是 0：按 id 升序
transfer(7, "carol", "bob", 500)       # -> 1500
transfer(8, "carol", "alice", 1000)    # -> 500
transfer(9, "alice", "bob", 1000)      # -> 0
top_spenders(10, 2)                    # -> ["carol(1500)", "alice(1000)"]
top_spenders(11, 5)                    # -> ["carol(1500)", "alice(1000)", "bob(0)"]  一共只有 3 个账户
top_spenders(12, 0)                    # -> []
```

### Level 3 —— 付款与返现

```py
class Bank:
    def pay(self, timestamp: int, account_id: str, amount: int) -> str | None:
        """Takes amount out of the account as a payment and returns the payment's id:
        "payment1", "payment2", ... The number comes from one counter shared by all accounts,
        and the counter advances only when a payment succeeds. Returns None if the account does
        not exist or amount is greater than its balance."""

    def get_payment_status(self, timestamp: int, account_id: str, payment: str) -> str | None:
        """Returns "IN_PROGRESS" if the payment's cashback has not been credited yet, and
        "CASHBACK_RECEIVED" once it has. Returns None if the account does not exist, or if
        payment is not the id of a payment belonging to this account (no such payment, or it
        was made from a different account)."""
```

付款的规则：

- 付款金额计入该账户的支出总额，`top_spenders` 要把它算进去。
- 每笔成功的付款都有*返现*（cashback）：金额是 `amount` 的 2%，向下取整，也就是 `amount * 2 // 100`。
  返现在 `timestamp + 86_400_000` 这一时刻（付款之后 24 小时）进入账户余额。
- 到点的返现先于那一时刻的任何操作。也就是说，一次调用的 `timestamp` 只要大于或等于某笔返现的到账时刻，
  这次调用开始之前那笔返现就已经在余额里了；`timestamp` 恰好等于到账时刻时也是如此。
- 返现不算支出，也不减少任何账户的支出总额。
- 返现金额是 $0$（`amount` 小于 50）时，到点后状态同样变成 `"CASHBACK_RECEIVED"`。

示例（`DAY = 86_400_000`）：

```text
create_account(1, "alice"); create_account(2, "bob")
deposit(3, "alice", 1000)                          # -> 1000
pay(4, "alice", 1001)                              # -> None          余额不够；编号不前进
pay(5, "alice", 450)                               # -> "payment1"    余额 550；返现 450*2//100 = 9，在 5+DAY 到账
deposit(6, "bob", 80)                              # -> 80
pay(7, "bob", 49)                                  # -> "payment2"    余额 31；返现 49*2//100 = 0
get_payment_status(8, "alice", "payment1")         # -> "IN_PROGRESS"
get_payment_status(9, "bob", "payment1")           # -> None          payment1 是 alice 的
get_payment_status(10, "alice", "payment7")        # -> None          没有这笔付款
get_payment_status(11, "carol", "payment1")        # -> None          没有这个账户
top_spenders(12, 2)                                # -> ["alice(450)", "bob(49)"]
deposit(4 + DAY, "alice", 0)                       # -> 550           还差 1 毫秒，返现没到
get_payment_status(5 + DAY, "alice", "payment1")   # -> "CASHBACK_RECEIVED"   恰好到点：先入账，再回答
deposit(6 + DAY, "alice", 0)                       # -> 559
get_payment_status(8 + DAY, "bob", "payment2")     # -> "CASHBACK_RECEIVED"   返现是 0，状态照样变
top_spenders(9 + DAY, 2)                           # -> ["alice(450)", "bob(49)"]   返现不改变支出总额
```

### Level 4 —— 合并账户与历史余额

```py
class Bank:
    def merge_accounts(self, timestamp: int, account_id_1: str, account_id_2: str) -> bool:
        """Merges account_id_2 into account_id_1. Returns False if the two ids are equal or if
        either account does not exist; otherwise applies the rules below and returns True."""

    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        """Returns the balance that the account named account_id had at time_at, or None if no
        account with that id existed at time_at. 1 <= time_at <= timestamp."""
```

`merge_accounts` 成功时：

- 账户 1 的余额加上账户 2 的余额；账户 1 的支出总额加上账户 2 的支出总额。
- 账户 2 还没到账的返现照常在原定时刻到账，但进的是账户 1。
- 账户 2 做过的所有付款从此属于账户 1：用 `account_id_1` 调 `get_payment_status` 能查到它们，状态规则
  不变。
- 账户 2 从系统里消失。之后用 `account_id_2` 做 `deposit`、`transfer`、`pay`、`get_payment_status`、
  `merge_accounts` 都按“账户不存在”处理，`top_spenders` 里也不再有它。`create_account` 可以再用这个 id
  开一个全新的账户：余额 $0$，支出总额 $0$，名下没有任何付款。

`get_balance` 的规则：

- 一个账户的存在时间从它被创建的时间戳开始（含这一刻），到它被合并掉的时间戳为止（不含这一刻）；没有
  被合并掉的账户一直存在。
- 判断的是 `time_at` 那一刻，不是现在。如果 `time_at` 时刻有一个叫 `account_id` 的账户存在，就返回那个
  账户当时的余额，哪怕它现在已经被合并掉了；如果那一刻没有，返回 `None`。
- “当时的余额”指：时间戳不超过 `time_at` 的所有操作都已生效，到账时刻不超过 `time_at` 的所有返现也都
  已入账——即使到账的那一刻没有任何调用发生。
- 合并不改写过去：账户 1 在合并之前各个时刻的余额仍是它自己当时的余额，从合并那一刻起才包含账户 2 的钱。
- 同一个 id 被合并掉之后又重新开户，前后是两个不同的账户，各有各的存在时间；两段之间的空档返回 `None`。

示例（`DAY = 86_400_000`）：

```text
create_account(1, "alice"); create_account(2, "bob")
deposit(3, "alice", 1000)                          # -> 1000
deposit(4, "bob", 800)                             # -> 800
pay(5, "bob", 300)                                 # -> "payment1"   bob 余额 500；返现 6，在 5+DAY 到账
transfer(6, "alice", "bob", 100)                   # -> 900          bob 余额 600
merge_accounts(7, "alice", "alice")                # -> False        不能和自己合并
merge_accounts(8, "alice", "carol")                # -> False        carol 不存在
merge_accounts(9, "alice", "bob")                  # -> True         alice 余额 900 + 600 = 1500
top_spenders(10, 5)                                # -> ["alice(400)"]   100（转账）+ 300（bob 的付款）
get_payment_status(11, "alice", "payment1")        # -> "IN_PROGRESS"    bob 的付款现在归 alice
get_payment_status(12, "bob", "payment1")          # -> None             bob 不存在了
deposit(13, "bob", 50)                             # -> None
get_balance(14, "alice", 8)                        # -> 900          合并之前：alice 自己当时的余额
get_balance(15, "alice", 9)                        # -> 1500         从合并那一刻起
get_balance(16, "bob", 8)                          # -> 600          时刻 8，bob 还在
get_balance(17, "bob", 9)                          # -> None         时刻 9，bob 已被合并掉
get_balance(18, "bob", 1)                          # -> None         bob 在时刻 2 才开户
create_account(20, "bob")                          # -> True         同一个 id，全新的账户
get_balance(21, "bob", 20)                         # -> 0
get_balance(22, "bob", 15)                         # -> None         时刻 15 没有叫 bob 的账户
get_balance(23, "bob", 5)                          # -> 500          旧 bob 在时刻 5 的余额
deposit(6 + DAY, "alice", 0)                       # -> 1506         bob 那笔付款的返现进了 alice
get_balance(7 + DAY, "alice", 4 + DAY)             # -> 1500
get_balance(8 + DAY, "alice", 5 + DAY)             # -> 1506         返现记在到账时刻，虽然那一刻没有调用
get_balance(9 + DAY, "bob", 5 + DAY)               # -> 0            新 bob 没有拿到返现
get_payment_status(10 + DAY, "alice", "payment1")  # -> "CASHBACK_RECEIVED"
```

## 参考解答

<details>
<summary>展开参考解答</summary>

值得事先确认的三点，因为合理的设计在这三点上可能不同：到点的返现是先于那一时刻的操作入账，还是之后
（这里：先入账，`timestamp` 恰好等于到账时刻也算到了）；`get_balance` 问的是 `time_at` 那一刻有没有这个账户，
还是现在有没有（这里：那一刻）；合并会不会改写过去（这里：不会——存下来的账户在合并之前的历史余额不变，被
合并掉的账户在合并之前的余额仍然查得到）。真实测评时以平台上的例子为准，另外两种读法见“追问”。

### 时间分配

90 分钟四关，分量前轻后重。一个可行的预算，剩下的时间留作机动：

| 关 | 预算 | 主要工作 |
| --- | --- | --- |
| Level 1 | 10 分钟以内 | 三个方法，各四五行 |
| Level 2 | 10 分钟以内 | 一次排序 |
| Level 3 | 20 分钟左右 | 返现队列，以及“每个方法先结算到点的返现” |
| Level 4 | 30 到 35 分钟 | 合并要搬的东西，按时间查余额 |

几条和具体题目无关的做法：

- 每一关动手之前，花两三分钟把这一关的签名和例子从头读到尾。这类题丢分多半不是不会写，而是返回值写错：
  这里 `transfer` 返回的是转出方的新余额，失败返回 `None` 而不是 `False`；`top_spenders` 返回的是
  `"id(金额)"` 字符串的列表。
- 写完一个方法就跑一次平台给的测试，不要攒到一关写完再跑。
- 后面的关卡事先看不到，但签名会给提示：第一关每个方法都带着一个用不上的 `timestamp`，后面多半有“到了
  某个时刻自动发生”的规则。所以下面从第一关起就把余额变化收进一个辅助方法 `_change`，并让每个方法第一行
  调用一个暂时为空的 `_settle(timestamp)`。多写这两行，第三、四关就不用回头改前面的方法。
- 最后一关没写完也有分：[面试流程](#/anthropic/process) 里记了候选人以 750/1000、800/1000 进入下一轮的
  例子。第四关里 `merge_accounts` 比 `get_balance` 容易，先写它，用到合并而不查历史余额的用例就能先通过。
- 不追求最优复杂度。能一眼看出对错的写法比省一个对数因子重要。

### Level 1

账户是一个小对象，`Bank` 里只有一张“现在存在的账户”的表。`_Account` 下面一次写成了四关做完后的样子，
每个字段旁边标了第几关才用到；考试时是做到那一关才加那个字段。同样，`transfer` 里记支出总额的那一行是
第二关加的，`_change` 里记历史的两行是第四关加的。这样安排是为了让后面每一关的代码块里只有新方法。

`transfer` 的四个失败条件写在一个 `if` 里。`source is target` 判断的是“同一个账户”，两个 id 相同且账户存在
时成立；id 相同但账户不存在的情况已经被前面的 `is None` 挡住了。

```python
import math


class _Account:
    def __init__(self, created_at: int) -> None:
        self.balance = 0
        self.outgoing = 0                      # Level 2: total sent out by transfer and pay
        self.payments: dict[str, int] = {}     # Level 3: payment id -> time its cashback lands
        self.created_at = created_at           # Level 4: the account exists on [created_at, closed_at)
        self.closed_at = math.inf
        self.times = [created_at]              # Level 4: balance history -- balances[i] holds from times[i] on
        self.balances = [0]


class Bank:
    def __init__(self) -> None:
        self._accounts: dict[str, _Account] = {}      # the accounts that exist right now

    def _settle(self, timestamp: int) -> None:
        """Whatever happens by itself as time passes. Nothing yet -- Level 3 fills this in."""

    def _change(self, account: _Account, delta: int, when: int) -> int:
        """The only place a balance changes."""
        account.balance += delta
        account.times.append(when)                    # Level 4
        account.balances.append(account.balance)      # Level 4
        return account.balance

    def create_account(self, timestamp: int, account_id: str) -> bool:
        self._settle(timestamp)
        if account_id in self._accounts:
            return False
        self._accounts[account_id] = _Account(timestamp)
        return True

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        self._settle(timestamp)
        account = self._accounts.get(account_id)
        if account is None:
            return None
        return self._change(account, amount, timestamp)

    def transfer(self, timestamp: int, source_account_id: str, target_account_id: str,
                 amount: int) -> int | None:
        self._settle(timestamp)
        source = self._accounts.get(source_account_id)
        target = self._accounts.get(target_account_id)
        if source is None or target is None or source is target or amount > source.balance:
            return None
        source.outgoing += amount                     # Level 2
        self._change(target, amount, timestamp)
        return self._change(source, -amount, timestamp)   # NOTE: the source's balance, not the target's
```

三个方法都是 $O(1)$。

### Level 2

对现有账户排一次序。排序键是 `(-支出总额, account_id)`：金额取负，所以大的在前；id 没有取负，所以并列时
按 id 升序。`ranked[:n]` 在 `n` 为 $0$ 或大于账户数时都自然正确，不需要单独判断。

```python
class Bank(Bank):
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        self._settle(timestamp)
        ranked = sorted(self._accounts.items(), key=lambda item: (-item[1].outgoing, item[0]))
        return [f"{account_id}({account.outgoing})" for account_id, account in ranked[:n]]
```

把 `Bank` 重新打开、定义成它自己的子类，只是这一页用来逐关展示代码的写法；考试时直接在原来的类里加方法。
账户数为 $k$ 时开销是 $O(k \log k)$。

### Level 3

返现是“将来某个时刻自动发生的事”。不需要定时器：没有任何调用的时候，也没有人能看到余额，所以只要保证
**每个方法开始时，先把到账时刻不晚于当前 `timestamp` 的返现全部入账**，从外面看就和准时入账没有区别。
这就是 `_settle`，也是第一关在每个方法开头留那一行的原因。如果第一关没有留，这一关要给前面四个方法各补
一行——漏掉任何一个，返现就会在那个方法里晚到。

等待入账的返现放在一个双端队列里。付款的时间戳严格递增，延迟又是同一个常数，所以到账时刻也是递增的：
先付款的一定先到账，队列天然按到账时刻排好，每次只看队首。条件是 `<=`，这对应“恰好到点时返现先入账”。

付款的状态不用单独存。每笔付款只记它的到账时刻，查询时和当前 `timestamp` 比较即可：`_settle` 已经保证了
“到账时刻不晚于现在”和“返现已经入账”是一回事。

返现金额用整数算：`amount * 2 // 100`。

```python
from collections import deque


class Bank(Bank):
    CASHBACK_DELAY = 86_400_000          # 24 hours in milliseconds

    def __init__(self) -> None:
        super().__init__()
        self._pending: deque[list] = deque()     # [due, account, cashback], earliest payment first
        self._payment_count = 0

    def _settle(self, timestamp: int) -> None:
        while self._pending and self._pending[0][0] <= timestamp:     # NOTE: <= -- a cashback due now lands first
            due, account, cashback = self._pending.popleft()
            self._change(account, cashback, due)                      # NOTE: booked at due, not at timestamp

    def pay(self, timestamp: int, account_id: str, amount: int) -> str | None:
        self._settle(timestamp)
        account = self._accounts.get(account_id)
        if account is None or amount > account.balance:
            return None
        account.outgoing += amount
        self._change(account, -amount, timestamp)
        self._payment_count += 1                                      # NOTE: only after the checks passed
        payment = f"payment{self._payment_count}"
        due = timestamp + self.CASHBACK_DELAY
        account.payments[payment] = due
        self._pending.append([due, account, amount * 2 // 100])
        return payment

    def get_payment_status(self, timestamp: int, account_id: str, payment: str) -> str | None:
        self._settle(timestamp)
        account = self._accounts.get(account_id)
        if account is None or payment not in account.payments:
            return None
        return "CASHBACK_RECEIVED" if account.payments[payment] <= timestamp else "IN_PROGRESS"
```

每笔返现只进队、出队各一次，所以 `_settle` 的开销摊到每次付款上是 $O(1)$；`pay` 和 `get_payment_status`
除此之外都是 $O(1)$。

`_settle` 把返现记在 `due` 这一时刻，而不是触发结算的那次调用的 `timestamp`。在这一关两种写法返回的结果
一样，区别要到第四关按时间查余额时才显出来，所以这里一次写对。

### Level 4

`merge_accounts` 要搬四样东西，对着题面的清单逐条写：余额（经过 `_change`，记在合并这一刻）、支出总额、
名下的付款记录、还在队列里的返现。队列里的每一项存的是账户**对象**而不是 id，把指向账户 2 的项改指向
账户 1 就行。用对象而不是 id 有一个实际的好处：账户 2 的 id 之后可以重新开户，如果队列里存的是 id 字符串，
旧账户的返现就会错进新账户。

被合并掉的账户不能直接丢掉，因为 `get_balance` 还会问到它合并之前的余额。做法是给它记下消失的时刻
`closed_at`，并按 id 存进 `_closed`。于是 `get_balance` 就是两步：先在“这个 id 曾经对应过的账户”里找出
`time_at` 落在其存在时间 `[created_at, closed_at)` 内的那一个；再在它的余额历史里二分，找最后一条时间不
晚于 `time_at` 的记录。

余额历史是两个平行的列表 `times` 和 `balances`，由 `_change` 追加。它的时间是不减的：普通操作的时间戳递增；
返现在 `_settle` 里按到账时刻从早到晚入账，而且一定晚于上一次调用、不晚于这一次调用。同一时刻可以有两条
记录（返现和恰好在到账时刻发生的操作），`bisect_right` 取到的是后一条，正是“那一刻所有事情做完之后”的
余额。

```python
from bisect import bisect_right


class Bank(Bank):
    def __init__(self) -> None:
        super().__init__()
        self._closed: dict[str, list[_Account]] = {}     # account_id -> accounts merged away under that id

    def merge_accounts(self, timestamp: int, account_id_1: str, account_id_2: str) -> bool:
        self._settle(timestamp)
        keep = self._accounts.get(account_id_1)
        gone = self._accounts.get(account_id_2)
        if keep is None or gone is None or keep is gone:
            return False
        self._change(keep, gone.balance, timestamp)
        keep.outgoing += gone.outgoing
        keep.payments.update(gone.payments)
        for entry in self._pending:                      # NOTE: cashback still on its way follows the money
            if entry[1] is gone:
                entry[1] = keep
        gone.closed_at = timestamp                       # NOTE: from this moment on the account does not exist
        self._closed.setdefault(account_id_2, []).append(gone)
        del self._accounts[account_id_2]
        return True

    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        self._settle(timestamp)
        candidates = list(self._closed.get(account_id, []))
        if account_id in self._accounts:
            candidates.append(self._accounts[account_id])
        for account in candidates:
            if account.created_at <= time_at < account.closed_at:
                return account.balances[bisect_right(account.times, time_at) - 1]
        return None
```

`merge_accounts` 的开销是 $O(q + p)$：$q$ 是队列里还没到账的返现数，$p$ 是账户 2 名下的付款数。
`get_balance` 是 $O(g + \log e)$：$g$ 是这个 id 先后对应过的账户数，$e$ 是找到的那个账户的余额变化次数。
`get_balance` 开头的 `_settle(timestamp)` 不能省：题目保证 `time_at` 不晚于 `timestamp`，所以结算之后，
到账时刻不晚于 `time_at` 的返现一定都已经写进历史了。

### 追问

- **为什么不在付款时就把返现加进余额，再用一个标记表示“还不能用”？** 因为余额本身是可见的：`deposit` 的
  返回值、`pay` 的余额检查、`get_balance` 都会读到它。提前加进去就要在每个读余额的地方再减掉，比推迟入账
  更容易错。
- **返现延迟不是常数（比如按商户不同）怎么办？** 到账时刻就不再随付款顺序递增，队列要换成按到账时刻排序的
  最小堆（`heapq`），`_settle` 改成不断弹出堆顶；其余不变。同一时刻到账的多笔返现还需要规定先后，比如按
  付款编号。
- **`get_balance` 还有哪些读法？** “继承余额历史”和“不存在时返回 `None`”可以有好几种理解，两份公开练习集
  的参考实现就各不相同。一份是“账户现在不存在就返回 `None`”（站内
  [另一套银行关卡](#/anthropic/coding/oa-bank-system) 也是这样），并且把两个账户的余额快照按时间混在一起，
  问合并之前的时刻时，返回两边快照里时间最晚的那一个。另一份把两个账户的收支流水归并，问合并之前的时刻
  得到的是两个账户余额之和；问被合并掉的 id 时，它转去查存下来的账户。分歧主要出现在问合并之前的时刻、
  或者问被合并掉的 id 的时候，所以考试时先找例子里有没有这样的一行。本页的读法只要把被合并的账户留着，
  不需要归并历史，改成上面任何一种都只动 `merge_accounts` 和 `get_balance`。
- **合并很频繁、排队的返现很多时，`merge_accounts` 里那个循环太慢怎么办？** 不去改队列，改成在被合并掉的
  账户上记一个“并入了谁”的指针，结算时顺着指针找到最终的账户（并查集的做法，可以顺手压缩路径）。合并本身
  就变成 $O(1)$ 加上付款记录的搬运。
- **如果允许 `time_at` 晚于 `timestamp`？** 那就是问未来：要把队列里到账时刻不晚于 `time_at` 的返现也算
  进去，但不能真的入账。这里题目排除了这种情况。
- **账户非常多、`n` 很小时的 `top_spenders`？** 用 `heapq.nsmallest(n, ..., key=...)`，开销从
  $O(k \log k)$ 降到 $O(k \log n)$。题面没有提性能要求，考试时不必这样做。

<details>
<summary>验证代码（可运行）</summary>

```python
import random
from collections import Counter

DAY = 86_400_000
assert Bank.CASHBACK_DELAY == DAY == 24 * 60 * 60 * 1000

# ---- Level 1: the statement's example ----
b = Bank()
assert b.create_account(1, "alice") is True
assert b.create_account(2, "alice") is False
assert b.create_account(3, "bob") is True
assert b.deposit(4, "carol", 500) is None
assert b.deposit(5, "alice", 1200) == 1200
assert b.transfer(6, "alice", "bob", 1201) is None
assert b.transfer(7, "alice", "bob", 1200) == 0
assert b.transfer(8, "bob", "bob", 10) is None
assert b.transfer(9, "bob", "carol", 10) is None
assert b.deposit(10, "bob", 0) == 1200

# ---- Level 2: the statement's example ----
b = Bank()
assert b.create_account(1, "carol") and b.create_account(2, "bob") and b.create_account(3, "alice")
assert b.deposit(4, "carol", 2000) == 2000
assert b.deposit(5, "bob", 3000) == 3000
assert b.top_spenders(6, 3) == ["alice(0)", "bob(0)", "carol(0)"]
assert b.transfer(7, "carol", "bob", 500) == 1500
assert b.transfer(8, "carol", "alice", 1000) == 500
assert b.transfer(9, "alice", "bob", 1000) == 0
assert b.top_spenders(10, 2) == ["carol(1500)", "alice(1000)"]
assert b.top_spenders(11, 5) == ["carol(1500)", "alice(1000)", "bob(0)"]
assert b.top_spenders(12, 0) == []

# ---- Level 3: the statement's example ----
b = Bank()
assert b.create_account(1, "alice") and b.create_account(2, "bob")
assert b.deposit(3, "alice", 1000) == 1000
assert b.pay(4, "alice", 1001) is None
assert b.pay(5, "alice", 450) == "payment1"
assert b.deposit(6, "bob", 80) == 80
assert b.pay(7, "bob", 49) == "payment2"
assert b.get_payment_status(8, "alice", "payment1") == "IN_PROGRESS"
assert b.get_payment_status(9, "bob", "payment1") is None
assert b.get_payment_status(10, "alice", "payment7") is None
assert b.get_payment_status(11, "carol", "payment1") is None
assert b.top_spenders(12, 2) == ["alice(450)", "bob(49)"]
assert b.deposit(4 + DAY, "alice", 0) == 550
assert b.get_payment_status(5 + DAY, "alice", "payment1") == "CASHBACK_RECEIVED"
assert b.deposit(6 + DAY, "alice", 0) == 559
assert b.get_payment_status(8 + DAY, "bob", "payment2") == "CASHBACK_RECEIVED"
assert b.top_spenders(9 + DAY, 2) == ["alice(450)", "bob(49)"]

# ---- Level 4: the statement's example ----
b = Bank()
assert b.create_account(1, "alice") and b.create_account(2, "bob")
assert b.deposit(3, "alice", 1000) == 1000
assert b.deposit(4, "bob", 800) == 800
assert b.pay(5, "bob", 300) == "payment1"
assert b.transfer(6, "alice", "bob", 100) == 900
assert b.merge_accounts(7, "alice", "alice") is False
assert b.merge_accounts(8, "alice", "carol") is False
assert b.merge_accounts(9, "alice", "bob") is True
assert b.top_spenders(10, 5) == ["alice(400)"]
assert b.get_payment_status(11, "alice", "payment1") == "IN_PROGRESS"
assert b.get_payment_status(12, "bob", "payment1") is None
assert b.deposit(13, "bob", 50) is None
assert b.get_balance(14, "alice", 8) == 900
assert b.get_balance(15, "alice", 9) == 1500
assert b.get_balance(16, "bob", 8) == 600
assert b.get_balance(17, "bob", 9) is None
assert b.get_balance(18, "bob", 1) is None
assert b.create_account(20, "bob") is True
assert b.get_balance(21, "bob", 20) == 0
assert b.get_balance(22, "bob", 15) is None
assert b.get_balance(23, "bob", 5) == 500
assert b.deposit(6 + DAY, "alice", 0) == 1506
assert b.get_balance(7 + DAY, "alice", 4 + DAY) == 1500
assert b.get_balance(8 + DAY, "alice", 5 + DAY) == 1506
assert b.get_balance(9 + DAY, "bob", 5 + DAY) == 0
assert b.get_payment_status(10 + DAY, "alice", "payment1") == "CASHBACK_RECEIVED"

# ---- the practice set's own example tables, transcribed (ids and numbers as in the source) ----
b = Bank()                                                    # its Level 1 table
assert b.create_account(1, "account1") is True
assert b.create_account(2, "account1") is False
assert b.create_account(3, "account2") is True
assert b.deposit(4, "non-existing", 2700) is None
assert b.deposit(5, "account1", 2700) == 2700
assert b.transfer(6, "account1", "account2", 2701) is None
assert b.transfer(7, "account1", "account2", 200) == 2500

b = Bank()                                                    # its Level 2 table
assert b.create_account(1, "account3") and b.create_account(2, "account2") and b.create_account(3, "account1")
assert b.deposit(4, "account3", 2000) == 2000
assert b.deposit(5, "account2", 3000) == 3000
assert b.deposit(6, "account3", 4000) == 6000
assert b.top_spenders(7, 3) == ["account1(0)", "account2(0)", "account3(0)"]
assert b.transfer(8, "account3", "account2", 500) == 5500
assert b.transfer(9, "account3", "account1", 1000) == 4500
# The table's last two rows do not follow from its own rules: account1 holds only 1000 at this point, so a
# transfer of 2500 fails. The table says it returns 500 and then ranks account1(2500) first.
assert b.transfer(10, "account1", "account2", 2500) is None
assert b.top_spenders(11, 3) == ["account3(1500)", "account1(0)", "account2(0)"]

b = Bank()                                                    # its Level 3 table
assert b.create_account(1, "account1") and b.create_account(2, "account2")
assert b.deposit(3, "account1", 2000) == 2000
assert b.pay(4, "account1", 1000) == "payment1"
assert b.pay(100, "account1", 1000) == "payment2"
assert b.get_payment_status(101, "non-existing", "payment1") is None
assert b.get_payment_status(102, "account2", "payment1") is None
assert b.get_payment_status(103, "account1", "payment1") == "IN_PROGRESS"
assert b.top_spenders(104, 2) == ["account1(2000)", "account2(0)"]
assert b.deposit(3 + DAY, "account1", 100) == 100
assert b.get_payment_status(4 + DAY, "account1", "payment1") == "CASHBACK_RECEIVED"
assert b.deposit(5 + DAY, "account1", 100) == 220
assert b.deposit(99 + DAY, "account1", 100) == 320
assert b.deposit(100 + DAY, "account1", 100) == 440

b = Bank()                                                    # its second Level 4 table
assert b.create_account(1, "account1") is True
assert b.deposit(2, "account1", 1000) == 1000
assert b.pay(3, "account1", 300) == "payment1"
assert b.get_balance(4, "account1", 3) == 700
assert b.get_balance(5 + DAY, "account1", 2 + DAY) == 700
assert b.get_balance(6 + DAY, "account1", 3 + DAY) == 706

# Its first Level 4 table does not add up: after pay(5, "account1", 500) the account holds 1500, so the
# transfer of 500 leaves 1000 (the table says 1500), the merged balance is 1000 + 2500 = 3500 (it says
# 3000), and the last deposit gives 3500 + 100 + 10 = 3610 (it says 3906). The calls are transcribed
# here with the values the stated rules give; every None / False / status in the table is unchanged.
b = Bank()
assert b.create_account(1, "account1") and b.create_account(2, "account2")
assert b.deposit(3, "account1", 2000) == 2000
assert b.deposit(4, "account2", 2000) == 2000
assert b.pay(5, "account1", 500) == "payment1"
assert b.transfer(6, "account1", "account2", 500) == 1000
assert b.merge_accounts(7, "account1", "non-existing") is False
assert b.merge_accounts(8, "account1", "account1") is False
assert b.merge_accounts(9, "account1", "account2") is True
assert b.get_balance(10, "account1", 10) == 3500
assert b.get_balance(11, "account2", 10) is None
assert b.get_payment_status(12, "account1", "payment1") == "IN_PROGRESS"
assert b.get_payment_status(13, "account2", "payment1") is None
assert b.get_balance(14, "account2", 1) is None               # "was not created yet"
assert b.get_balance(15, "account2", 9) is None               # "was already merged"
assert b.get_balance(16, "account1", 11) == 3500
assert b.deposit(5 + DAY, "account1", 100) == 3610

# ---- the practice set's unit tests, transcribed. Where a test reused or decreased a timestamp, the
#      timestamp is fixed here (marked); no expected value depends on it. ----
b = Bank()
assert b.create_account(1, "acc1") is True and b.create_account(2, "acc1") is False
assert b.create_account(3, "acc2") is True

b = Bank()
b.create_account(1, "acc1")
assert b.deposit(2, "acc1", 500) == 500
assert b.deposit(3, "acc1", 300) == 800
assert b.deposit(4, "non_existent", 100) is None

b = Bank()
assert b.create_account(1, "acc1") and b.create_account(2, "acc2")
assert b.deposit(3, "acc1", 1000) == 1000
assert b.transfer(4, "acc1", "acc2", 300) == 700
assert b.transfer(5, "acc1", "acc2", 800) is None
assert b.transfer(6, "acc1", "non_existent", 100) is None
assert b.transfer(7, "acc1", "acc1", 100) is None

b = Bank()
assert b.create_account(1, "A") and b.create_account(2, "B")
assert b.deposit(3, "A", 500) == 500
assert b.transfer(4, "A", "B", 300) == 200
assert b.deposit(5, "B", 200) == 500
assert b.transfer(6, "B", "A", 600) is None
assert b.transfer(7, "B", "A", 400) == 100

b = Bank()
assert b.create_account(1, "X") is True
assert b.deposit(2, "X", 1000) == 1000
assert b.create_account(3, "Y") is True
assert b.transfer(4, "X", "Y", 500) == 500
assert b.transfer(5, "Y", "X", 600) is None
assert b.deposit(6, "Y", 300) == 800
assert b.transfer(7, "Y", "X", 400) == 400

b = Bank()
assert b.top_spenders(1, 0) == []
assert b.top_spenders(2, 5) == []

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000); b.create_account(3, "acc2")
b.transfer(4, "acc1", "acc2", 500)
assert b.top_spenders(5, 1) == ["acc1(500)"]

b = Bank()
b.create_account(1, "acc1"); b.create_account(2, "acc2"); b.create_account(3, "acc3")
b.deposit(4, "acc1", 1000); b.deposit(5, "acc2", 1500); b.deposit(6, "acc3", 1200)
b.transfer(7, "acc2", "acc3", 500)          # source: timestamps 8 then 7
b.transfer(8, "acc1", "acc2", 500)
b.transfer(9, "acc3", "acc1", 300)
assert b.top_spenders(10, 3) == ["acc1(500)", "acc2(500)", "acc3(300)"]

b = Bank()
assert b.pay(1, "non_existent", 100) is None
assert b.get_payment_status(2, "non_existent", "payment1") is None

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 100)
assert b.pay(3, "acc1", 200) is None

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000)
assert b.pay(3, "acc1", 500) == "payment1"
assert b.pay(4, "acc1", 300) == "payment2"
b.create_account(5, "acc2"); b.deposit(6, "acc2", 800); b.transfer(7, "acc2", "acc1", 200)
assert b.top_spenders(8, 2) == ["acc1(800)", "acc2(200)"]          # source: timestamp 5

b = Bank()
b.create_account(1, "acc1")
assert b.get_payment_status(2, "acc1", "payment1") is None

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000)
assert b.pay(3, "acc1", 500) == "payment1"
b.create_account(4, "acc2")
assert b.get_payment_status(5, "acc2", "payment1") is None          # source: timestamp 4 again

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000)
assert b.pay(3, "acc1", 500) == "payment1"
assert b.get_payment_status(4, "acc1", "payment1") == "IN_PROGRESS"
assert b.get_payment_status(26 * 3600, "acc1", "payment1") == "IN_PROGRESS"   # the source does not assert this
assert b.get_payment_status(DAY + 3, "acc1", "payment1") == "CASHBACK_RECEIVED"
assert b.deposit(28 * 60 * 60 * 1000, "acc1", 0) == 1000 - 500 + 10

b = Bank()
b.create_account(1, "acc2")
assert b.merge_accounts(2, "acc1", "acc2") is False
b = Bank()
b.create_account(1, "acc1")
assert b.merge_accounts(2, "acc1", "acc2") is False

b = Bank()                                   # the payer is merged away; its cashback follows
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000)
assert b.pay(3, "acc1", 500) == "payment1"
b.create_account(4, "acc2")
assert b.merge_accounts(5, "acc2", "acc1") is True
assert b.get_payment_status(6, "acc2", "payment1") == "IN_PROGRESS"
assert b.get_payment_status(DAY + 3, "acc2", "payment1") == "CASHBACK_RECEIVED"
assert b.deposit(DAY + 5, "acc2", 0) == 510

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000); b.pay(3, "acc1", 500)
b.create_account(4, "acc2"); b.deposit(5, "acc2", 2000); b.pay(6, "acc2", 800)
assert b.merge_accounts(7, "acc1", "acc2") is True
assert b.top_spenders(8, 1) == ["acc1(1300)"]

b = Bank()
b.create_account(1, "acc1"); b.deposit(2, "acc1", 1000); b.pay(3, "acc1", 300)
assert b.get_balance(4, "acc1", 3) == 700
assert b.get_balance(DAY + 5, "acc1", DAY + 2) == 700
assert b.get_balance(DAY + 6, "acc1", DAY + 3) == 706               # source: DAY + 5 again

# ---- a second public practice set with the same four levels (a trainer repo's "mock-bank-4"), its unit
#      tests transcribed. They put several calls on one timestamp (e.g. two create_account at 1); those
#      calls are moved to the next free integers, in the same order. ----
b = Bank(); assert b.create_account(1, "a") is True and b.create_account(2, "a") is False
b = Bank(); b.create_account(1, "a"); assert b.deposit(2, "a", 100) == 100 and b.deposit(3, "a", 50) == 150
b = Bank(); assert b.deposit(1, "ghost", 100) is None
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 200)
assert b.transfer(4, "a", "b", 50) == 150
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 30)
assert b.transfer(4, "a", "b", 50) is None and b.deposit(5, "a", 0) == 30
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 100)
assert b.transfer(3, "a", "ghost", 10) is None and b.transfer(4, "ghost", "a", 10) is None
assert b.transfer(5, "a", "a", 10) is None

b = Bank(); assert b.top_spenders(1, 5) == []
b = Bank()
for t, aid in enumerate("abc", start=1):
    b.create_account(t, aid); b.deposit(t + 3, aid, 1000)
b.transfer(7, "a", "b", 300); b.transfer(8, "b", "c", 200); b.transfer(9, "c", "a", 100)
assert b.top_spenders(10, 3) == ["a(300)", "b(200)", "c(100)"]
b = Bank(); b.create_account(1, "x"); b.create_account(2, "y"); b.deposit(3, "x", 100); b.deposit(4, "y", 100)
b.transfer(5, "x", "y", 25)
assert b.top_spenders(6, 1) == ["x(25)"]
b = Bank()
for t, aid in enumerate("zam", start=1):
    b.create_account(t, aid); b.deposit(t + 3, aid, 100)
b.transfer(7, "z", "a", 50); b.transfer(8, "a", "m", 50); b.transfer(9, "m", "z", 50)
assert b.top_spenders(10, 3) == ["a(50)", "m(50)", "z(50)"]
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 100); assert b.top_spenders(3, 99) == ["a(0)"]

b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 1000)
assert b.pay(3, "a", 100) == "payment1" and b.pay(4, "a", 50) == "payment2"
b = Bank(); assert b.pay(1, "ghost", 10) is None
b = Bank(); b.create_account(1, "a"); assert b.pay(2, "a", 10) is None
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 1000); pid = b.pay(3, "a", 100)
assert b.get_payment_status(4, "a", pid) == "IN_PROGRESS"
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 1000); pid = b.pay(10, "a", 100)
b.deposit(10 + DAY + 1, "a", 0)
assert b.get_payment_status(10 + DAY + 2, "a", pid) == "CASHBACK_RECEIVED"
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 1000); b.pay(3, "a", 500)
b.deposit(3 + DAY + 1, "a", 0)
assert b.deposit(3 + DAY + 2, "a", 0) == 510
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 500); pid = b.pay(4, "a", 100)
assert b.get_payment_status(5, "b", pid) is None
b = Bank(); b.create_account(1, "a"); assert b.get_payment_status(2, "a", "nope") is None

b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 100); b.deposit(4, "b", 50)
assert b.merge_accounts(5, "a", "b") is True and b.deposit(6, "b", 10) is None and b.deposit(7, "a", 0) == 150
b = Bank(); b.create_account(1, "a")
assert b.merge_accounts(2, "a", "ghost") is False and b.merge_accounts(3, "a", "a") is False
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "b", 1000); b.pay(4, "b", 200)
b.merge_accounts(5, "a", "b"); b.deposit(10 + DAY, "a", 0)
assert b.get_balance(11 + DAY, "a", 10 + DAY) == 800 + 4            # the cashback of b's payment reached a
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "b", 1000); pid = b.pay(4, "b", 100)
b.merge_accounts(5, "a", "b")
assert b.get_payment_status(6, "b", pid) is None and b.get_payment_status(7, "a", pid) == "IN_PROGRESS"
b = Bank(); b.create_account(1, "a"); b.deposit(2, "a", 100); b.deposit(3, "a", 50); b.deposit(4, "a", 25)
assert [b.get_balance(t, "a", at) for t, at in ((5, 2), (6, 3), (7, 4))] == [100, 150, 175]
assert b.get_balance(8, "a", 99) == 175      # time_at > timestamp: outside this page's guarantee, same answer here
b = Bank(); assert b.get_balance(1, "ghost", 1) is None
b = Bank(); b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 500); b.transfer(4, "a", "b", 200)
assert b.get_balance(5, "a", 3) == 500 and b.get_balance(6, "a", 4) == 300 and b.get_balance(7, "b", 4) == 200

# ---- edge cases beyond the source ----
b = Bank()                                   # a cashback due exactly at the merge lands on the old account first
b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "b", 1000)
assert b.pay(4, "b", 500) == "payment1"                             # cashback 10 at 4 + DAY
assert b.merge_accounts(4 + DAY, "a", "b") is True
assert b.get_balance(5 + DAY, "a", 4 + DAY) == 510                  # 500 + 10, all of it moved by the merge
assert b.get_balance(6 + DAY, "b", 3 + DAY) == 500                  # b one millisecond before: no cashback yet
assert b.get_balance(7 + DAY, "b", 4 + DAY) is None
assert b.get_payment_status(8 + DAY, "a", "payment1") == "CASHBACK_RECEIVED"

b = Bank()                                   # two merges in a row: the cashback follows the money twice
b.create_account(1, "a"); b.create_account(2, "b"); b.create_account(3, "c"); b.deposit(4, "c", 1000)
assert b.pay(5, "c", 250) == "payment1"                             # cashback 5
assert b.merge_accounts(6, "b", "c") is True
assert b.merge_accounts(7, "a", "b") is True
assert b.top_spenders(8, 9) == ["a(250)"]
assert b.deposit(5 + DAY, "a", 0) == 755
assert b.get_payment_status(6 + DAY, "a", "payment1") == "CASHBACK_RECEIVED"
assert b.get_balance(7 + DAY, "c", 5) == 750 and b.get_balance(8 + DAY, "b", 6) == 750
assert b.get_balance(9 + DAY, "b", 5) == 0 and b.get_balance(10 + DAY, "a", 6) == 0

b = Bank()                                   # a reused id does not inherit payments, cashback or history
b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "b", 400)
assert b.pay(4, "b", 400) == "payment1"                             # cashback 8
assert b.merge_accounts(5, "a", "b") is True
assert b.create_account(6, "b") is True
assert b.get_payment_status(7, "b", "payment1") is None
assert b.pay(8, "b", 0) == "payment2"                               # a payment of 0 is still a payment
assert b.top_spenders(9, 9) == ["a(400)", "b(0)"]
assert b.merge_accounts(10, "b", "a") is True                       # and now the other way round
assert b.get_payment_status(11, "b", "payment1") == "IN_PROGRESS"
assert b.deposit(4 + DAY, "b", 0) == 8
assert [b.get_balance(5 + DAY, "b", t) for t in (1, 2, 4, 5, 6, 9, 10, 3 + DAY, 4 + DAY)] == \
    [None, 0, 0, None, 0, 0, 0, 0, 8]
assert [b.get_balance(6 + DAY, "a", t) for t in (1, 4, 5, 9, 10)] == [0, 0, 0, 0, None]

b = Bank()                                   # the payment counter is shared and skips failed payments
b.create_account(1, "a"); b.create_account(2, "b"); b.deposit(3, "a", 10); b.deposit(4, "b", 10)
assert b.pay(5, "a", 11) is None and b.pay(6, "zzz", 1) is None
assert b.pay(7, "b", 10) == "payment1" and b.pay(8, "a", 10) == "payment2"
assert b.pay(9, "a", 1) is None


# ---- an independent model. It has no queue and no per-account history: it keeps every payment in one
#      list, and after every event it stores a full copy of all balances. get_balance reads the last copy
#      made at or before time_at. ----
class _SlowBank:
    def __init__(self):
        self.balance = {}            # account_id -> balance, for the accounts that exist now
        self.spent = {}              # account_id -> outgoing total
        self.opened = {}             # account_id -> when the account now using this id was created
        self.merged_away = set()     # ids that have been merged away at least once
        self.payments = []           # dicts: id, owner (an account_id), due, cashback, done, payer
        self.snapshots = []          # (time, {account_id: balance}), in time order
        self.seen = Counter()        # how often the awkward situations came up

    def call(self, name, timestamp, *args):
        for p in self.payments:                                  # payment order is also due order
            if not p["done"] and p["due"] <= timestamp:
                self.balance[p["owner"]] += p["cashback"]
                p["done"] = True
                self.snapshots.append((p["due"], dict(self.balance)))
                self.seen["cashback lands on another account"] += (p["owner"], self.opened[p["owner"]]) != p["payer"]
                self.seen["call exactly at a due time"] += p["due"] == timestamp
        self.seen["call names a merged-away id"] += any(a in self.merged_away and a not in self.balance
                                                        for a in args if isinstance(a, str))
        result = getattr(self, "_" + name)(timestamp, *args)
        self.snapshots.append((timestamp, dict(self.balance)))
        return result

    def _create_account(self, timestamp, account_id):
        if account_id in self.balance:
            return False
        self.balance[account_id] = 0
        self.spent[account_id] = 0
        self.opened[account_id] = timestamp
        return True

    def _deposit(self, timestamp, account_id, amount):
        if account_id not in self.balance:
            return None
        self.balance[account_id] += amount
        return self.balance[account_id]

    def _transfer(self, timestamp, source, target, amount):
        if source not in self.balance or target not in self.balance or source == target:
            return None
        if self.balance[source] < amount:
            return None
        self.balance[source] -= amount
        self.balance[target] += amount
        self.spent[source] += amount
        return self.balance[source]

    def _top_spenders(self, timestamp, n):
        order = sorted(self.balance)                                   # by id ...
        order.sort(key=lambda a: self.spent[a], reverse=True)          # ... then a stable sort by amount
        return ["%s(%d)" % (a, self.spent[a]) for a in order][:n]

    def _pay(self, timestamp, account_id, amount):
        if account_id not in self.balance or self.balance[account_id] < amount:
            return None
        self.balance[account_id] -= amount
        self.spent[account_id] += amount
        self.payments.append({"id": "payment%d" % (len(self.payments) + 1), "owner": account_id,
                              "due": timestamp + 24 * 60 * 60 * 1000, "cashback": amount // 50, "done": False,
                              "payer": (account_id, self.opened[account_id])})
        return self.payments[-1]["id"]

    def _get_payment_status(self, timestamp, account_id, payment):
        if account_id not in self.balance:
            return None
        for p in self.payments:
            if p["id"] == payment and p["owner"] == account_id:
                return "CASHBACK_RECEIVED" if p["done"] else "IN_PROGRESS"
        return None

    def _merge_accounts(self, timestamp, id1, id2):
        if id1 == id2 or id1 not in self.balance or id2 not in self.balance:
            return False
        self.balance[id1] += self.balance.pop(id2)
        self.spent[id1] += self.spent.pop(id2)
        for p in self.payments:
            if p["owner"] == id2:
                p["owner"] = id1
        self.merged_away.add(id2)
        return True

    def _get_balance(self, timestamp, account_id, time_at):
        answer = None
        for when, balances in self.snapshots:
            if when <= time_at:
                answer = balances.get(account_id)
        gone_now = account_id not in self.balance or time_at < self.opened[account_id]
        self.seen["past balance of an account that is gone"] += answer is not None and gone_now
        return answer


def _random_run(bank_class, seed, steps=120):
    """Drive bank_class and the model with the same random calls. Returns the model, or None on a mismatch."""
    rng = random.Random(seed)
    fast, slow = bank_class(), _SlowBank()
    ids = ["a", "b", "c", "d"]
    now, moments = 0, []
    jump = rng.choice([0.03, 0.08, 0.25])            # how eagerly this run skips ahead to the next cashback
    for _ in range(steps):
        dues = [p["due"] + d for p in slow.payments if not p["done"] for d in (-1, 0, 0, 1) if p["due"] + d > now]
        if dues and now < 9 * DAY and rng.random() < jump:
            now = rng.choice(dues)                   # just before, exactly on, or just after a cashback
        else:
            now += rng.randint(1, 3)
        moments.append(now)
        kind = rng.choice(["create", "create", "deposit", "deposit", "transfer", "transfer", "top", "pay",
                           "pay", "pay", "status", "merge", "merge", "balance", "balance", "balance"])
        x, y = rng.choice(ids), rng.choice(ids)
        if kind == "create":
            call = ("create_account", now, x)
        elif kind == "deposit":
            call = ("deposit", now, x, rng.choice([0, 30, 120, 500]))
        elif kind == "transfer":
            call = ("transfer", now, x, y, rng.choice([0, 20, 49, 130, 400]))
        elif kind == "top":
            call = ("top_spenders", now, rng.randint(0, 5))
        elif kind == "pay":
            call = ("pay", now, x, rng.choice([0, 49, 50, 99, 150, 333]))
        elif kind == "status":
            call = ("get_payment_status", now, x, "payment%d" % rng.randint(1, len(slow.payments) + 1))
        elif kind == "merge":
            call = ("merge_accounts", now, x, y)
        else:
            near = [t + d for t in moments + [p["due"] for p in slow.payments] for d in (-1, 0, 1)]
            call = ("get_balance", now, x, rng.choice([t for t in near if 1 <= t <= now]))
        if getattr(fast, call[0])(*call[1:]) != slow.call(*call):
            return None
    assert now <= 10 ** 9
    return slow


runs = [_random_run(Bank, seed) for seed in range(1000)]
assert all(run is not None for run in runs)
seen = sum((run.seen for run in runs), Counter())
# the random calls really reach the awkward situations, each of them hundreds of times
assert seen["cashback lands on another account"] > 300, seen
assert seen["call exactly at a due time"] > 300, seen
assert seen["call names a merged-away id"] > 300, seen
assert seen["past balance of an account that is gone"] > 300, seen


# ---- the comparison has teeth: each of these one-line mistakes is caught by the same random calls ----
class _LateCashback(Bank):                   # "<" instead of "<=": a cashback due now arrives one call late
    def _settle(self, timestamp):
        while self._pending and self._pending[0][0] < timestamp:
            due, account, cashback = self._pending.popleft()
            self._change(account, cashback, due)


class _BookedAtCall(Bank):                   # the cashback is booked at the call that noticed it, not at its due time
    def _settle(self, timestamp):
        while self._pending and self._pending[0][0] <= timestamp:
            due, account, cashback = self._pending.popleft()
            self._change(account, cashback, timestamp)


class _LostCashback(Bank):                   # the merge forgets to redirect the cashback still on its way
    def merge_accounts(self, timestamp, account_id_1, account_id_2):
        self._settle(timestamp)
        gone = self._accounts.get(account_id_2)
        waiting = [entry for entry in self._pending if entry[1] is gone]
        merged = super().merge_accounts(timestamp, account_id_1, account_id_2)
        for entry in waiting:
            entry[1] = gone
        return merged


class _LiveOnly(Bank):                       # get_balance answers only for accounts that exist now
    def get_balance(self, timestamp, account_id, time_at):
        if account_id not in self._accounts:
            self._settle(timestamp)
            return None
        return super().get_balance(timestamp, account_id, time_at)


for wrong in (_LateCashback, _BookedAtCall, _LostCashback, _LiveOnly):
    assert any(_random_run(wrong, seed) is None for seed in range(60)), wrong.__name__
```

</details>

</details>
