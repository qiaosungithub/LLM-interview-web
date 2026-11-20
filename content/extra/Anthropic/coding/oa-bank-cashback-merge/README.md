# Online Assessment: Bank System with Instant Transfers, Cashback and Merges

## Problem

Implement the four levels below in order: the tests for a level run only after every test for the previous
level passes. Each level adds methods to the same `Bank` class; you do not start a new class.

This is a different set of levels for the "bank system" type of problem, not the same set as
[Online Assessment: Bank System](#/anthropic/coding/oa-bank-system): here transfers land immediately, Level 3 is
payments with delayed cashback, and Level 4 answers historical balances by "did this account exist at that
moment". Don't mix the rules of the two pages; when working on this page, go only by this page's statement.

General conventions:

- The first parameter of every method is `timestamp`: an integer in milliseconds, in the range $1$ to $10^9$.
  On a given instance, each call's `timestamp` is greater than the previous one (strictly increasing, never
  repeated).
- `account_id` is a non-empty string. `amount` is a non-negative integer and may be $0$.
- At any moment, an `account_id` corresponds to at most one existing account. Accounts are created by
  `create_account`; only Level 4's `merge_accounts` can make an account disappear.
- A failed call (one that returns `None` or `False`) changes no state.

### Level 1 — Opening accounts, deposits and transfers

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

Example:

```text
create_account(1, "alice")             # -> True
create_account(2, "alice")             # -> False   the account already exists
create_account(3, "bob")               # -> True
deposit(4, "carol", 500)               # -> None    no such account
deposit(5, "alice", 1200)              # -> 1200
transfer(6, "alice", "bob", 1201)      # -> None    alice only has 1200
transfer(7, "alice", "bob", 1200)      # -> 0       sending the entire balance is allowed; the source's new balance is returned
transfer(8, "bob", "bob", 10)          # -> None    can't transfer to yourself
transfer(9, "bob", "carol", 10)        # -> None    the target account doesn't exist
deposit(10, "bob", 0)                  # -> 1200    depositing 0 is valid; the result is the current balance
```

### Level 2 — Top spenders

Each account has an *outgoing total*: the sum of the amounts it has successfully transferred out; from Level 3
on, the amounts it has successfully paid are added too. Money received doesn't count, and neither does Level
3's cashback. A new account's outgoing total is $0$.

```py
class Bank:
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        """Returns the n accounts with the largest outgoing totals, largest first. Accounts with
        equal totals are ordered by account_id ascending (ordinary string comparison). Each
        entry is the string "<account_id>(<outgoing total>)", e.g. "alice(450)". If fewer than
        n accounts exist, returns all of them in the same order. n is a non-negative integer;
        n == 0, or a bank with no accounts, gives []."""
```

Example:

```text
create_account(1, "carol"); create_account(2, "bob"); create_account(3, "alice")
deposit(4, "carol", 2000)              # -> 2000
deposit(5, "bob", 3000)                # -> 3000
top_spenders(6, 3)                     # -> ["alice(0)", "bob(0)", "carol(0)"]        all 0: by id ascending
transfer(7, "carol", "bob", 500)       # -> 1500
transfer(8, "carol", "alice", 1000)    # -> 500
transfer(9, "alice", "bob", 1000)      # -> 0
top_spenders(10, 2)                    # -> ["carol(1500)", "alice(1000)"]
top_spenders(11, 5)                    # -> ["carol(1500)", "alice(1000)", "bob(0)"]  there are only 3 accounts
top_spenders(12, 0)                    # -> []
```

### Level 3 — Payments and cashback

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

Payment rules:

- The payment amount counts toward the account's outgoing total, and `top_spenders` must include it.
- Every successful payment earns *cashback*: 2% of `amount`, rounded down, i.e. `amount * 2 // 100`. The
  cashback is added to the account's balance at time `timestamp + 86_400_000` (24 hours after the payment).
- Cashback that has come due is applied before any operation at that moment. That is, as soon as a call's
  `timestamp` is greater than or equal to a cashback's due time, that cashback is already in the balance before
  the call starts; this includes the case where `timestamp` equals the due time exactly.
- Cashback is not spending and does not reduce any account's outgoing total.
- When the cashback amount is $0$ (`amount` less than 50), the status still changes to `"CASHBACK_RECEIVED"`
  once it comes due.

Example (`DAY = 86_400_000`):

```text
create_account(1, "alice"); create_account(2, "bob")
deposit(3, "alice", 1000)                          # -> 1000
pay(4, "alice", 1001)                              # -> None          insufficient balance; the counter doesn't advance
pay(5, "alice", 450)                               # -> "payment1"    balance 550; cashback 450*2//100 = 9, due at 5+DAY
deposit(6, "bob", 80)                              # -> 80
pay(7, "bob", 49)                                  # -> "payment2"    balance 31; cashback 49*2//100 = 0
get_payment_status(8, "alice", "payment1")         # -> "IN_PROGRESS"
get_payment_status(9, "bob", "payment1")           # -> None          payment1 belongs to alice
get_payment_status(10, "alice", "payment7")        # -> None          no such payment
get_payment_status(11, "carol", "payment1")        # -> None          no such account
top_spenders(12, 2)                                # -> ["alice(450)", "bob(49)"]
deposit(4 + DAY, "alice", 0)                       # -> 550           1 millisecond early, the cashback hasn't landed
get_payment_status(5 + DAY, "alice", "payment1")   # -> "CASHBACK_RECEIVED"   exactly due: credit first, then answer
deposit(6 + DAY, "alice", 0)                       # -> 559
get_payment_status(8 + DAY, "bob", "payment2")     # -> "CASHBACK_RECEIVED"   cashback is 0, the status changes anyway
top_spenders(9 + DAY, 2)                           # -> ["alice(450)", "bob(49)"]   cashback doesn't change outgoing totals
```

### Level 4 — Merging accounts and historical balances

```py
class Bank:
    def merge_accounts(self, timestamp: int, account_id_1: str, account_id_2: str) -> bool:
        """Merges account_id_2 into account_id_1. Returns False if the two ids are equal or if
        either account does not exist; otherwise applies the rules below and returns True."""

    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        """Returns the balance that the account named account_id had at time_at, or None if no
        account with that id existed at time_at. 1 <= time_at <= timestamp."""
```

When `merge_accounts` succeeds:

- Account 2's balance is added to account 1's balance; account 2's outgoing total is added to account 1's
  outgoing total.
- Account 2's cashback that hasn't landed yet still lands at its scheduled time, but into account 1.
- All payments made by account 2 now belong to account 1: calling `get_payment_status` with `account_id_1`
  finds them, and the status rules are unchanged.
- Account 2 disappears from the system. From then on, `deposit`, `transfer`, `pay`, `get_payment_status` and
  `merge_accounts` with `account_id_2` are all treated as "the account doesn't exist", and it no longer appears
  in `top_spenders`. `create_account` may open a brand-new account with that id again: balance $0$, outgoing
  total $0$, and no payments under it.

Rules for `get_balance`:

- An account exists from the timestamp at which it was created (inclusive) until the timestamp at which it was
  merged away (exclusive); an account that was never merged away exists from then on.
- What matters is the moment `time_at`, not now. If an account named `account_id` existed at `time_at`, return
  that account's balance at that time, even if it has since been merged away; if none existed at that moment,
  return `None`.
- "The balance at that time" means: every operation with a timestamp no later than `time_at` has taken effect,
  and every cashback with a due time no later than `time_at` has been credited, even if no call happened at the
  moment it landed.
- Merging doesn't rewrite the past: account 1's balance at every moment before the merge is still its own
  balance at that time; only from the moment of the merge does it include account 2's money.
- If an id is merged away and then opened again, those are two different accounts, each with its own lifetime;
  the gap between the two returns `None`.

Example (`DAY = 86_400_000`):

```text
create_account(1, "alice"); create_account(2, "bob")
deposit(3, "alice", 1000)                          # -> 1000
deposit(4, "bob", 800)                             # -> 800
pay(5, "bob", 300)                                 # -> "payment1"   bob's balance 500; cashback 6, due at 5+DAY
transfer(6, "alice", "bob", 100)                   # -> 900          bob's balance 600
merge_accounts(7, "alice", "alice")                # -> False        can't merge with yourself
merge_accounts(8, "alice", "carol")                # -> False        carol doesn't exist
merge_accounts(9, "alice", "bob")                  # -> True         alice's balance 900 + 600 = 1500
top_spenders(10, 5)                                # -> ["alice(400)"]   100 (transfer) + 300 (bob's payment)
get_payment_status(11, "alice", "payment1")        # -> "IN_PROGRESS"    bob's payment now belongs to alice
get_payment_status(12, "bob", "payment1")          # -> None             bob no longer exists
deposit(13, "bob", 50)                             # -> None
get_balance(14, "alice", 8)                        # -> 900          before the merge: alice's own balance then
get_balance(15, "alice", 9)                        # -> 1500         from the moment of the merge
get_balance(16, "bob", 8)                          # -> 600          at time 8, bob still exists
get_balance(17, "bob", 9)                          # -> None         at time 9, bob has been merged away
get_balance(18, "bob", 1)                          # -> None         bob was only opened at time 2
create_account(20, "bob")                          # -> True         same id, brand-new account
get_balance(21, "bob", 20)                         # -> 0
get_balance(22, "bob", 15)                         # -> None         no account named bob at time 15
get_balance(23, "bob", 5)                          # -> 500          the old bob's balance at time 5
deposit(6 + DAY, "alice", 0)                       # -> 1506         the cashback for bob's payment went to alice
get_balance(7 + DAY, "alice", 4 + DAY)             # -> 1500
get_balance(8 + DAY, "alice", 5 + DAY)             # -> 1506         cashback is booked at its due time, even though no call happened then
get_balance(9 + DAY, "bob", 5 + DAY)               # -> 0            the new bob didn't get the cashback
get_payment_status(10 + DAY, "alice", "payment1")  # -> "CASHBACK_RECEIVED"
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Three things are worth confirming up front, because reasonable designs can differ on them: whether cashback
that has come due is credited before or after the operations at that moment (here: before, and a `timestamp`
exactly equal to the due time counts as due); whether `get_balance` asks if the account existed at `time_at` or
exists now (here: at that moment); and whether a merge rewrites the past (here: no, the surviving account's
historical balances before the merge are unchanged, and the merged-away account's balances before the merge can
still be queried). In the real assessment, go by the platform's examples; the other two readings are covered in
the follow-ups.

### Time budget

Four levels in 90 minutes, light at the start and heavy at the end. A workable budget, with the rest kept in
reserve:

| Level | Budget | Main work |
| --- | --- | --- |
| Level 1 | under 10 minutes | Three methods, four or five lines each |
| Level 2 | under 10 minutes | One sort |
| Level 3 | about 20 minutes | The cashback queue, and "every method first settles the cashback that has come due" |
| Level 4 | 30 to 35 minutes | What a merge has to move, and balances queried by time |

A few practices that don't depend on the specific problem:

- Before starting each level, spend two or three minutes reading that level's signatures and examples from top
  to bottom. Points in this kind of problem are mostly lost not because you can't write the code but because of
  wrong return values: here `transfer` returns the source's new balance and returns `None` on failure, not
  `False`; `top_spenders` returns a list of `"id(amount)"` strings.
- Run the platform's tests after each method you write; don't save them up until the whole level is done.
- You can't see the later levels in advance, but the signatures give hints: in Level 1 every method takes a
  `timestamp` it doesn't use, so later levels most likely have rules where "something happens automatically
  at a certain time". So from Level 1 on, the code below routes balance changes through a helper `_change`,
  and makes the first line of every method a call to `_settle(timestamp)`, which is empty for now. With those
  two extra lines, Levels 3 and 4 don't require going back to edit earlier methods.
- An unfinished last level still earns points: the [interview process](#/anthropic/process) page records
  candidates advancing with 750/1000 and 800/1000. In Level 4, `merge_accounts` is easier than `get_balance`,
  so write it first; the test cases that use merges without querying historical balances can then pass.
- Don't chase optimal complexity. Code whose correctness you can see at a glance matters more than saving a log
  factor.

### Level 1

An account is a small object, and `Bank` holds only one table of "accounts that exist now". `_Account` below
is written in its final shape after all four levels, with each field labeled by the level that first uses it;
in the assessment you add a field only when you reach that level. Likewise, the line in `transfer` that records
the outgoing total is added in Level 2, and the two history lines in `_change` are added in Level 4. This
arrangement keeps each later level's code block down to just its new methods.

`transfer`'s four failure conditions are written in one `if`. `source is target` checks for "the same
account", which holds when the two ids are equal and the account exists; the case of equal ids for an account
that doesn't exist has already been caught by the earlier `is None`.

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

All three methods are $O(1)$.

### Level 2

Sort the existing accounts once. The sort key is `(-outgoing total, account_id)`: the amount is negated, so
larger amounts come first; the id is not negated, so ties are broken by id ascending. `ranked[:n]` is
naturally correct when `n` is $0$ or larger than the number of accounts, with no special case needed.

```python
class Bank(Bank):
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        self._settle(timestamp)
        ranked = sorted(self._accounts.items(), key=lambda item: (-item[1].outgoing, item[0]))
        return [f"{account_id}({account.outgoing})" for account_id, account in ranked[:n]]
```

Reopening `Bank` by defining it as a subclass of itself is just how this page shows the code level by level;
in the assessment, add the methods directly to the original class. With $k$ accounts the cost is
$O(k \log k)$.

### Level 3

Cashback is "something that happens automatically at some future moment". No timer is needed: when no call is
happening, nobody can see the balance either, so as long as **at the start of every method, all cashback due no
later than the current `timestamp` is credited first**, from the outside it is indistinguishable from crediting
it on time. That is `_settle`, and it is why Level 1 left that line at the start of every method. If Level 1
didn't leave it, this level has to add one line to each of the four earlier methods, and missing any one of
them makes cashback arrive late in that method.

Cashback waiting to be credited goes into a double-ended queue. Payment timestamps are strictly increasing and
the delay is the same constant, so due times are increasing too: an earlier payment always lands earlier, the
queue is naturally sorted by due time, and you only ever look at the front. The condition is `<=`, which
corresponds to "when exactly due, the cashback is credited first".

Payment status doesn't need to be stored separately. Each payment records only its due time, and a query
compares it with the current `timestamp`: `_settle` has already made "the due time is no later than now" and
"the cashback has been credited" the same thing.

The cashback amount is computed with integers: `amount * 2 // 100`.

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

Each cashback is enqueued once and dequeued once, so the cost of `_settle` amortizes to $O(1)$ per payment;
apart from that, `pay` and `get_payment_status` are both $O(1)$.

`_settle` books the cashback at the moment `due`, not at the `timestamp` of the call that triggered the
settlement. At this level both versions return the same results; the difference only shows up in Level 4 when
balances are queried by time, so get it right here once.

### Level 4

`merge_accounts` has to move four things; write them one by one against the list in the statement: the balance
(through `_change`, booked at the moment of the merge), the outgoing total, the payment records, and the
cashback still in the queue. Each queue entry stores the account **object** rather than the id, so you just
repoint the entries that point to account 2 at account 1. Using the object rather than the id has a practical
benefit: account 2's id can later be used to open a new account, and if the queue stored id strings, the old
account's cashback would wrongly go into the new account.

A merged-away account can't simply be thrown away, because `get_balance` may still ask about its balance before
the merge. Instead, record the moment it disappeared in `closed_at`, and store it by id in `_closed`. Then
`get_balance` is two steps: first, among "the accounts this id has ever referred to", find the one whose
lifetime `[created_at, closed_at)` contains `time_at`; then binary-search its balance history for the last
record whose time is no later than `time_at`.

The balance history is two parallel lists `times` and `balances`, appended to by `_change`. Its times are
non-decreasing: ordinary operations have increasing timestamps; cashback is credited in `_settle` in due-time
order from earliest to latest, always later than the previous call and no later than the current one. There
can be two records at the same moment (a cashback and an operation that happens exactly at its due time);
`bisect_right` picks the later one, which is exactly the balance "after everything at that moment is done".

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

`merge_accounts` costs $O(q + p)$: $q$ is the number of cashbacks in the queue that haven't landed yet, and
$p$ is the number of payments under account 2. `get_balance` is $O(g + \log e)$: $g$ is the number of accounts
this id has referred to over time, and $e$ is the number of balance changes of the account found. The
`_settle(timestamp)` at the start of `get_balance` can't be skipped: the statement guarantees `time_at` is no
later than `timestamp`, so after settling, every cashback due no later than `time_at` has definitely been
written into the history.

### Follow-ups

- **Why not add the cashback to the balance at payment time and use a flag to mark it as "not usable yet"?**
  Because the balance itself is visible: `deposit`'s return value, `pay`'s balance check and `get_balance` all
  read it. Adding it early means subtracting it again everywhere the balance is read, which is easier to get
  wrong than deferring the credit.
- **What if the cashback delay isn't constant (say, it varies by merchant)?** Due times then no longer increase
  with payment order, so the queue has to become a min-heap ordered by due time (`heapq`), and `_settle` keeps
  popping the top of the heap; everything else stays the same. Several cashbacks landing at the same moment
  also need a defined order, for example by payment number.
- **What other readings of `get_balance` are there?** "Inheriting the balance history" and "return `None` if it
  doesn't exist" can be understood in several ways, and the reference implementations of the two public
  practice sets differ from each other. One returns `None` if the account doesn't exist now (as does the
  site's [other set of bank levels](#/anthropic/coding/oa-bank-system)), and it mixes the two accounts' balance
  snapshots together by time, so a query for a moment before the merge returns whichever of the two sides'
  snapshots is latest. The other merges the two accounts' transaction streams, so a query for a moment before
  the merge gives the sum of the two accounts' balances; a query for the merged-away id is redirected to the
  surviving account. The disagreements mostly show up when asking about a moment before the merge or about the
  merged-away id, so in the assessment, first look for such a line in the examples. This page's reading only
  needs to keep the merged-away account around, without merging histories; switching to either reading above
  touches only `merge_accounts` and `get_balance`.
- **When merges are frequent and many cashbacks are queued, what if the loop in `merge_accounts` is too
  slow?** Don't modify the queue; instead record on the merged-away account a pointer to "who it was merged
  into", and when settling, follow the pointers to the final account (the union-find approach, with path
  compression along the way). The merge itself then becomes $O(1)$ plus moving the payment records.
- **What if `time_at` were allowed to be later than `timestamp`?** That is asking about the future: you would
  have to include cashback in the queue due no later than `time_at`, without actually crediting it. The
  statement rules this case out here.
- **`top_spenders` with a huge number of accounts and a small `n`?** Use `heapq.nsmallest(n, ..., key=...)`,
  which brings the cost down from $O(k \log k)$ to $O(k \log n)$. The statement says nothing about performance,
  so this isn't necessary in the assessment.

<details>
<summary>Checks (runnable)</summary>

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
