# expect: WA
# get_balance treats an account as existing at the merge moment too ([created_at, closed_at] instead of [created_at, closed_at)).
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


class Bank(Bank):
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        self._settle(timestamp)
        ranked = sorted(self._accounts.items(), key=lambda item: (-item[1].outgoing, item[0]))
        return [f"{account_id}({account.outgoing})" for account_id, account in ranked[:n]]


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
            if account.created_at <= time_at <= account.closed_at:
                return account.balances[bisect_right(account.times, time_at) - 1]
        return None
