import bisect


class _Account:
    def __init__(self, created_at: int):
        self.balance = 0
        self.outgoing_total = 0
        self.events: list[tuple[int, int]] = [(created_at, 0)]   # (timestamp, balance) after each change


class Bank:
    TRANSFER_WINDOW = 1_000

    def __init__(self):
        self._live: dict[str, _Account] = {}
        self._transfers: dict[str, "_Transfer"] = {}
        self._holds: dict[str, list[str]] = {}      # account_id -> ids of transfers it is the source of
        self._incoming: dict[str, list[str]] = {}    # account_id -> ids of transfers it is the target of
        self._transfer_seq = 0

    def create_account(self, timestamp: int, account_id: str) -> bool:
        if account_id in self._live:
            return False
        self._live[account_id] = _Account(timestamp)   # NOTE: a fresh object -- see Level 4
        return True

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        account = self._live.get(account_id)
        return None if account is None else self._apply(account, timestamp, amount)

    def withdraw(self, timestamp: int, account_id: str, amount: int) -> int | None:
        account = self._live.get(account_id)
        if account is None or amount > self._available(account_id, timestamp):
            return None
        account.outgoing_total += amount
        return self._apply(account, timestamp, -amount)

    def _apply(self, account: _Account, timestamp: int, delta: int) -> int:
        account.balance += delta
        account.events.append((timestamp, account.balance))
        return account.balance

    def _available(self, account_id: str, timestamp: int) -> int:
        account = self._live[account_id]
        held = sum(self._transfers[tid].amount for tid in self._holds.get(account_id, [])
                   if not self._transfers[tid].accepted
                   and timestamp <= self._transfers[tid].created_at + self.TRANSFER_WINDOW)
        return account.balance - held   # NOTE: a hold never touches balance -- only what may be taken from it


def top_spenders(self, timestamp: int, n: int) -> list[str]:
    ranked = sorted(self._live.items(), key=lambda item: (-item[1].outgoing_total, item[0]))
    return [f"{account_id}({account.outgoing_total})" for account_id, account in ranked[:n]]


Bank.top_spenders = top_spenders


class _Transfer:
    def __init__(self, transfer_id: str, source_id: str, target_id: str, amount: int, created_at: int):
        self.id = transfer_id
        self.source_id = source_id
        self.target_id = target_id
        self.amount = amount
        self.created_at = created_at
        self.accepted = False


def transfer(self, timestamp: int, source_id: str, target_id: str, amount: int) -> str | None:
    if (source_id == target_id or source_id not in self._live or target_id not in self._live
            or amount > self._available(source_id, timestamp)):
        return None
    self._transfer_seq += 1
    transfer_id = f"transfer{self._transfer_seq}"
    self._transfers[transfer_id] = _Transfer(transfer_id, source_id, target_id, amount, timestamp)
    self._holds.setdefault(source_id, []).append(transfer_id)
    self._incoming.setdefault(target_id, []).append(transfer_id)
    return transfer_id


def accept_transfer(self, timestamp: int, account_id: str, transfer_id: str) -> bool:
    t = self._transfers.get(transfer_id)
    if (t is None or t.accepted or t.target_id != account_id
            or timestamp > t.created_at + self.TRANSFER_WINDOW):
        return False                     # NOTE: expiry is a plain comparison, re-checked on every call --
    t.accepted = True                    # nothing is swept or scheduled ahead of time
    self._apply(self._live[t.source_id], timestamp, -t.amount)
    self._apply(self._live[t.target_id], timestamp, t.amount)
    return True


Bank.transfer = transfer
Bank.accept_transfer = accept_transfer


def merge_accounts(self, timestamp: int, a: str, b: str) -> bool:
    if a == b or a not in self._live or b not in self._live:
        return False
    acc_a, acc_b = self._live[a], self._live[b]
    self._apply(acc_a, timestamp, acc_b.balance)
    acc_a.outgoing_total += acc_b.outgoing_total
    for tid in self._holds.pop(b, []):              # NOTE: both directions -- b as source ...
        self._transfers[tid].source_id = a
        self._holds.setdefault(a, []).append(tid)
    for tid in self._incoming.pop(b, []):            # ... and b as target
        self._transfers[tid].target_id = a
        self._incoming.setdefault(a, []).append(tid)
    del self._live[b]                                # NOTE: a later create_account(b) builds a fresh _Account --
    return True                                       # nothing here keeps that new object linked to this one


def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
    account = self._live.get(account_id)
    if account is None:
        return None
    i = bisect.bisect_right(account.events, time_at, key=lambda e: e[0]) - 1
    return account.events[i][1] if i >= 0 else None


Bank.merge_accounts = merge_accounts
Bank.get_balance = get_balance
