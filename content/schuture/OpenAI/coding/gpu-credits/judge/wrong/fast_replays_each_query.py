# expect: TLE
# Part 3 is Part 2 again: every get_balance sorts and replays the whole log (correct but too slow).
import bisect
import heapq


class _LedgerState:
    """The ledger right after replaying some events in increasing order of timestamp."""

    def __init__(self):
        self.active = []    # heap of [expire_at, remaining]; the earliest-expiring grant is on top
        self.total = 0      # sum of remaining over self.active
        self.debt = 0

    def _drop_expired(self, ts):
        while self.active and self.active[0][0] < ts:     # NOTE: '<', not '<=' -- expiration is inclusive
            self.total -= heapq.heappop(self.active)[1]

    def apply(self, ts, event):
        """Applies one event as of its own timestamp ts."""
        if event[0] == "add":
            _, _credit_id, amount, expire_at = event
            paid = min(self.debt, amount)     # NOTE: a new grant repays outstanding debt before it can be spent
            self.debt -= paid
            if amount > paid:
                heapq.heappush(self.active, [expire_at, amount - paid])
                self.total += amount - paid
            return
        _, amount = event
        self._drop_expired(ts)
        while amount > 0 and self.active:
            grant = self.active[0]
            take = min(amount, grant[1])
            grant[1] -= take
            self.total -= take
            amount -= take
            if grant[1] == 0:
                heapq.heappop(self.active)
        self.debt += amount           # NOTE: whatever no valid grant could cover becomes debt

    def balance(self, ts):
        self._drop_expired(ts)
        net = self.total - self.debt      # NOTE: O(1); summing the heap here would cost O(k) on every query
        return None if net < 0 else net


class GPUCreditLedger:
    def __init__(self):
        self._state = _LedgerState()

    def add_credit(self, credit_id, amount, timestamp, expiration):
        self._state.apply(timestamp, ("add", credit_id, amount, timestamp + expiration))

    def subtract(self, amount, timestamp):
        self._state.apply(timestamp, ("sub", amount))

    def get_balance(self, timestamp):
        return self._state.balance(timestamp)

class GPUCreditLedgerAnyOrder:
    def __init__(self):
        self._events = []   # append-only, call order -- NOT necessarily sorted by timestamp

    def add_credit(self, credit_id, amount, timestamp, expiration):
        self._events.append((timestamp, ("add", credit_id, amount, timestamp + expiration)))

    def subtract(self, amount, timestamp):
        self._events.append((timestamp, ("sub", amount)))

    def get_balance(self, timestamp):
        state = _LedgerState()
        for ts, event in sorted(e for e in self._events if e[0] <= timestamp):   # NOTE: sorted by
            state.apply(ts, event)                                              # timestamp, not call order
        return state.balance(timestamp)



class GPUCreditLedgerFast(GPUCreditLedgerAnyOrder):
    pass
