"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator. It keeps no per-account history: after every event (a call, or a cashback
landing) it stores a full copy of every existing account's balance, and get_balance reads the last copy made
at or before time_at. Every query re-scans what it needs; nothing is indexed."""

DAY = 86_400_000


class _Account:
    def __init__(self, account_id, created_at):
        self.id = account_id
        self.created_at = created_at
        self.balance = 0
        self.outgoing = 0


class _Payment:
    def __init__(self, pid, owner, due, cashback):
        self.pid, self.owner, self.due, self.cashback = pid, owner, due, cashback
        self.landed = False


class Bank:
    def __init__(self):
        self.live = {}              # account_id -> the _Account that uses this id now
        self.payments = []          # every successful payment, in payment order (= due order)
        self.snapshots = []         # (time, {account_id: balance}) after every event, in time order
        self.events = []            # (time, account_id) of creations and merges, for the generator

    def settle(self, t):
        for p in self.payments:
            if not p.landed and p.due <= t:
                p.landed = True
                p.owner.balance += p.cashback
                self._snap(p.due)

    def _snap(self, t):
        self.snapshots.append((t, {aid: acc.balance for aid, acc in self.live.items()}))

    def _call(self, t, fn, *args):
        self.settle(t)
        out = fn(t, *args)
        self._snap(t)
        return out

    # Level 1
    def create_account(self, t, aid):
        return self._call(t, self._create_account, aid)

    def _create_account(self, t, aid):
        if aid in self.live:
            return False
        self.live[aid] = _Account(aid, t)
        self.events.append((t, aid))
        return True

    def deposit(self, t, aid, amount):
        return self._call(t, self._deposit, aid, amount)

    def _deposit(self, t, aid, amount):
        if aid not in self.live:
            return None
        self.live[aid].balance += amount
        return self.live[aid].balance

    def transfer(self, t, source, target, amount):
        return self._call(t, self._transfer, source, target, amount)

    def _transfer(self, t, source, target, amount):
        if source == target or source not in self.live or target not in self.live:
            return None
        if amount > self.live[source].balance:
            return None
        self.live[source].balance -= amount
        self.live[source].outgoing += amount
        self.live[target].balance += amount
        return self.live[source].balance

    # Level 2
    def top_spenders(self, t, n):
        return self._call(t, self._top_spenders, n)

    def _top_spenders(self, t, n):
        ranked = sorted(self.live, key=lambda aid: (-self.live[aid].outgoing, aid))
        return [f"{aid}({self.live[aid].outgoing})" for aid in ranked[:n]]

    # Level 3
    def pay(self, t, aid, amount):
        return self._call(t, self._pay, aid, amount)

    def _pay(self, t, aid, amount):
        if aid not in self.live or amount > self.live[aid].balance:
            return None
        acc = self.live[aid]
        acc.balance -= amount
        acc.outgoing += amount
        pid = f"payment{len(self.payments) + 1}"
        self.payments.append(_Payment(pid, acc, t + DAY, amount * 2 // 100))
        return pid

    def get_payment_status(self, t, aid, pid):
        return self._call(t, self._get_payment_status, aid, pid)

    def _get_payment_status(self, t, aid, pid):
        acc = self.live.get(aid)
        if acc is None:
            return None
        for p in self.payments:
            if p.pid == pid and p.owner is acc:
                return "CASHBACK_RECEIVED" if p.landed else "IN_PROGRESS"
        return None

    # Level 4
    def merge_accounts(self, t, a, b):
        return self._call(t, self._merge_accounts, a, b)

    def _merge_accounts(self, t, a, b):
        if a == b or a not in self.live or b not in self.live:
            return False
        keep, gone = self.live[a], self.live.pop(b)
        keep.balance += gone.balance
        keep.outgoing += gone.outgoing
        for p in self.payments:
            if p.owner is gone:
                p.owner = keep
        self.events.append((t, b))
        return True

    def get_balance(self, t, aid, time_at):
        return self._call(t, self._get_balance, aid, time_at)

    def _get_balance(self, t, aid, time_at):
        answer = None
        for when, balances in self.snapshots:
            if when > time_at:
                break
            answer = balances.get(aid)
        return answer


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight; each level also uses every earlier level's operations
    1: {"create_account": 2, "deposit": 3, "transfer": 6},
    2: {"create_account": 2, "deposit": 3, "transfer": 6, "top_spenders": 3},
    3: {"create_account": 1, "deposit": 3, "transfer": 4, "top_spenders": 2, "pay": 6,
        "get_payment_status": 4},
    4: {"create_account": 3, "deposit": 3, "transfer": 3, "top_spenders": 2, "pay": 5,
        "get_payment_status": 3, "merge_accounts": 4, "get_balance": 6},
}
PAY_AMOUNTS = [0, 1, 49, 50, 51, 99, 100, 149, 150, 199, 250, 333]


def random_ops(r, level, n_ops, ids):
    """A sequence of (method, args) calls for one Bank, drawn while a model follows along so that the
    draws hit the interesting cases often: amounts equal to the balance (cashback included), calls exactly
    on, just before and just after a cashback's due time, real payment ids, merges of an account whose
    cashback is still on its way, ids reopened after a merge, and time_at on the edges of each account's
    existence."""
    m = Bank()
    kinds, weights = zip(*KINDS[level].items())
    ops, t = [], r.randint(1, 4)
    gone = None         # the id the last merge ended

    def pick(live_bias=0.85):
        if m.live and r.random() < live_bias:
            return r.choice(sorted(m.live))
        return r.choice(ids)

    def amount_for(aid, choices=None):
        if aid not in m.live or r.random() < 0.1:
            return r.choice(choices) if choices else r.randint(0, 300)
        bal = m.live[aid].balance
        x = r.random()
        if x < 0.3:
            return bal
        if x < 0.45:
            return bal + 1
        if choices and x < 0.8:
            return r.choice(choices)
        return r.randint(0, max(bal, 1))

    for _ in range(n_ops):
        waiting = [p.due + d for p in m.payments if not p.landed for d in (-1, 0, 0, 1) if p.due + d > t]
        if waiting and t < 9 * DAY and r.random() < 0.2:
            t = r.choice(waiting)
        else:
            t += r.choice([1, 1, 1, 2, 3, 10])
        m.settle(t)         # the call below settles at t anyway; settling first lets amounts see the cashback
        if len(m.live) < 2 and r.random() < 0.6:
            kind = "create_account"
        else:
            kind = r.choices(kinds, weights)[0]
            if kind in ("transfer", "pay") and m.live and max(a.balance for a in m.live.values()) == 0 \
                    and r.random() < 0.7:
                kind = "deposit"
        if kind == "create_account":
            aid = gone if gone is not None and gone not in m.live and r.random() < 0.5 else pick(0.2)
            args = (t, aid)
        elif kind == "deposit":
            args = (t, pick(), r.choice([0, r.randint(1, 500), r.randint(1, 500)]))
        elif kind == "transfer":
            rich = [aid for aid in sorted(m.live) if m.live[aid].balance > 0]
            source = r.choice(rich) if rich and r.random() < 0.7 else pick()
            target = source if r.random() < 0.08 else pick()
            args = (t, source, target, amount_for(source))
        elif kind == "top_spenders":
            args = (t, r.randint(0, len(ids) + 1))
        elif kind == "pay":
            rich = [aid for aid in sorted(m.live) if m.live[aid].balance > 0]
            aid = r.choice(rich) if rich and r.random() < 0.7 else pick()
            args = (t, aid, amount_for(aid, PAY_AMOUNTS))
        elif kind == "get_payment_status":
            if m.payments and r.random() < 0.85:
                p = r.choice(m.payments)
                owner_id = next((aid for aid, acc in m.live.items() if acc is p.owner), None)
                aid = owner_id if owner_id is not None and r.random() < 0.7 else pick()
                pid = p.pid
            else:
                aid, pid = pick(), f"payment{len(m.payments) + r.choice([0, 1, 2])}"
            args = (t, aid, pid)
        elif kind == "merge_accounts":
            owners = [aid for aid, acc in sorted(m.live.items())
                      if any(p.owner is acc and not p.landed for p in m.payments)]
            b = r.choice(owners) if owners and r.random() < 0.6 else pick(0.95)
            a = b if r.random() < 0.06 else pick(0.95)
            args = (t, a, b)
            if a != b and a in m.live and b in m.live:
                gone = b
        else:   # get_balance
            aid = gone if gone is not None and r.random() < 0.3 else pick(0.6)
            near = [t, r.randint(1, t)]
            near += [w + d for w, x in m.events if x == aid for d in (-1, 0, 1)]
            near += [p.due + d for p in m.payments if p.landed for d in (-1, 0)]
            near += [w for w, _ in m.snapshots[-6:]]
            args = (t, aid, r.choice([x for x in near if 1 <= x <= t]))
        ops.append((kind, args))
        getattr(m, kind)(*args)
    assert t <= 10 ** 9
    return ops
