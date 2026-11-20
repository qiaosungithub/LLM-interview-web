"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator. Every query re-scans what it needs; nothing is indexed or cached."""

WINDOW = 1_000


class _Generation:
    """One generation of an account id: from its create_account call until a merge ends it."""

    def __init__(self, created_at):
        self.created_at = created_at
        self.balance = 0
        self.outgoing = 0
        self.history = [(created_at, 0)]        # (timestamp, balance right after the change)


class _Transfer:
    def __init__(self, source, target, amount, created_at):
        self.source, self.target, self.amount, self.created_at = source, target, amount, created_at
        self.completed = False

    def pending(self, t):
        return not self.completed and self.created_at <= t <= self.created_at + WINDOW


class Bank:
    def __init__(self):
        self.live = {}              # account_id -> its live _Generation
        self.transfers = {}         # transfer_id -> _Transfer, in creation order

    def _set_balance(self, aid, t, value):
        g = self.live[aid]
        g.balance = value
        g.history.append((t, value))

    def available(self, aid, t):
        held = sum(x.amount for x in self.transfers.values() if x.source == aid and x.pending(t))
        return self.live[aid].balance - held

    # Level 1
    def create_account(self, t, aid):
        if aid in self.live:
            return False
        self.live[aid] = _Generation(t)
        return True

    def deposit(self, t, aid, amount):
        if aid not in self.live:
            return None
        self._set_balance(aid, t, self.live[aid].balance + amount)
        return self.live[aid].balance

    def withdraw(self, t, aid, amount):
        if aid not in self.live or amount > self.available(aid, t):
            return None
        self.live[aid].outgoing += amount
        self._set_balance(aid, t, self.live[aid].balance - amount)
        return self.live[aid].balance

    # Level 2
    def top_spenders(self, t, n):
        ranked = sorted(self.live, key=lambda aid: (-self.live[aid].outgoing, aid))
        return [f"{aid}({self.live[aid].outgoing})" for aid in ranked[:n]]

    # Level 3
    def transfer(self, t, source, target, amount):
        if source == target or source not in self.live or target not in self.live:
            return None
        if amount > self.available(source, t):
            return None
        tid = f"transfer{len(self.transfers) + 1}"
        self.transfers[tid] = _Transfer(source, target, amount, t)
        return tid

    def accept_transfer(self, t, aid, tid):
        x = self.transfers.get(tid)
        if x is None or x.completed or x.target != aid or not x.pending(t):
            return False
        x.completed = True
        self._set_balance(x.source, t, self.live[x.source].balance - x.amount)
        self._set_balance(x.target, t, self.live[x.target].balance + x.amount)
        return True

    # Level 4
    def merge_accounts(self, t, a, b):
        if a == b or a not in self.live or b not in self.live:
            return False
        ga, gb = self.live[a], self.live[b]
        self._set_balance(a, t, ga.balance + gb.balance)
        ga.outgoing += gb.outgoing
        for x in self.transfers.values():
            if x.pending(t):
                if x.source == b:
                    x.source = a
                if x.target == b:
                    x.target = a
        del self.live[b]
        return True

    def get_balance(self, t, aid, time_at):
        g = self.live.get(aid)
        if g is None or time_at < g.created_at:
            return None
        return [bal for ts, bal in g.history if ts <= time_at][-1]


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight; each level also uses every earlier level's operations
    1: {"create_account": 2, "deposit": 4, "withdraw": 5},
    2: {"create_account": 2, "deposit": 4, "withdraw": 5, "top_spenders": 3},
    3: {"create_account": 1, "deposit": 3, "withdraw": 4, "top_spenders": 1, "transfer": 5, "accept_transfer": 4},
    4: {"create_account": 3, "deposit": 3, "withdraw": 3, "top_spenders": 3, "transfer": 5, "accept_transfer": 4,
        "merge_accounts": 3, "get_balance": 5},
}


def random_ops(r, level, n_ops, ids):
    """A sequence of (method, args) calls for one Bank, drawn while a model follows along so that the
    draws hit the interesting cases often: exact available balances, window edges, real transfer ids,
    and the account whose holds just changed."""
    m = Bank()
    kinds, weights = zip(*KINDS[level].items())
    ops, t = [], r.randint(0, 3)
    focus = None        # an account worth poking next: its holds just changed, or it was just merged or re-created
    gone = None         # the id the last merge ended

    def pick(live_bias=0.85):
        if focus in m.live and r.random() < 0.4:
            return focus
        if m.live and r.random() < live_bias:
            return r.choice(sorted(m.live))
        return r.choice(ids)

    def amount_for(aid):
        if aid not in m.live or r.random() < 0.15:
            return r.randint(1, 100)
        avail = m.available(aid, t)
        if avail <= 0 or r.random() < 0.35:
            return avail + 1 if avail >= 0 else r.randint(1, 100)
        return avail if r.random() < 0.5 else r.randint(1, avail)

    for _ in range(n_ops):
        pending = [x for x in m.transfers.values() if x.pending(t)]
        if pending and r.random() < 0.15:
            x = r.choice(pending)
            t, focus = max(t, x.created_at + WINDOW + r.choice([0, 1])), x.source
        else:
            t += r.choice([0, 0, 0, 1, 1, 2, 5, 20, 100, 400])
        if len(m.live) < 2 and r.random() < 0.6:
            kind = "create_account"
        elif level == 1 or not m.live or max(g.balance for g in m.live.values()) > 0 or r.random() < 0.4:
            kind = r.choices(kinds, weights)[0]
        else:
            kind = "deposit"
        if kind == "create_account":
            aid = gone if gone is not None and gone not in m.live and r.random() < 0.5 else pick(0.2)
            args = (t, aid)
            if aid == gone and aid not in m.live:
                focus = aid     # a new generation of an id that a merge just ended
        elif kind == "deposit":
            args = (t, pick(), r.randint(1, 100))
        elif kind == "withdraw":
            aid = pick()
            args = (t, aid, amount_for(aid))
        elif kind == "top_spenders":
            args = (t, r.randint(1, len(ids) + 1))
        elif kind == "transfer":
            rich = [aid for aid in sorted(m.live) if m.available(aid, t) > 0]
            source = r.choice(rich) if rich and r.random() < 0.7 else pick()
            target = source if r.random() < 0.08 else pick()
            args = (t, source, target, amount_for(source))
        elif kind == "accept_transfer":
            pending = [k for k, x in m.transfers.items() if x.pending(t)]
            mine = [k for k in pending if m.transfers[k].target == focus]
            if mine and r.random() < 0.5:
                tid = r.choice(mine)
            elif pending and r.random() < 0.7:
                tid = r.choice(pending)
            elif m.transfers and r.random() < 0.85:
                tid = r.choice(list(m.transfers))
            else:
                tid = f"transfer{len(m.transfers) + 1}"
            x, who = m.transfers.get(tid), r.random()
            aid = (x.target if who < 0.75 else x.source if who < 0.87 else pick()) if x else pick()
            args = (t, aid, tid)
            if x and x.target == aid and x.pending(t):
                focus = x.source
        elif kind == "merge_accounts":
            pending = [x for x in m.transfers.values() if x.pending(t)]
            if pending and r.random() < 0.5:     # merge away one end of a pending transfer
                x = r.choice(pending)
                a, b = r.choice([(pick(0.95), x.target), (x.source, x.target), (pick(0.95), x.source),
                                 (x.target, x.source)])
            else:
                a = pick(0.95)
                b = a if r.random() < 0.08 else pick(0.95)
            args = (t, a, b)
            if a != b and a in m.live and b in m.live:
                focus, gone = a, b
        else:   # get_balance
            aid = pick()
            g = m.live.get(aid)
            choices = [r.randint(0, t), t]
            if g is not None:
                choices += [g.created_at, max(0, g.created_at - 1), r.randint(0, g.created_at),
                            r.choice(g.history)[0], r.choice(g.history)[0], t + r.randint(1, 5)]
            args = (t, aid, r.choice(choices))
        ops.append((kind, args))
        getattr(m, kind)(*args)
    return ops
