# expect: WA
# The transfer counter is advanced before the checks, so a rejected transfer uses up an id.
class Bank:
    WINDOW = 1000

    def __init__(self):
        self.accounts = {}      # account_id -> live account
        self.transfers = {}     # transfer_id -> transfer
        self.count = 0

    def _new_account(self, timestamp):
        return {"balance": 0, "outgoing": 0, "created": timestamp, "history": [(timestamp, 0)]}

    def _set_balance(self, account_id, timestamp, balance):
        acc = self.accounts[account_id]
        acc["balance"] = balance
        acc["history"].append((timestamp, balance))
        return balance

    def _pending(self, x, timestamp):
        return not x["done"] and x["created"] <= timestamp <= x["created"] + self.WINDOW

    def _available(self, account_id, timestamp):
        held = sum(x["amount"] for x in self.transfers.values()
                   if x["source"] == account_id and self._pending(x, timestamp))
        return self.accounts[account_id]["balance"] - held

    def create_account(self, timestamp, account_id):
        if account_id in self.accounts:
            return False
        self.accounts[account_id] = self._new_account(timestamp)
        return True

    def deposit(self, timestamp, account_id, amount):
        if account_id not in self.accounts:
            return None
        return self._set_balance(account_id, timestamp, self.accounts[account_id]["balance"] + amount)

    def withdraw(self, timestamp, account_id, amount):
        if account_id not in self.accounts or amount > self._available(account_id, timestamp):
            return None
        self.accounts[account_id]["outgoing"] += amount
        return self._set_balance(account_id, timestamp, self.accounts[account_id]["balance"] - amount)

    def top_spenders(self, timestamp, n):
        ranked = sorted(self.accounts.items(), key=lambda kv: (-kv[1]["outgoing"], kv[0]))
        return [f"{aid}({acc['outgoing']})" for aid, acc in ranked[:n]]

    def transfer(self, timestamp, source_id, target_id, amount):
        self.count += 1
        if source_id == target_id or source_id not in self.accounts or target_id not in self.accounts:
            return None
        if amount > self._available(source_id, timestamp):
            return None
        transfer_id = f"transfer{self.count}"
        self.transfers[transfer_id] = {"source": source_id, "target": target_id, "amount": amount,
                                       "created": timestamp, "done": False}
        return transfer_id

    def accept_transfer(self, timestamp, account_id, transfer_id):
        x = self.transfers.get(transfer_id)
        if x is None or x["target"] != account_id or not self._pending(x, timestamp):
            return False
        x["done"] = True
        self._set_balance(x["source"], timestamp, self.accounts[x["source"]]["balance"] - x["amount"])
        self._set_balance(x["target"], timestamp, self.accounts[x["target"]]["balance"] + x["amount"])
        return True

    def merge_accounts(self, timestamp, a, b):
        if a == b or a not in self.accounts or b not in self.accounts:
            return False
        acc_a, acc_b = self.accounts[a], self.accounts[b]
        self._set_balance(a, timestamp, acc_a["balance"] + acc_b["balance"])
        acc_a["outgoing"] += acc_b["outgoing"]
        for x in self.transfers.values():
            if self._pending(x, timestamp):
                if x["source"] == b:
                    x["source"] = a
                if x["target"] == b:
                    x["target"] = a
        del self.accounts[b]
        return True

    def get_balance(self, timestamp, account_id, time_at):
        acc = self.accounts.get(account_id)
        if acc is None or time_at < acc["created"]:
            return None
        balance = None
        for ts, bal in acc["history"]:
            if ts <= time_at:
                balance = bal
        return balance
