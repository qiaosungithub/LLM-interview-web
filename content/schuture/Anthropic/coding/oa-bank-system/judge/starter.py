class Bank:
    def __init__(self) -> None:
        """Creates an empty bank: no accounts and no transfers yet."""

    def create_account(self, timestamp: int, account_id: str) -> bool:
        """Level 1: opens a new account with balance 0. Returns True, or False, without effect, if
        account_id is already live."""
        raise NotImplementedError

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Level 1: adds amount to account_id's balance. Returns the new balance, or None, without
        effect, if account_id is not live."""
        raise NotImplementedError

    def withdraw(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Level 1: subtracts amount from account_id's balance. Returns the new balance. Returns None,
        without effect, if account_id is not live, or if amount is greater than its balance."""
        raise NotImplementedError

    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        """Level 2: returns the n live accounts with the largest outgoing total, ranked highest first,
        ties by account_id ascending, each as f"{account_id}({outgoing_total})"."""
        raise NotImplementedError

    def transfer(self, timestamp: int, source_id: str, target_id: str, amount: int) -> str | None:
        """Level 3: starts a pending transfer of amount from source_id to target_id and returns its
        transfer_id ("transfer1", "transfer2", ...), or None, without effect, if it is not allowed."""
        raise NotImplementedError

    def accept_transfer(self, timestamp: int, account_id: str, transfer_id: str) -> bool:
        """Level 3: completes a pending transfer (only its target_id may); returns True on success,
        or False, without effect."""
        raise NotImplementedError

    def merge_accounts(self, timestamp: int, a: str, b: str) -> bool:
        """Level 4: merges b into a. Returns True, or False, without effect, if a equals b, or if a
        or b is not live."""
        raise NotImplementedError

    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        """Level 4: returns the balance that account_id's current generation had at time_at, or None
        if account_id is not live or time_at is before that generation's create_account call."""
        raise NotImplementedError
