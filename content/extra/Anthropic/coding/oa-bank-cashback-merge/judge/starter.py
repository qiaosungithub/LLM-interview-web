class Bank:
    def __init__(self) -> None:
        """Creates an empty bank: no accounts and no payments yet."""

    def create_account(self, timestamp: int, account_id: str) -> bool:
        """Level 1: opens an account with balance 0. Returns True, or False if an account with this id
        already exists."""
        raise NotImplementedError

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Level 1: adds amount to the account's balance and returns the new balance. Returns None if the
        account does not exist."""
        raise NotImplementedError

    def transfer(self, timestamp: int, source_account_id: str, target_account_id: str,
                 amount: int) -> int | None:
        """Level 1: moves amount from source to target immediately and returns the SOURCE account's new
        balance; None if either account does not exist, the ids are the same, or amount > source balance."""
        raise NotImplementedError

    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        """Level 2: the n accounts with the largest outgoing totals, largest first, ties by account_id
        ascending, each as "<account_id>(<outgoing total>)"."""
        raise NotImplementedError

    def pay(self, timestamp: int, account_id: str, amount: int) -> str | None:
        """Level 3: takes amount out of the account as a payment and returns its id ("payment1", ...);
        None if the account does not exist or amount is greater than its balance."""
        raise NotImplementedError

    def get_payment_status(self, timestamp: int, account_id: str, payment: str) -> str | None:
        """Level 3: "IN_PROGRESS" until the payment's cashback is credited, then "CASHBACK_RECEIVED";
        None if the account does not exist or the payment is not this account's."""
        raise NotImplementedError

    def merge_accounts(self, timestamp: int, account_id_1: str, account_id_2: str) -> bool:
        """Level 4: merges account_id_2 into account_id_1. Returns False if the two ids are equal or if
        either account does not exist; otherwise True."""
        raise NotImplementedError

    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        """Level 4: the balance that the account named account_id had at time_at, or None if no account
        with that id existed at time_at."""
        raise NotImplementedError
