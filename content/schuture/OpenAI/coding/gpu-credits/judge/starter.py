class GPUCreditLedger:
    """Part 1: add_credit, subtract and get_balance are called in non-decreasing order of timestamp."""

    def __init__(self):
        """Creates an empty ledger."""

    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None:
        """Issues a grant of amount credits, valid on [timestamp, timestamp + expiration]; it repays debt first."""
        raise NotImplementedError

    def subtract(self, amount: int, timestamp: int) -> None:
        """Spends amount at timestamp from the valid grants, soonest expiry first; any shortfall becomes debt."""
        raise NotImplementedError

    def get_balance(self, timestamp: int) -> int | None:
        """Remaining credit of the grants valid at timestamp minus outstanding debt; None if that is negative."""
        raise NotImplementedError


class GPUCreditLedgerAnyOrder:
    """Part 2: add_credit and subtract may be called in any order; get_balance replays them by timestamp."""

    def __init__(self):
        """Creates an empty ledger."""

    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None:
        """Issues a grant of amount credits, valid on [timestamp, timestamp + expiration]; it repays debt first."""
        raise NotImplementedError

    def subtract(self, amount: int, timestamp: int) -> None:
        """Spends amount at timestamp from the valid grants, soonest expiry first; any shortfall becomes debt."""
        raise NotImplementedError

    def get_balance(self, timestamp: int) -> int | None:
        """The balance after replaying, in timestamp order, every call so far with timestamp <= the query."""
        raise NotImplementedError


class GPUCreditLedgerFast:
    """Part 3: same behaviour as GPUCreditLedgerAnyOrder; O(U log U + Q) for non-decreasing queries."""

    def __init__(self):
        """Creates an empty ledger."""

    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None:
        """Issues a grant of amount credits, valid on [timestamp, timestamp + expiration]; it repays debt first."""
        raise NotImplementedError

    def subtract(self, amount: int, timestamp: int) -> None:
        """Spends amount at timestamp from the valid grants, soonest expiry first; any shortfall becomes debt."""
        raise NotImplementedError

    def get_balance(self, timestamp: int) -> int | None:
        """The balance after replaying, in timestamp order, every call so far with timestamp <= the query."""
        raise NotImplementedError
