class ReservationService:
    def __init__(self, resources: list[str]):
        """Part 1: a service over a fixed list of resource names."""

    def reserve(self, resource: str, start: int, end: int, owner: str) -> int | None:
        """Book [start, end) on `resource`. Returns the new booking id,
        or None if it overlaps a booking that is already there."""
        raise NotImplementedError

    def cancel(self, booking_id: int) -> bool:
        """True if the booking existed and is now cancelled.
        False if the id was never issued or is already cancelled."""
        raise NotImplementedError

    def query(self, resource: str, start: int, end: int) -> list[tuple[int, int, int, str]]:
        """The bookings on `resource` that overlap [start, end),
        as (start, end, booking_id, owner), sorted by start."""
        raise NotImplementedError

    def reserve_all(self, requests: list[tuple[str, int, int]], owner: str) -> int | None:
        """Part 3: book every (resource, start, end) in `requests`, or none of them.
        One booking id covers all of them."""
        raise NotImplementedError
