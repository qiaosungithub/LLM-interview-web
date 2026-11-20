# expect: WA
# The Part 1 code used from many threads: check-then-act with no lock, so two threads book the same slot.
import bisect
from dataclasses import dataclass


@dataclass
class Booking:
    id: int
    owner: str
    legs: tuple                  # ((resource, start, end), ...)
    active: bool = True          # guarded by the locks of the resources in legs


def check_interval(start, end):
    if start >= end:
        raise ValueError(f"empty interval [{start}, {end})")


def is_free(cal, start, end):
    """cal: sorted, pairwise disjoint [(start, end, booking_id, owner)]. True iff [start, end) hits none."""
    i = bisect.bisect_left(cal, (end,))          # cal[:i] are exactly the intervals that start before `end`
    return i == 0 or cal[i - 1][1] <= start      # disjoint + sorted: the last of them also ends last


def overlapping(cal, start, end):
    """The intervals of cal that overlap [start, end), as a new list (still sorted by start)."""
    lo = bisect.bisect_right(cal, start, key=lambda item: item[1])   # first interval that ends after `start`
    hi = bisect.bisect_left(cal, (end,))                              # first interval that starts at/after `end`
    return cal[lo:hi]


def remove_leg(cal, start, end, booking_id):
    i = bisect.bisect_left(cal, (start, end, booking_id))
    assert cal[i][:3] == (start, end, booking_id)
    del cal[i]


class ReservationService:
    """Part 1: correct for one caller at a time. No locks."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}     # resource -> sorted [(start, end, booking_id, owner)]
        self._bookings = {}                              # booking_id -> Booking, active ones only
        self._next_id = 1

    def reserve(self, resource, start, end, owner):
        cal = self._cal[resource]                        # KeyError for an unknown resource
        check_interval(start, end)
        if not is_free(cal, start, end):                 # check ...
            return None
        return self._commit(owner, ((resource, start, end),))    # ... then act

    def _commit(self, owner, legs):
        booking = Booking(self._next_id, owner, legs)
        self._next_id += 1
        self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        booking = self._bookings.get(booking_id)
        if booking is None:                              # check ...
            return False
        self._uncommit(booking)                          # ... then act
        return True

    def _uncommit(self, booking):
        del self._bookings[booking.id]
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)

    def query(self, resource, start, end):
        cal = self._cal[resource]
        check_interval(start, end)
        return overlapping(cal, start, end)

    def reserve_all(self, requests, owner):
        legs = tuple((resource, start, end) for resource, start, end in requests)
        if not legs:
            raise ValueError("empty request")
        for resource, start, end in legs:
            self._cal[resource]
            check_interval(start, end)
        if not all(is_free(self._cal[resource], start, end) for resource, start, end in legs):
            return None
        ordered = sorted(legs)
        if not all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(ordered, ordered[1:])):
            return None
        return self._commit(owner, legs)
