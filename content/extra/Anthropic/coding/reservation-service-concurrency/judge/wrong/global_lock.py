# expect: WA
# Correct results, but every resource shares one lock, so booking room A waits for a slow booking of room B.
import bisect
import threading
from contextlib import contextmanager
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


class ConcurrentReservationService:
    """Part 2: one lock per resource; a small inner lock for the id counter and the id index."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}
        big = threading.Lock()
        self._locks = {name: big for name in self._cal}                 # one lock shared by every resource
        self._bookings = {}
        self._next_id = 1
        self._meta = threading.Lock()        # guards _bookings and _next_id; always the innermost lock

    def reserve(self, resource, start, end, owner):
        lock = self._locks[resource]                     # KeyError before any lock is taken
        check_interval(start, end)
        with lock:                                       # check and insert are one atomic step
            if not is_free(self._cal[resource], start, end):
                return None
            return self._commit(owner, ((resource, start, end),))

    def _commit(self, owner, legs):
        """Caller holds the lock of every resource in legs."""
        with self._meta:
            booking = Booking(self._next_id, owner, legs)
            self._next_id += 1
            self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locks[booking.legs[0][0]]:
            if not booking.active:                       # another cancel got in between the lookup and the lock
                return False
            self._uncommit(booking)
        return True

    def _uncommit(self, booking):
        """Caller holds the lock of every resource in booking.legs."""
        booking.active = False
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)
        with self._meta:
            del self._bookings[booking.id]

    def query(self, resource, start, end):
        lock = self._locks[resource]
        check_interval(start, end)
        with lock:
            return overlapping(self._cal[resource], start, end)      # a slice: a private copy of immutable tuples


class AtomicReservationService(ConcurrentReservationService):
    """Part 3: all-or-nothing bookings across several resources, locks taken in one global order."""

    def _lock_order(self, resources):
        return sorted(set(resources))[:1]    # every name maps to the same lock: take it once

    @contextmanager
    def _locked(self, resources):
        held = []
        try:
            for name in self._lock_order(resources):
                self._locks[name].acquire()
                held.append(name)
            yield
        finally:
            for name in reversed(held):
                self._locks[name].release()

    def reserve_all(self, requests, owner):
        legs = tuple((resource, start, end) for resource, start, end in requests)
        if not legs:
            raise ValueError("empty request")
        for resource, start, end in legs:                # validate everything before taking any lock
            self._locks[resource]
            check_interval(start, end)
        with self._locked([resource for resource, _, _ in legs]):
            if not self._all_free(legs):
                return None                              # nothing has been written yet: no rollback needed
            return self._commit(owner, legs)

    def _all_free(self, legs):
        if not all(is_free(self._cal[resource], start, end) for resource, start, end in legs):
            return False
        ordered = sorted(legs)                           # legs of one request must not hit each other either
        return all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(ordered, ordered[1:]))

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locked([resource for resource, _, _ in booking.legs]):
            if not booking.active:
                return False
            self._uncommit(booking)
        return True


ReservationService = AtomicReservationService
