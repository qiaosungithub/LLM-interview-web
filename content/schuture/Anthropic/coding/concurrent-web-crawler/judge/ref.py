"""Helpers for tests.py.

- expected_crawl: the crawl straight from the statement's definition (strip the fragment at the first '#',
  compare urlparse(...).hostname), a plain search over a fixed link graph.
- FakeParser: the HtmlParser the tests hand to the answer. There is no network; the graph is a dict. It counts
  every get_urls call per normalized URL, the number of calls running at once (and the peak), and notes calls
  made on a thread that is running an asyncio event loop (a blocking call that stalls the loop).
- Gates that hold get_urls calls inside the fake, so that what the tests look for happens on every run instead
  of once in a while:
    Crowd     holds fetches of some pages until more than `cap` calls are running (the cap is broken: release at
              once) or a short wait ends. A correct answer starts as many calls as it is going to while the first
              ones are held, so the peak it reaches is its real concurrency, and an answer that ignores the cap is
              caught every time.
    Trap      holds the first fetch of one page until a second fetch of the same page arrives (fetched twice) or
              a short wait ends: a duplicate claim made while the page is still being fetched always lands.
    Lockstep  makes the fetches of a group of pages return together (a barrier with a timeout), so several
              threads walk the same list of new links at the same moment.
    Slow      makes some pages slow (a plain sleep).
    SlowLinks makes the list some pages return pause when it is first walked: the caller stops between the end
              of the fetch and handling its links (where a frontier.task_done() called too early opens a gap).
    Jitter    random short sleeps, seeded, for the stress rounds.
- slow_queue_get: while active, queue.Queue.get (and get_nowait) sleeps a moment after taking an item and
  before handing it back, as if the thread were descheduled right there (always legal). Meanwhile the URL is in
  no queue and the thread that took it has not run a line of the answer yet, so a crawl that decides it is done
  from "the queue is empty and no worker is busy" returns early; Queue.join()/task_done(), or any count of URLs
  claimed and not yet finished, is not fooled.
- call_with_timeout: run the answer in a helper thread and give up after a few seconds, so that a
  crawl that never returns is reported with an explanation instead of hanging the judge.
"""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlparse


def normalize(url: str) -> str:
    return url.split("#", 1)[0]


def host_of(url: str):
    return urlparse(url).hostname


def expected_crawl(start_url: str, graph: dict, fail=()) -> list[str]:
    """Every normalized in-scope URL reachable from start_url; a failing page contributes no links."""
    host = host_of(start_url)
    start = normalize(start_url)
    seen, stack = {start}, [start]
    while stack:
        url = stack.pop()
        if url in fail:
            continue
        for link in graph.get(url, ()):
            v = normalize(link)
            if host_of(v) == host and v not in seen:
                seen.add(v)
                stack.append(v)
    return sorted(seen)


# ---------------------------------------------------------------- the fake parser

def fail_map(fail) -> dict:
    return dict(fail) if isinstance(fail, dict) else {u: RuntimeError for u in fail}


class FakeParser:
    """HtmlParser over {normalized url: [links]}. Pages in `fail` raise (fail: {url: exception class}, or urls that
    raise RuntimeError); pages not in the graph have no links."""

    def __init__(self, graph, fail=(), gates=()):
        self.graph, self.fail, self.gates = graph, fail_map(fail), list(gates)
        self.cond = threading.Condition()
        self.calls = {}             # normalized url -> number of get_urls calls
        self.in_flight = 0
        self.peak = 0
        self.on_loop = []           # urls fetched on a thread that runs an event loop

    def get_urls(self, url):
        key = normalize(url) if isinstance(url, str) else url
        on_loop = asyncio._get_running_loop() is not None
        with self.cond:
            self.calls[key] = self.calls.get(key, 0) + 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            if on_loop:
                self.on_loop.append(key)
            self.cond.notify_all()
            for g in self.gates:
                g.hold(self, key)
        try:
            for g in self.gates:
                g.pause(self, key)
            if key in self.fail:
                raise self.fail[key](f"模拟的页面加载失败：{key}")
            links = list(self.graph.get(key, ()))
            for g in self.gates:
                links = g.links(self, key, links)
            return links
        finally:
            with self.cond:
                self.in_flight -= 1
                self.cond.notify_all()


class Gate:
    def hold(self, parser, key):       # called with parser.cond held; may wait on it
        pass

    def pause(self, parser, key):      # called without any lock
        pass

    def links(self, parser, key, links):   # may hand back a different list object
        return links


class Crowd(Gate):
    """Fetches of `urls` wait (one batch, at most `wait` seconds after the first) until more than `cap` calls are
    running. `over` says the cap was broken."""

    def __init__(self, urls, cap, wait=0.4):
        self.urls, self.cap, self.wait = set(urls), cap, wait
        self.deadline = None
        self.over = False

    def hold(self, parser, key):
        if key not in self.urls:
            return
        if parser.in_flight > self.cap:
            self.over = True
            parser.cond.notify_all()
            return
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now + self.wait
        while not self.over and parser.in_flight <= self.cap and now < self.deadline:
            parser.cond.wait(self.deadline - now)
            now = time.monotonic()
        if parser.in_flight > self.cap:
            self.over = True


class Trap(Gate):
    """The first fetch of `url` waits until a second fetch of it arrives, at most `wait` seconds."""

    def __init__(self, url, wait=0.4):
        self.url, self.wait, self.arrived = url, wait, 0

    def hold(self, parser, key):
        if key != self.url:
            return
        self.arrived += 1
        if self.arrived > 1:
            parser.cond.notify_all()
            return
        end = time.monotonic() + self.wait
        while self.arrived < 2 and time.monotonic() < end:
            parser.cond.wait(end - time.monotonic())


class Lockstep(Gate):
    """Fetches of the pages of each group return together (all of the group, or after `wait` seconds)."""

    def __init__(self, groups, wait=0.3):
        self.where = {}
        for urls in groups:
            barrier = threading.Barrier(len(urls))
            for u in urls:
                self.where[u] = barrier
        self.wait = wait
        self.filled = 0

    def pause(self, parser, key):
        b = self.where.get(key)
        if b is None or b.broken:
            return
        try:
            if b.wait(self.wait) == 0:
                self.filled += 1
        except threading.BrokenBarrierError:
            pass


class Slow(Gate):
    def __init__(self, urls, seconds):
        self.urls, self.seconds = set(urls), seconds

    def pause(self, parser, key):
        if key in self.urls:
            time.sleep(self.seconds)


class SlowList(list):
    """A list whose iteration starts only after a short sleep (in whichever thread iterates it)."""

    def __init__(self, items, delay):
        super().__init__(items)
        self.delay = delay

    def __iter__(self):
        time.sleep(self.delay)
        return super().__iter__()


class SlowLinks(Gate):
    """get_urls for `urls` returns a SlowList: the caller pauses when it starts walking the links, i.e. after the
    fetch has returned and before it handles the first link."""

    def __init__(self, urls, delay):
        self.urls, self.delay = set(urls), delay

    def links(self, parser, key, links):
        return SlowList(links, self.delay) if key in self.urls else links


class Jitter(Gate):
    """Random short sleeps; r is a judgelib.rng() generator (draws are serialized by a lock)."""

    def __init__(self, r, p=0.4, longest=1e-3):
        self.r, self.p, self.longest, self.lock = r, p, longest, threading.Lock()

    def pause(self, parser, key):
        with self.lock:
            x = self.r.random()
        if x < self.p:
            time.sleep(0.0 if x < self.p / 3 else self.longest * x / self.p)


# ---------------------------------------------------------------- running the answer

class Stuck(Exception):
    pass


def call_with_timeout(fn, wait=4.0):
    """fn() in a daemon helper thread; raises Stuck if it has not returned after `wait` seconds. An exception
    raised by fn is raised again here, with the answer's own traceback."""
    box, done = {}, threading.Event()

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:      # noqa: BLE001 - handed to the caller below
            box["error"] = e
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(wait):
        raise Stuck()
    if "error" in box:
        raise box["error"]
    return box["value"]


@contextmanager
def fast_switching(interval=1e-5):
    """Ask the interpreter to switch threads every `interval` seconds (default 5 ms): many more interleavings."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


@contextmanager
def slow_queue_get(delay):
    """See the module docstring."""
    orig = queue.Queue.get

    def get(self, block=True, timeout=None):
        item = orig(self, block, timeout)
        time.sleep(delay)
        return item
    queue.Queue.get = get
    try:
        yield
    finally:
        queue.Queue.get = orig


# ---------------------------------------------------------------- graphs

def star(host, n, scheme="http"):
    root = f"{scheme}://{host}/root"
    leaves = [f"{scheme}://{host}/leaf{i}" for i in range(n)]
    return root, {root: list(leaves)}, leaves


def diamond(host="d.test", width=2):
    """start -> A0..A{width-1}; every A links to the same page B."""
    start = f"http://{host}/start"
    a = [f"http://{host}/a{i}" for i in range(width)]
    b = f"http://{host}/b"
    graph = {start: list(a), b: [start]}
    for u in a:
        graph[u] = [b, start]
    return start, graph, a, b


def chain(host, n):
    pages = [f"http://{host}/c{i}" for i in range(n)]
    graph = {pages[i]: [pages[i + 1]] for i in range(n - 1)}
    return pages[0], graph, pages


def layered(r, host, width, fanout, layers):
    """start -> width pages (group 0); each of them links to the same `fanout` new pages in the same (random)
    order, so that threads released together walk the same list side by side; one of those pages links to the
    next group of `width` pages, and so on. Returns start, graph, groups."""
    start = f"http://{host}/s"
    graph, groups, prev = {}, [], start
    for L in range(layers):
        group = [f"http://{host}/g{L}/{i}" for i in range(width)]
        nxt = [f"http://{host}/n{L}/{j}" for j in range(fanout)]
        graph[prev] = graph.get(prev, []) + group
        links = list(nxt) + [start]
        r.shuffle(links)
        for u in group:
            graph[u] = links + [u]
        groups.append(group)
        prev = nxt[r.randrange(fanout)]
    return start, graph, groups


def random_graph(r, n_pages=None):
    """A random graph around host a.test: other schemes, ports, userinfo and letter case of the same host (in
    scope), subdomains and other hosts (out of scope), fragments ('#', '#x', '#x#y'), duplicate links, self-links,
    failing pages. Returns start_url (sometimes with a fragment), graph, fail."""
    n = n_pages or r.randint(3, 14)
    paths = [f"/p{i}" for i in range(n)] + ["/", "/p0/", "/p1?q=1", "/p1?"]
    origins = ["http://a.test", "http://a.test", "http://a.test", "https://a.test", "http://a.test:8080",
               "http://u@a.test", "http://A.Test", "HTTP://a.test", "http://shop.a.test", "http://b.test",
               "http://a.test.b.test"]
    pages = sorted({o + p for o in origins for p in paths if r.random() < 0.35} | {"http://a.test/p0"})
    graph, fail = {}, set()
    for u in pages:
        links = []
        for _ in range(r.randint(0, 5)):
            v = r.choice(pages)
            x = r.random()
            if x < 0.15:
                v += "#"
            elif x < 0.35:
                v += f"#f{r.randint(0, 2)}"
            elif x < 0.40:
                v += "#a#b"
            links.append(v)
        if r.random() < 0.2:
            links.append(u)
        if links and r.random() < 0.2:
            links.append(links[0])
        graph[u] = links
        if r.random() < 0.12:
            fail.add(u)
    start = r.choice([p for p in pages if host_of(p) == "a.test"])
    if r.random() < 0.15:
        start += "#top"
    return start, graph, fail
