# expect: WA
# crawl_concurrent stops when the queue is empty and no worker is busy, but a worker counts itself busy only after q.get() returns: in that gap the crawl looks finished and returns early.
import asyncio
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.parse import urlparse


def _norm(u):
    return u.partition("#")[0]


def _children(links, host, seen):
    out = []
    for link in links:
        v = _norm(link)
        if v not in seen and urlparse(v).hostname == host:
            seen.add(v)
            out.append(v)
    return out


def _fetch(parser, url):
    try:
        return parser.get_urls(url)
    except Exception:
        return []


def crawl(start_url, html_parser):
    host = urlparse(start_url).hostname
    start = _norm(start_url)
    seen = {start}
    todo = [start]
    while todo:
        todo.extend(_children(_fetch(html_parser, todo.pop()), host, seen))
    return sorted(seen)


import threading
import time
from queue import Empty, Queue


def crawl_concurrent(start_url, html_parser, max_workers=8):
    host = urlparse(start_url).hostname
    start = _norm(start_url)
    seen = {start}
    lock = threading.Lock()
    q = Queue()
    q.put(start)
    busy = [0]
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                url = q.get(timeout=0.01)
            except Empty:
                continue
            with lock:
                busy[0] += 1
            try:
                for link in _fetch(html_parser, url):
                    v = _norm(link)
                    if urlparse(v).hostname != host:
                        continue
                    with lock:
                        if v in seen:
                            continue
                        seen.add(v)
                    q.put(v)
            finally:
                with lock:
                    busy[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(max_workers)]
    for t in threads:
        t.start()
    while True:
        with lock:
            if q.empty() and busy[0] == 0:
                break
        time.sleep(0.001)
    stop.set()
    for t in threads:
        t.join()
    return sorted(seen)


async def crawl_async(start_url, html_parser, concurrency=8):
    host = urlparse(start_url).hostname
    start = _norm(start_url)
    seen = {start}
    sem = asyncio.Semaphore(concurrency)

    async def fetch(url):
        async with sem:
            return await asyncio.to_thread(_fetch, html_parser, url)

    pending = {asyncio.create_task(fetch(start))}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            for v in _children(t.result(), host, seen):
                pending.add(asyncio.create_task(fetch(v)))
    return sorted(seen)
