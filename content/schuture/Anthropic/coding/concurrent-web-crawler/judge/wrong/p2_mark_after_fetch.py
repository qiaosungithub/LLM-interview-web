# expect: WA
# A page is marked visited only after its fetch returns, so a second copy taken from the queue while the first fetch is running is fetched again.
from typing import Protocol
from urllib.parse import urlparse
from collections import deque
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor


class HtmlParser(Protocol):
    def get_urls(self, url: str) -> list[str]: ...


def normalize(url: str) -> str:
    return url.split("#", 1)[0]        # NOTE: split on the FIRST '#' -- a fragment cannot contain one itself


def same_host(url: str, host: str) -> bool:
    return urlparse(url).hostname == host


def crawl(start_url: str, html_parser: HtmlParser) -> list[str]:
    host = urlparse(start_url).hostname
    start = normalize(start_url)
    visited = {start}
    queue = deque([start])
    while queue:
        url = queue.popleft()
        try:
            links = html_parser.get_urls(url)
        except Exception:
            continue            # NOTE: `url` is already in `visited` -- it stays in the result, it just
                                  #  contributes no further links
        for link in links:
            norm = normalize(link)
            if same_host(norm, host) and norm not in visited:
                visited.add(norm)
                queue.append(norm)
    return sorted(visited)


def crawl_concurrent(start_url: str, html_parser: HtmlParser, max_workers: int = 8) -> list[str]:
    host = urlparse(start_url).hostname
    start = normalize(start_url)

    visited: set[str] = set()
    visited_lock = threading.Lock()
    frontier: Queue = Queue()
    frontier.put(start)

    def worker() -> None:
        while True:
            url = frontier.get()
            if url is None:
                frontier.task_done()
                return
            with visited_lock:
                done = url in visited
            if done:
                frontier.task_done()
                continue
            try:
                links = html_parser.get_urls(url)
            except Exception:
                links = []
            with visited_lock:
                visited.add(url)
            for link in links:
                norm = normalize(link)
                if not same_host(norm, host):
                    continue
                with visited_lock:
                    if norm in visited:
                        continue
                frontier.put(norm)
            frontier.task_done()

    threads = [threading.Thread(target=worker) for _ in range(max_workers)]
    for t in threads:
        t.start()
    frontier.join()
    for _ in threads:
        frontier.put(None)
    for t in threads:
        t.join()
    return sorted(visited)


async def crawl_async(start_url: str, html_parser: HtmlParser, concurrency: int = 8) -> list[str]:
    host = urlparse(start_url).hostname
    start = normalize(start_url)
    visited = {start}
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)

    async def fetch(url: str) -> list[str]:
        async with semaphore:                           # NOTE: caps calls actually running, not tasks created
            try:
                return await loop.run_in_executor(executor, html_parser.get_urls, url)
            except Exception:
                return []                                 # NOTE: `url` stays in `visited`; see Part 1

    async def visit(url: str) -> None:
        links = await fetch(url)
        children = []
        for link in links:
            norm = normalize(link)
            if not same_host(norm, host):
                continue
            if norm not in visited:                     # NOTE: no `await` between this check and the add
                visited.add(norm)                          #  below -- see the paragraph above
                children.append(asyncio.create_task(visit(norm)))
        if children:
            await asyncio.gather(*children)

    try:
        await visit(start)
    finally:
        executor.shutdown(wait=True)                     # NOTE: every pool the page opens is shut down
    return sorted(visited)
