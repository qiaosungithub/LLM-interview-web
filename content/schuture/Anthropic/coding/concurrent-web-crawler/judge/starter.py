from typing import Protocol


class HtmlParser(Protocol):
    def get_urls(self, url: str) -> list[str]:
        """Returns the absolute URLs of every link found on the page at `url`."""
        ...


def crawl(start_url: str, html_parser: HtmlParser) -> list[str]:
    """Part 1: every normalized in-scope URL reachable from start_url, fetched once each, sorted."""
    raise NotImplementedError


def crawl_concurrent(start_url: str, html_parser: HtmlParser, max_workers: int = 8) -> list[str]:
    """Part 2: same contract as crawl, overlapping get_urls with at most max_workers calls running at once."""
    raise NotImplementedError


async def crawl_async(start_url: str, html_parser: HtmlParser, concurrency: int = 8) -> list[str]:
    """Part 3: same contract as crawl, with asyncio and at most concurrency get_urls calls in flight."""
    raise NotImplementedError
