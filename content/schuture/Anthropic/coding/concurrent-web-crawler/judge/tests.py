import asyncio
import threading

from judgelib import calling, case, expect, expect_true, info, rng
import ref
from ref import Crowd, FakeParser, Jitter, Lockstep, Slow, SlowLinks, Stuck, Trap

WAIT = 3.0          # seconds a crawl may take before it counts as stuck; the reference needs well under 1 s here
CROWD_WAIT = 0.5    # how long the first fetches of a crowd are held (see ref.Crowd)
GET_PAUSE = 0.05    # how long queue.Queue.get pauses after taking an item, in p2_take_gap (see ref.slow_queue_get)

PARSER_SRC = (
    "class Parser:  # 测试用的 HtmlParser：按 graph 返回链接，fail 里的页面抛出对应的异常\n"
    "    def get_urls(self, url):\n"
    "        url = url.split('#', 1)[0]\n"
    "        if url in fail:\n"
    "            raise fail[url](url)\n"
    "        return list(graph.get(url, []))"
)

EXAMPLE_START = "http://a.com/index"
EXAMPLE_GRAPH = {
    "http://a.com/index": ["http://a.com/about#team", "http://b.com/x", "http://a.com/index", "https://a.com/index"],
    "http://a.com/about": ["http://a.com/index#top", "http://a.com/products"],
    "https://a.com/index": ["http://a.com/products"],
    "http://a.com/products": ["http://a.com/products"],
}
EXAMPLE_WANT = ["http://a.com/about", "http://a.com/index", "http://a.com/products", "https://a.com/index"]


# ---------------------------------------------------------------- showing a call

def graph_src(graph):
    if not graph:
        return "graph = {}"
    lines = [f"    {u!r}: {links!r}," for u, links in graph.items()]
    if sum(map(len, lines)) > 2500:
        return f"graph = ...  # 随机生成的 {len(graph)} 个页面的链接图，太长，这里不显示"
    return "graph = {\n" + "\n".join(lines) + "\n}"


def fail_src(fail):
    fail = ref.fail_map(fail)
    return "fail = {" + ", ".join(f"{u!r}: {e.__name__}" for u, e in sorted(fail.items())) + "}"


def repro(fn, start, graph, fail, kw=None, note=""):
    lines = ["import asyncio"] if fn == "crawl_async" else []
    if note:
        lines.append(f"# {note}")
    lines += [graph_src(graph), fail_src(fail), PARSER_SRC]
    call = f"{fn}({start!r}, Parser()" + (f", {kw[0]}={kw[1]}" if kw else "") + ")"
    lines.append(f"asyncio.run({call})" if fn == "crawl_async" else call)
    return "\n".join(lines)


# ---------------------------------------------------------------- checking a result

def fail_here(msg, call, got=None):
    expect_true(False, msg, call=call, got=got)


def verify(got, parser, start, graph, fail, call, cap=None, cap_name=None, why=""):
    """The result is crawl's result, every crawled page was fetched exactly once, nothing else was fetched, and
    (Part 2/3) the number of get_urls calls running at once never went over the cap. `why` explains a wrong list."""
    want = ref.expected_crawl(start, graph, fail)
    if isinstance(got, tuple):
        got = list(got)
    if why and got != want:
        expect(got, want, call=call, msg=f"返回的列表和 crawl 在这张图上的结果不同。{why}")
    expect(got, want, call=call)
    calls = dict(parser.calls)
    total = sum(calls.values())
    for u, n in sorted(calls.items(), key=lambda kv: str(kv[0])):
        if n > 1:
            fail_here(f"get_urls 对同一个页面 {u!r} 调用了 {n} 次（一共调用了 {total} 次）：每个规范化后的 URL 最多只能抓取一次", call)
    for u in want:
        if u not in calls:
            fail_here(f"{u!r} 在返回的列表里，但从来没有为它调用过 get_urls：每个被爬取的页面都要恰好抓取一次", call)
    for u in sorted(calls, key=str):
        if u not in want:
            fail_here(f"调用了 get_urls({u!r})，但这个页面不该被抓取：它不属于爬取范围，或者从 start_url 出发不可达", call)
    if cap is not None and parser.peak > cap:
        fail_here(f"同一时刻有 {parser.peak} 个 get_urls 调用在运行，超过了 {cap_name}={cap}", call)
    if parser.on_loop:
        fail_here(f"get_urls({parser.on_loop[0]!r}) 是在运行事件循环的线程里直接调用的：它是阻塞调用，会在调用期间卡住"
                  "整个事件循环。要把它交给别的线程去跑（例如 loop.run_in_executor 或 asyncio.to_thread）", call)


def overlapped(parser, n, cap, cap_name, call):
    """With n pages that could all be fetched at once and a cap >= 2, at least two fetches must overlap."""
    info(f"同一时刻最多有 {parser.peak} 个 get_urls 在运行（{cap_name}={cap}）")
    if parser.peak < 2:
        fail_here(f"{n} 个页面可以同时抓取、{cap_name}={cap}，但 get_urls 同一时刻最多只有 {parser.peak} 个在运行："
                  f"题目要求同时进行多个 get_urls 调用来重叠延迟，而不是一次只抓取一个页面（测试让这些页面的 "
                  f"get_urls 先停 {CROWD_WAIT:g} 秒，期间你的代码一直没有开始抓取别的页面）", call)


# ---------------------------------------------------------------- running the answer

def room(call):
    n = threading.active_count()
    if n > 200:
        fail_here(f"前面的调用返回以后，还有 {n} 个线程在运行：评测沙箱最多只允许 256 个线程，再这样下去就无法创建新线程了。"
                  "返回之前要让自己启动的线程都退出", call)


def stuck(fn, wait, fail, hint=""):
    msg = (f"{fn} 过了 {wait:g} 秒还没有返回（测试里的 get_urls 都很快返回，正常爬完这张图用不了 1 秒）："
           "可能死锁了，或者在等一个永远不会满足的条件。")
    if fail:
        msg += "注意：这张图里有 get_urls 会抛出异常的页面。"
    return msg + hint


def run_p1(sol, start, graph, fail=()):
    parser = FakeParser(graph, fail)
    call = calling(repro("crawl", start, graph, fail))
    got = sol.crawl(start, parser)
    verify(got, parser, start, graph, fail, call)
    return parser


def run_p2(sol, start, graph, fail=(), gates=(), k=4, hint="", note="", why=""):
    parser = FakeParser(graph, fail, gates)
    call = calling(repro("crawl_concurrent", start, graph, fail, ("max_workers", k), note=note))
    room(call)
    try:
        got = ref.call_with_timeout(lambda: sol.crawl_concurrent(start, parser, max_workers=k), WAIT)
    except Stuck:
        fail_here(stuck("crawl_concurrent", WAIT, fail, hint), call)
    verify(got, parser, start, graph, fail, call, cap=k, cap_name="max_workers", why=why)
    return parser, call


def run_p3(sol, start, graph, fail=(), gates=(), k=4, hint=""):
    parser = FakeParser(graph, fail, gates)
    call = calling(repro("crawl_async", start, graph, fail, ("concurrency", k)))
    room(call)
    try:
        got = ref.call_with_timeout(lambda: asyncio.run(sol.crawl_async(start, parser, concurrency=k)), WAIT)
    except Stuck:
        fail_here(stuck("crawl_async", WAIT, fail, hint), call)
    verify(got, parser, start, graph, fail, call, cap=k, cap_name="concurrency")
    return parser, call


# ---------------------------------------------------------------- fixed graphs for the edge cases

SAME_HOST_START = "http://a.test/"
SAME_HOST = {
    "http://a.test/": ["https://a.test/", "http://a.test:8080/", "http://a.test:80/", "http://u@a.test/me",
                       "http://A.Test/Up", "HTTP://a.test/scheme", "https://a.test:8443/s"],
    "https://a.test/": ["http://a.test/", "https://a.test/#"],
    "http://a.test:8080/": ["http://a.test:8080/next", "http://a.test:8080/"],
    "http://A.Test/Up": ["http://a.test/Up"],
}

OTHER_HOSTS_START = "http://a.test/home"
OTHER_HOSTS = {
    "http://a.test/home": ["http://shop.a.test/", "http://a.test.evil.test/", "http://evil.test/?next=http://a.test/x",
                           "http://evil.test/a.test", "http://a.test@evil.test/", "http://xa.test/", "https://b.test/",
                           "http://a.test/ok"],
    "http://shop.a.test/": ["http://a.test/via-shop"],
    "http://a.test.evil.test/": ["http://a.test/via-suffix"],
    "http://a.test@evil.test/": ["http://a.test/via-userinfo"],
    "http://xa.test/": ["http://a.test/via-xa"],
    "https://b.test/": ["http://a.test/via-b"],
    "http://a.test/ok": [],
}

FRAGMENTS_START = "http://a.test/s#top"
FRAGMENTS = {
    "http://a.test/s": ["http://a.test/x#", "http://a.test/x#a#b", "http://a.test/y#sec", "http://a.test/s#again",
                        "http://a.test/q?#f", "HTTP://a.test/U#f", "http://a.test/p?a=1#x", "http://a.test/s"],
    "http://a.test/x": ["http://a.test/x#self", "http://a.test/x#a#b#c"],
    "http://a.test/q?": ["http://a.test/q"],
    "HTTP://a.test/U": [],
}

FAIL_GRAPH = {   # the page's own example: b fails, so e (reachable only through b) is never found
    "http://x.com/a": ["http://x.com/b", "http://x.com/c"],
    "http://x.com/b": ["http://x.com/d", "http://x.com/e"],
    "http://x.com/c": ["http://x.com/d"],
    "http://x.com/d": [],
    "http://x.com/e": [],
}
FAIL_MANY = {
    "http://f.test/": ["http://f.test/1", "http://f.test/2", "http://f.test/3", "http://f.test/dead"],
    "http://f.test/1": ["http://f.test/only-via-1", "http://f.test/2#x"],
    "http://f.test/2": ["http://f.test/deep", "http://f.test/1"],
    "http://f.test/3": ["http://f.test/deep#y", "http://f.test/"],
    "http://f.test/deep": ["http://f.test/deeper"],
    "http://f.test/deeper": [],
    "http://f.test/only-via-1": ["http://f.test/never"],
}
FAIL_MANY_FAIL = {"http://f.test/1": ConnectionError, "http://f.test/2": TimeoutError,
                  "http://f.test/dead": ValueError, "http://f.test/deeper": OSError}

CYCLES_START = "http://c.test/a"
CYCLES = {
    "http://c.test/a": ["http://c.test/b", "http://c.test/b", "http://c.test/a#me", "http://c.test/a"],
    "http://c.test/b": ["http://c.test/c", "http://c.test/a", "http://c.test/b#x", "http://c.test/c#y"],
    "http://c.test/c": ["http://c.test/a", "http://c.test/b", "http://c.test/d", "http://c.test/d"],
    "http://c.test/d": ["http://c.test/d", "http://c.test/c", "http://c.test/dangling"],
}

DETOUR_START = "https://a.test/1"
DETOUR = {
    "https://a.test/1": ["http://b.test/x", "https://a.test/2"],
    "http://b.test/x": ["https://a.test/secret"],
    "https://a.test/2": ["https://www.a.test/", "https://a.test/1#back"],
    "https://www.a.test/": ["https://a.test/hidden"],
}


# ================================================================ Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    parser = FakeParser(EXAMPLE_GRAPH)
    call = calling(repro("crawl", EXAMPLE_START, EXAMPLE_GRAPH, {}))
    got = sol.crawl(EXAMPLE_START, parser)
    expect(list(got) if isinstance(got, tuple) else got, EXAMPLE_WANT, call=call)
    verify(got, parser, EXAMPLE_START, EXAMPLE_GRAPH, {}, call)


@case("Part 1", title="协议、端口、用户名、大小写不同的同一主机")
def p1_same_host(sol):
    run_p1(sol, SAME_HOST_START, SAME_HOST)
    run_p1(sol, "http://p.com/root", {"http://p.com/root": ["http://p.com:8080/admin", "http://p.com/root#frag",
                                                            "http://p.com/root"],
                                      "http://p.com:8080/admin": ["http://p.com/root"]})


@case("Part 1", title="子域名和其它主机都不属于爬取范围")
def p1_other_hosts(sol):
    run_p1(sol, OTHER_HOSTS_START, OTHER_HOSTS)
    run_p1(sol, "http://shop.a.test/", {"http://shop.a.test/": ["http://a.test/", "http://shop.a.test/2"],
                                        "http://a.test/": ["http://shop.a.test/3"]})


@case("Part 1", title="片段：空片段、多个 #、起始 URL 带片段")
def p1_fragments(sol):
    run_p1(sol, FRAGMENTS_START, FRAGMENTS)


@case("Part 1", title="抓取失败的页面仍算已爬取")
def p1_failures(sol):
    run_p1(sol, "http://x.com/a", FAIL_GRAPH, {"http://x.com/b"})
    run_p1(sol, "http://f.test/", FAIL_MANY, FAIL_MANY_FAIL)
    run_p1(sol, "http://s.test/", {"http://s.test/": ["http://s.test/x"]}, {"http://s.test/": ConnectionError})


@case("Part 1", title="环、重复链接、自链接")
def p1_cycles(sol):
    run_p1(sol, CYCLES_START, CYCLES)


@case("Part 1", title="起始页没有链接；经由其它主机的链接")
def p1_detour(sol):
    run_p1(sol, "http://lonely.test/", {})
    run_p1(sol, "http://lonely.test/#x", {"http://lonely.test/": ["http://lonely.test/#y", "http://other.test/"]})
    run_p1(sol, DETOUR_START, DETOUR)


@case("Part 1", title="随机图，与暴力结果比较")
def p1_random(sol):
    r = rng()
    for _ in range(60):
        start, graph, fail = ref.random_graph(r)
        run_p1(sol, start, graph, fail)


# ================================================================ Part 2

@case("Part 2", sample=True, title="题目例子，max_workers=2")
def p2_example(sol):
    parser, call = run_p2(sol, EXAMPLE_START, EXAMPLE_GRAPH, k=2)
    expect(sum(parser.calls.values()), 4, call=call, msg="get_urls 一共应该恰好调用 4 次")


@case("Part 2", title="max_workers=1：一次只能有一个调用")
def p2_cap_one(sol):
    root, graph, leaves = ref.star("one.test", 5)
    crowd = Crowd(leaves, 1, CROWD_WAIT)
    run_p2(sol, root, graph, gates=[crowd], k=1)


@case("Part 2", title="不超过 max_workers，且确实并发")
def p2_cap(sol):
    for k in (2, 3, 5):
        root, graph, leaves = ref.star(f"cap{k}.test", 2 * k + 1)
        crowd = Crowd(leaves, k, CROWD_WAIT)
        parser, call = run_p2(sol, root, graph, gates=[crowd], k=k)
        overlapped(parser, len(leaves), k, "max_workers", call)


@case("Part 2", title="两个页面同时链接到同一个新页面")
def p2_shared_child(sol):
    start, graph, a, b = ref.diamond("dup.test", 3)
    run_p2(sol, start, graph, gates=[Trap(b, 0.4)], k=4)


@case("Part 2", title="深链：每层只有一个页面")
def p2_chain(sol):
    start, graph, pages = ref.chain("chain.test", 12)
    run_p2(sol, start, graph, gates=[SlowLinks(pages[:4], 0.03)], k=4)
    start, graph, pages = ref.chain("chain2.test", 6)
    run_p2(sol, start, graph, k=1)


@case("Part 2", title="取出 URL 到开始处理之间的空档")
def p2_take_gap(sol):
    ms = f"{GET_PAUSE * 1000:g}"
    note = f"测试让 queue.Queue.get() 取出元素以后先停 {ms} 毫秒再返回"
    why = (f"测试让线程从 queue.Queue 取出 URL 以后先停 {ms} 毫秒再返回（线程随时可能在这里被系统暂停）：这段时间里队列"
           "是空的，取走 URL 的线程也还没来得及把自己记成忙碌，但爬取并没有结束。判断爬完不能只看「队列空了、没有线程"
           "在忙」，而要在 URL 入队时就把它算作未完成（例如 Queue.join() 配合 task_done()）。")
    with ref.slow_queue_get(GET_PAUSE):
        start, graph, pages = ref.chain("gap.test", 5)
        run_p2(sol, start, graph, k=3, note=note, why=why)
        root, graph, leaves = ref.star("gap2.test", 6)
        run_p2(sol, root, graph, k=2, note=note, why=why)


@case("Part 2", title="抓取失败的页面")
def p2_failures(sol):
    run_p2(sol, "http://x.com/a", FAIL_GRAPH, {"http://x.com/b"}, k=4)
    run_p2(sol, "http://f.test/", FAIL_MANY, FAIL_MANY_FAIL, k=3)
    run_p2(sol, "http://s.test/", {"http://s.test/": ["http://s.test/x"]}, {"http://s.test/": ConnectionError}, k=2)


@case("Part 2", title="边界：同主机不同协议端口、片段、环")
def p2_edges(sol):
    for start, graph in ((SAME_HOST_START, SAME_HOST), (OTHER_HOSTS_START, OTHER_HOSTS),
                         (FRAGMENTS_START, FRAGMENTS), (CYCLES_START, CYCLES), (DETOUR_START, DETOUR)):
        run_p2(sol, start, graph, k=3)


@case("Part 2", title="随机图，随机延迟")
def p2_random(sol):
    r = rng()
    with ref.fast_switching():
        for _ in range(12):
            start, graph, fail = ref.random_graph(r)
            run_p2(sol, start, graph, fail, gates=[Jitter(r)], k=r.choice([1, 2, 3, 4, 8]))


@case("Part 2", title="多个线程同时处理同一批新链接")
def p2_lockstep(sol):
    r = rng()
    width = 8
    with ref.fast_switching():
        for _ in range(4):
            start, graph, groups = ref.layered(r, "race.test", width, 60, 3)
            run_p2(sol, start, graph, gates=[Lockstep(groups, 0.3)], k=width)


# ================================================================ Part 3

@case("Part 3", sample=True, title="题目例子，concurrency=2")
def p3_example(sol):
    parser, call = run_p3(sol, EXAMPLE_START, EXAMPLE_GRAPH, k=2)
    expect(sum(parser.calls.values()), 4, call=call, msg="get_urls 一共应该恰好调用 4 次")


@case("Part 3", sample=True, title="十个页面，最多两个同时运行")
def p3_ten_pages(sol):
    root, graph, leaves = ref.star("ten.test", 10)
    crowd = Crowd(leaves, 2, CROWD_WAIT)
    parser, call = run_p3(sol, root, graph, gates=[crowd], k=2)


@case("Part 3", title="不超过 concurrency，且确实并发")
def p3_cap(sol):
    root, graph, leaves = ref.star("cap1.test", 5)
    run_p3(sol, root, graph, gates=[Crowd(leaves, 1, CROWD_WAIT)], k=1)
    for k in (3, 5):
        root, graph, leaves = ref.star(f"cap{k}.test", 2 * k + 1)
        parser, call = run_p3(sol, root, graph, gates=[Crowd(leaves, k, CROWD_WAIT)], k=k)
        overlapped(parser, len(leaves), k, "concurrency", call)


@case("Part 3", title="链比 concurrency 深")
def p3_deep(sol):
    hint = "注意：这条链比 concurrency 深，每个页面要等它的子页面抓完。"
    start, graph, pages = ref.chain("deep.test", 10)
    run_p3(sol, start, graph, k=2, hint=hint)
    start, graph, pages = ref.chain("deep1.test", 4)
    run_p3(sol, start, graph, k=1, hint=hint)


@case("Part 3", title="get_urls 不能阻塞事件循环")
def p3_loop_free(sol):
    root, graph, leaves = ref.star("loop.test", 4)
    run_p3(sol, root, graph, gates=[Slow(leaves, 0.02)], k=4)


@case("Part 3", title="两个页面同时链接到同一个新页面")
def p3_shared_child(sol):
    start, graph, a, b = ref.diamond("dup.test", 3)
    run_p3(sol, start, graph, gates=[Trap(b, 0.4)], k=4)


@case("Part 3", title="抓取失败的页面")
def p3_failures(sol):
    run_p3(sol, "http://x.com/a", FAIL_GRAPH, {"http://x.com/b"}, k=4)
    run_p3(sol, "http://f.test/", FAIL_MANY, FAIL_MANY_FAIL, k=3)
    run_p3(sol, "http://s.test/", {"http://s.test/": ["http://s.test/x"]}, {"http://s.test/": ConnectionError}, k=2)


@case("Part 3", title="边界：同主机不同协议端口、片段、环")
def p3_edges(sol):
    for start, graph in ((SAME_HOST_START, SAME_HOST), (OTHER_HOSTS_START, OTHER_HOSTS),
                         (FRAGMENTS_START, FRAGMENTS), (CYCLES_START, CYCLES), (DETOUR_START, DETOUR)):
        run_p3(sol, start, graph, k=3)


@case("Part 3", title="随机图，随机延迟")
def p3_random(sol):
    r = rng()
    for _ in range(12):
        start, graph, fail = ref.random_graph(r)
        run_p3(sol, start, graph, fail, gates=[Jitter(r)], k=r.choice([1, 2, 3, 4, 8]))
