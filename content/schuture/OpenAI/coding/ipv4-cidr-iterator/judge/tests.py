import itertools

from judgelib import calling, case, expect, expect_raises, expect_true, rng
import ref

SHOW = 12           # a replay lists at most this many calls; earlier ones are folded into one loop
STOP = object()     # the expected outcome of next(): StopIteration
TOP = ref.TOP


# ---------------------------------------------------------------- how calls are shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement where possible."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _ctor(s, **kw):
    return "IPV4Iterator(" + ", ".join([_lit(s)] + [f"{k}={v!r}" for k, v in kw.items()]) + ")"


def _brief(xs):
    if len(xs) <= 4:
        return "[" + ", ".join(_lit(x) for x in xs) + "]"
    return f"{len(xs)} 个地址：{_lit(xs[0])}, {_lit(xs[1])}, …, {_lit(xs[-1])}"


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _why(text):
    return f"（{text}）" if text else ""


class Session:
    """One IPV4Iterator of the answer, driven call by call. Each call is checked as soon as it returns; a failure
    shows the calls so far as code to paste into 自定义运行."""

    def __init__(self, sol, s, trace=True, **kw):
        self.head = "it = " + _ctor(s, **kw)
        self.trace = trace          # False: the replay is built only when a call fails (long random runs)
        self.done = []              # ["next", count, last address] / ["stop"] / ["batch", size, addresses]
        calling(self.head)
        try:
            self.it = sol.IPV4Iterator(s, **kw)
        except ValueError as e:
            expect_true(False, f"{_lit(s)} 是合法的参数，构造函数不应抛出 ValueError（抛出了：{e}）", call=self.head)

    def replay(self, current, mark="# <- 这一步"):
        lines, done = [self.head], self.done
        if len(done) > SHOW:
            folded, done = done[:-SHOW], done[-SHOW:]
            taken = sum(d[1] if d[0] == "next" else len(d[2]) if d[0] == "batch" else 0 for d in folded)
            if taken:
                lines.append(f"for _ in range({taken}): next(it)    # 前面 {len(folded)} 组调用一共取走 {taken} 个地址")
        rows = []
        for d in done:
            if d[0] == "next":
                rows.append(("next(it)", f"# -> {_lit(d[2])}") if d[1] == 1 else
                            (f"for _ in range({d[1]}): next(it)", f"# 最后一次返回 {_lit(d[2])}"))
            elif d[0] == "stop":
                rows.append(("next(it, None)", "# 抛出 StopIteration，所以得到默认值 None"))
            else:
                rows.append((f"it.next_batch({d[1]})", f"# -> {_brief(d[2])}"))
        rows.append((f"print({current})", mark))
        width = max(len(code) for code, _ in rows) + 4
        return "\n".join(lines + [code.ljust(width) + note for code, note in rows])

    def _start(self, code):
        if self.trace:
            calling(self.replay(code))

    def _raised(self, code):
        calling(self.replay(code, "# <- 这一步抛出异常"))

    def next(self, want, why=""):
        """One next(it); want is an address, or STOP for StopIteration."""
        code = "next(it)"
        self._start(code)
        try:
            got = next(self.it)
        except StopIteration:
            if want is STOP:
                self.done.append(["stop"])
                return
            expect_true(False, f"迭代过早结束：这一次应当返回 {_lit(want)}，却抛出了 StopIteration" + _why(why),
                        call=self.replay(code))
        except Exception:
            self._raised(code)
            raise
        if want is STOP:
            expect_true(False, "应当抛出 StopIteration：已经没有可以产生的地址了" + _why(why), call=self.replay(code),
                        got=got)
        if not _same(got, want):
            expect(got, want, call=self.replay(code),
                   msg=why or (f"这一次应当返回 {_lit(want)}，返回了 {_lit(got)}" if isinstance(got, str) else ""))
        if self.done and self.done[-1][0] == "next":
            self.done[-1][1:] = [self.done[-1][1] + 1, got]
        else:
            self.done.append(["next", 1, got])

    def drain(self, wants, why=""):
        """next(it) for each address in wants, then StopIteration; `why` explains the last two steps."""
        for i, want in enumerate(wants):
            self.next(want, why if i == len(wants) - 1 else "")
        self.next(STOP, why)

    def batch(self, size, want, why=""):
        code = f"it.next_batch({size})"
        self._start(code)
        try:
            got = self.it.next_batch(size)
        except StopIteration:
            expect_true(False, "next_batch 本身从不抛出 StopIteration：剩下的地址不够时返回不足 size 个，"
                               "没有地址时返回 []", call=self.replay(code))
        except Exception:
            self._raised(code)
            raise
        norm = list(got) if isinstance(got, (list, tuple)) else got
        if not _same(norm, want):
            expect(got, want, call=self.replay(code), msg=why)
        self.done.append(["batch", size, norm])

    def bad_batch(self, size):
        """next_batch(size) with size < 0 must raise ValueError; the session ends there."""
        code = f"it.next_batch({size})"
        self._start(code)
        expect_raises(ValueError, self.it.next_batch, size, call=self.replay(code),
                      msg=f"size 必须是非负整数，next_batch({size}) 应当抛出 ValueError")


def rejects(sol, s, why=None, **kw):
    """The constructor must raise ValueError; `why` defaults to ref's reason."""
    if why is None:
        try:
            ref.parse(s)
            why = ""
        except ValueError as e:
            why = str(e)
    call = _ctor(s, **kw)
    calling(f"{call}    # 非法参数，应当抛出 ValueError")     # shown with the traceback if another exception escapes
    try:
        sol.IPV4Iterator(s, **kw)
    except ValueError:
        return
    what = "参数不合法" if kw else f"{_lit(s)} 不是合法的参数"
    expect_true(False, f"{what}（{why}），构造函数应当抛出 ValueError", call=call)


def compare(sol, s, count, **kw):
    """The first `count` addresses against ref, and StopIteration right after the last one if it comes that soon."""
    wants = ref.addresses(s, count=count + 1, **kw)
    it = Session(sol, s, trace=False, **kw)
    for want in wants[:count]:
        it.next(want)
    if len(wants) <= count:
        it.next(STOP, "迭代在允许范围的一端停止")


def accepts(sol, s):
    compare(sol, s, 2)


def fuzz(sol, r, count, cidr):
    """Valid strings with one to three characters inserted, deleted or replaced: rejected exactly when ref rejects."""
    alphabet = "0123456789.0123456789. -+x_\n" + ("/" if cidr else "")
    for _ in range(count):
        chars = list(ref.text(r.getrandbits(32)) + (f"/{r.randint(0, 32)}" if cidr and r.random() < 0.7 else ""))
        for _ in range(r.randint(1, 3)):
            i, op = r.randrange(len(chars) + 1), r.randrange(3)
            if op == 0:
                chars.insert(i, r.choice(alphabet))
            elif i < len(chars):
                chars[i:i + 1] = [] if op == 1 else [r.choice(alphabet)]
        s = "".join(chars)
        try:
            ref.parse(s)
        except ValueError as e:
            rejects(sol, s, str(e))
        else:
            compare(sol, s, 3)


def near_end(r, suffix, reverse, step, reach):
    """A start within `reach` steps of the end of the range that iteration walks toward."""
    _, lo, hi = ref.parse(ref.text(r.getrandbits(32)) + suffix)
    gap = r.randint(0, min(reach * step, hi - lo))
    return ref.text(lo + gap if reverse else hi - gap) + suffix


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：从 192.0.2.254 开始")
def p1_example(sol):
    it = Session(sol, "192.0.2.254")
    it.next("192.0.2.254")
    it.next("192.0.2.255")
    it.next("192.0.3.0", "最后一段超过 255，向前一段进位")
    it.next("192.0.3.1")


@case("Part 1", title="进位跨过好几段")
def p1_carry(sol):
    why = "地址按 32 位整数递增，某一段超过 255 时向前一段进位；每段写成不带前导零的十进制"
    for start, wants in [("1.2.255.255", ["1.2.255.255", "1.3.0.0", "1.3.0.1"]),
                         ("0.255.255.254", ["0.255.255.254", "0.255.255.255", "1.0.0.0"]),
                         ("254.255.255.255", ["254.255.255.255", "255.0.0.0", "255.0.0.1"]),
                         ("10.0.99.255", ["10.0.99.255", "10.0.100.0"]),
                         ("9.9.9.9", ["9.9.9.9", "9.9.9.10"]),
                         ("0.0.0.0", ["0.0.0.0", "0.0.0.1", "0.0.0.2"])]:
        it = Session(sol, start)
        for want in wants:
            it.next(want, why)


@case("Part 1", title="到 255.255.255.255 为止，含它")
def p1_last_address(sol):
    it = Session(sol, "255.255.255.255")
    it.next("255.255.255.255", "ip_or_cidr 本身就是 255.255.255.255 时，只产生这一个地址")
    for _ in range(3):
        it.next(STOP, "迭代到 255.255.255.255 为止，此后每次调用都抛出 StopIteration")
    it = Session(sol, "255.255.255.250")
    it.drain([f"255.255.255.{i}" for i in range(250, 256)], "迭代到 255.255.255.255 为止（含它）")
    want = ["255.255.255.252", "255.255.255.253", "255.255.255.254", "255.255.255.255"]
    call = 'import itertools\nprint(list(itertools.islice(IPV4Iterator("255.255.255.252"), 10)))'
    calling(call)
    expect(list(itertools.islice(sol.IPV4Iterator("255.255.255.252"), 10)), want, call=call,
           msg="for 循环和 list() 通过 __iter__ 取得迭代器，再反复调用 __next__ 直到 StopIteration")


@case("Part 1", title="段数不对、空段、非数字")
def p1_bad_format(sol):
    for s in ["1.2.3", "1.2.3.4.5", "1..3.4", "", "1.2.3.", ".1.2.3", "1.2.3.4.", "1234", "....", "a.b.c.d",
              "1.2.3.x", "1,2,3,4", "1.2.3.4a", "0x1.2.3.4", "1.2.3.4e0", "1.2.3.4.0"]:
        rejects(sol, s)


@case("Part 1", title="某一段超过 255")
def p1_out_of_range(sol):
    for s in ["256.0.0.0", "1.256.3.4", "1.2.3.256", "999.1.1.1", "1.2.3.1000", "300.300.300.300",
              "1.2.3.4294967296"]:
        rejects(sol, s)
    for s in ["255.0.0.0", "1.2.3.255", "249.250.251.252"]:
        accepts(sol, s)


@case("Part 1", title="前导零")
def p1_leading_zero(sol):
    for s in ["01.2.3.4", "1.2.03.4", "1.2.00.4", "1.2.3.00", "1.2.3.04", "001.2.3.4", "0.0.0.000", "1.2.3.010",
              "1.02.3.4"]:
        rejects(sol, s)
    for s in ["0.0.0.0", "10.0.100.0", "100.200.0.10", "1.0.0.255"]:
        accepts(sol, s)


@case("Part 1", title="符号、空白、下划线")
def p1_signs_and_spaces(sol):
    for s in ["+1.2.3.4", "1.2.3.+4", "-1.2.3.4", "1.-0.3.4", " 1.2.3.4", "1.2.3.4 ", "1. 2.3.4", "1.2 .3.4",
              "1.2.3.\t4", "1.2.3.1_0", "1_0.2.3.4"]:
        rejects(sol, s)


@case("Part 1", title="结尾的换行符")
def p1_trailing_newline(sol):
    for s in ["1.2.3.4\n", "255.255.255.255\n", "1.2.3\n.4", "1.2.3.4\r\n", "\n1.2.3.4"]:
        rejects(sol, s)


@case("Part 1", title="for 循环从当前位置继续")
def p1_iter_continues(sol):
    head = 'it = IPV4Iterator("255.255.255.250")\nnext(it)\nnext(it)'
    calling(head)
    it = sol.IPV4Iterator("255.255.255.250")
    next(it)
    next(it)
    call = head + "\nprint(list(it))"
    calling(call)
    expect(list(it), ["255.255.255.252", "255.255.255.253", "255.255.255.254", "255.255.255.255"], call=call,
           msg="__iter__ 返回迭代器本身，所以 for 循环和 list() 从当前位置接着取，不会从 ip_or_cidr 重新开始")
    call += "\nprint(list(it))    # 已经耗尽"
    calling(call)
    expect(list(it), [], call=call, msg="迭代器耗尽之后再用 for 循环或 list() 也取不到地址，__next__ 继续抛出 StopIteration")


@case("Part 1", title="随机起点")
def p1_random(sol):
    r = rng("p1")
    for _ in range(150):
        v = r.choice([r.getrandbits(32), r.randint(TOP - 30, TOP), r.getrandbits(8) << 24 | r.randint(2 ** 24 - 5, 2 ** 24 - 1)])
        compare(sol, ref.text(v), 40)


@case("Part 1", title="随机的合法与非法字符串")
def p1_random_strings(sol):
    fuzz(sol, rng("p1s"), 1200, cidr=False)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：从 192.0.2.1 反向")
def p2_example(sol):
    it = Session(sol, "192.0.2.1", reverse=True)
    it.next("192.0.2.1")
    it.next("192.0.2.0")
    it.next("192.0.1.255", "向前一段借位")
    it.next("192.0.1.254")


@case("Part 2", title="到 0.0.0.0 为止，含它")
def p2_first_address(sol):
    it = Session(sol, "0.0.0.0", reverse=True)
    it.next("0.0.0.0", "ip_or_cidr 本身就是 0.0.0.0 时，只产生这一个地址")
    for _ in range(3):
        it.next(STOP, "反向迭代到 0.0.0.0 为止，此后每次调用都抛出 StopIteration")
    it = Session(sol, "0.0.0.4", reverse=True)
    it.drain(["0.0.0.4", "0.0.0.3", "0.0.0.2", "0.0.0.1", "0.0.0.0"], "反向迭代到 0.0.0.0 为止（含它）")


@case("Part 2", title="借位跨过好几段")
def p2_borrow(sol):
    why = "反向时地址按 32 位整数递减，某一段低于 0 时向前一段借位"
    for start, wants in [("1.0.0.0", ["1.0.0.0", "0.255.255.255", "0.255.255.254"]),
                         ("10.1.0.0", ["10.1.0.0", "10.0.255.255"]),
                         ("128.0.0.0", ["128.0.0.0", "127.255.255.255"]),
                         ("255.255.255.255", ["255.255.255.255", "255.255.255.254"]),
                         ("9.9.9.10", ["9.9.9.10", "9.9.9.9"])]:
        it = Session(sol, start, reverse=True)
        for want in wants:
            it.next(want, why)


@case("Part 2", title="reverse=False 与 Part 1 相同")
def p2_reverse_false(sol):
    it = Session(sol, "255.255.255.254", reverse=False)
    it.drain(["255.255.255.254", "255.255.255.255"], "reverse=False 保持 Part 1 的行为")
    it = Session(sol, "0.0.0.0", reverse=False)
    it.next("0.0.0.0")
    it.next("0.0.0.1", "reverse=False 保持 Part 1 的行为")


@case("Part 2", title="随机起点")
def p2_random(sol):
    r = rng("p2")
    for _ in range(150):
        v = r.choice([r.getrandbits(32), r.randint(0, 30), r.getrandbits(8) << 24 | r.randint(0, 4)])
        compare(sol, ref.text(v), 40, reverse=True)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：203.0.113.100/28")
def p3_example(sol):
    it = Session(sol, "203.0.113.100/28")
    it.drain([f"203.0.113.{i}" for i in range(100, 112)], "正向在广播地址 203.0.113.111 处停止（含广播地址），共 12 个地址")
    it = Session(sol, "203.0.113.100/28", reverse=True)
    it.drain([f"203.0.113.{i}" for i in range(100, 95, -1)], "反向在网络地址 203.0.113.96 处停止（含网络地址），共 5 个地址")


@case("Part 3", sample=True, title="题目例子：/32、/31、/0")
def p3_special_prefixes(sol):
    it = Session(sol, "198.51.100.77/32")
    it.drain(["198.51.100.77"], "/32 块只有一个地址")
    it = Session(sol, "198.51.100.16/31")
    it.drain(["198.51.100.16", "198.51.100.17"], "/31 块有两个地址")
    it = Session(sol, "10.20.30.40/0")
    for want in ["10.20.30.40", "10.20.30.41", "10.20.30.42"]:
        it.next(want, "/0 块是整个地址空间，迭代从 ip_or_cidr 开始")


@case("Part 3", title="起点在块中间：从它开始")
def p3_seed_inside_block(sol):
    why = "网络地址和广播地址由 ip_or_cidr 掩码得到，迭代从 ip_or_cidr 本身开始，在块的一端停止"
    for s, reverse, wants in [
            ("10.1.2.3/24", True, ["10.1.2.3", "10.1.2.2", "10.1.2.1", "10.1.2.0"]),
            ("10.1.2.253/24", False, ["10.1.2.253", "10.1.2.254", "10.1.2.255"]),
            ("10.1.2.255/24", False, ["10.1.2.255"]),
            ("10.1.2.0/24", True, ["10.1.2.0"]),
            ("172.16.255.254/16", False, ["172.16.255.254", "172.16.255.255"]),
            ("172.16.0.2/15", True, ["172.16.0.2", "172.16.0.1", "172.16.0.0"]),
            ("192.168.77.130/25", True, ["192.168.77.130", "192.168.77.129", "192.168.77.128"]),
            ("192.168.77.126/25", False, ["192.168.77.126", "192.168.77.127"]),
            ("100.64.13.37/30", False, ["100.64.13.37", "100.64.13.38", "100.64.13.39"]),
            ("100.64.13.37/30", True, ["100.64.13.37", "100.64.13.36"])]:
        Session(sol, s, reverse=reverse).drain(wants, why)


@case("Part 3", title="/32 和 /31")
def p3_tiny_blocks(sol):
    for s, reverse, wants in [("0.0.0.0/32", False, ["0.0.0.0"]),
                              ("0.0.0.0/32", True, ["0.0.0.0"]),
                              ("255.255.255.255/32", False, ["255.255.255.255"]),
                              ("255.255.255.255/32", True, ["255.255.255.255"]),
                              ("1.2.3.4/32", True, ["1.2.3.4"]),
                              ("198.51.100.17/31", False, ["198.51.100.17"]),
                              ("198.51.100.17/31", True, ["198.51.100.17", "198.51.100.16"]),
                              ("198.51.100.16/31", True, ["198.51.100.16"]),
                              ("255.255.255.255/31", True, ["255.255.255.255", "255.255.255.254"])]:
        Session(sol, s, reverse=reverse).drain(wants, "/32 块只有 ip_or_cidr 一个地址；/31 块有两个地址")


@case("Part 3", title="/0、/1、/2：块很大")
def p3_huge_blocks(sol):
    for s, reverse, wants in [("255.255.255.250/0", False, [f"255.255.255.{i}" for i in range(250, 256)]),
                              ("0.0.0.3/0", True, ["0.0.0.3", "0.0.0.2", "0.0.0.1", "0.0.0.0"]),
                              ("127.255.255.253/1", False, ["127.255.255.253", "127.255.255.254", "127.255.255.255"]),
                              ("128.0.0.2/1", True, ["128.0.0.2", "128.0.0.1", "128.0.0.0"]),
                              ("63.255.255.254/2", False, ["63.255.255.254", "63.255.255.255"]),
                              ("192.0.0.1/2", True, ["192.0.0.1", "192.0.0.0"])]:
        Session(sol, s, reverse=reverse).drain(wants, "块是网络地址到广播地址（两端都含）的全部地址")
    it = Session(sol, "10.20.30.40/0", reverse=True)
    for want in ["10.20.30.40", "10.20.30.39", "10.20.30.38"]:
        it.next(want)


@case("Part 3", title="每一个前缀长度 0 到 32")
def p3_every_prefix(sol):
    r = rng("prefix")
    for p in range(33):
        for _ in range(4):
            compare(sol, near_end(r, f"/{p}", False, 1, 5), 8)
            compare(sol, near_end(r, f"/{p}", True, 1, 5), 8, reverse=True)
            s = f"{ref.text(r.getrandbits(32))}/{p}"
            compare(sol, s, 4)
            compare(sol, s, 4, reverse=True)


@case("Part 3", title="前缀超出 0-32")
def p3_prefix_range(sol):
    for s in ["1.2.3.4/33", "1.2.3.4/34", "1.2.3.4/40", "1.2.3.4/64", "1.2.3.4/99", "1.2.3.4/100", "1.2.3.4/320"]:
        rejects(sol, s)
    for s in ["1.2.3.4/32", "1.2.3.4/0", "1.2.3.4/9", "1.2.3.4/10", "1.2.3.4/30"]:
        accepts(sol, s)


@case("Part 3", title="前缀不带前导零，也不能写成掩码")
def p3_prefix_form(sol):
    for s in ["198.51.100.0/08", "198.51.100.0/00", "198.51.100.0/032", "1.2.3.4/01", "198.51.100.0/255.0.0.0",
              "1.2.3.4/255.255.255.0", "1.2.3.4/0.0.0.255", "1.2.3.4/255.255.255.255"]:
        rejects(sol, s)


@case("Part 3", title="缺少前缀或地址、'/' 不止一个")
def p3_prefix_missing(sol):
    for s in ["1.2.3.4/", "/8", "/", "1.2.3.4//8", "1.2.3.4/8/", "1.2.3.4/1/2", "/1.2.3.4", "1.2.3/8",
              "1.2.3.4.5/8", "01.2.3.4/8", "1.2.3.256/8"]:
        rejects(sol, s)


@case("Part 3", title="前缀里的符号和空白")
def p3_prefix_chars(sol):
    for s in ["1.2.3.4/+8", "1.2.3.4/-0", "1.2.3.4/-1", "1.2.3.4/ 8", "1.2.3.4/8 ", "1.2.3.4 /8", "1.2.3.4/8\n",
              "1.2.3.4/0x8", "1.2.3.4/8.0", "1.2.3.4/a", "1.2.3.4/1_6", "1.2.3.4/\t8"]:
        rejects(sol, s)


@case("Part 3", title="随机块：走到块的一端")
def p3_random(sol):
    r = rng("p3")
    for _ in range(200):
        s = f"{ref.text(r.getrandbits(32))}/{r.randint(22, 32)}"
        compare(sol, s, 1100, reverse=r.random() < 0.5)


@case("Part 3", title="随机的合法与非法字符串")
def p3_random_strings(sol):
    fuzz(sol, rng("p3s"), 1500, cidr=True)


# ---------------------------------------------------------------- Part 4
@case("Part 4", sample=True, title="题目例子：/28，step=3")
def p4_example(sol):
    it = Session(sol, "203.0.113.16/28", step=3)
    it.drain(["203.0.113.16", "203.0.113.19", "203.0.113.22", "203.0.113.25", "203.0.113.28", "203.0.113.31"],
             "下一个候选地址 203.0.113.34 已经超过广播地址")


@case("Part 4", sample=True, title="题目例子：反向，step=4")
def p4_example_reverse(sol):
    it = Session(sol, "203.0.113.30/28", reverse=True, step=4)
    it.drain(["203.0.113.30", "203.0.113.26", "203.0.113.22", "203.0.113.18"],
             "下一个候选地址 203.0.113.14 已经低于网络地址 203.0.113.16")


@case("Part 4", sample=True, title="题目例子：step=0")
def p4_step_zero(sol):
    rejects(sol, "203.0.113.16/28", "step 必须是正整数", step=0)


@case("Part 4", title="step 为负数")
def p4_negative_step(sol):
    why = "step 必须是正整数，step <= 0 抛出 ValueError"
    for kw in [dict(step=-1), dict(step=-3), dict(step=-100), dict(reverse=True, step=-1), dict(step=-(2 ** 32))]:
        rejects(sol, "203.0.113.16/28", why, **kw)
    rejects(sol, "1.2.3.4", why, step=-1)
    rejects(sol, "255.255.255.255", why, reverse=True, step=-2)


@case("Part 4", title="越过边界的地址不会产生")
def p4_overshoot(sol):
    why = "下一个待产生的地址落到允许范围之外时迭代就停止，越过之后的地址永远不会被产生"
    for s, kw, wants in [("10.0.0.0/29", dict(step=3), ["10.0.0.0", "10.0.0.3", "10.0.0.6"]),
                         ("10.0.0.13/29", dict(reverse=True, step=3), ["10.0.0.13", "10.0.0.10"]),
                         ("192.168.1.200/24", dict(step=50), ["192.168.1.200", "192.168.1.250"]),
                         ("255.255.255.250", dict(step=4), ["255.255.255.250", "255.255.255.254"]),
                         ("255.255.255.255", dict(step=2), ["255.255.255.255"]),
                         ("0.0.0.5", dict(reverse=True, step=2), ["0.0.0.5", "0.0.0.3", "0.0.0.1"]),
                         ("0.0.0.5", dict(reverse=True, step=3), ["0.0.0.5", "0.0.0.2"])]:
        Session(sol, s, **kw).drain(wants, why)


@case("Part 4", title="第一个地址不受 step 影响")
def p4_first_is_seed(sol):
    why = "第一个产生的地址是 ip_or_cidr 本身，之后每次前进 step 个地址"
    for s, kw, wants in [("10.0.0.5/24", dict(step=100), ["10.0.0.5", "10.0.0.105", "10.0.0.205"]),
                         ("10.0.0.5/24", dict(reverse=True, step=2), ["10.0.0.5", "10.0.0.3", "10.0.0.1"]),
                         ("10.0.0.255/24", dict(step=7), ["10.0.0.255"]),
                         ("10.0.0.0/24", dict(reverse=True, step=7), ["10.0.0.0"])]:
        Session(sol, s, **kw).drain(wants, why)


@case("Part 4", title="步长比整个范围还大")
def p4_huge_step(sol):
    why = "第一个地址之后的下一跳已经落到允许范围之外"
    for s, kw, wants in [("203.0.113.20/28", dict(step=16), ["203.0.113.20"]),
                         ("203.0.113.20/28", dict(reverse=True, step=16), ["203.0.113.20"]),
                         ("1.2.3.4", dict(step=2 ** 32), ["1.2.3.4"]),
                         ("1.2.3.4", dict(reverse=True, step=2 ** 32), ["1.2.3.4"]),
                         ("0.0.0.0", dict(step=2 ** 40), ["0.0.0.0"]),
                         ("1.2.3.4/0", dict(step=2 ** 31), ["1.2.3.4", "129.2.3.4"])]:
        Session(sol, s, **kw).drain(wants, why)


@case("Part 4", title="大步长跨段")
def p4_big_steps(sol):
    it = Session(sol, "10.0.0.7", step=256)
    for want in ["10.0.0.7", "10.0.1.7", "10.0.2.7"]:
        it.next(want)
    for s, kw in [("250.1.2.3", dict(step=2 ** 24)), ("5.1.2.3", dict(reverse=True, step=2 ** 24)),
                  ("172.16.5.5/16", dict(step=4096)), ("172.16.5.5/16", dict(step=65536)),
                  ("172.16.250.5/16", dict(reverse=True, step=10000))]:
        Session(sol, s, **kw).drain(ref.addresses(s, **kw))


@case("Part 4", title="随机步长")
def p4_random(sol):
    r = rng("p4")
    for _ in range(300):
        suffix = r.choice(["", f"/{r.randint(0, 32)}", f"/{r.randint(20, 32)}"])
        step = r.choice([1, 2, 3, 5, 7, 16, 100, 256, 1000, 65536, 2 ** 24, r.randint(1, 2 ** 33)])
        reverse = r.random() < 0.5
        compare(sol, near_end(r, suffix, reverse, step, 30), 40, reverse=reverse, step=step)
        compare(sol, ref.text(r.getrandbits(32)) + suffix, 10, reverse=reverse, step=step)


# ---------------------------------------------------------------- Part 5
@case("Part 5", sample=True, title="题目例子：/28 批量读取")
def p5_example(sol):
    it = Session(sol, "203.0.113.16/28")
    it.batch(5, [f"203.0.113.{i}" for i in range(16, 21)])
    it.batch(100, [f"203.0.113.{i}" for i in range(21, 32)], "只剩下 11 个地址，返回不足 size 个")
    it.batch(5, [], "没有地址剩余时返回空列表")


@case("Part 5", title="size = 0：返回 []，位置不变")
def p5_size_zero(sol):
    why = "size == 0 总是返回 []，既不抛出异常，也不改变当前位置"
    it = Session(sol, "203.0.113.16/28")
    it.batch(0, [], why)
    it.next("203.0.113.16", why)
    it.batch(0, [], why)
    it.batch(3, ["203.0.113.17", "203.0.113.18", "203.0.113.19"], why)
    it.batch(0, [], why)
    it.next("203.0.113.20", why)
    it = Session(sol, "203.0.113.31/28")
    it.next("203.0.113.31")
    it.batch(0, [], why)
    it.next(STOP)
    it.batch(0, [], why + "，迭代器已经耗尽时也一样")
    it = Session(sol, "1.2.3.4")
    it.batch(0, [], why)
    it.next("1.2.3.4", why)


@case("Part 5", title="size < 0 抛出 ValueError")
def p5_negative_size(sol):
    Session(sol, "1.2.3.4").bad_batch(-1)
    Session(sol, "203.0.113.16/28").bad_batch(-5)
    it = Session(sol, "10.0.0.0/30")
    it.batch(2, ["10.0.0.0", "10.0.0.1"])
    it.bad_batch(-1)
    it = Session(sol, "203.0.113.31/28")
    it.next("203.0.113.31")
    it.next(STOP)
    it.bad_batch(-1)


@case("Part 5", title="与 next() 交替调用，耗尽之后")
def p5_mixed(sol):
    it = Session(sol, "198.51.100.250/29")
    it.next("198.51.100.250")
    it.batch(2, ["198.51.100.251", "198.51.100.252"], "next_batch 从当前位置开始")
    it.next("198.51.100.253", "next_batch 之后 next 接着它停下的位置")
    it.batch(1, ["198.51.100.254"])
    it.next("198.51.100.255")
    it.batch(4, [], "没有地址剩余时返回空列表")
    it.next(STOP)
    it.batch(2, [], "迭代器耗尽之后返回空列表")
    it = Session(sol, "255.255.255.252")
    it.batch(2, ["255.255.255.252", "255.255.255.253"])
    it.batch(5, ["255.255.255.254", "255.255.255.255"], "地址不够时返回不足 size 个")
    it.next(STOP, "next_batch 取完了最后的地址")
    it.batch(1, [])


@case("Part 5", title="批量读取也遵守 reverse 和 step")
def p5_reverse_and_step(sol):
    why = "next_batch 按 __next__ 会产生的顺序返回地址，同样遵守 reverse、step 和块的边界"
    it = Session(sol, "10.0.0.20/27", reverse=True, step=3)
    it.batch(4, ["10.0.0.20", "10.0.0.17", "10.0.0.14", "10.0.0.11"], why)
    it.batch(10, ["10.0.0.8", "10.0.0.5", "10.0.0.2"], why)
    it.batch(1, [], why)
    it = Session(sol, "10.0.0.20/27", step=5)
    it.batch(2, ["10.0.0.20", "10.0.0.25"], why)
    it.next("10.0.0.30", why)
    it.batch(3, [], why)
    it = Session(sol, "0.0.0.6", reverse=True)
    it.batch(10, ["0.0.0.6", "0.0.0.5", "0.0.0.4", "0.0.0.3", "0.0.0.2", "0.0.0.1", "0.0.0.0"], why)
    it = Session(sol, "203.0.113.16/28", step=3)
    it.batch(10, ["203.0.113.16", "203.0.113.19", "203.0.113.22", "203.0.113.25", "203.0.113.28", "203.0.113.31"],
             why + "；这是 Part 4 例子里的 6 个地址")
    it = Session(sol, "255.255.255.250", step=2)
    it.batch(5, ["255.255.255.250", "255.255.255.252", "255.255.255.254"], why)


@case("Part 5", title="在整个地址空间里批量读取")
def p5_whole_space(sol):
    Session(sol, "0.0.0.0").batch(3, ["0.0.0.0", "0.0.0.1", "0.0.0.2"])
    Session(sol, "10.20.30.40/0").batch(4, ["10.20.30.40", "10.20.30.41", "10.20.30.42", "10.20.30.43"])
    Session(sol, "255.255.255.255", reverse=True).batch(2, ["255.255.255.255", "255.255.255.254"])
    Session(sol, "128.0.0.0/1", step=2 ** 20).batch(3, ["128.0.0.0", "128.16.0.0", "128.32.0.0"])
    Session(sol, "255.255.255.253").batch(10, ["255.255.255.253", "255.255.255.254", "255.255.255.255"])
    Session(sol, "0.0.0.1", reverse=True).batch(5, ["0.0.0.1", "0.0.0.0"])


@case("Part 5", title="随机交替调用")
def p5_random(sol):
    r = rng("p5")
    for _ in range(250):
        reverse, step = r.random() < 0.5, r.choice([1, 1, 1, 2, 3, 4, 7])
        if r.random() < 0.8:
            s = f"{ref.text(r.getrandbits(32))}/{r.randint(25, 32)}"
        else:
            s = near_end(r, r.choice(["", "/0", f"/{r.randint(1, 24)}"]), reverse, step, 40)
        model, it = ref.Model(s, reverse, step), Session(sol, s, trace=False, reverse=reverse, step=step)
        for _ in range(40):
            if r.random() < 0.3:
                want = model.take(1)
                it.next(want[0] if want else STOP)
                if not want:
                    break
            else:
                size = r.randint(0, 12)
                want = model.take(size)
                it.batch(size, want)
                if len(want) < size:
                    break
