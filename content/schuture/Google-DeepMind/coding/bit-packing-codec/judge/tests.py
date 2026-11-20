from judgelib import Mismatch, calling, case, expect, expect_raises, rng, show_call
import ref


# ---------------------------------------------------------------- helpers

def bytes_msg(got, want):
    """Where two byte strings first differ, with the bits of that byte (packing errors are about bits)."""
    if not isinstance(got, (bytes, bytearray)) or got == want:
        return ""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            extra = f"；长度期望 {len(want)}，得到 {len(got)}" if len(got) != len(want) else ""
            return f"第 {i} 个字节（下标从 0 开始）不同：期望 {b:08b}，得到 {a:08b}{extra}"
    return f"长度不对：期望 {len(want)} 个字节，得到 {len(got)} 个（前 {min(len(got), len(want))} 个相同）"


def as_list(got):
    return list(got) if isinstance(got, tuple) else got   # the statement says list; a tuple carries the same values


def check_bytes(got, want, call):
    expect(got, want, call=call, msg=bytes_msg(got, want))


def check_pack(sol, values, k, call=None):
    want = ref.pack(values, k)
    call = calling(call or show_call("pack", values, k))
    check_bytes(sol.pack(list(values), k), want, call)


def check_unpack(sol, data, k, count, call=None):
    want = ref.unpack(data, k, count)
    call = calling(call or show_call("unpack", data, k, count))
    expect(as_list(sol.unpack(data, k, count)), want, call=call)


def check_pack_both(sol, values, k):
    check_pack(sol, values, k)
    check_unpack(sol, ref.pack(values, k), k, len(values))


def raises_value_error(sol, fname, *args):
    call = calling(show_call(fname, *args))
    expect_raises(ValueError, getattr(sol, fname), *args, call=call)


def check_encode(sol, values, call=None):
    want = ref.uvarints(values)
    call = calling(call or show_call("encode_uvarints", values))
    check_bytes(sol.encode_uvarints(list(values)), want, call)


def check_decode(sol, data, want, call=None):
    call = calling(call or show_call("decode_uvarints", data))
    expect(as_list(sol.decode_uvarints(data)), want, call=call)


def check_zigzag(sol, n):
    call = calling(show_call("zigzag", n))
    expect(sol.zigzag(n), ref.zigzag(n), call=call)


def check_unzigzag(sol, z):
    call = calling(show_call("unzigzag", z))
    expect(sol.unzigzag(z), ref.unzigzag(z), call=call)


def check_encode_for(sol, values, call=None):
    want = ref.encode_for(values)
    call = calling(call or show_call("encode_for", values))
    check_bytes(sol.encode_for(list(values)), want, call)


def check_decode_for(sol, values, call=None):
    data = ref.encode_for(values)                          # the correct bytes, not the answer's own encoder output
    call = calling(call or show_call("decode_for", data))
    expect(as_list(sol.decode_for(data)), list(values), call=call)


def check_for_size(sol, values, call=None):
    want = len(ref.encode_for(values))
    call = calling(call or show_call("for_size", values))
    expect(sol.for_size(list(values)), want, call=call)


def check_best(sol, values, call=None):
    want = ref.best_encoding(values)
    call = calling(call or show_call("best_encoding", values))
    got = sol.best_encoding(list(values))
    sizes = f"（frame-of-reference {len(ref.encode_for(values))} 个字节，varint 方案 {len(ref.varint_block(values))} 个字节）"
    expect(got, want, call=call, msg="" if got == want else f"选错了{sizes}")


def check_block(sol, values):
    """Every Part 3 function on one block."""
    check_encode_for(sol, values)
    check_decode_for(sol, values)
    check_for_size(sol, values)
    check_best(sol, values)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 pack([5, 1, 7], 3)")
def p1_example_pack(sol):
    calling('pack([5, 1, 7], 3)')
    check_bytes(sol.pack([5, 1, 7], 3), b"\xa7\x80", 'pack([5, 1, 7], 3)')


@case("Part 1", sample=True, title="题目例子 unpack(b\"\\xa7\\x80\", 3, 3)")
def p1_example_unpack(sol):
    calling('unpack(b"\\xa7\\x80", 3, 3)')
    expect(as_list(sol.unpack(b"\xa7\x80", 3, 3)), [5, 1, 7], call='unpack(b"\\xa7\\x80", 3, 3)')


@case("Part 1", title="空列表：0 个字节")
def p1_empty(sol):
    for k in (1, 3, 8, 32):
        check_pack(sol, [], k)
        check_unpack(sol, b"", k, 0)


@case("Part 1", title="k 是 8 的倍数或约数")
def p1_byte_aligned(sol):
    for values, k in [([0, 255, 17, 128], 8), ([0x1234, 0xFFFF, 1], 16), ([0xDEADBEEF, 0, 1], 32),
                      ([1, 2, 3, 15], 4), ([3, 0, 1, 2, 3, 3, 0, 1], 2), ([1, 0, 1, 1, 0, 0, 0, 1, 1], 1)]:
        check_pack_both(sol, values, k)


@case("Part 1", title="最后一个字节的低位补 0")
def p1_padding(sol):
    for values, k in [([1], 1), ([1], 7), ([3], 2), ([5, 1, 7], 3), ([1, 1, 1], 5), ([0x7FFFFFFF], 31),
                      ([1], 9), ([2 ** 32 - 1], 32), ([2 ** 31 + 5, 3], 32)]:
        check_pack_both(sol, values, k)


@case("Part 1", title="字段跨字节，每个 k 都试全 1 和最高位")
def p1_every_width(sol):
    for k in range(1, 33):
        for values in ([2 ** k - 1] * 3, [2 ** (k - 1), 0, 2 ** (k - 1)], [0, 2 ** k - 1, 0, 1, 2 ** k - 2]):
            check_pack_both(sol, values, k)


@case("Part 1", title="随机 k 和随机值（对照逐位拼接）")
def p1_random(sol):
    r = rng()
    for _ in range(600):
        k = r.randint(1, 32)
        values = [r.randrange(1 << k) for _ in range(r.randint(0, 40))]
        check_pack_both(sol, values, k)


@case("Part 1", title="几千个值")
def p1_long(sol):
    r = rng()
    for k in (1, 5, 13, 32):
        values = [r.randrange(1 << k) for _ in range(3000)]
        data = ref.pack(values, k)
        check_pack(sol, values, k, call=f"pack(<{len(values)} 个 {k} 比特的值>, {k})")
        check_unpack(sol, data, k, len(values), call=f"unpack(<{len(data)} 个字节>, {k}, {len(values)})")


@case("Part 1", title="pack 的 k 越界或值放不下")
def p1_pack_errors(sol):
    for values, k in [([1], 0), ([1], 33), ([], 0), ([], 33), ([0], -1),            # k outside 1..32
                      ([8], 3), ([0, 1, 8], 3), ([2], 1), ([2 ** 32], 32), ([256], 8),  # 2**k does not fit
                      ([-1], 3), ([5, -1], 8), ([-(2 ** 31)], 32)]:                   # negative values
        raises_value_error(sol, "pack", list(values), k)


@case("Part 1", title="unpack 的 k 越界或长度不对")
def p1_unpack_errors(sol):
    for data, k, count in [(b"\x00", 0, 1), (b"\x00", 33, 1), (b"", 0, 0),        # k outside 1..32
                           (b"", 3, 1), (b"\xa7", 3, 3), (b"\x00" * 3, 32, 1),     # one byte short
                           (b"\xa7\x80\x00", 3, 3), (b"\x00", 5, 0), (b"\x00\x00", 8, 1),  # one byte long
                           (b"\x00" * 5, 32, 1), (b"\xff\xff", 1, 8)]:
        raises_value_error(sol, "unpack", data, k, count)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 encode_uvarints([300])")
def p2_example_encode(sol):
    calling("encode_uvarints([300])")
    check_bytes(sol.encode_uvarints([300]), b"\xac\x02", "encode_uvarints([300])")


@case("Part 2", sample=True, title="题目例子 decode_uvarints(b\"\\xac\\x02\")")
def p2_example_decode(sol):
    check_decode(sol, b"\xac\x02", [300], 'decode_uvarints(b"\\xac\\x02")')


@case("Part 2", sample=True, title="题目例子 zigzag 0, -1, 1, -2, 2")
def p2_example_zigzag(sol):
    for n, z in [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4)]:
        calling(f"zigzag({n})")
        expect(sol.zigzag(n), z, call=f"zigzag({n})")


@case("Part 2", title="0 也要写出一个字节；空列表")
def p2_zero(sol):
    check_encode(sol, [0])
    check_encode(sol, [])
    check_encode(sol, [0, 0, 5, 0])
    for data, want in [(b"", []), (b"\x00", [0]), (b"\x00\x00", [0, 0]), (b"\x00\x05\x00", [0, 5, 0])]:
        check_decode(sol, data, want)


@case("Part 2", title="7 比特分组的边界：127、128、16383、16384 …")
def p2_group_boundaries(sol):
    values = [1, 127, 128, 129, 255, 256, 16383, 16384, 2 ** 21 - 1, 2 ** 21, 2 ** 28, 2 ** 32 - 1, 2 ** 63, 2 ** 64 - 1]
    for v in values:
        check_encode(sol, [v])
        check_decode(sol, ref.uvarint(v), [v])
    check_encode(sol, values)
    check_decode(sol, ref.uvarints(values), values)


@case("Part 2", title="超过 64 位的大整数")
def p2_huge(sol):
    for v in (2 ** 64, 2 ** 70 + 12345, 3 ** 100, 2 ** 200 - 1, 10 ** 60):
        check_encode(sol, [v])
        check_decode(sol, ref.uvarint(v), [v])


@case("Part 2", title="随机整数序列（对照 divmod 实现）")
def p2_random_streams(sol):
    r = rng()
    for _ in range(300):
        values = [ref.random_uvarint_value(r) for _ in range(r.randint(0, 30))]
        check_encode(sol, values)
        check_decode(sol, ref.uvarints(values), values)


@case("Part 2", title="有负数时 encode_uvarints 抛 ValueError")
def p2_encode_negative(sol):
    for values in ([-1], [5, -3], [0, 2 ** 70, -(2 ** 70)], [-128]):
        raises_value_error(sol, "encode_uvarints", list(values))


@case("Part 2", title="数据在某个数中间结束")
def p2_truncated(sol):
    for data in (b"\x80", b"\xac", b"\xff\xff", b"\x05\xac", b"\x00\x81\x80"):
        raises_value_error(sol, "decode_uvarints", data)
    for values in ([300], [300, 5], [2 ** 64 - 1, 7, 128], [0, 16384, 1]):
        data = ref.uvarints(values)
        ends = [len(ref.uvarints(values[:i])) for i in range(len(values) + 1)]
        for cut in range(len(data)):                      # every proper prefix, at a number boundary or inside one
            if cut in ends:
                check_decode(sol, data[:cut], values[:ends.index(cut)])
            else:
                raises_value_error(sol, "decode_uvarints", data[:cut])


@case("Part 2", title="zigzag / unzigzag 在小范围内逐个对照")
def p2_zigzag_small(sol):
    for n in range(-300, 301):
        check_zigzag(sol, n)
    for z in range(0, 601):
        check_unzigzag(sol, z)


@case("Part 2", title="zigzag 超出 64 位范围的整数")
def p2_zigzag_big(sol):
    for n in (2 ** 63 - 1, -(2 ** 63), 2 ** 63, -(2 ** 63) - 1, 2 ** 64, -(2 ** 64), 2 ** 100 + 7, -(10 ** 30)):
        check_zigzag(sol, n)
        check_unzigzag(sol, ref.zigzag(n))


# ---------------------------------------------------------------- Part 3
EXAMPLE = [998, 1000, 1005, 1000]


@case("Part 3", sample=True, title="题目例子 for_size == 6")
def p3_example_size(sol):
    calling("for_size([998, 1000, 1005, 1000])")
    expect(sol.for_size(list(EXAMPLE)), 6, call="for_size([998, 1000, 1005, 1000])")


@case("Part 3", sample=True, title="题目例子 编码后再解码")
def p3_example_roundtrip(sol):
    call = calling("decode_for(encode_for([998, 1000, 1005, 1000]))")
    expect(as_list(sol.decode_for(sol.encode_for(list(EXAMPLE)))), [998, 1000, 1005, 1000], call=call)


@case("Part 3", title="题目例子的字节：头部 + 负载")
def p3_example_bytes(sol):
    check_encode_for(sol, EXAMPLE)
    check_decode_for(sol, EXAMPLE)


@case("Part 3", title="全部相等（w = 0）：没有负载")
def p3_all_equal(sol):
    for values in ([7, 7, 7, 7], [0], [-5], [123456789] * 3, [-1] * 200, [64] * 2):
        check_block(sol, values)


@case("Part 3", title="负数和 m 跨 varint 字节数边界")
def p3_negative_and_m(sol):
    for values in ([-3, 0, -2, -1], [-1000, -990, -999], [63, 70, 66], [64, 70, 66], [-64, -60], [-65, -60], [0, 5, 3], [0, 1]):
        check_block(sol, values)


@case("Part 3", title="m = 0：zigzag(0) 也占 1 个字节")
def p3_m_zero(sol):
    for values in ([0], [0, 0, 0], [0, 1, 2, 3], [0, 255], [0] * 130, [5, 0]):
        check_for_size(sol, values)
        check_best(sol, values)


@case("Part 3", title="m 超出 64 位，跨度仍很小")
def p3_big_m(sol):
    for values in ([10 ** 30, 10 ** 30 + 5], [-(2 ** 70), -(2 ** 70) + 1000], [2 ** 64] * 3):
        check_block(sol, values)


@case("Part 3", title="w = 32 可以，w = 33 抛 ValueError")
def p3_width_limit(sol):
    for values in ([0, 2 ** 32 - 1], [-(2 ** 31), 2 ** 31 - 1, 0], [5, 5 + 2 ** 31]):
        check_block(sol, values)
    for values in ([0, 2 ** 32], [-(2 ** 31), 2 ** 31], [0, 2 ** 40], [1, 2, 3, -(2 ** 40)]):
        for fname in ("encode_for", "for_size", "best_encoding"):
            raises_value_error(sol, fname, list(values))


@case("Part 3", title="空块抛 ValueError")
def p3_empty(sol):
    for fname in ("encode_for", "for_size", "best_encoding"):
        raises_value_error(sol, fname, [])


@case("Part 3", title="随机块（对照逐位拼接的编码）")
def p3_random(sol):
    r = rng()
    for _ in range(300):
        values = ref.random_block(r)
        if len(values) <= 40:
            check_block(sol, values)
        else:
            shown = f"<{len(values)} 个值的块，min {min(values)}，max {max(values)}>"
            check_encode_for(sol, values, f"encode_for({shown})")
            check_decode_for(sol, values, f"decode_for(<encode_for({shown}) 的正确输出，"
                                          f"{len(ref.encode_for(values))} 个字节>)")
            check_for_size(sol, values, f"for_size({shown})")
            check_best(sol, values, f"best_encoding({shown})")


@case("Part 3", title="decode_for：数据被截断")
def p3_decode_truncated(sol):
    for values in (EXAMPLE, [7, 7, 7, 7], [-(2 ** 40), -(2 ** 40) + 3], [0, 2 ** 32 - 1], [1] + [2] * 129):
        data = ref.encode_for(values)
        for cut in range(len(data)):
            raises_value_error(sol, "decode_for", data[:cut])


@case("Part 3", title="decode_for：负载后面多出字节")
def p3_decode_trailing(sol):
    for values in (EXAMPLE, [0, 1], [0, 2 ** 32 - 1], [1, 2, 3]):
        for extra in (b"\x00", b"\x01\x02"):
            raises_value_error(sol, "decode_for", ref.encode_for(values) + extra)


@case("Part 3", title="decode_for：w = 0 时头部后面不能再有字节")
def p3_decode_trailing_w0(sol):
    for values in ([7, 7, 7, 7], [0], [-3, -3]):
        for extra in (b"\x00", b"\x07\x07"):
            raises_value_error(sol, "decode_for", ref.encode_for(values) + extra)


class Trap:
    """Stands in for an encoder while for_size runs: for_size must use the size formulas, not encode."""

    def __init__(self, name, call):
        self.name, self.call, self.hit = name, call, False

    def __call__(self, *args, **kwargs):
        self.hit = True
        raise Mismatch(self.message(), self.call, show=False)

    def message(self):
        return f"for_size 应当只用头部和负载的大小公式算出字节数，不构造任何字节；它却调用了 {self.name}()"


@case("Part 3", title="for_size 只用公式，不真的编码")
def p3_for_size_formula(sol):
    for values in (EXAMPLE, [7, 7, 7, 7], [-5, 300, 2 ** 20]):
        call = calling(show_call("for_size", values))
        names = [name for name in ("encode_for", "pack", "encode_uvarints") if hasattr(sol, name)]
        saved = {name: getattr(sol, name) for name in names}
        traps = [Trap(name, call) for name in names]
        try:
            for trap in traps:
                setattr(sol, trap.name, trap)
            got = sol.for_size(list(values))
        finally:
            for name, fn in saved.items():
                setattr(sol, name, fn)
        for trap in traps:
            if trap.hit:                                   # raised again in case the answer caught the first one
                raise Mismatch(trap.message(), call, show=False)
        expect(got, len(ref.encode_for(values)), call=call)


@case("Part 3", title="best_encoding：参考解答里的三个块")
def p3_best_examples(sol):
    for values in (EXAMPLE, [100000, 100001, 100000, 100003, 100002], [1, 2, 1, 3, 1000000000]):
        check_best(sol, values)


@case("Part 3", title="best_encoding：恰好打平时选 \"for\"")
def p3_best_ties(sol):
    for values in ([5, 5], [1, 2, 3], [0, 0], [-3, 0, -2]):
        check_best(sol, values)
    check_best(sol, [-64] + [0] * 125 + [100, 191], "best_encoding([-64] + [0] * 125 + [100, 191])")


@case("Part 3", title="best_encoding：varint 方案也要算上个数")
def p3_best_count_varint(sol):
    # 128 values: the count varint takes 2 bytes, and frame-of-reference wins by exactly 1 byte
    check_best(sol, [-64] + [0] * 124 + [100, 150, 191], "best_encoding([-64] + [0] * 124 + [100, 150, 191])")
    # 64 values: the count is a plain varint (1 byte); zigzagging it would cost 2 bytes and turn "varint" into a tie
    check_best(sol, [0] * 63 + [200], "best_encoding([0] * 63 + [200])")
    check_best(sol, [0], "best_encoding([0])")
    check_best(sol, [64], "best_encoding([64])")
