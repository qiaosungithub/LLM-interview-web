"""Tests for the sharded matmul problem: Part 2 (the sharded linear layers) and Part 3 (the debugged sharded MLP).

Part 1 is a derivation on paper and is not judged. Every expected value is the unsharded float64 product (ref.py), and
every shard is cut by the statement's rule (contiguous, the remainder one apiece to the first shards).

Tolerances. Part 2: a float64 dot product of length m <= 50 is off by at most about m * 1.1e-16 times the sum of the
absolute values of its terms, and cutting it into n partial sums adds at most n more additions; the tests allow 1e-11
times the largest such sum (ref.abs_product), about 100 times the worst case, while a missing, extra or misplaced term
is of the size of the terms themselves. all_gather only copies, so it must be exact. Part 3: the statement's own
bound, 1e-8, on quantities of size about 1 (the inputs are drawn like check_gradients draws them).

Where the two reductions of Part 3 go. The statement asks only that check_gradients() pass; it does not say whether the
missing all_reduce of Y belongs in layer2_forward or in check_gradients (which reads Y_shards[0]), nor whether the one
of dX belongs in layer1_backward or where check_gradients reads dX_shards[0]. So when layer2_forward returns exactly
the per-device partial products H_k W2_k, or layer1_backward returns exactly each device's own term of dX, the tests
accept it provided the answer's own check_gradients() gets that quantity right (it then sums them itself), and the
pipeline test sums them too. Without that, the partial results are the unfixed bug and get the targeted message.
"""
import contextlib
import io

import numpy as np

from judgelib import Mismatch, calling, case, expect_close, expect_true, rng
import ref

P2_ATOL = 1e-11     # times ref.abs_product of the product being checked
P3_ATOL = 1e-8      # check_gradients() should return errors below 1e-8 for any seed


# ---------------------------------------------------------------- helpers

def copies(arrays):
    return [np.array(a, copy=True) for a in arrays]


def shape_of(x):
    try:
        return np.shape(x)
    except Exception:
        return None


def per_device(got, n, what, call):
    """got as a list of n per-device entries; anything else is a wrong answer."""
    if isinstance(got, (list, tuple)) or (isinstance(got, np.ndarray) and got.ndim >= 1):
        items = list(got)
        expect_true(len(items) == n, f"{what} 应是长度为 {n} 的列表（每个设备一项），得到的长度是 {len(items)}", call, got)
        return items
    expect_true(False, f"{what} 应是长度为 {n} 的列表（每个设备一项），得到的是 {type(got).__name__}", call, got)


def pair(got, names, call):
    expect_true(isinstance(got, (list, tuple)) and len(got) == 2, f"应返回二元组 {names}", call, got)
    return got


def same(got, alt, atol):
    """got equals the alternative array alt within atol (used only to name a mistake)."""
    try:
        a = np.asarray(got, dtype=float)
    except Exception:
        return False
    return a.shape == alt.shape and bool(np.allclose(a, alt, rtol=0, atol=atol))


def close(got, want, atol, call, label, hint=None):
    """expect_close with a label in front of the message; hint() may return a sentence that names the mistake."""
    try:
        expect_close(got, want, rtol=0, atol=atol, call=call)
    except Mismatch as e:
        better = ""
        if hint is not None:
            try:
                better = hint() or ""
            except Exception:
                better = ""
        e.msg = label + (better or e.msg)
        raise


def transposed_msg(got, want, name):
    s = shape_of(got)
    if s is not None and s != want.shape and s == want.T.shape:
        return f"形状应与 {name} 相同，是 {want.shape}；得到的 {s} 是它的转置"
    return ""


# ---------------------------------------------------------------- Part 2 checks, one input each

def check_column_forward(sol, X, W, n, setup):
    call = calling(setup + f"\ncolumn_parallel_forward(X, np.array_split(W, {n}, axis=1))")
    W_shards = ref.split(W, n, 1)
    want = ref.split(X @ W, n, 1)
    atol = P2_ATOL * ref.abs_product(X, W)
    got = per_device(sol.column_parallel_forward(X.copy(), copies(W_shards)), n, "返回值", call)
    full = (X.shape[0], W.shape[1])
    for k in range(n):
        close(got[k], want[k], atol, call, f"Y_shards[{k}]：",
              lambda: (f"设备 {k} 应只持有自己那几列的 Y，形状 {want[k].shape}；得到的 {full} 是拼好的完整 Y，"
                       f"这个函数不做 all_gather" if n > 1 and shape_of(got[k]) == full else ""))
    return got


def check_column_backward(sol, X, W, G, n, setup):
    call = calling(setup + f"\ncolumn_parallel_backward(X, np.array_split(W, {n}, axis=1), "
                           f"np.array_split(G, {n}, axis=1))")
    W_shards, G_shards = ref.split(W, n, 1), ref.split(G, n, 1)
    dW_want, dX_want = ref.split(X.T @ G, n, 1), G @ W.T
    own = [Gk @ Wk.T for Gk, Wk in zip(G_shards, W_shards)]
    atol_dW, atol_dX = P2_ATOL * ref.abs_product(X.T, G), P2_ATOL * ref.abs_product(G, W.T)
    got = pair(sol.column_parallel_backward(X.copy(), copies(W_shards), copies(G_shards)),
               "(dW_shards, dX_shards)", call)
    dW = per_device(got[0], n, "dW_shards", call)
    dX = per_device(got[1], n, "dX_shards", call)
    for k in range(n):
        close(dW[k], dW_want[k], atol_dW, call, f"dW_shards[{k}]：",
              lambda: transposed_msg(dW[k], dW_want[k], f"W_shards[{k}]"))
    for k in range(n):
        close(dX[k], dX_want, atol_dX, call, f"dX_shards[{k}]：",
              lambda: (f"它只是设备 {k} 自己那一项 dY_{k} W_{k}ᵀ；每个设备都持有完整的 X，"
                       f"dX 应是所有设备这一项之和" if n > 1 and same(dX[k], own[k], atol_dX) else ""))


def check_row_forward(sol, X, W, n, setup):
    call = calling(setup + f"\nrow_parallel_forward(np.array_split(X, {n}, axis=1), np.array_split(W, {n}, axis=0))")
    X_shards, W_shards = ref.split(X, n, 1), ref.split(W, n, 0)
    want = X @ W
    partial = [Xk @ Wk for Xk, Wk in zip(X_shards, W_shards)]
    atol = P2_ATOL * ref.abs_product(X, W)
    got = per_device(sol.row_parallel_forward(copies(X_shards), copies(W_shards)), n, "返回值", call)
    for k in range(n):
        close(got[k], want, atol, call, f"Y_shards[{k}]：",
              lambda: (f"它只是设备 {k} 的部分和 X_{k} W_{k}；按行切时每个设备上都应得到完整的 Y = Σₖ Xₖ Wₖ"
                       if n > 1 and same(got[k], partial[k], atol) else ""))


def check_row_backward(sol, X, W, G, n, setup):
    call = calling(setup + f"\nrow_parallel_backward(np.array_split(X, {n}, axis=1), "
                           f"np.array_split(W, {n}, axis=0), G)")
    X_shards, W_shards = ref.split(X, n, 1), ref.split(W, n, 0)
    dW_want, dX_want = ref.split(X.T @ G, n, 0), ref.split(G @ W.T, n, 1)
    atol_dW, atol_dX = P2_ATOL * ref.abs_product(X.T, G), P2_ATOL * ref.abs_product(G, W.T)
    got = pair(sol.row_parallel_backward(copies(X_shards), copies(W_shards), G.copy()), "(dW_shards, dX_shards)", call)
    dW = per_device(got[0], n, "dW_shards", call)
    dX = per_device(got[1], n, "dX_shards", call)
    full = X.shape
    for k in range(n):
        close(dW[k], dW_want[k], atol_dW, call, f"dW_shards[{k}]：",
              lambda: transposed_msg(dW[k], dW_want[k], f"W_shards[{k}]"))
    for k in range(n):
        close(dX[k], dX_want[k], atol_dX, call, f"dX_shards[{k}]：",
              lambda: (f"设备 {k} 只持有 X_{k}，它的 dX 也只是对应的那几列，形状 {dX_want[k].shape}；"
                       f"得到的 {full} 是完整的 dX" if n > 1 and shape_of(dX[k]) == full else ""))


def column_inputs(r):
    """(seed, B, d_in, d_out, n, a) with n <= d_out: the statement's 7 = 3 + 2 + 2, the extremes, then random."""
    fixed = [(5, 4, 7, 3, 1), (1, 1, 1, 1, 100), (3, 2, 5, 5, 100), (4, 3, 6, 1, 1), (50, 50, 50, 7, 100),
             (50, 50, 50, 50, 100), (2, 50, 9, 4, 100)]
    out = [(r.randrange(2 ** 31), *c) for c in fixed]
    for _ in range(12):
        d_out = r.randint(1, 50)
        out.append((r.randrange(2 ** 31), r.randint(1, 50), r.randint(1, 50), d_out, r.randint(1, d_out),
                    r.choice([1, 100])))
    return out


def row_inputs(r):
    """(seed, B, d_in, d_out, n, a) with n <= d_in."""
    fixed = [(5, 7, 4, 3, 1), (1, 1, 1, 1, 100), (3, 5, 2, 5, 100), (4, 6, 3, 1, 1), (50, 50, 50, 7, 100),
             (50, 50, 50, 50, 100), (2, 9, 50, 4, 100)]
    out = [(r.randrange(2 ** 31), *c) for c in fixed]
    for _ in range(12):
        d_in = r.randint(1, 50)
        out.append((r.randrange(2 ** 31), r.randint(1, 50), d_in, r.randint(1, 50), r.randint(1, d_in),
                    r.choice([1, 100])))
    return out


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="验证代码 check() 的三组规模")
def p2_check(sol):
    for B, d_in, d_out, n in [(5, 4, 6, 3), (5, 4, 7, 3), (8, 9, 5, 4)]:
        g = np.random.default_rng(0)
        X = g.standard_normal((B, d_in))
        W = g.standard_normal((d_in, d_out))
        G = g.standard_normal((B, d_out))
        setup = (f"rng = np.random.default_rng(0); X = rng.standard_normal(({B}, {d_in})); "
                 f"W = rng.standard_normal(({d_in}, {d_out})); G = rng.standard_normal(({B}, {d_out}))")
        Y_shards = check_column_forward(sol, X, W, n, setup)
        call = calling(setup + f"\nall_gather(column_parallel_forward(X, np.array_split(W, {n}, axis=1)), axis=1)[0]")
        gathered = per_device(sol.all_gather(copies(Y_shards), 1), n, "all_gather 的返回值", call)
        close(gathered[0], X @ W, P2_ATOL * ref.abs_product(X, W), call, "拼起来的 Y：")
        check_column_backward(sol, X, W, G, n, setup)
        check_row_forward(sol, X, W, n, setup)
        check_row_backward(sol, X, W, G, n, setup)


@case("Part 2", title="all_gather 沿给定的轴拼接")
def p2_all_gather(sol):
    r = rng()
    tests = [((5, 7), 3, 1), ((7, 4), 3, 0), ((2, 3), 1, 0), ((1, 1), 1, 1), ((4, 50), 50, 1), ((7, 6), 7, 0)]
    for _ in range(8):
        axis = r.randint(0, 1)
        shape = (r.randint(1, 50), r.randint(1, 50))
        tests.append((shape, r.randint(1, shape[axis]), axis))
    for shape, n, axis in tests:
        seed, a = r.randrange(2 ** 31), r.choice([1, 100])
        full = ref.full_input(seed, shape, a)
        call = calling(ref.full_recipe(seed, shape, a) + f"; shards = np.array_split(full, {n}, axis={axis})"
                       f"\nall_gather(shards, axis={axis})")
        got = per_device(sol.all_gather(ref.split(full, n, axis), axis), n, "返回值", call)
        for k in range(n):
            close(got[k], full, 0.0, call, f"第 {k} 项：")


@case("Part 2", title="all_reduce 按元素求和")
def p2_all_reduce(sol):
    r = rng()
    tests = [(3, (5, 7)), (1, (2, 3)), (4, (1, 1)), (50, (2, 2)), (2, (50, 50))]
    for _ in range(8):
        tests.append((r.randint(1, 10), (r.randint(1, 50), r.randint(1, 50))))
    for n, shape in tests:
        seed, a = r.randrange(2 ** 31), r.choice([1, 100])
        shards = ref.summand_inputs(seed, n, shape, a)
        want = np.sum(shards, axis=0)
        atol = P2_ATOL * float(np.sum(np.abs(shards), axis=0).max())
        call = calling(ref.summand_recipe(seed, n, shape, a) + "\nall_reduce(shards)")
        got = per_device(sol.all_reduce(copies(shards)), n, "返回值", call)
        for k in range(n):
            close(got[k], want, atol, call, f"第 {k} 项：")


@case("Part 2", title="按列切的前向：每个设备自己那几列")
def p2_column_forward(sol):
    for seed, B, d_in, d_out, n, a in column_inputs(rng()):
        X, W, _ = ref.linear_inputs(seed, B, d_in, d_out, a)
        check_column_forward(sol, X, W, n, ref.linear_recipe(seed, B, d_in, d_out, a))


@case("Part 2", title="按列切的反向：dW 本地、dX 求和")
def p2_column_backward(sol):
    for seed, B, d_in, d_out, n, a in column_inputs(rng()):
        X, W, G = ref.linear_inputs(seed, B, d_in, d_out, a)
        check_column_backward(sol, X, W, G, n, ref.linear_recipe(seed, B, d_in, d_out, a))


@case("Part 2", title="按行切的前向：每个设备完整的 Y")
def p2_row_forward(sol):
    for seed, B, d_in, d_out, n, a in row_inputs(rng()):
        X, W, _ = ref.linear_inputs(seed, B, d_in, d_out, a)
        check_row_forward(sol, X, W, n, ref.linear_recipe(seed, B, d_in, d_out, a))


@case("Part 2", title="按行切的反向：两个梯度都是本地的")
def p2_row_backward(sol):
    for seed, B, d_in, d_out, n, a in row_inputs(rng()):
        X, W, G = ref.linear_inputs(seed, B, d_in, d_out, a)
        check_row_backward(sol, X, W, G, n, ref.linear_recipe(seed, B, d_in, d_out, a))


# ---------------------------------------------------------------- Part 3

# (B, d_in, d_hidden, d_out, n) with n <= d_hidden; the first is the given file's own constants.
MLP_SIZES = [(5, 4, 7, 3, 3), (5, 4, 6, 3, 3), (6, 5, 10, 4, 4), (3, 2, 5, 2, 5), (4, 3, 4, 2, 1), (8, 6, 9, 5, 2),
             (2, 3, 7, 2, 7)]
ACTIVATION = "数值与用 1 − Z1² 当作 tanh 的导数时一致，而 tanh′(z) = 1 − tanh(z)²"


def mlp_cases(r):
    return [(r.randrange(2 ** 31), *size) for size in MLP_SIZES] + [(r.randrange(2 ** 31), *MLP_SIZES[0])]


def reduced_in_check_gradients(sol, key):
    """True when the answer's own check_gradients() gets `key` ("Y" or "dX") below 1e-8: the reduction was moved into
    check_gradients, so per-device partial results from the layer function are acceptable (see the module docstring)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return float(sol.check_gradients()[key]) < P3_ATOL
    except Exception:
        return False


def partial_results(sol, got, partial, key, n):
    """got is exactly the list of per-device partial results and the answer sums them in check_gradients."""
    return n > 1 and all(same(g, p, P3_ATOL) for g, p in zip(got, partial)) and reduced_in_check_gradients(sol, key)


def summed(items):
    """The elementwise sum of the per-device arrays, or None if they do not add up."""
    try:
        return np.sum([np.asarray(x, dtype=float) for x in items], axis=0)
    except Exception:
        return None


def moved_reduction(sol, items, want, key, n):
    """In the pipeline: items[0] is not the full array but the items sum to it, and the answer's check_gradients
    sums them too."""
    total = summed(items)
    return (n > 1 and not same(items[0], want, P3_ATOL) and total is not None and same(total, want, P3_ATOL)
            and reduced_in_check_gradients(sol, key))


def check_errors(errs, call):
    expect_true(isinstance(errs, dict), f"check_gradients 应返回一个 dict，得到的是 {type(errs).__name__}", call, errs)
    for key in ("Y", "dW1", "dW2", "dX"):
        expect_true(key in errs, f"返回的 dict 里没有 {key!r}", call, errs)
        try:
            value = float(errs[key])
        except Exception:
            value = None
        expect_true(value is not None, f"{key!r} 对应的值不是一个数", call, errs)
        expect_true(value < P3_ATOL, f"{key} 的误差是 {value:.3g}，应低于 1e-8", call, errs)


@case("Part 3", sample=True, title="check_gradients() 四项误差都低于 1e-8")
def p3_check_gradients(sol):
    call = calling("check_gradients()")
    check_errors(sol.check_gradients(), call)


@case("Part 3", title="任意随机种子：check_gradients(seed)")
def p3_seeds(sol):
    r = rng()
    for seed in list(range(1, 11)) + [r.randrange(2 ** 32) for _ in range(5)]:
        call = calling(f"check_gradients({seed})")
        check_errors(sol.check_gradients(seed), call)


@case("Part 3", title="shard_columns / shard_rows 的分片边界")
def p3_shard(sol):
    for fn, axis in (("shard_columns", 1), ("shard_rows", 0)):
        for length, n in [(7, 3), (6, 3), (8, 3), (10, 4), (5, 5), (4, 1), (9, 2), (50, 7), (50, 50)]:
            shape = (length, 3) if axis == 0 else (3, length)
            A = np.arange(float(length * 3)).reshape(shape)
            call = calling(f"{fn}(np.arange({float(length * 3)}).reshape{shape}, {n})")
            want = ref.split(A, n, axis)
            got = per_device(getattr(sol, fn)(A.copy(), n), n, "返回值", call)
            sizes = ref.sizes(length, n)
            for k in range(n):
                start = sum(sizes[:k])
                unit = "行" if axis == 0 else "列"
                close(got[k], want[k], 0.0, call, f"第 {k} 片：",
                      lambda: (f"长度 {length} 分成 {n} 片应是 {sizes} {unit}（余数按每片一{unit}分给前面的分片），"
                               f"第 {k} 片是第 {start}–{start + sizes[k] - 1} {unit}"))


@case("Part 3", title="layer1_forward：按列切的第一层")
def p3_layer1_forward(sol):
    for seed, B, d_in, d_h, d_out, n in mlp_cases(rng()):
        X, W1, _, _, _ = ref.mlp_inputs(seed, B, d_in, d_h, d_out)
        call = calling(ref.mlp_recipe(seed, B, d_in, d_h, d_out) + f"; W1_shards = np.array_split(W1, {n}, axis=1)"
                       "\nlayer1_forward(X, W1_shards)")
        got = pair(sol.layer1_forward(X.copy(), ref.split(W1, n, 1)), "(Z1_shards, H_shards)", call)
        Z1 = per_device(got[0], n, "Z1_shards", call)
        H = per_device(got[1], n, "H_shards", call)
        want_Z1 = ref.split(X @ W1, n, 1)
        for k in range(n):
            close(Z1[k], want_Z1[k], P3_ATOL, call, f"Z1_shards[{k}]：")
            close(H[k], np.tanh(want_Z1[k]), P3_ATOL, call, f"H_shards[{k}]：")


@case("Part 3", title="layer2_forward：按行切的第二层")
def p3_layer2_forward(sol):
    for seed, B, d_in, d_h, d_out, n in mlp_cases(rng()):
        X, W1, W2, _, _ = ref.mlp_inputs(seed, B, d_in, d_h, d_out)
        H = np.tanh(X @ W1)
        H_shards, W2_shards = ref.split(H, n, 1), ref.split(W2, n, 0)
        call = calling(ref.mlp_recipe(seed, B, d_in, d_h, d_out) + f"; H_shards = np.array_split(np.tanh(X @ W1), {n}, "
                       f"axis=1); W2_shards = np.array_split(W2, {n}, axis=0)\nlayer2_forward(H_shards, W2_shards)")
        want = H @ W2
        partial = [Hk @ W2k for Hk, W2k in zip(H_shards, W2_shards)]
        got = per_device(sol.layer2_forward(copies(H_shards), copies(W2_shards)), n, "返回值", call)
        if partial_results(sol, got, partial, "Y", n):
            continue                    # the answer's check_gradients does the all_reduce of Y
        for k in range(n):
            close(got[k], want, P3_ATOL, call, f"Y_shards[{k}]：",
                  lambda: (f"它只是设备 {k} 的部分和 H_{k} W2_{k}；第二层按行切，每个设备上都应得到完整的 Y"
                           if n > 1 and same(got[k], partial[k], P3_ATOL) else ""))


@case("Part 3", title="layer2_backward：第二层的梯度")
def p3_layer2_backward(sol):
    for seed, B, d_in, d_h, d_out, n in mlp_cases(rng()):
        X, W1, W2, target, _ = ref.mlp_inputs(seed, B, d_in, d_h, d_out)
        full = ref.mlp(X, W1, W2, target)
        H, dY = full["H"], full["dY"]
        call = calling(ref.mlp_recipe(seed, B, d_in, d_h, d_out) + f"; H = np.tanh(X @ W1); "
                       f"dY = 2 * (H @ W2 - target) / target.size; H_shards = np.array_split(H, {n}, axis=1); "
                       f"W2_shards = np.array_split(W2, {n}, axis=0)\nlayer2_backward(H_shards, W2_shards, dY)")
        got = pair(sol.layer2_backward(ref.split(H, n, 1), ref.split(W2, n, 0), dY.copy()),
                   "(dW2_shards, dH_shards)", call)
        dW2 = per_device(got[0], n, "dW2_shards", call)
        dH = per_device(got[1], n, "dH_shards", call)
        want_dW2, want_dH = ref.split(full["dW2"], n, 0), ref.split(dY @ W2.T, n, 1)
        for k in range(n):
            close(dW2[k], want_dW2[k], P3_ATOL, call, f"dW2_shards[{k}]：",
                  lambda: transposed_msg(dW2[k], want_dW2[k], f"W2_shards[{k}]"))
        for k in range(n):
            close(dH[k], want_dH[k], P3_ATOL, call, f"dH_shards[{k}]：")


@case("Part 3", title="layer1_backward：dW1、激活导数和 dX")
def p3_layer1_backward(sol):
    for seed, B, d_in, d_h, d_out, n in mlp_cases(rng()):
        X, W1, _, _, dH = ref.mlp_inputs(seed, B, d_in, d_h, d_out)
        W1_shards = ref.split(W1, n, 1)
        Z1_shards = [X @ W1k for W1k in W1_shards]
        H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]
        dH_shards = ref.split(dH, n, 1)
        call = calling(ref.mlp_recipe(seed, B, d_in, d_h, d_out) + f"; W1_shards = np.array_split(W1, {n}, axis=1); "
                       "Z1_shards = [X @ W1k for W1k in W1_shards]; H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]; "
                       f"dH_shards = np.array_split(dH, {n}, axis=1)"
                       "\nlayer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards)")
        dW1_full, dX_want = ref.layer1_grads(X, W1, dH)
        alt_dW1_full, alt_dX = ref.layer1_grads(X, W1, dH, derivative_from_z=True)
        want_dW1, alt_dW1 = ref.split(dW1_full, n, 1), ref.split(alt_dW1_full, n, 1)
        dZ1_shards = ref.split(dH * (1 - np.tanh(X @ W1) ** 2), n, 1)
        own = [dZ1k @ W1k.T for dZ1k, W1k in zip(dZ1_shards, W1_shards)]
        got = pair(sol.layer1_backward(X.copy(), copies(W1_shards), copies(Z1_shards), copies(H_shards),
                                       copies(dH_shards)), "(dW1_shards, dX_shards)", call)
        dW1 = per_device(got[0], n, "dW1_shards", call)
        dX = per_device(got[1], n, "dX_shards", call)
        for k in range(n):
            close(dW1[k], want_dW1[k], P3_ATOL, call, f"dW1_shards[{k}]：",
                  lambda: (transposed_msg(dW1[k], want_dW1[k], f"W1_shards[{k}]")
                           or (ACTIVATION if same(dW1[k], alt_dW1[k], P3_ATOL) else "")))
        if partial_results(sol, dX, own, "dX", n):
            continue                    # the answer's check_gradients does the all_reduce of dX
        for k in range(n):
            close(dX[k], dX_want, P3_ATOL, call, f"dX_shards[{k}]：",
                  lambda: (ACTIVATION if same(dX[k], alt_dX, P3_ATOL) else
                           f"它只是设备 {k} 自己那一项 dZ1_{k} W1_{k}ᵀ；X 在每个设备上都是完整副本，dX 应是所有设备"
                           f"这一项之和" if n > 1 and same(dX[k], own[k], P3_ATOL) else ""))


@case("Part 3", title="其他规模下整条流水线")
def p3_pipeline(sol):
    for seed, B, d_in, d_h, d_out, n in mlp_cases(rng()):
        X, W1, W2, target, _ = ref.mlp_inputs(seed, B, d_in, d_h, d_out)
        want = ref.mlp(X, W1, W2, target)
        call = calling(ref.mlp_recipe(seed, B, d_in, d_h, d_out) + f"\nW1_shards = shard_columns(W1, {n})\n"
                       f"W2_shards = shard_rows(W2, {n})\n"
                       "Z1_shards, H_shards = layer1_forward(X, W1_shards)\n"
                       "Y = layer2_forward(H_shards, W2_shards)[0]\n"
                       "loss, dY = mse_loss_and_grad(Y, target)\n"
                       "dW2_shards, dH_shards = layer2_backward(H_shards, W2_shards, dY)\n"
                       "dW1_shards, dX_shards = layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards)")
        W1_shards = per_device(sol.shard_columns(W1.copy(), n), n, "W1_shards", call)
        W2_shards = per_device(sol.shard_rows(W2.copy(), n), n, "W2_shards", call)
        Z1_shards, H_shards = pair(sol.layer1_forward(X.copy(), W1_shards), "(Z1_shards, H_shards)", call)
        Y_shards = per_device(sol.layer2_forward(H_shards, W2_shards), n, "layer2_forward 的返回值", call)
        Y = Y_shards[0]
        if moved_reduction(sol, Y_shards, want["Y"], "Y", n):
            Y = summed(Y_shards)        # the answer's check_gradients does the all_reduce of Y
            call = calling(call.replace("Y = layer2_forward(H_shards, W2_shards)[0]",
                                        "Y = all_reduce(layer2_forward(H_shards, W2_shards))[0]"))
        close(Y, want["Y"], P3_ATOL, call, "Y：")
        loss, dY = pair(sol.mse_loss_and_grad(Y, target.copy()), "(loss, dY)", call)
        close(loss, want["loss"], P3_ATOL, call, "loss：")
        dW2_shards, dH_shards = pair(sol.layer2_backward(H_shards, W2_shards, dY), "(dW2_shards, dH_shards)", call)
        dW1_shards, dX_shards = pair(sol.layer1_backward(X.copy(), W1_shards, Z1_shards, H_shards, dH_shards),
                                     "(dW1_shards, dX_shards)", call)
        for name, got, full, axis in (("dW1", dW1_shards, want["dW1"], 1), ("dW2", dW2_shards, want["dW2"], 0)):
            got = per_device(got, n, f"{name}_shards", call)
            for k, wk in enumerate(ref.split(full, n, axis)):
                close(got[k], wk, P3_ATOL, call, f"{name}_shards[{k}]：")
        dX_shards = per_device(dX_shards, n, "dX_shards", call)
        if moved_reduction(sol, dX_shards, want["dX"], "dX", n):
            continue                    # they sum to dX, and the answer's check_gradients does that all_reduce
        for k, dXk in enumerate(dX_shards):
            close(dXk, want["dX"], P3_ATOL, call, f"dX_shards[{k}]：")
