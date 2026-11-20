import numpy as np
import torch
from torch.overrides import TorchFunctionMode

import judgelib
from judgelib import Mismatch, case, expect_close, expect_true, info, rng
import ref

torch.set_num_threads(1)
F64 = torch.float64

# The worked example: small integers, which float32 represents and multiplies exactly in any order.
EX_TOL = 1e-6
# Random inputs are float64. Every correct method adds or multiplies the same O(N D) terms in some order, so it
# differs from the loop in ref.py by about 1e-15 of the largest entry; a mistake changes entries by O(1) of it.
TOL = 1e-9
# The statement's own float32 snippet (Part 1(b)): unit roundoff 6e-8 times a few dozen operations. The statement
# does not fix the output dtype, so a float32 result for a float64 input is compared at this tolerance too.
F32_TOL = 1e-4
GUESS_TOL = 1e-7          # an answer "looks like" a known mistake when it matches it this closely

W_EX_CODE = "W = torch.tensor([[[1., 1.], [0., 1.]], [[2., 0.], [0., 1.]], [[1., 0.], [3., 1.]]])"
W_EX = [[[1., 1.], [0., 1.]], [[2., 0.], [0., 1.]], [[1., 0.], [3., 1.]]]
P_EX = [[[1., 1.], [0., 1.]], [[2., 1.], [0., 1.]], [[5., 1.], [3., 1.]]]       # printed in the statement
# dL/dW for L = P.sum() (every G_i is all ones), worked out by hand with Part 3's formulas:
# A_2 = J, A_1 = J + J W_2^T, A_0 = J + A_1 W_1^T; dW_0 = A_0, dW_1 = P_0^T A_1, dW_2 = P_1^T A_2.
DW_EX = [[[5., 6.], [5., 6.]], [[2., 5.], [4., 10.]], [[2., 2.], [2., 2.]]]

# Calls that multiply matrices, by the name torch passes to __torch_function__ (`A @ B` arrives as "matmul"), plus
# elementwise products, so that a loop over i written as (A[:, :, None] * B[None]).sum(1) is counted as well.
MATMUL_NAMES = {"matmul", "__matmul__", "__rmatmul__", "linalg_matmul", "bmm", "mm", "mv", "einsum", "baddbmm",
                "addbmm", "addmm", "tensordot", "linalg_multi_dot", "chain_matmul",
                "mul", "__mul__", "__rmul__", "mul_", "__imul__", "multiply"}


def build(code):
    """Run `code` (which builds W, G, ...) and return its names and the same code as the start of the call."""
    ns = {"torch": torch}
    exec(code, ns)
    return ns, f"{code}; "


def seeds(n):
    r = rng()
    return [r.randrange(2 ** 31) for _ in range(n)]


def random_code(s, N, D, grad=False):
    code = (f"g = torch.Generator().manual_seed({s}); "
            f"W = torch.randn({N}, {D}, {D}, generator=g, dtype=torch.float64) / {D} ** 0.5")
    if grad:
        code += f"; G = torch.randn({N}, {D}, {D}, generator=g, dtype=torch.float64)"
    return code


def as_np(x):
    try:
        if torch.is_tensor(x):
            return x.detach().to(F64).cpu().numpy()
        return np.asarray(x, dtype=float)
    except Exception:
        return None


def looks_like(got, variant):
    a, b = as_np(got), variant.detach().to(F64).numpy()
    if a is None or a.shape != b.shape or not np.isfinite(a).all():
        return False
    scale = max(float(np.abs(b).max()), 1e-300)
    return bool(np.allclose(a, b, rtol=GUESS_TOL, atol=GUESS_TOL * scale))


def check(got, want, call, guesses=(), tol=TOL):
    """got == want within tol, relative to want's largest entry. On a mismatch, name the first known mistake that
    got matches, or say that the result lost precision to a narrower dtype."""
    want = torch.as_tensor(want).detach()
    scale = float(want.abs().max()) or 1.0
    if tol < F32_TOL and narrower_float(got, want):
        note_once(f"提示：输入是 {want.dtype}，返回的是 {got.dtype}，按 float32 的精度（{F32_TOL:g}）比较。"
                  "题目没有规定返回的 dtype，但最好用 torch.zeros_like / torch.empty_like 保持输入的 dtype")
        tol = F32_TOL
    try:
        expect_close(got, want, rtol=tol, atol=tol * scale, call=call)
    except Mismatch as e:
        hint = next((h for v, h in guesses if looks_like(got, v)), None)
        if hint is None and torch.is_tensor(got) and got.is_floating_point() and got.dtype != want.dtype:
            hint = (f"返回的 dtype 是 {got.dtype}，输入是 {want.dtype}：中间结果被转成了更低的精度"
                    "（例如 torch.zeros(...) 默认是 float32，应当用 torch.zeros_like 或指定 dtype）")
        if hint:
            e.msg = f"{hint}。{e.msg}"
            e.args = (e.msg,)
        raise


def narrower_float(got, want):
    """got is a floating tensor or array with fewer bits than want (e.g. float32 for a float64 input)."""
    try:
        if torch.is_tensor(got):
            return got.is_floating_point() and got.element_size() < want.element_size()
        a = np.asarray(got)
        return a.dtype.kind == "f" and a.dtype.itemsize < want.element_size()
    except Exception:
        return False


def note_once(text):
    if text not in judgelib._current["info"]:
        info(text)


def forward_guesses(W, scan=False):
    order = ("乘法顺序反了：每一轮应当是 x_i ← x_{i−s} x_i，下标小的矩阵在左边" if scan else
             "乘法顺序反了：新的矩阵要乘在右边，P_i = P_{i−1} W_i")
    out = [(ref.prefix_reversed(W), f"{order}；现在的结果像是 W_i ⋯ W_1 W_0")]
    R = ref.n_rounds(W.shape[0])
    if scan and R > 0:
        out.append((ref.scan_rounds(W, R - 1),
                    f"N = {W.shape[0]} 时少做了最后一轮：步长 s = 1, 2, 4, … 只要 s < N 都要做，一共 ⌈log₂N⌉ = {R} 轮"))
    return out


class CountMatmul(TorchFunctionMode):
    """Counts the matrix-multiplication (and elementwise-product) calls made while it is active."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") in MATMUL_NAMES:
            self.n += 1
        return func(*args, **(kwargs or {}))


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：N = 3、D = 2")
def p1_example(sol):
    W = torch.tensor(W_EX)
    check(sol.prefix_products_inplace(W.clone()), P_EX, f"{W_EX_CODE}; prefix_products_inplace(W)",
          forward_guesses(W), tol=EX_TOL)


@case("Part 1", title="随机方阵：N = 1…11、D = 1…3")
def p1_random(sol):
    for s, (N, D) in zip(seeds(5), [(1, 3), (2, 2), (5, 3), (8, 1), (11, 3)]):
        ns, call = build(random_code(s, N, D))
        W = ns["W"]
        check(sol.prefix_products_inplace(W.clone()), ref.prefix(W), call + "prefix_products_inplace(W)",
              forward_guesses(W))


@case("Part 1", title="不改写输入 W")
def p1_input_unchanged(sol):
    ns, call = build(random_code(seeds(1)[0], 4, 3))
    W = ns["W"]
    want, before = ref.prefix(W), W.clone()
    call += "W0 = W.clone(); prefix_products_inplace(W); torch.equal(W, W0)"
    got = sol.prefix_products_inplace(W)
    expect_true(torch.equal(W, before), "prefix_products_inplace 改写了输入 W：题目要求用 P = torch.empty_like(W) "
                "分配新的输出，再往 P 里填，W 本身不能变", call)
    check(got, want, call)


# ---------------------------------------------------------------- Part 2
def run_autograd(sol, W, G, call):
    """P = prefix_products(W) on a leaf copy of W, then backward of sum(P * G) (of P.sum() when G is None)."""
    Wv = W.detach().clone().requires_grad_(True)
    P = sol.prefix_products(Wv)
    if not (torch.is_tensor(P) and P.requires_grad):
        raise Mismatch("prefix_products 返回的 P 不在计算图里（P.requires_grad 是 False）：P 要由 W 经过可求导的运算得到，"
                       "不能用 detach()、torch.no_grad() 或 torch.tensor(...) 构造", call, show=False)
    loss = P.sum() if G is None else (P * G).sum()
    try:
        loss.backward()
    except RuntimeError as e:
        text = str(e).splitlines()[0]
        why = "：prefix_products 里还有原地操作，autograd 保存的张量后来被改写了" if "inplace" in text else ""
        raise Mismatch(f"loss.backward() 报错：{text}{why}", call, show=False) from None
    if Wv.grad is None:
        raise Mismatch("backward() 之后 W.grad 是 None：P 没有连到 W 上", call, show=False)
    return P, Wv.grad


def check_grad(g, want, call, tol=TOL):
    """W.grad == want, first naming the case where only W[0] got a gradient (the graph was cut after P[0])."""
    a, b = as_np(g), torch.as_tensor(want).detach().to(F64).numpy()
    if a is not None and a.shape == b.shape and b.shape[0] > 1 and not (a[1:] != 0).any() and (b[1:] != 0).any():
        raise Mismatch("W.grad 只有 W[0] 那一项非零：其余 W_i 到 P 的计算图被切断了（例如用了 detach() 或 torch.no_grad()）",
                       call, want=want, got=g)
    check(g, want, call, tol=tol)


@case("Part 2", sample=True, title="题目例子：P.sum().backward()")
def p2_example(sol):
    W = torch.tensor(W_EX)
    call = f"{W_EX_CODE}.requires_grad_(); P = prefix_products(W); P.sum().backward(); W.grad"
    P, g = run_autograd(sol, W, None, call)
    check(P, P_EX, call, forward_guesses(W), tol=EX_TOL)
    check_grad(g, torch.tensor(DW_EX), call, tol=EX_TOL)


@case("Part 2", title="题目 Part 1(b) 的代码（float32）")
def p2_statement_snippet(sol):
    s = seeds(1)[0]
    ns, call = build(f"g = torch.Generator().manual_seed({s}); W = torch.randn(4, 3, 3, generator=g)")
    W = ns["W"]
    call += "W.requires_grad_(); P = prefix_products(W); P.sum().backward(); W.grad"
    P, g = run_autograd(sol, W, None, call)
    W64 = W.to(F64)
    want = ref.grad(W64, torch.ones_like(W64))
    check(P, ref.prefix(W64).float(), call, tol=F32_TOL)
    check_grad(g, want.float(), call, tol=F32_TOL)


@case("Part 2", title="随机上游梯度：W.grad 和公式一致")
def p2_random(sol):
    for s, (N, D) in zip(seeds(4), [(1, 3), (2, 2), (6, 3), (9, 2)]):
        ns, call = build(random_code(s, N, D, grad=True))
        W, G = ns["W"], ns["G"]
        call += "W.requires_grad_(); P = prefix_products(W); (P * G).sum().backward(); W.grad"
        P, g = run_autograd(sol, W, G, call)
        check(P, ref.prefix(W), call, forward_guesses(W))
        want = ref.grad(W, G)
        check_grad(g, want, call)


# ---------------------------------------------------------------- Part 3
def check_backward(sol, W, G, call, tol=TOL, want=None):
    P = ref.prefix(W)
    want = ref.grad(W, G) if want is None else want
    got = sol.prefix_products_backward(W.clone(), P.clone(), G.clone())
    guesses = [
        (ref.backward_no_accumulation(W, P, G), "只用了 G_i：P_i 除了被损失直接使用，还是 P_{i+1} = P_i W_{i+1} 的左因子，"
         "到达 P_i 的梯度是两项之和 A_i = G_i + A_{i+1} W_{i+1}ᵀ"),
        (ref.backward_w0_wraps(W, P, G), "只有 dW[0] 不对：W_0 前面没有左因子，∂L/∂W_0 = A_0。i = 0 时 P[i − 1] 取到的是 "
         "P[−1]，也就是最后一个前缀积"),
        (ref.backward_no_transpose(W, P, G), "梯度往前传时 W_{i+1} 少了转置：A_i = G_i + A_{i+1} W_{i+1}ᵀ"),
    ]
    check(got, want, call, [(v, h) for v, h in guesses if not torch.allclose(v, want)], tol=tol)


@case("Part 3", sample=True, title="题目例子：L = P.sum()")
def p3_example(sol):
    W = torch.tensor(W_EX)
    call = f"{W_EX_CODE}; P = torch.tensor({P_EX}); G = torch.ones(3, 2, 2); prefix_products_backward(W, P, G)"
    check_backward(sol, W, torch.ones(3, 2, 2), call, tol=EX_TOL, want=torch.tensor(DW_EX))


@case("Part 3", title="有奇异矩阵（不可逆）")
def p3_singular(sol):
    s1, s2 = seeds(2)
    for code in [f"g = torch.Generator().manual_seed({s1}); "
                 "W = torch.tensor([[[2.]], [[0.]], [[3.]], [[-1.]]], dtype=torch.float64); "
                 "G = torch.randn(4, 1, 1, generator=g, dtype=torch.float64)",
                 random_code(s2, 5, 3, grad=True) + "; W[2, 1] = 0"]:
        ns, call = build(code)
        check_backward(sol, ns["W"], ns["G"], call + "P = prefix_products(W); prefix_products_backward(W, P, G)")


@case("Part 3", title="随机方阵：N = 1…11、D = 1…3")
def p3_random(sol):
    for s, (N, D) in zip(seeds(6), [(1, 3), (2, 2), (3, 3), (6, 1), (7, 3), (11, 2)]):
        ns, call = build(random_code(s, N, D, grad=True))
        check_backward(sol, ns["W"], ns["G"], call + "P = prefix_products(W); prefix_products_backward(W, P, G)")


# ---------------------------------------------------------------- Part 4
def run_scan(sol, W, call):
    out = sol.scan_forward(W.clone())
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise Mismatch("scan_forward 应当返回 (P, tape) 两个值：P 是全部前缀积，tape 留给 scan_backward 用", call,
                       got=type(out).__name__)
    return out


def scan_backward_guesses(W, G, want):
    out = [(ref.scan_backward_variant(W, G, assign=True),
            "像是用 = 覆盖而不是 += 累加：同一轮里大多数 X_j 被用了两次（Y_j 的右因子、Y_{j+s} 的左因子），两项梯度要相加"),
           (ref.scan_backward_variant(W, G, final_x=True),
            "反向用到的 X 像是最终结果 P，而不是每一轮开始之前的值：tape 里保存的张量后来被原地改写了")]
    return [(v, h) for v, h in out if not torch.allclose(v, want)]


@case("Part 4", sample=True, title="题目例子：scan_forward")
def p4_example_forward(sol):
    W = torch.tensor(W_EX)
    call = f"{W_EX_CODE}; P, tape = scan_forward(W); P"
    P, _ = run_scan(sol, W, call)
    check(P, P_EX, call, forward_guesses(W, scan=True), tol=EX_TOL)


@case("Part 4", sample=True, title="题目例子：scan_backward，L = P.sum()")
def p4_example_backward(sol):
    W = torch.tensor(W_EX)
    call = f"{W_EX_CODE}; P, tape = scan_forward(W); scan_backward(tape, torch.ones(3, 2, 2))"
    _, tape = run_scan(sol, W, call)
    G = torch.ones(3, 2, 2)
    check(sol.scan_backward(tape, G.clone()), DW_EX, call, scan_backward_guesses(W, G, torch.tensor(DW_EX)),
          tol=EX_TOL)


SCAN_SHAPES = [(1, 2), (2, 3), (3, 1), (4, 3), (5, 2), (7, 3), (8, 2), (9, 3), (16, 2), (17, 3)]


@case("Part 4", title="前向：N 取 2 的幂和非 2 的幂")
def p4_forward(sol):
    for s, (N, D) in zip(seeds(len(SCAN_SHAPES)), SCAN_SHAPES):
        ns, call = build(random_code(s, N, D))
        W = ns["W"]
        P, _ = run_scan(sol, W, call + "P, tape = scan_forward(W); P")
        check(P, ref.prefix(W), call + "P, tape = scan_forward(W); P", forward_guesses(W, scan=True))


@case("Part 4", title="反向：N 取 2 的幂和非 2 的幂")
def p4_backward(sol):
    for s, (N, D) in zip(seeds(len(SCAN_SHAPES)), SCAN_SHAPES):
        ns, call = build(random_code(s, N, D, grad=True))
        W, G = ns["W"], ns["G"]
        call += "P, tape = scan_forward(W); scan_backward(tape, G)"
        _, tape = run_scan(sol, W, call)
        want = ref.grad(W, G)
        check(sol.scan_backward(tape, G.clone()), want, call, scan_backward_guesses(W, G, want))


@case("Part 4", title="N = 128：每一轮一次批量矩阵乘法")
def p4_rounds(sol):
    N, R = 128, ref.n_rounds(128)
    ns, call = build(f"g = torch.Generator().manual_seed({seeds(1)[0]}); "
                     "W = torch.eye(3) + 0.1 * torch.randn(128, 3, 3, generator=g, dtype=torch.float64); "
                     "G = torch.randn(128, 3, 3, generator=g, dtype=torch.float64)")
    W, G = ns["W"], ns["G"]
    with CountMatmul() as fwd:
        P, tape = run_scan(sol, W, call + "P, tape = scan_forward(W)")
    expect_true(fwd.n <= 4 * R, f"N = {N} 时 scan_forward 调用了 {fwd.n} 次乘法（矩阵乘法和逐元素乘法），而 ⌈log₂N⌉ = {R}："
                f"每一轮应当是一次批量矩阵乘法（例如 X[:-s] @ X[s:]），不能对 i 写 Python 循环", call + "P, tape = scan_forward(W)")
    with CountMatmul() as bwd:
        dW = sol.scan_backward(tape, G.clone())
    call += "P, tape = scan_forward(W); scan_backward(tape, G)"
    expect_true(bwd.n <= 8 * R, f"N = {N} 时 scan_backward 调用了 {bwd.n} 次乘法（矩阵乘法和逐元素乘法），而 ⌈log₂N⌉ = {R}："
                f"反向也要一轮一轮地批量计算，不能像 Part 3 那样对 i 逐个循环", call)
    check(P, ref.prefix(W), call)
    check(dW, ref.grad(W, G), call)
