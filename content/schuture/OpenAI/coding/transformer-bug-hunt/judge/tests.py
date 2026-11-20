import copy
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from judgelib import Mismatch, case, expect, expect_true, expect_valid, info, rng
import ref

torch.set_num_threads(1)        # the tensors are tiny: a second thread only adds CPU time

VOCAB, N_DIGITS = ref.VOCAB, ref.N_DIGITS
ANSWER = "/work/answer.py"


def seeds(r, n=2):
    return [r.randrange(2 ** 31) for _ in range(n)]


def shown(kwargs):
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def from_answer(e):
    return any(f.filename == ANSWER for f in traceback.extract_tb(e.__traceback__))


def as_logits(out, shape, call):
    expect_true(isinstance(out, torch.Tensor), f"应当返回 logits tensor，得到的是 {type(out).__name__}", call, got=out)
    expect(tuple(out.shape), shape, call=call, msg=f"形状不对：期望 {shape}，得到 {tuple(out.shape)}")
    return out


# ---------------------------------------------------------------- Part 1
SCRIPT = "torch.manual_seed(0); model = TinyGPT(); final_loss = train(model); run_tests(model, final_loss)"


@case("Part 1", sample=True, cpu=20, title="python tiny_gpt.py：三项测试都通过")
def p1_script(sol):
    torch.manual_seed(0)
    model = sol.TinyGPT()
    final_loss = float(sol.train(model))
    seq = ref.make_batch(500, torch.Generator().manual_seed(1))
    a, b = seq[:, :-1].clone(), seq[:, :-1].clone()
    b[:, -1] = (b[:, -1] + 1) % 10                      # a and b differ in the last token only
    with torch.no_grad():
        early_a, early_b = model(a)[:, :-1], model(b)[:, :-1]
    info(f"训练结束时的损失 {final_loss:.4f}")
    if not torch.allclose(early_a, early_b, atol=1e-5):
        worst = (early_a - early_b).abs().max().item()
        raise Mismatch(f"测试 1 没有通过：只改动最后一个输入 token，前面位置的 logits 也变了（最大变化 {worst:.3g}）",
                       SCRIPT, show=False)
    expect_true(final_loss < 0.05, f"测试 2 没有通过：训练 400 步后损失是 {final_loss:.4f}，没有降到 0.05 以下", SCRIPT)
    completed = ref.generate(model, seq[:, :N_DIGITS + 1], N_DIGITS)     # the statement's original generate
    accuracy = (completed == seq).all(dim=1).float().mean().item()
    info(f"500 条新序列里补全完全正确的占 {accuracy:.1%}")
    expect_true(accuracy > 0.99, f"测试 3 没有通过：500 条新序列里只有 {accuracy:.1%} 补全得一个 token 都不错（要求 99% 以上）",
                SCRIPT)


@case("Part 1", title="前向输出的形状是 (B, T, 11)")
def p1_shapes(sol):
    r = rng()
    for kwargs, B, T in [({}, 2, 12), ({}, 3, 1), ({}, 1, 13), ({"d_model": 32, "n_head": 2, "n_layer": 2}, 3, 5)]:
        s, g = seeds(r)
        torch.manual_seed(s)
        model = sol.TinyGPT(**kwargs)
        idx = torch.randint(0, VOCAB, (B, T), generator=torch.Generator().manual_seed(g))
        call = f"model = TinyGPT({shown(kwargs)}); model(torch.randint(0, 11, ({B}, {T})))"
        with torch.no_grad():
            out = as_logits(model(idx), (B, T, VOCAB), call)
        expect_true(torch.isfinite(out).all(), "输出里有 NaN 或 inf", call)


@case("Part 1", title="后面的 token 不应影响前面的输出")
def p1_causal(sol):
    r = rng()
    for kwargs in ({}, {"n_layer": 2}):
        s, g = seeds(r)
        torch.manual_seed(s)
        model = sol.TinyGPT(**kwargs)
        gen = torch.Generator().manual_seed(g)
        a = ref.make_batch(16, gen)[:, :-1]
        for k in (11, 7, 3, 1):                         # positions k.. change, positions < k must not notice
            b = a.clone()
            b[:, k:] = (a[:, k:] + torch.randint(1, VOCAB, a[:, k:].shape, generator=gen)) % VOCAB
            with torch.no_grad():
                la, lb = model(a), model(b)
            diff = (la[:, :k] - lb[:, :k]).abs().amax(dim=(0, 2))
            if diff.max().item() > 1e-5:
                pos = int((diff > 1e-5).nonzero()[0])
                raise Mismatch(f"a 和 b 只在位置 {k} 及以后不同，可位置 {pos} 的 logits 变了（最大变化 {diff[pos].item():.3g}）："
                               f"位置 t 的输出只能依赖位置 ≤ t 的 token",
                               f"torch.manual_seed({s}); model = TinyGPT({shown(kwargs)}); a, b = <16 条长 12 的序列，"
                               f"只在位置 {k} 及以后不同>; 比较 model(a)[:, :{k}] 和 model(b)[:, :{k}]", show=False)


@case("Part 1", title="train 要真的更新模型参数")
def p1_train_updates(sol):
    r = rng()
    torch.manual_seed(seeds(r, 1)[0])
    model = ref.TinyGPT()
    twin = copy.deepcopy(model)
    before = snapshot(model)
    call = "model = <一个没有 bug 的 TinyGPT>; train(model, steps=40, batch_size=16)"
    got = sol.train(model, steps=40, batch_size=16)
    want = ref.train(twin, steps=40, batch_size=16)
    after = snapshot(model)
    if all(torch.equal(x, y) for x, y in zip(before, after)):
        raise Mismatch("train 跑完 40 步，模型的参数一个都没有变", call, show=False)
    worst = max((x - y).abs().max().item() for x, y in zip(after, snapshot(twin)))
    expect_true(worst <= 1e-4, f"训练 40 步后的参数与正确的训练循环得到的不一致（最大相差 {worst:.3g}）", call)
    expect_true(isinstance(got, (int, float)) and abs(got - want) <= 1e-4 * abs(want) + 1e-5,
                f"train 应当返回最后一步的损失 {want:.6f}，返回的是 {got!r}", call)


@case("Part 1", title="模型要能分辨数字的先后顺序")
def p1_positions(sol):
    r = rng()
    s = seeds(r, 1)[0]
    torch.manual_seed(s)
    model = sol.TinyGPT()
    ref.train(model, steps=30)                          # the judge's own (correct) training loop
    prompts = torch.tensor([[7, 0, 2, 2, 5, 8, 10], [8, 5, 2, 2, 0, 7, 10]])
    with torch.no_grad():
        last = model(prompts)[:, -1]
    diff = (last[0] - last[1]).abs().max().item()
    expect_true(diff > 1e-3,
                f"提示 7 0 2 2 5 8 10 和 8 5 2 2 0 7 10 在分隔符处的 logits 几乎相同（最大相差 {diff:.2g}），"
                "可正确的下一个 token 分别是 8 和 7：模型分不出数字的先后顺序",
                f"torch.manual_seed({s}); model = TinyGPT(); <用正确的训练循环训练 30 步>; "
                "model(torch.tensor([[7, 0, 2, 2, 5, 8, 10], [8, 5, 2, 2, 0, 7, 10]]))[:, -1]")


# ---------------------------------------------------------------- Part 2
EXAMPLES = [[7, 0, 2, 2, 5, 8], [8, 5, 2, 2, 0, 7]]
MAX_STEPS = 600
TRAINING = ("clf = ParityClassifier(); opt = torch.optim.Adam(clf.parameters(), lr=3e-3); 每步取 64 个随机六位数 digits，"
            "loss = F.cross_entropy(clf(digits), digits[:, -1] % 2)")
_trained = {}


def new_classifier(sol):
    try:
        return sol.ParityClassifier()
    except TypeError as e:
        if from_answer(e):
            raise
        raise Mismatch(f"ParityClassifier() 构造失败（{e}）：题目给出的接口是不带参数的 __init__(self)",
                       "clf = ParityClassifier()", show=False) from None


def trained_classifier(sol):
    """Trains the answer's classifier with the reference answer's recipe, at most MAX_STEPS steps, until it labels
    1000 new numbers right. Done once per run and shared by the cases that need it."""
    if id(sol) in _trained:
        return _trained[id(sol)]
    s, g = seeds(rng("parity"))
    torch.manual_seed(s)
    clf = new_classifier(sol)
    gen = torch.Generator().manual_seed(g)
    test = torch.randint(0, 10, (1000, N_DIGITS), generator=gen)
    as_logits(clf(torch.tensor(EXAMPLES)), (2, 2), "clf = ParityClassifier(); clf(torch.tensor(" + repr(EXAMPLES) + "))")
    params = [p for p in clf.parameters() if p.requires_grad]
    expect_true(params, "ParityClassifier() 没有可训练的参数", "list(ParityClassifier().parameters())")
    opt = torch.optim.Adam(params, lr=3e-3)
    steps = wrong = 0
    while steps < MAX_STEPS:
        clf.train()
        for _ in range(100):
            digits = torch.randint(0, 10, (64, N_DIGITS), generator=gen)
            loss = F.cross_entropy(clf(digits), digits[:, -1] % 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        steps += 100
        clf.eval()
        with torch.no_grad():
            wrong = int((clf(test).argmax(dim=-1) != test[:, -1] % 2).sum())
        if wrong == 0:
            break
    _trained[id(sol)] = clf, steps, wrong
    return _trained[id(sol)]


@case("Part 2", sample=True, cpu=12, title="题目例子：702258 是偶数，852207 是奇数")
def p2_examples(sol):
    clf, steps, wrong = trained_classifier(sol)
    info(f"按参考解答的设置训练了 {steps} 步，1000 个新的数里判断错 {wrong} 个")
    with torch.no_grad():
        pred = clf(torch.tensor(EXAMPLES)).argmax(dim=-1)
    expect(pred.tolist(), [0, 1], call=f"{TRAINING}; 训练 {steps} 步后 clf(torch.tensor({EXAMPLES})).argmax(dim=-1)")


@case("Part 2", title="输出形状是 (B, 2)，B、T 可以变")
def p2_shapes(sol):
    r = rng()
    s, g = seeds(r)
    torch.manual_seed(s)
    clf = new_classifier(sol)
    gen = torch.Generator().manual_seed(g)
    for B, T in [(5, 6), (1, 6), (3, 4)]:
        idx = torch.randint(0, 10, (B, T), generator=gen)
        with torch.no_grad():
            as_logits(clf(idx), (B, 2), f"clf = ParityClassifier(); clf(torch.randint(0, 10, ({B}, {T})))")


@case("Part 2", cpu=12, title="训练后 1000 个新的数全部判断正确")
def p2_learns(sol):
    clf, steps, wrong = trained_classifier(sol)
    info(f"按参考解答的设置训练了 {steps} 步，1000 个新的数里判断错 {wrong} 个")
    expect_true(wrong == 0, f"训练 {steps} 步之后，1000 个新的六位数里仍有 {wrong} 个判断错误", f"{TRAINING}; 最多训练 {MAX_STEPS} 步")


# ---------------------------------------------------------------- Part 3
def call_cached(model, idx, cache, call):
    try:
        return model(idx, cache)
    except TypeError as e:
        if from_answer(e):
            raise
        raise Mismatch(f"model(idx, cache) 调用失败（{e}）：Part 3 要给 TinyGPT.forward 加一个可选参数 cache",
                       call, show=False) from None


def check_chunks(sol, model, seq, chunks, setup):
    B = seq.shape[0]
    with torch.no_grad():
        full = model(seq)
        cache = sol.KVCache(n_layer=len(model.blocks))
        for j, (a, b) in enumerate(chunks):
            recent = chunks[max(0, j - 7):j + 1]
            call = (f"{setup}; cache = KVCache(n_layer=len(model.blocks)); " + ("…; " if j >= 8 else "")
                    + "; ".join(f"model(seq[:, {x}:{y}], cache)" for x, y in recent))
            out = as_logits(call_cached(model, seq[:, a:b], cache, call), (B, b - a, VOCAB), call)
            if not torch.allclose(out, full[:, a:b], rtol=1e-5, atol=1e-5):
                worst = (out - full[:, a:b]).abs().max().item()
                raise Mismatch(f"第 {j + 1} 块 seq[:, {a}:{b}] 的 logits 与整条序列前向 model(seq) 的对应切片不一致"
                               f"（最大误差 {worst:.3g}；下面是第 0 行）", call, full[0, a:b], out[0])
            expect(cache.length, b, call=call, msg=f"处理完 seq[:, {a}:{b}] 之后 cache.length 应当是 {b}")


def random_chunks(r, n=13):
    cuts = sorted(r.sample(range(1, n), r.randint(1, 6)))
    edges = [0] + cuts + [n]
    return list(zip(edges, edges[1:]))


@case("Part 3", sample=True, title="题目例子：分 3、1、5、2、2 块与整条前向一致")
def p3_example(sol):
    torch.manual_seed(3)
    model = sol.TinyGPT()
    seq = ref.make_batch(8, torch.Generator().manual_seed(4))
    check_chunks(sol, model, seq, [(0, 3), (3, 4), (4, 9), (9, 11), (11, 13)],
                 "torch.manual_seed(3); model = TinyGPT(); seq = make_batch(8, torch.Generator().manual_seed(4))")


@case("Part 3", title="KVCache.update 追加新位置并返回全部位置")
def p3_update(sol):
    gen = torch.Generator().manual_seed(seeds(rng(), 1)[0])
    new = {(layer, t): (torch.randn(2, 4, n, 8, generator=gen), torch.randn(2, 4, n, 8, generator=gen))
           for layer in (0, 1) for t, n in enumerate((3, 1, 2))}
    cache = sol.KVCache(2)
    expect(cache.length, 0, call="KVCache(2).length")
    steps = []
    for t, total in enumerate((3, 4, 6)):
        for layer in (0, 1):
            k, v = new[layer, t]
            steps.append(f"cache.update({layer}, <k, v 形状 {tuple(k.shape)}>)")
            call = "cache = KVCache(2); " + "; ".join(steps[-6:])
            got = cache.update(layer, k.clone(), v.clone())
            expect_true(isinstance(got, (tuple, list)) and len(got) == 2, "update 应当返回 (k, v) 两个 tensor", call, got=got)
            want_k = torch.cat([new[layer, i][0] for i in range(t + 1)], dim=2)
            want_v = torch.cat([new[layer, i][1] for i in range(t + 1)], dim=2)
            for name, g, w in (("k", got[0], want_k), ("v", got[1], want_v)):
                expect_true(isinstance(g, torch.Tensor), f"返回的 {name} 应当是 tensor", call, got=g)
                expect(tuple(g.shape), tuple(w.shape), call=call,
                       msg=f"返回的 {name} 形状不对：这一层已有 {total} 个位置，期望 {tuple(w.shape)}，得到 {tuple(g.shape)}")
                expect(g, w, call=call, msg=f"返回的 {name} 应当是这一层到目前为止所有位置按时间顺序拼起来")
            expect(cache.length, total, call=call, msg=f"cache.length 应当是 {total}")


@case("Part 3", title="随机分块、多层多头：与整条前向一致")
def p3_random_chunks(sol):
    r = rng()
    for n_layer, n_head in [(1, 1), (2, 2), (3, 4)]:
        s, g = seeds(r)
        B = r.randint(1, 4)
        torch.manual_seed(s)
        model = sol.TinyGPT(d_model=32, n_head=n_head, n_layer=n_layer)
        seq = ref.make_batch(B, torch.Generator().manual_seed(g))
        setup = (f"torch.manual_seed({s}); model = TinyGPT(d_model=32, n_head={n_head}, n_layer={n_layer}); "
                 f"seq = make_batch({B}, torch.Generator().manual_seed({g}))")
        for chunks in ([(0, 13)], [(i, i + 1) for i in range(13)], random_chunks(r), random_chunks(r)):
            check_chunks(sol, model, seq, chunks, setup)


@case("Part 3", title="带缓存的调用只计算新位置，不重算")
def p3_no_recompute(sol):
    r = rng()
    s, g = seeds(r)
    torch.manual_seed(s)
    model = sol.TinyGPT(n_layer=2)
    seq = ref.make_batch(4, torch.Generator().manual_seed(g))
    rows = []

    def count(module, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].dim() >= 2:
            rows.append(args[0].numel() // args[0].shape[-1])

    hooks = [m.register_forward_pre_hook(count) for m in model.modules() if isinstance(m, nn.Linear)]
    try:
        cache = sol.KVCache(2)
        done = []
        for a, b in [(0, 6), (6, 7), (7, 10), (10, 13)]:
            done.append(f"model(seq[:, {a}:{b}], cache)")
            call = (f"torch.manual_seed({s}); model = TinyGPT(n_layer=2); seq = make_batch(4, torch.Generator()"
                    f".manual_seed({g})); cache = KVCache(2); " + "; ".join(done))
            rows.clear()
            with torch.no_grad():
                call_cached(model, seq[:, a:b], cache, call)
            most = max(rows, default=0) // 4
            expect_true(most <= b - a, f"这次调用只有 {b - a} 个新位置，模型里却有线性层一次算了 {most} 个位置："
                                       "更早位置的键和值应当从缓存里读，不重算", call)
    finally:
        for h in hooks:
            h.remove()


@case("Part 3", title="generate 用缓存：先处理整个提示，之后每次只喂最新的 token")
def p3_generate(sol):
    r = rng()
    s, g = seeds(r)
    torch.manual_seed(s)
    model = sol.TinyGPT(n_layer=2)
    prompt = ref.make_batch(16, torch.Generator().manual_seed(g))[:, :N_DIGITS + 1]
    setup = (f"torch.manual_seed({s}); model = TinyGPT(n_layer=2); "
             f"prompt = make_batch(16, torch.Generator().manual_seed({g}))[:, :7]")

    call = f"{setup}; generate(model, prompt, 6)"
    out = sol.generate(model, prompt.clone(), N_DIGITS)
    expect_valid(out, lambda o: ref.greedy_problem(model, prompt, o, N_DIGITS), call)

    call = f"{setup}; generate(model, prompt, 6, use_cache=True)"
    calls, forward = [], model.forward

    def recorder(idx, *args, **kwargs):
        cache = args[0] if args else kwargs.get("cache")
        calls.append((tuple(idx.shape), cache is not None))
        return forward(idx, *args, **kwargs)

    model.forward = recorder                            # nn.Module.__call__ looks forward up on the instance first
    try:
        out = sol.generate(model, prompt.clone(), N_DIGITS, use_cache=True)
    except TypeError as e:
        if from_answer(e):
            raise
        raise Mismatch(f"调用失败（{e}）：Part 3 要给 generate 加一个开关 use_cache", call, show=False) from None
    finally:
        del model.forward
    expect_valid(out, lambda o: ref.greedy_problem(model, prompt, o, N_DIGITS), call)
    seen = "；".join(f"第 {j + 1} 次 {n} 个 token" + ("" if c else "（没传 cache）") for j, ((_, n), c) in enumerate(calls[:8]))
    for j, ((_, n), with_cache) in enumerate(calls):
        expect_true(with_cache, f"第 {j + 1} 次调用模型时没有传入 cache。各次调用：{seen}", call)
        if j == 0:
            expect_true(n == prompt.shape[1], f"第 1 次调用应当传入整个提示（7 个 token），实际传入 {n} 个。各次调用：{seen}", call)
        else:
            expect_true(n == 1, f"第 {j + 1} 次调用应当只传入最新的 1 个 token，实际传入 {n} 个。各次调用：{seen}", call)
