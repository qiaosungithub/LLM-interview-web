"""Tests for the config system. Results are compared strictly: types count (7 is not 7.0 or "7", True is not 1),
0.0 and -0.0 differ, dict order does not matter. Each failure shows code that can be pasted into 自定义运行; there
DELETE is the answer's own sentinel and the Part 3 schemas come from ref (`from ref import ExperimentConfig`)."""
import os
import subprocess
import sys
import tempfile
from types import MappingProxyType

from judgelib import Mismatch, calling, case, expect, expect_raises, expect_true, rng
import ref
from ref import D, src

# ---------------------------------------------------------------- shared checks


def check(sol, got, want):
    m = ref.diff(got, want, sol.DELETE)
    if m:
        raise Mismatch(m, want=want, got=got)


def untouched(sol, after, before, what):
    m = ref.diff(after, before, sol.DELETE, root=what)
    if m:
        raise Mismatch(f"传入的 {what} 被修改了：{m}。题目要求不修改传入的任何配置", show=False)


# ---------------------------------------------------------------- Part 1

STORE = {
    "base": {"trainer": {"epochs": 10, "batch_size": 32, "lr": 0.1},
             "optimizer": {"name": "sgd", "momentum": 0.9}},
    "wide": {"_base_": ["base"], "trainer": {"batch_size": 128}},
    "regularized": {"trainer": {"weight_decay": 0.01}},
    "experiment": {"_base_": ["wide", "regularized"], "trainer": {"epochs": 20},
                   "optimizer": {"name": "adam", "momentum": D}},
}


def merge_case(sol, base, overlay, want):
    calling(f"base = {src(base)}\noverlay = {src(overlay)}\nprint(deep_merge(base, overlay))")
    b, o = ref.inject(base, sol.DELETE), ref.inject(overlay, sol.DELETE)
    got = sol.deep_merge(b, o)
    check(sol, got, want)
    untouched(sol, b, base, "base")
    untouched(sol, o, overlay, "overlay")
    return got, b, o


def store_call(store, name):
    return calling(ref.store_src(store) + f"\nprint(resolve_bases(store, {name!r}))")


def resolve_case(sol, store, name, want):
    store_call(store, name)
    passed = ref.inject(store, sol.DELETE)
    got = sol.resolve_bases(passed, name)
    check(sol, got, want)
    untouched(sol, passed, store, "store")
    return got, passed


def cycle_case(sol, store, name, ring):
    store_call(store, name)
    e = expect_raises(sol.ConfigCycleError, sol.resolve_bases, ref.inject(store, sol.DELETE), name)
    expect_true(isinstance(e, sol.ConfigError), "ConfigCycleError 应当是 ConfigError 的子类")
    m = ref.cycle_problem(str(e), ring)
    if m:
        raise Mismatch(m, show=False)


def not_found_case(sol, store, name):
    store_call(store, name)
    e = expect_raises(sol.ConfigNotFoundError, sol.resolve_bases, ref.inject(store, sol.DELETE), name)
    expect_true(isinstance(e, sol.ConfigError), "ConfigNotFoundError 应当是 ConfigError 的子类")


@case("Part 1", sample=True, title="题目例子：experiment")
def p1_example(sol):
    resolve_case(sol, STORE, "experiment",
                 {"trainer": {"epochs": 20, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01},
                  "optimizer": {"name": "adam"}})


@case("Part 1", title="deep_merge 的四条规则")
def p1_merge_rules(sol):
    for base, overlay, want in [
        ({"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 1, "b": 3, "c": 4}),
        ({"x": {"y": 1, "z": 2}}, {"x": {"z": 3, "w": 4}}, {"x": {"y": 1, "z": 3, "w": 4}}),
        ({"a": {"b": {"c": 1, "d": 2}}, "e": 0}, {"a": {"b": {"d": 3}}}, {"a": {"b": {"c": 1, "d": 3}}, "e": 0}),
        ({"x": {"y": 1}}, {"x": 5}, {"x": 5}),
        ({"x": 5}, {"x": {"y": 1}}, {"x": {"y": 1}}),
        ({"x": [1, 2, 3]}, {"x": [4]}, {"x": [4]}),
        ({"x": {"y": 1}}, {"x": [1]}, {"x": [1]}),
        ({"x": {"y": 1}, "z": 1}, {"x": None}, {"x": None, "z": 1}),
        ({"x": None}, {"x": {"a": 1}}, {"x": {"a": 1}}),
        ({"a": 1}, {"n": {"m": 1}}, {"a": 1, "n": {"m": 1}}),
        ({"a": 1, "b": {"c": 1}}, {"a": D, "b": D}, {}),
        ({"a": 1}, {"z": D}, {"a": 1}),
        ({"x": {"y": 1, "z": 2}, "w": 0}, {"x": {"y": D}}, {"x": {"z": 2}, "w": 0}),
        ({"x": {"y": {"p": 1, "q": 2}}}, {"x": {"y": {"q": D, "r": 3}}}, {"x": {"y": {"p": 1, "r": 3}}}),
        ({}, {}, {}),
        ({"x": {"y": 1}}, {"x": {}}, {"x": {"y": 1}}),
        ({"x": {}}, {}, {"x": {}}),
        ({"x": 1}, {"x": {}}, {"x": {}}),
        ({"s": "a", "f": 0.5, "t": True}, {"s": "", "f": -0.0, "t": False}, {"s": "", "f": -0.0, "t": False}),
    ]:
        merge_case(sol, base, overlay, want)


@case("Part 1", title="base 处没有 dict 时，嵌套的 DELETE")
def p1_merge_delete_nested(sol):
    for base, overlay, want in [
        ({"a": 1}, {"n": {"m": D, "k": 2}}, {"a": 1, "n": {"k": 2}}),
        ({"x": 5}, {"x": {"y": D}}, {"x": {}}),
        ({"x": [1]}, {"x": {"y": D, "z": 1}}, {"x": {"z": 1}}),
        ({}, {"b": {"c": {"d": D}, "e": [1]}}, {"b": {"c": {}, "e": [1]}}),
        ({"b": {"c": 5}}, {"b": {"c": {"d": D, "f": None}}}, {"b": {"c": {"f": None}}}),
    ]:
        merge_case(sol, base, overlay, want)
    resolve_case(sol, {"base": {"trainer": {"lr": 0.1}},
                       "exp": {"_base_": ["base"], "opt": {"momentum": D, "name": "adam"}}},
                 "exp", {"trainer": {"lr": 0.1}, "opt": {"name": "adam"}})
    resolve_case(sol, {"solo": {"a": D, "b": {"c": D}}}, "solo", {"b": {}})


@case("Part 1", title="结果不与 base、overlay 共享对象")
def p1_merge_no_sharing(sol):
    base = {"keep": {"deep": [1, {"z": 2}]}, "lst": [1, 2], "both": {"x": [1], "y": {"u": 1}}, "rep": {"q": 1}}
    overlay = {"both": {"y": {"v": [2]}, "z": {"w": 1}}, "new": [3, [4]], "newd": {"k": [5]}, "rep": [9]}
    got, b, o = merge_case(sol, base, overlay, ref.merge(base, overlay))
    hit = ref.shared_with(got, {"base": b, "overlay": o})
    if hit:
        raise Mismatch(f"结果里的 {hit[0]} 就是 {hit[1]} 里的同一个对象，没有拷贝：题目要求结果中的任何值都不与 base 或 "
                       "overlay 共享，否则修改结果会改到传入的参数", show=False)


@case("Part 1", title="链式 base 和 base 的先后顺序")
def p1_chain_and_order(sol):
    chain = {
        "c0": {"v": 0, "a": {"x": 0}},
        "c1": {"_base_": ["c0"], "v": 1, "a": {"y": 1}},
        "c2": {"_base_": ["c1"], "a": {"x": 2}},
        "c3": {"_base_": ["c2"], "w": 3},
    }
    resolve_case(sol, chain, "c3", {"v": 1, "a": {"x": 2, "y": 1}, "w": 3})
    resolve_case(sol, chain, "c1", {"v": 1, "a": {"x": 0, "y": 1}})
    resolve_case(sol, chain, "c0", {"v": 0, "a": {"x": 0}})
    order = {
        "z": {"k": "z", "only_z": 1, "d": {"p": 1}},
        "w": {"k": "w", "only_w": 2, "d": {"p": 2, "q": 2}},
        "zw": {"_base_": ["z", "w"], "own": 0},
        "wz": {"_base_": ["w", "z"]},
        "own": {"_base_": ["z", "w"], "k": "own", "d": {"q": 3}},
        "drop": {"_base_": ["z"], "only_z": D, "d": {"p": D}},
        "empty": {"_base_": [], "a": 1},
    }
    resolve_case(sol, order, "zw", {"k": "w", "only_z": 1, "only_w": 2, "d": {"p": 2, "q": 2}, "own": 0})
    resolve_case(sol, order, "wz", {"k": "z", "only_w": 2, "only_z": 1, "d": {"p": 1, "q": 2}})
    resolve_case(sol, order, "own", {"k": "own", "only_z": 1, "only_w": 2, "d": {"p": 2, "q": 3}})
    resolve_case(sol, order, "drop", {"k": "z", "d": {}})
    resolve_case(sol, order, "empty", {"a": 1})


@case("Part 1", title="菱形：同一个 base 被用到两次")
def p1_diamond(sol):
    diamond = {
        "root": {"opt": {"name": "sgd", "momentum": 0.9}, "lr": 0.1},
        "left": {"_base_": ["root"], "opt": {"momentum": D}},
        "right": {"_base_": ["root"], "lr": 0.2},
        "top": {"_base_": ["left", "right"]},
        "top2": {"_base_": ["right", "left"], "extra": [1]},
        "top3": {"_base_": ["right", "left"], "opt": {"momentum": D}},
    }
    # left's own DELETE only acts while left is composed: its resolved config simply lacks momentum.
    resolve_case(sol, diamond, "top", {"opt": {"name": "sgd", "momentum": 0.9}, "lr": 0.2})
    resolve_case(sol, diamond, "top2", {"opt": {"name": "sgd", "momentum": 0.9}, "lr": 0.1, "extra": [1]})
    resolve_case(sol, diamond, "top3", {"opt": {"name": "sgd"}, "lr": 0.1})
    resolve_case(sol, {"a": {"x": 1, "y": {"z": 2}}, "twice": {"_base_": ["a", "a"], "w": 0}},
                 "twice", {"x": 1, "y": {"z": 2}, "w": 0})
    lattice = {
        "a": {"v": "a", "a": 1, "n": {"a": 1}},
        "b": {"_base_": ["a"], "v": "b", "b": 1, "n": {"a": D, "b": 1}},
        "c": {"_base_": ["a"], "v": "c", "c": 1, "n": {"c": 1}},
        "d": {"_base_": ["b", "c"], "d": 1},
        "e": {"_base_": ["c", "b", "a"], "e": 1},
        "f": {"_base_": ["d", "e"], "n": {"f": 1}},
    }
    for name in ("d", "e", "f"):
        resolve_case(sol, lattice, name, ref.resolve(lattice, name))


@case("Part 1", title="环：ConfigCycleError，信息列出环")
def p1_cycle(sol):
    cycle_case(sol, {"a": {"_base_": ["b"], "x": 1}, "b": {"_base_": ["a"], "y": 2}}, "a", ["a", "b"])
    cycle_case(sol, {"loop": {"_base_": ["loop"], "x": 1}}, "loop", ["loop"])
    ring = {
        "top": {"_base_": ["ok", "p1"]},
        "ok": {"x": 1},
        "p1": {"_base_": ["r1"]},
        "r1": {"_base_": ["ok", "r2"]},
        "r2": {"_base_": ["r3"]},
        "r3": {"_base_": ["r1"], "y": 2},
    }
    cycle_case(sol, ring, "top", ["r1", "r2", "r3"])
    cycle_case(sol, ring, "r2", ["r2", "r3", "r1"])


@case("Part 1", title="缺少名字：ConfigNotFoundError")
def p1_not_found(sol):
    not_found_case(sol, {"a": {"x": 1}}, "zz")
    not_found_case(sol, {}, "x")
    not_found_case(sol, {"a": {"_base_": ["ghost"], "x": 1}}, "a")
    not_found_case(sol, {"a": {"_base_": ["b"]}, "b": {"_base_": ["c"]}, "c": {"_base_": ["ok", "ghost"]}, "ok": {}},
                   "a")


@case("Part 1", title="到不了的环和缺失名字不影响结果")
def p1_unreachable(sol):
    store = {
        "x": {"_base_": ["y"]},
        "y": {"_base_": ["x"]},
        "g": {"_base_": ["ghost"]},
        "b": {"v": 1},
        "a": {"_base_": ["b"], "w": 2},
    }
    resolve_case(sol, store, "a", {"v": 1, "w": 2})
    resolve_case(sol, store, "b", {"v": 1})


@case("Part 1", title="不修改、不共享 store")
def p1_store_untouched(sol):
    store = {
        "base": {"data": {"paths": ["a", "b"], "aug": {"flip": True}}, "tags": ["x"], "opt": {"name": "sgd"}},
        "mid": {"_base_": ["base"], "data": {"aug": {"crop": 4}}, "seeds": [1, [2, 3]]},
        "top": {"_base_": ["mid"], "opt": {"name": D, "lr": 0.1}, "extra": {"k": {}}},
    }
    want = ref.resolve(store, "top")
    got, passed = resolve_case(sol, store, "top", want)
    hit = ref.shared_with(got, {"store": passed})
    if hit:
        raise Mismatch(f"结果里的 {hit[0]} 就是 store 里的同一个对象，没有拷贝：resolve_bases 是用 deep_merge 组合出来的，"
                       "结果不应与 store 共享任何值，否则修改结果会改到 store", show=False)
    got["data"]["paths"].append("changed")
    got["data"]["aug"].clear()
    got["seeds"][1].append(4)
    got.clear()
    calling(ref.store_src(store) + "\nfirst = resolve_bases(store, 'top')\nfirst['data']['paths'].append('changed')\n"
            "first['data']['aug'].clear()\nfirst['seeds'][1].append(4)\nfirst.clear()\n"
            "print(resolve_bases(store, 'top'))   # 第二次解析，不应受到第一次结果被修改的影响")
    check(sol, sol.resolve_bases(passed, "top"), want)
    untouched(sol, passed, store, "store")


@case("Part 1", title="随机配置库，和逐条按规则计算比对")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        store, names = ref.rand_store(r, r.randint(1, 7))
        name = r.choice(names)
        resolve_case(sol, store, name, ref.resolve(store, name))
    for _ in range(100):
        store, start, ring = ref.rand_ring_store(r)
        cycle_case(sol, store, start, ring)
    for _ in range(100):
        store, start = ref.rand_missing_store(r)
        not_found_case(sol, store, start)


# ---------------------------------------------------------------- Part 2: overrides

RESOLVED = {"trainer": {"epochs": 20, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01},
            "optimizer": {"name": "adam"},
            "run_name": "exp-${optimizer.name}-bs${trainer.batch_size}",
            "eval_batch_size": "${trainer.batch_size}"}
OVERRIDDEN = {"trainer": {"epochs": 30, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01, "warmup_steps": 100},
              "optimizer": {"name": "adamw"},
              "run_name": "exp-${optimizer.name}-bs${trainer.batch_size}",
              "eval_batch_size": "${trainer.batch_size}"}
FINAL = {"trainer": {"epochs": 30, "batch_size": 128, "lr": 0.1, "weight_decay": 0.01, "warmup_steps": 100},
         "optimizer": {"name": "adamw"},
         "run_name": "exp-adamw-bs128",
         "eval_batch_size": 128}


def override_call(config, overrides):
    return calling(f"config = {src(config)}\nprint(apply_overrides(config, {overrides!r}))")


def override_case(sol, config, overrides, want):
    override_call(config, overrides)
    passed = ref.clone(config)
    got = sol.apply_overrides(passed, list(overrides))
    check(sol, got, want)
    untouched(sol, passed, config, "config")
    expect_true(got is not passed, "apply_overrides 应当返回一份拷贝，而不是传入的 config 本身")


def override_raises(sol, exc, config, overrides):
    override_call(config, overrides)
    passed = ref.clone(config)
    expect_raises(getattr(sol, exc), sol.apply_overrides, passed, list(overrides))
    untouched(sol, passed, config, "config")


@case("Part 2", sample=True, title="题目例子：三个覆盖")
def p2_example_overrides(sol):
    override_case(sol, RESOLVED, ["trainer.epochs=30", "+trainer.warmup_steps=100", 'optimizer.name="adamw"'],
                  OVERRIDDEN)


@case("Part 2", sample=True, title="题目例子：插值")
def p2_example_interp(sol):
    interp_case(sol, OVERRIDDEN, FINAL)


VALUES = [
    ("007", 7), ("0", 0), ("-0", 0), ("-12", -12), ("-007", -7), ("12345678901234567890", 12345678901234567890),
    ("1.5", 1.5), ("2.0", 2.0), ("-0.0", -0.0), ("0.001", 0.001), ("01.5", 1.5), ("1e5", 100000.0),
    ("-2e-2", -0.02), ("1.5e3", 1500.0), ("7e0", 7.0),
    ("null", None), ("true", True), ("false", False),
    ("'adamw'", "adamw"), ('"adamw"', "adamw"), ("''", ""), ('""', ""), ('"a=b"', "a=b"), ("'x,y'", "x,y"),
    ("'s3://bucket/x-1_2.txt'", "s3://bucket/x-1_2.txt"), ('"it\'s"', "it's"), ("'say\"hi\"'", 'say"hi"'),
    ("'007'", "007"), ("'true'", "true"), ("'null'", "null"), ("'[1]'", "[1]"), ("'${a.b}'", "${a.b}"),
    ("[]", []), ("[ ]", []), ("[1,2]", [1, 2]), ("[ 1 , 2 ]", [1, 2]), ("[007]", [7]),
    ("[1, 2.5, true, null, 'x,y']", [1, 2.5, True, None, "x,y"]), ("['a]', \"b,c\"]", ["a]", "b,c"]),
    ("[  'a'  ,  'b'  ]", ["a", "b"]), ("[-0.0,-0]", [-0.0, 0]),
]


@case("Part 2", title="各种合法的值")
def p2_values(sol):
    for text, want in VALUES:
        override_case(sol, {}, ["+v=" + text], {"v": want})


BAD_OVERRIDES = [
    "v=adamw", "v=True", "v=None", "v=NULL", "v=truee", "v=1.", "v=.5", "v=1e", "v=e5", "v=1.e5", "v=+1", "v=+1.5",
    "v=--1", "v=-", "v=1_000", "v=0x1F", "v=nan", "v=inf", "v=-inf", "v=NaN", "v=Infinity", "v=1.5.2", "v=1e5.0",
    "v=1.5e", "v='abc", "v=abc'", "v='abc\"", "v='a'b'", 'v="a""b"', "v=", "v= 1", "v =1", " v=1", "v=1 ",
    "+ v=1", "v=1 2", "v='a' ", "v= [1]", "v=[1] ", "v=[1 2]", "v=[[1]]", "v=[1, [2]]", "v=[1,,2]", "v=[1,]",
    "v=[,]", "v=[,1]", "v=[ ,1]", "v=[1]]", "v=[1", "v=1]", "v=[1,'a]", "v=['a',]", "v", "", "=1", "+=1", "++v=1",
    "+v", "1v=1", "w..x=1", ".v=1", "v.=1", "v-w=1", "v[0]=1", "w.0=1", "w.x =1", "v==1", "v=1=2", "v=1,2",
    "v={}", "v={'a': 1}", "v=(1)",
]


@case("Part 2", title="不合法的覆盖：OverrideSyntaxError")
def p2_syntax_errors(sol):
    for text in BAD_OVERRIDES:
        override_raises(sol, "OverrideSyntaxError", {"v": 0, "w": {"x": 0}}, [text])


OVR_CFG = {"trainer": {"lr": 0.1, "opt": {"name": "sgd"}}, "tags": ["a"], "seed": 1}


@case("Part 2", title="没有 + 时 path 必须存在")
def p2_missing_keys(sol):
    for text in ["trainer.epochs=10", "trainer.opt.beta=0.9", "model.dim=3", "Trainer.lr=1", "seed.x=1",
                 "tags.x=1", "trainer.lr.x=1"]:
        override_raises(sol, "OverrideKeyError", OVR_CFG, [text])
    override_case(sol, OVR_CFG, ["trainer=5"], {"trainer": 5, "tags": ["a"], "seed": 1})
    override_case(sol, OVR_CFG, ["trainer.opt='x'", "tags=[]", "seed=null"],
                  {"trainer": {"lr": 0.1, "opt": "x"}, "tags": [], "seed": None})
    override_case(sol, OVR_CFG, ["trainer.opt.name='adam'"],
                  {"trainer": {"lr": 0.1, "opt": {"name": "adam"}}, "tags": ["a"], "seed": 1})


@case("Part 2", title="+ 创建缺失的段")
def p2_plus_paths(sol):
    override_case(sol, OVR_CFG, ["+trainer.epochs=10"],
                  {"trainer": {"lr": 0.1, "opt": {"name": "sgd"}, "epochs": 10}, "tags": ["a"], "seed": 1})
    override_case(sol, OVR_CFG, ["+model.enc.dim=3"],
                  {"trainer": {"lr": 0.1, "opt": {"name": "sgd"}}, "tags": ["a"], "seed": 1,
                   "model": {"enc": {"dim": 3}}})
    override_case(sol, OVR_CFG, ["+trainer.lr=0.5", "+trainer.opt=1", "+a=[1, 2]"],
                  {"trainer": {"lr": 0.5, "opt": 1}, "tags": ["a"], "seed": 1, "a": [1, 2]})
    override_case(sol, OVR_CFG, ["+trainer.opt.betas=[0.9, 0.999]"],
                  {"trainer": {"lr": 0.1, "opt": {"name": "sgd", "betas": [0.9, 0.999]}}, "tags": ["a"], "seed": 1})
    override_case(sol, {}, ["+a.b.c.d='deep'"], {"a": {"b": {"c": {"d": "deep"}}}})
    for text in ["+seed.x=1", "+tags.x=1", "+trainer.lr.x=1", "+trainer.opt.name.first='a'"]:
        override_raises(sol, "OverrideKeyError", OVR_CFG, [text])


@case("Part 2", title="按顺序应用；返回拷贝")
def p2_override_order_and_copy(sol):
    override_case(sol, {}, ["+x=1", "x=2"], {"x": 2})
    override_case(sol, {"x": 0}, ["x=1", "x='a'", "x=[true]"], {"x": [True]})
    override_case(sol, {}, ["+x.y=1", "+x.z=2", "x.y=3"], {"x": {"y": 3, "z": 2}})
    override_raises(sol, "OverrideKeyError", {}, ["x=1", "+x=2"])
    override_raises(sol, "OverrideKeyError", {"trainer": {"lr": 0.1}}, ["trainer=5", "+trainer.lr=1"])
    override_raises(sol, "OverrideKeyError", {"a": {"b": 1}}, ["a.b=2", "+a.c=3", "a.d=4"])
    override_case(sol, {"a": {"b": [1]}, "c": 1}, [], {"a": {"b": [1]}, "c": 1})


@case("Part 2", title="随机覆盖，和逐条按规则计算比对")
def p2_override_random(sol):
    r = rng()
    for _ in range(400):
        cfg = ref.rand_config(r)
        texts, ops = ref.rand_ops(r, cfg)
        try:
            want = ref.apply_ops(cfg, ops)
        except ref.OverrideKey:
            override_raises(sol, "OverrideKeyError", cfg, texts)
        else:
            override_case(sol, cfg, texts, want)


# ---------------------------------------------------------------- Part 2: interpolation

def interp_call(config):
    return calling(f"config = {src(config)}\nprint(resolve_interpolations(config))")


def interp_case(sol, config, want):
    interp_call(config)
    passed = ref.clone(config)
    got = sol.resolve_interpolations(passed)
    check(sol, got, want)
    untouched(sol, passed, config, "config")


def interp_raises(sol, config):
    interp_call(config)
    passed = ref.clone(config)
    expect_raises(sol.InterpolationError, sol.resolve_interpolations, passed)
    untouched(sol, passed, config, "config")


def interp_cycle(sol, config, ring):
    interp_call(config)
    e = expect_raises(sol.InterpolationCycleError, sol.resolve_interpolations, ref.clone(config))
    expect_true(isinstance(e, sol.ConfigError), "InterpolationCycleError 应当是 ConfigError 的子类")
    if ring:
        m = ref.cycle_problem(str(e), ring)
        if m:
            raise Mismatch(m, show=False)


@case("Part 2", title="bool 嵌进字符串是 true/false")
def p2_interp_bool(sol):
    cfg = {"flag": True, "off": False, "name": "run", "a": "debug=${flag}", "b": "${off}-${flag}", "c": "${flag}",
           "d": ["${off}", "x${off}"], "e": "${name}_${flag}", "x": "${flag}", "y": "v${x}"}
    interp_case(sol, cfg, {"flag": True, "off": False, "name": "run", "a": "debug=true", "b": "false-true",
                           "c": True, "d": [False, "xfalse"], "e": "run_true", "x": True, "y": "vtrue"})


@case("Part 2", title="整串引用保留类型，嵌入用 str(x)")
def p2_interp_types(sol):
    cfg = {"i": 3, "f": 0.5, "s": "adam", "n": None, "l": [1, "x"], "d": {"k": 1, "j": [2]}, "b": False,
           "tiny": 1e-07, "huge": 1e22, "neg": -0.0, "empty": "",
           "ri": "${i}", "rf": "${f}", "rs": "${s}", "rn": "${n}", "rl": "${l}", "rd": "${d}", "rb": "${b}",
           "rdk": "${d.k}", "rdj": "${d.j}", "rneg": "${neg}", "rempty": "${empty}",
           "e1": "i=${i}", "e2": "${f}x", "e3": "[${s}]", "e4": "${i}${i}", "e5": "${s}.${i}.${f}",
           "e6": "lr=${tiny}", "e7": "${huge}!", "e8": "<${empty}>", "e9": "${neg}/${d.k}",
           "lit": "costs $5", "lit2": "a{b}", "lit3": "$$"}
    want = {"i": 3, "f": 0.5, "s": "adam", "n": None, "l": [1, "x"], "d": {"k": 1, "j": [2]}, "b": False,
            "tiny": 1e-07, "huge": 1e22, "neg": -0.0, "empty": "",
            "ri": 3, "rf": 0.5, "rs": "adam", "rn": None, "rl": [1, "x"], "rd": {"k": 1, "j": [2]}, "rb": False,
            "rdk": 1, "rdj": [2], "rneg": -0.0, "rempty": "",
            "e1": "i=3", "e2": "0.5x", "e3": "[adam]", "e4": "33", "e5": "adam.3.0.5",
            "e6": "lr=1e-07", "e7": "1e+22!", "e8": "<>", "e9": "-0.0/1",
            "lit": "costs $5", "lit2": "a{b}", "lit3": "$$"}
    interp_case(sol, cfg, want)


@case("Part 2", title="引用链、列表和 dict 里的引用")
def p2_interp_nested(sol):
    interp_case(sol, {"a": "${b}", "b": "${c}", "c": 5}, {"a": 5, "b": 5, "c": 5})
    interp_case(sol, {"base": "/data", "train": "${base}/train", "file": "${train}/x.bin"},
                {"base": "/data", "train": "/data/train", "file": "/data/train/x.bin"})
    interp_case(sol, {"opt": {"name": "${n}", "lr": "${lr}"}, "n": "adam", "lr": 0.1, "copy": "${opt}"},
                {"opt": {"name": "adam", "lr": 0.1}, "n": "adam", "lr": 0.1, "copy": {"name": "adam", "lr": 0.1}})
    interp_case(sol, {"a": 2, "xs": ["${a}", "p${a}", 3, ["${a}", {"k": "${a}"}]]},
                {"a": 2, "xs": [2, "p2", 3, [2, {"k": 2}]]})
    interp_case(sol, {"m": {"enc": {"dim": 64}}, "dec": {"dim": "${m.enc.dim}", "hid": "${dec.dim}", "s": "h${dec.dim}"}},
                {"m": {"enc": {"dim": 64}}, "dec": {"dim": 64, "hid": 64, "s": "h64"}})
    interp_case(sol, {"a": "${b}-${c}", "b": "${c}x", "c": 1}, {"a": "1x-1", "b": "1x", "c": 1})
    interp_case(sol, {"x": "${y.z}", "y": {"z": "${w}"}, "w": [1, "${v}"], "v": "${u}", "u": None},
                {"x": [1, None], "y": {"z": [1, None]}, "w": [1, None], "v": None, "u": None})


@case("Part 2", title="不存在的 path、嵌入非标量：InterpolationError")
def p2_interp_errors(sol):
    for cfg in [
        {"a": "${missing}"},
        {"a": "${t.nope}", "t": {"x": 1}},
        {"a": "${t.x.y}", "t": {"x": 1}},
        {"a": "${l.x}", "l": [1, 2]},
        {"a": "${x.y.z}"},
        {"a": "x${n}", "n": None},
        {"a": "x${l}", "l": [1]},
        {"a": "x${d}", "d": {"k": 1}},
        {"a": "${d}${d}", "d": {}},
        {"a": ["ok", "${nope}"]},
        {"a": {"b": {"c": "v-${n}"}}, "n": None},
        {"a": "${b}", "b": "y${n}", "n": None},
        {"a": "${b}", "b": "${c.k}", "c": "text"},
    ]:
        interp_raises(sol, cfg)


@case("Part 2", title="插值的环：InterpolationCycleError")
def p2_interp_cycle(sol):
    interp_cycle(sol, {"a": "${b}", "b": "${a}"}, ["a", "b"])
    interp_cycle(sol, {"x": "${x}"}, ["x"])
    interp_cycle(sol, {"lr": "lr=${lr}"}, ["lr"])
    interp_cycle(sol, {"t": {"a": "${t.b}", "b": "x${t.a}"}, "u": 1}, ["t.a", "t.b"])
    interp_cycle(sol, {"p": "${q}", "q": "${r}", "r": "-${s}", "s": "${q}"}, ["q", "r", "s"])
    interp_cycle(sol, {"a": {"b": "${a}"}}, None)


@case("Part 2", title="随机插值，构造时同步算出结果")
def p2_interp_random(sol):
    r = rng()
    for _ in range(250):
        cfg, want = ref.rand_interp(r)
        interp_case(sol, cfg, want)
    for _ in range(150):
        cfg, _ = ref.rand_interp(r, bad=True)
        interp_raises(sol, cfg)
    for _ in range(100):
        cfg, ring = ref.rand_interp_ring(r)
        interp_cycle(sol, cfg, ring)


# ---------------------------------------------------------------- Part 3

BROKEN = {"trainer": {"epochs": 30, "batch_size": 128, "lr": "0.1", "weight_decay": 0.01,
                      "warmup_steps": 100, "debug": True},
          "optimizer": {"name": "adamw"},
          "run_name": "exp-adamw-bs128"}


def validate_case(sol, config, schema, want):
    calling(f"from ref import {schema.__name__}\nconfig = {src(config)}\n"
            f"print(validate_config(config, {schema.__name__}))")
    got = sol.validate_config(ref.clone(config), schema)
    m = ref.errors_problem(got, want)
    if m:
        raise Mismatch(m, want=ref.ShowErrors(want), got=got)


@case("Part 3", sample=True, title="题目例子：final 合法")
def p3_example_valid(sol):
    validate_case(sol, FINAL, ref.ExperimentConfig, [])


@case("Part 3", sample=True, title="题目例子：broken 的三处错误")
def p3_example_broken(sol):
    validate_case(sol, BROKEN, ref.ExperimentConfig,
                  [("trainer.lr", "type"), ("trainer.debug", "unexpected"), ("eval_batch_size", "missing")])


@case("Part 3", title="bool 不能当 int 或 float")
def p3_bool_not_number(sol):
    validate_case(sol, {"i": True, "f": 1.5, "b": True, "s": "x"}, ref.Prims, [("i", "type")])
    validate_case(sol, {"i": 1, "f": False, "b": False, "s": "x"}, ref.Prims, [("f", "type")])
    validate_case(sol, {"i": False, "f": True, "b": 1, "s": 5}, ref.Prims,
                  [("i", "type"), ("f", "type"), ("b", "type"), ("s", "type")])
    validate_case(sol, {"ints": [1, True, 2], "floats": [0.5, False], "flags": [True, 0], "names": ["a"]}, ref.Lists,
                  [("ints.1", "type"), ("floats.1", "type"), ("flags.1", "type")])
    validate_case(sol, {"epochs": True, "batch_size": 32, "lr": 0.5}, ref.TrainerConfig, [("epochs", "type")])


@case("Part 3", title="float 字段接受 int")
def p3_int_for_float(sol):
    validate_case(sol, {"i": 1, "f": 3, "b": True, "s": ""}, ref.Prims, [])
    validate_case(sol, {"floats": [1, 2.5, 0, -4]}, ref.Lists, [])
    validate_case(sol, {"epochs": 10, "batch_size": 32, "lr": 1, "weight_decay": 0}, ref.TrainerConfig, [])
    validate_case(sol, {"name": "sgd", "momentum": 0}, ref.OptimizerConfig, [])
    validate_case(sol, {"i": 3.0, "f": 3, "b": True, "s": "x"}, ref.Prims, [("i", "type")])
    validate_case(sol, {"i": 1, "f": "1.5", "b": True, "s": "x"}, ref.Prims, [("f", "type")])


RUN_OK = {"data": {"path": "p", "aug": {}}, "model": {"layers": [{"hidden_size": 8}]}, "seeds": [1], "lr": 0.1}


@case("Part 3", title="嵌套模式和完整路径")
def p3_nested(sol):
    run_name = {"run_name": "r", "eval_batch_size": 1, "optimizer": {"name": "sgd"}}
    validate_case(sol, run_name, ref.ExperimentConfig, [("trainer", "missing")])
    for bad in (5, [], None, "trainer"):
        validate_case(sol, {**run_name, "trainer": bad}, ref.ExperimentConfig, [("trainer", "type")])
    validate_case(sol, {**run_name, "trainer": {}}, ref.ExperimentConfig,
                  [("trainer.epochs", "missing"), ("trainer.batch_size", "missing"), ("trainer.lr", "missing")])
    validate_case(sol, RUN_OK, ref.RunConfig, [])
    deep = {"data": {"path": "p", "aug": {"flip": 1, "zoom": 2, "scales": [1, "x"]}, "shards": [1]},
            "model": {"layers": [{"hidden_size": 64}, {"activation": "gelu", "dim": 3}, 7], "dropout": 0.1},
            "seeds": [1, 2], "lr": 0.1,
            "stages": [{"path": "s", "aug": {}}, {"aug": {"crop": True}}]}
    validate_case(sol, deep, ref.RunConfig,
                  [("data.aug.flip", "type"), ("data.aug.zoom", "unexpected"), ("data.aug.scales.1", "type"),
                   ("model.layers.1.hidden_size", "missing"), ("model.layers.1.dim", "unexpected"),
                   ("model.layers.2", "type"), ("stages.1.path", "missing"), ("stages.1.aug.crop", "type")])


@case("Part 3", title="list[X] 字段")
def p3_lists(sol):
    validate_case(sol, {"names": "abc"}, ref.Lists, [("names", "type")])
    validate_case(sol, {"ints": {"a": 1}}, ref.Lists, [("ints", "type")])
    validate_case(sol, {"ints": None, "flags": True}, ref.Lists, [("ints", "type"), ("flags", "type")])
    validate_case(sol, {"ints": [], "names": []}, ref.Lists, [])
    validate_case(sol, {}, ref.Lists, [])
    validate_case(sol, {"names": [1, "a", None]}, ref.Lists, [("names.0", "type"), ("names.2", "type")])
    validate_case(sol, {k: v for k, v in RUN_OK.items() if k != "seeds"}, ref.RunConfig, [("seeds", "missing")])
    validate_case(sol, {"layers": "abc"}, ref.ModelConfig, [("layers", "type")])
    validate_case(sol, {"layers": []}, ref.ModelConfig, [])


@case("Part 3", title="config 本身不是 dict")
def p3_root_not_dict(sol):
    for bad in ([], "text", None, 5, [{"trainer": {}}]):
        validate_case(sol, bad, ref.ExperimentConfig, [("<root>", "type")])


@case("Part 3", title="随机模式和损坏的配置")
def p3_random(sol):
    r = rng()
    for schema in (ref.ExperimentConfig, ref.RunConfig, ref.ModelConfig, ref.Prims, ref.Lists):
        for _ in range(120):
            cfg = ref.sample_value(r, schema)
            if r.random() < 0.7:
                cfg = ref.corrupt(r, cfg, schema)
            validate_case(sol, cfg, schema, ref.validate(cfg, schema))


# ---------------------------------------------------------------- Part 4

def setitem(obj, key, value):
    obj[key] = value


@case("Part 4", sample=True, title="题目例子：freeze_config")
def p4_freeze_example(sol):
    calling(f"final = {src(FINAL)}\nfrozen = freeze_config(final)\nprint(frozen['trainer']['epochs'])")
    frozen = sol.freeze_config(ref.clone(FINAL))
    expect(frozen["trainer"]["epochs"], 30)
    calling(f"final = {src(FINAL)}\nfrozen = freeze_config(final)\nfrozen['trainer']['epochs'] = 999")
    expect_raises(TypeError, setitem, frozen["trainer"], "epochs", 999)


@case("Part 4", sample=True, title="题目例子：config_hash")
def p4_hash_example(sol):
    c1 = {"trainer": {"epochs": 30, "batch_size": 128}, "optimizer": {"name": "adamw"}}
    c2 = {"optimizer": {"name": "adamw"}, "trainer": {"batch_size": 128, "epochs": 30}}
    c3 = {"trainer": {"epochs": 31, "batch_size": 128}, "optimizer": {"name": "adamw"}}
    hash_same(sol, c1, c2)
    hash_differ(sol, c1, c3)


FREEZE_CFG = {"a": {"b": [1, {"c": [2, 3], "d": {}}], "e": {}}, "l": [], "s": "x", "n": None, "f": -0.0, "t": True,
              "nested": [[1, [2]], []]}


def mappings(frozen, path=()):
    if isinstance(frozen, MappingProxyType):
        yield path, frozen
        items = frozen.items()
    elif isinstance(frozen, tuple):
        items = enumerate(frozen)
    else:
        return
    for k, v in items:
        yield from mappings(v, path + (k,))


def tuples(frozen, path=()):
    if isinstance(frozen, tuple):
        yield path, frozen
    if isinstance(frozen, (MappingProxyType, tuple)):
        for k, v in (frozen.items() if isinstance(frozen, MappingProxyType) else enumerate(frozen)):
            yield from tuples(v, path + (k,))


@case("Part 4", title="每层都是只读映射和 tuple")
def p4_freeze_structure(sol):
    head = f"config = {src(FREEZE_CFG)}\nfrozen = freeze_config(config)\n"
    calling(head + "print(frozen)")
    frozen = sol.freeze_config(ref.clone(FREEZE_CFG))
    m = ref.frozen_problem(frozen, FREEZE_CFG)
    if m:
        raise Mismatch(m, show=False)
    for path, node in mappings(frozen):
        for key in list(node.keys())[:1] + ["new_key"]:
            calling(head + f"frozen{ref.index_src(path)}[{key!r}] = 0")
            expect_raises(TypeError, setitem, node, key, 0)
    for path, node in tuples(frozen):
        if node:
            calling(head + f"frozen{ref.index_src(path)}[0] = 0")
            expect_raises(TypeError, setitem, node, 0, 0)
    calling("print(freeze_config({}))")
    m = ref.frozen_problem(sol.freeze_config({}), {})
    if m:
        raise Mismatch(m, show=False)


@case("Part 4", title="快照：之后修改原配置不影响结果")
def p4_freeze_snapshot(sol):
    cfg = {"a": {"b": [1, 2], "c": {"d": 1}}, "flat": {"k": 1}, "s": "x", "lst": [{"z": 0}]}
    head = f"config = {src(cfg)}\nfrozen = freeze_config(config)\n"
    passed = ref.clone(cfg)
    calling(head + "print(config)")
    sol.freeze_config(passed)
    untouched(sol, passed, cfg, "config")
    passed = ref.clone(cfg)
    frozen = sol.freeze_config(passed)
    passed["a"]["b"].append(9)
    passed["a"]["c"]["d"] = 2
    passed["flat"]["k"] = 99
    passed["flat"]["new"] = 1
    passed["lst"][0]["z"] = 5
    passed["s"] = "changed"
    passed["added"] = 1
    calling(head + "config['a']['b'].append(9)\nconfig['a']['c']['d'] = 2\nconfig['flat']['k'] = 99\n"
            "config['flat']['new'] = 1\nconfig['lst'][0]['z'] = 5\nconfig['s'] = 'changed'\nconfig['added'] = 1\n"
            "print(frozen)   # 应当还是冻结那一刻的内容")
    m = ref.frozen_problem(frozen, cfg)
    if m:
        raise Mismatch(f"freeze_config 返回的应当是快照，之后修改原配置不应影响它：{m}", show=False)


def hash_of(sol, config):
    h = sol.config_hash(ref.clone(config))
    expect_true(isinstance(h, str), f"config_hash 应当返回 str，得到 {type(h).__name__}", got=h)
    return h


def hash_same(sol, a, b):
    calling(f"a = {src(a)}\nb = {src(b)}\nprint(config_hash(a) == config_hash(b))   # 应当是 True")
    ha, hb = hash_of(sol, a), hash_of(sol, b)
    if ha != hb:
        raise Mismatch("两个内容相同的配置（只是键的顺序不同）应当得到相同的哈希", want=ha, got=hb)


def hash_differ(sol, a, b):
    calling(f"a = {src(a)}\nb = {src(b)}\nprint(config_hash(a) != config_hash(b))   # 应当是 True")
    ha, hb = hash_of(sol, a), hash_of(sol, b)
    if ha == hb:
        raise Mismatch("这两个配置的内容不同，哈希却相同：只要某个值不同，哈希值就应当不同", got=ha)


@case("Part 4", title="键的顺序不影响哈希")
def p4_hash_equal(sol):
    hash_same(sol, {"a": 1, "b": 2}, {"b": 2, "a": 1})
    hash_same(sol, {"x": {"p": [1, {"q": 1, "r": 2}], "s": None}, "y": "z"},
              {"y": "z", "x": {"s": None, "p": [1, {"r": 2, "q": 1}]}})
    hash_same(sol, {"f": 0.1, "g": 1e22, "e": {}, "l": []}, {"l": [], "e": {}, "g": 1e+22, "f": 1e-1})
    hash_same(sol, {}, {})
    calling(f"config = {src(FINAL)}\nprint(config_hash(config) == config_hash(config))   # 应当是 True")
    expect(hash_of(sol, FINAL), hash_of(sol, FINAL), msg="同一个配置算两次，哈希应当相同")


DISTINCT_PAIRS = [
    ({"a": 2}, {"a": 2.0}),
    ({"a": 0.0}, {"a": -0.0}),
    ({"a": True}, {"a": 1}),
    ({"a": False}, {"a": 0}),
    ({"a": "1"}, {"a": 1}),
    ({"a": None}, {}),
    ({"a": None}, {"a": "None"}),
    ({"a": None}, {"a": "null"}),
    ({"a": [1, 2]}, {"a": [2, 1]}),
    ({"x": {"y": 1}}, {"x.y": 1}),
    ({"a": []}, {"a": {}}),
    ({"a": [1, [2]]}, {"a": [[1], 2]}),
    ({"a": {"b": 1}, "c": 2}, {"a": {"b": 1, "c": 2}}),
    ({"a": 0.1}, {"a": 0.10000000000000002}),
    ({"a": 10 ** 20}, {"a": 10 ** 20 + 1}),
    ({"a": 1}, {"b": 1}),
    ({"a": "x", "b": "y"}, {"a": "x\", \"b\": \"y"}),
    ({"a": [1, 2]}, {"a": "[1, 2]"}),
    ({"a": ""}, {"a": None}),
]


@case("Part 4", title="值不同哈希就不同（含 2/2.0、0.0/-0.0）")
def p4_hash_distinct(sol):
    for a, b in DISTINCT_PAIRS:
        hash_differ(sol, a, b)


@case("Part 4", title="NaN 和无穷大：ValueError")
def p4_hash_nan(sol):
    for cfg in ({"x": float("nan")}, {"x": float("inf")}, {"x": float("-inf")},
                {"a": [1.0, {"b": float("nan")}], "c": 1}):
        calling(f"print(config_hash({src(cfg)}))")
        expect_raises(ValueError, sol.config_hash, ref.clone(cfg))


CROSS_CFG = {"trainer": {"epochs": 30, "lr": 0.1, "betas": [0.9, 0.999]}, "name": "run", "flag": True, "none": None,
             "zero": -0.0, "tags": ["a", "数据"]}
CHILD = """import runpy, sys
sys.path[:0] = sys.argv[3].split("\\n")
h = runpy.run_path(sys.argv[1], run_name="sol")["config_hash"](%s)
open(sys.argv[2], "w", encoding="utf-8").write(repr(h))
"""


@case("Part 4", cpu=4, title="换一个 Python 进程哈希不变")
def p4_hash_cross_process(sol):
    calling("\n".join([
        "import os, subprocess, sys",
        f"config = {src(CROSS_CFG)}",
        "child = (f'import runpy; print(runpy.run_path({os.path.abspath(__file__)!r}, run_name=\"sol\")'",
        "         f'[\"config_hash\"]({config!r}))')",
        "for seed in ('12345', '67890'):",
        "    env = {**os.environ, 'PYTHONHASHSEED': seed}",
        "    out = subprocess.run([sys.executable, '-c', child], env=env, capture_output=True, text=True)",
        "    print('PYTHONHASHSEED', seed, out.stdout.strip() or out.stderr)",
        "print('this process      ', config_hash(config))",
    ]))
    here = hash_of(sol, CROSS_CFG)
    work = tempfile.mkdtemp()
    for seed in ("12345", "67890"):
        out = os.path.join(work, f"hash_{seed}.txt")
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", CHILD % src(CROSS_CFG), os.path.abspath(sol.__file__), out,
                               "\n".join(p for p in sys.path if p)],
                              env=env, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0 or not os.path.exists(out):
            raise Mismatch(f"在另一个 Python 进程（PYTHONHASHSEED={seed}）里载入答案并调用 config_hash 失败："
                           f"{(proc.stderr or '').strip()[-600:]}", show=False)
        with open(out, encoding="utf-8") as f:
            there = f.read()
        if there != repr(here):
            raise Mismatch(f"同一个配置在另一个 Python 进程（PYTHONHASHSEED={seed}）里得到的哈希不同：这个哈希要在几周后"
                           "复现运行时用来比对，不能依赖每个进程随机的 hash()", want=repr(here), got=there, show=True)


@case("Part 4", title="随机配置：打乱键序相同，改一个值不同")
def p4_hash_random(sol):
    r = rng()
    seen = {}
    for _ in range(200):
        cfg = ref.rand_config(r, scalar=ref.hash_scalar)
        hash_same(sol, cfg, ref.shuffled(r, cfg))
        changed = ref.mutated(r, cfg)
        if ref.canon(changed) != ref.canon(cfg):
            hash_differ(sol, cfg, changed)
        h = hash_of(sol, cfg)
        key = ref.canon(cfg)
        if h in seen and seen[h][0] != key:
            hash_differ(sol, seen[h][1], cfg)
        seen.setdefault(h, (key, cfg))
