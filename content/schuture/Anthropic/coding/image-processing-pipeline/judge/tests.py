import os
import pickle
from contextlib import contextmanager

from PIL import Image

from judgelib import calling, case, expect_true, info, rng

import ref

GRAY = {"type": "grayscale"}
FLIP_H = {"type": "flip_horizontal"}
FLIP_V = {"type": "flip_vertical"}


def scale(f):
    return {"type": "scale", "factor": f}


def blur(r):
    return {"type": "blur", "radius": r}


def rotate(a):
    return {"type": "rotate", "angle": a}


def noise(r, w, h):
    return ref.noise_image(r, w, h)


@contextmanager
def workspace(images, pipelines, label=None):
    ws = ref.Workspace(images, pipelines, label)
    try:
        yield ws
    finally:
        ws.cleanup()


P1 = "process_images(image_dir, transformation_dir, output_dir, get_output_path)"


def p23(fn, workers):
    return f"{fn}(image_dir, transformation_dir, output_dir, get_output_path, max_workers={workers})"


def _pickle_trouble(e):
    text = str(e).lower()
    return isinstance(e, pickle.PicklingError) or "pickle" in text or "can't get attribute" in text


def run(fn, ws, call, outputs=True, **kwargs):
    """Run the answer on the workspace; check every output file unless outputs=False. Returns ref.Run."""
    shown = calling(ws.describe(call))
    try:
        got = ref.run_answer(fn, ws, **kwargs)
    except Exception as e:
        if _pickle_trouble(e):
            info("提示：跨进程边界只能传文件路径和 JSON 大小的数据。get_output_path 是一个不能 pickle 的闭包，"
                 "要在主进程里调用它，把算好的路径交给工作进程；工作进程的函数要定义在模块顶层。")
        elif type(e).__name__ == "BrokenProcessPool":
            info("提示：有工作进程异常退出（例如被系统杀掉、调用了 os._exit / sys.exit，或者在工作进程里"
                 "反序列化任务时出错）。")
        raise
    if outputs:
        problem = ws.check_outputs()
        expect_true(problem is None, problem, call=shown)
    return got


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子 sample.png")
def p1_example(sol):
    images = {"sample.png": noise(rng(), 60, 40)}
    pipelines = {"p.json": [GRAY, scale(0.5), rotate(90)]}
    with workspace(images, pipelines) as ws:
        shown = ws.describe(P1)
        run(sol.process_images, ws, P1, outputs=False)
        path = ws.get_output_path("sample.png", "p.json")
        expect_true(os.path.exists(path), "没有写出 get_output_path(\"sample.png\", \"p.json\")", call=shown)
        with Image.open(path) as im:
            expect_true((im.mode, im.size) == ("L", (20, 30)),
                        f"结果应为模式 \"L\"、尺寸 (20, 30)，得到模式 {im.mode!r}、尺寸 {im.size}", call=shown)
        problem = ws.check_outputs()
        expect_true(problem is None, problem, call=shown)


@case("Part 1", sample=True, title="题目例子 2 张图 × 2 条流水线")
def p1_example_four(sol):
    r = rng()
    images = {"a.png": noise(r, 8, 6), "b.png": noise(r, 8, 6)}
    pipelines = {"x.json": [GRAY], "y.json": [FLIP_H]}
    with workspace(images, pipelines) as ws:
        run(sol.process_images, ws, P1, outputs=False)
        got = sorted(os.listdir(ws.output_dir))
        want = sorted(os.path.basename(ws.get_output_path(i, j)) for i in ("a.png", "b.png")
                      for j in ("x.json", "y.json"))
        expect_true(got == want, f"应当恰好写出 4 个文件 {want}，实际 output_dir 里是 {got}",
                    call=ws.describe(P1))
        problem = ws.check_outputs()
        expect_true(problem is None, problem, call=ws.describe(P1))


@case("Part 1", title="空流水线：原样保存为 PNG")
def p1_empty_pipeline(sol):
    r = rng()
    images = {"one.png": noise(r, 7, 5), "two.png": noise(r, 1, 1), "three.png": noise(r, 13, 2)}
    with workspace(images, {"nothing.json": [], "also_nothing.json": []}) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="灰度与两种翻转")
def p1_gray_flips(sol):
    r = rng()
    images = {"wide.png": noise(r, 9, 4), "tall.png": noise(r, 3, 8)}
    pipelines = {"g.json": [GRAY], "h.json": [FLIP_H], "v.json": [FLIP_V], "hv.json": [FLIP_H, FLIP_V],
                 "gh.json": [GRAY, FLIP_H], "vg.json": [FLIP_V, GRAY]}
    with workspace(images, pipelines) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="scale：四舍五入到偶数")
def p1_scale_round(sol):
    r = rng()
    # 7*0.5=3.5->4, 5*0.5=2.5->2, 9*0.5=4.5->4, 3*0.5=1.5->2, 6*0.75=4.5->4, 10*0.75=7.5->8, 3*1.5=4.5->4
    images = {"w7h5.png": noise(r, 7, 5), "w9h3.png": noise(r, 9, 3), "w6h10.png": noise(r, 6, 10)}
    pipelines = {"half.json": [scale(0.5)], "three_quarters.json": [scale(0.75)], "one_and_half.json": [scale(1.5)]}
    with workspace(images, pipelines, label="例如 7×5 缩放 0.5：round(3.5)=4，round(2.5)=2") as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="scale：放大与最小 1 像素")
def p1_scale_clamp(sol):
    r = rng()
    images = {"w5h5.png": noise(r, 5, 5), "w3h1.png": noise(r, 3, 1), "w2h7.png": noise(r, 2, 7)}
    pipelines = {"tiny.json": [scale(0.1)], "small.json": [scale(0.3)], "quarter.json": [scale(0.25)],
                 "double.json": [scale(2.0)], "int_factor.json": [scale(3)], "same.json": [scale(1.0)]}
    with workspace(images, pipelines, label="5×5 缩放 0.1 应得到 1×1：两边都 round 成 0，再取 max(1, …)") as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="blur：半径 0 到 3")
def p1_blur(sol):
    r = rng()
    images = {"rgb.png": noise(r, 11, 9)}
    pipelines = {"b0.json": [blur(0)], "b1.json": [blur(1)], "b2.json": [blur(2)], "b3.json": [blur(3)],
                 "gb2.json": [GRAY, blur(2)]}
    with workspace(images, pipelines) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="rotate：逆时针、扩展画布、双线性")
def p1_rotate(sol):
    r = rng()
    images = {"rect.png": noise(r, 9, 6)}
    pipelines = {"r45.json": [rotate(45)], "r33_5.json": [rotate(33.5)], "rm30.json": [rotate(-30)],
                 "r90.json": [rotate(90)], "r180.json": [rotate(180)], "r270.json": [rotate(270)],
                 "r360.json": [rotate(360)], "r0.json": [rotate(0)], "g_r45.json": [GRAY, rotate(45)],
                 "r10_r10.json": [rotate(10), rotate(10)]}
    with workspace(images, pipelines) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="JPEG、BMP 等任意格式的源图")
def p1_formats(sol):
    r = rng()
    images = {"photo.jpg": noise(r, 10, 8), "scan.bmp": noise(r, 7, 9), "shot.JPEG": noise(r, 6, 6),
              "plain.png": noise(r, 5, 4)}
    pipelines = {"none.json": [], "mix.json": [GRAY, scale(0.5)], "turn.json": [FLIP_V, rotate(45)]}
    with workspace(images, pipelines, label="非 PNG 的源图同样要处理，结果一律存成 PNG") as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="没有图片或没有流水线")
def p1_empty_dirs(sol):
    r = rng()
    with workspace({}, {"g.json": [GRAY]}) as ws:
        run(sol.process_images, ws, P1)
    with workspace({"a.png": noise(r, 4, 4), "b.png": noise(r, 3, 5)}, {}) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="名字里有空格和中文")
def p1_unicode_names(sol):
    r = rng()
    images = {"猫 cat.png": noise(r, 6, 5), "a b.png": noise(r, 4, 7)}
    pipelines = {"流水线 1.json": [GRAY, FLIP_H], "two words.json": [scale(1.5)]}
    with workspace(images, pipelines) as ws:
        run(sol.process_images, ws, P1)


@case("Part 1", title="随机图片与流水线")
def p1_random(sol):
    r = rng()
    for trial in range(3):
        images = {f"img{i}.png": noise(r, r.randint(1, 20), r.randint(1, 20)) for i in range(3)}
        with workspace(images, ref.random_pipelines(r, 6), label=f"随机第 {trial + 1} 组") as ws:
            run(sol.process_images, ws, P1)


# ---------------------------------------------------------------- Parts 2 and 3: shared checks

def check_no_images_crossed(got, shown):
    expect_true(got.images_sent == 0 and got.images_received == 0,
                f"有 PIL 图片对象被 pickle 后跨进程传递（主进程发出 {got.images_sent} 次、收到 "
                f"{got.images_received} 次）：跨进程边界只能传文件路径和 JSON 大小的数据，"
                "图片要在工作进程里从文件解码", call=shown)


def check_decodes_fewer(got, ws, shown):
    """Statement: no image may be decoded as many times as there are pipelines using it."""
    n_pipes = len(ws.pipelines)
    counts = got.decodes()
    for name, k in sorted(counts.items(), key=lambda kv: -kv[1]):
        if name in ws.images and k >= n_pipes:
            expect_true(False, f"{name} 被解码了 {k} 次，而用到它的流水线只有 {n_pipes} 条：同一张源图片"
                               "不能按流水线数量重复解码", call=shown)
    total = sum(counts.values())
    expect_true(total < len(ws.images) * n_pipes,
                f"一共解码了 {total} 次，等于 图片数 × 流水线数 = {len(ws.images) * n_pipes}：同一张源图片"
                "不能按流水线数量重复解码", call=shown)


def check_workers_used(got, shown, workers):
    expect_true(got.worker_pids(),
                f"max_workers={workers}，但所有解码和变换都发生在主进程里：Part 2 要把工作分摊到多个工作进程"
                "（max_workers 限定的是同时运行的工作进程数）", call=shown)


def check_parallel_limit(got, shown, workers):
    k = got.max_parallel_workers()
    expect_true(k <= workers,
                f"max_workers={workers}，但同一时刻有 {k} 个工作进程在处理图片（按每个进程第一次到最后一次"
                "解码或变换的时间段统计）", call=shown)


def check_small_messages(got, ws, shown):
    msgs = got.sent + got.received
    biggest = max(msgs, default=0)
    expect_true(biggest <= ref.BIG_PAYLOAD,
                f"主进程和工作进程之间传了一条 {biggest} 字节的消息（本测试的全部路径和流水线写成 JSON 一共只有"
                f"约 {ws.json_bytes()} 字节，上限按 {ref.BIG_PAYLOAD} 字节算）：跨进程边界传递的看来是解码后的"
                "图片、像素字节或文件内容，而不是路径", call=shown)


def check_shared_count(got, ws, shown, want=None):
    per_image = ref.distinct_prefixes(ws.pipelines)
    bound = per_image * len(ws.images) if want is None else want
    naive = sum(len(s) for s in ws.pipelines.values()) * len(ws.images)
    ops = got.ops()
    expect_true(ops <= bound,
                f"共享前缀后应当总共只执行 {bound} 次步骤（{len(ws.images)} 张图，每张 {per_image} 个不同的前缀），"
                f"实际执行了 {ops} 次（每条流水线各自从头执行是 {naive} 次）", call=shown)


def mixed_images(r, n, lo=2, hi=16):
    exts = [".png", ".png", ".jpg", ".bmp"]
    return {f"img{i}{exts[i % 4]}": noise(r, r.randint(lo, hi), r.randint(lo, hi)) for i in range(n)}


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, cpu=4, title="题目例子 200 张图 × 5 条流水线")
def p2_example(sol):
    r = rng()
    images = {f"big{i:03d}.png": noise(r, 24, 16) for i in range(200)}
    pipelines = {"p1.json": [GRAY], "p2.json": [FLIP_H, scale(0.5)], "p3.json": [rotate(90)],
                 "p4.json": [blur(1)], "p5.json": [FLIP_V, GRAY, rotate(30)]}
    call = p23("process_images_parallel", 2)
    with workspace(images, pipelines, label="应写出 200 * 5 = 1000 个输出，总共只解码 200 次") as ws:
        got = run(sol.process_images_parallel, ws, call, max_workers=2)
        shown = ws.describe(call)
        counts = got.decodes()
        total = sum(counts.values())
        worst = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
        expect_true(total <= 200, f"题目要求总共只解码 200 张图片——每张一次，实际解码了 {total} 次"
                                  f"（例如 {worst[0]} 解码了 {worst[1]} 次）", call=shown)
        check_no_images_crossed(got, shown)


@case("Part 2", title="max_workers=1 也要输出正确")
def p2_one_worker(sol):
    r = rng()
    with workspace(mixed_images(r, 5), ref.random_pipelines(r, 5)) as ws:
        call = p23("process_images_parallel", 1)
        got = run(sol.process_images_parallel, ws, call, max_workers=1)
        check_no_images_crossed(got, ws.describe(call))


@case("Part 2", title="同一张图不能按流水线数重复解码")
def p2_decode_once(sol):
    r = rng()
    images = {f"src{i}.png": noise(r, 12, 10) for i in range(3)}
    pipelines = {f"q{i}.json": [scale(0.5 + 0.25 * i)] for i in range(6)}
    with workspace(images, pipelines) as ws:
        call = p23("process_images_parallel", 3)
        got = run(sol.process_images_parallel, ws, call, max_workers=3)
        check_decodes_fewer(got, ws, ws.describe(call))


@case("Part 2", title="工作交给工作进程，而不是主进程")
def p2_uses_processes(sol):
    r = rng()
    images = {f"pic{i}.png": noise(r, 10, 10) for i in range(6)}
    with workspace(images, {"g.json": [GRAY], "r.json": [rotate(45)]}) as ws:
        call = p23("process_images_parallel", 2)
        got = run(sol.process_images_parallel, ws, call, max_workers=2)
        check_workers_used(got, ws.describe(call), 2)


@case("Part 2", title="max_workers=2：最多 2 个进程同时工作")
def p2_limit_two(sol):
    r = rng()
    images = {f"job{i}.png": noise(r, 8, 8) for i in range(8)}
    with workspace(images, {"g.json": [GRAY], "h.json": [FLIP_H]},
                   label="每个工作进程解码后都会被评测故意拖慢 0.15 秒，好让同时工作的进程数看得出来") as ws:
        call = p23("process_images_parallel", 2)
        got = run(sol.process_images_parallel, ws, call, max_workers=2, slow_decode=0.15)
        shown = ws.describe(call)
        check_workers_used(got, shown, 2)
        check_parallel_limit(got, shown, 2)


@case("Part 2", title="max_workers=3：最多 3 个进程同时工作")
def p2_limit_three(sol):
    r = rng()
    images = {f"job{i}.png": noise(r, 8, 8) for i in range(9)}
    with workspace(images, {"v.json": [FLIP_V]},
                   label="每个工作进程解码后都会被评测故意拖慢 0.15 秒，好让同时工作的进程数看得出来") as ws:
        call = p23("process_images_parallel", 3)
        got = run(sol.process_images_parallel, ws, call, max_workers=3, slow_decode=0.15)
        check_parallel_limit(got, ws.describe(call), 3)


@case("Part 2", title="跨进程只传路径和 JSON 数据")
def p2_payload(sol):
    r = rng()
    images = {f"photo{i}.png": noise(r, 160, 120) for i in range(4)}
    pipelines = {"keep.json": [], "mirror.json": [FLIP_H], "gray.json": [GRAY]}
    with workspace(images, pipelines, label="每张图 160×120，解码后的 RGB 像素 57600 字节") as ws:
        call = p23("process_images_parallel", 2)
        got = run(sol.process_images_parallel, ws, call, max_workers=2)
        shown = ws.describe(call)
        check_no_images_crossed(got, shown)
        check_small_messages(got, ws, shown)


@case("Part 2", title="没有图片或没有流水线")
def p2_empty_dirs(sol):
    r = rng()
    call = p23("process_images_parallel", 2)
    with workspace({}, {"g.json": [GRAY]}) as ws:
        run(sol.process_images_parallel, ws, call, max_workers=2)
    with workspace({"a.png": noise(r, 4, 4), "b.jpg": noise(r, 5, 3)}, {}) as ws:
        run(sol.process_images_parallel, ws, call, max_workers=2)


@case("Part 2", title="随机：输出与 Part 1 相同")
def p2_random(sol):
    r = rng()
    for trial, workers in enumerate((2, 3)):
        with workspace(mixed_images(r, 6), ref.random_pipelines(r, 6), label=f"随机第 {trial + 1} 组") as ws:
            call = p23("process_images_parallel", workers)
            got = run(sol.process_images_parallel, ws, call, max_workers=workers)
            check_no_images_crossed(got, ws.describe(call))


# ---------------------------------------------------------------- Part 3

PART3_EXAMPLE = {
    "p1.json": [GRAY, scale(0.5), rotate(90)],
    "p2.json": [GRAY, scale(0.5), blur(2)],
    "p3.json": [GRAY, scale(0.5), rotate(90)],
    "p4.json": [FLIP_H, blur(2)],
}


@case("Part 3", sample=True, title="题目例子：11 次步骤共享后只要 6 次")
def p3_example(sol):
    images = {"leaf.png": noise(rng(), 30, 20)}
    with workspace(images, PART3_EXAMPLE) as ws:
        call = p23("process_images_shared_prefix", 2)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=2)
        shown = ws.describe(call)
        ops = got.ops()
        expect_true(ops <= 6, f"题目例子共享前缀后总计只执行 6 次步骤，实际执行了 {ops} 次（不共享是 11 次）",
                    call=shown)
        check_no_images_crossed(got, shown)


@case("Part 3", title="一条流水线是另一条的前缀，兄弟分支互不影响")
def p3_prefix_and_siblings(sol):
    r = rng()
    pipelines = {
        "short.json": [GRAY],
        "long.json": [GRAY, scale(0.5)],
        "longer.json": [GRAY, scale(0.5), rotate(45)],
        "sibling_a.json": [GRAY, FLIP_H],
        "sibling_b.json": [GRAY, FLIP_V],
        "sibling_c.json": [GRAY, blur(1)],
        "empty.json": [],
        "empty_again.json": [],
        "same_as_long.json": [GRAY, scale(0.5)],
    }
    images = {"one.png": noise(r, 14, 9), "two.png": noise(r, 6, 11)}
    with workspace(images, pipelines) as ws:
        call = p23("process_images_shared_prefix", 2)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=2)
        check_shared_count(got, ws, ws.describe(call))


@case("Part 3", title="步骤的 JSON 键顺序不影响共享")
def p3_key_order(sol):
    r = rng()
    pipelines = {
        "a.json": [{"type": "scale", "factor": 0.5}, {"type": "blur", "radius": 2}, FLIP_H],
        "b.json": [{"factor": 0.5, "type": "scale"}, {"radius": 2, "type": "blur"}, FLIP_V],
        "c.json": [{"factor": 0.5, "type": "scale"}, {"type": "blur", "radius": 2}, {"angle": 45, "type": "rotate"}],
        "d.json": [{"type": "scale", "factor": 0.5}, {"radius": 2, "type": "blur"}],
    }
    images = {"keys.png": noise(r, 16, 12)}
    with workspace(images, pipelines, label="四条流水线都以 scale(0.5)、blur(2) 开头，只是键的书写顺序不同") as ws:
        call = p23("process_images_shared_prefix", 2)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=2)
        check_shared_count(got, ws, ws.describe(call))


@case("Part 3", title="不同图片之间不共享")
def p3_no_cross_image(sol):
    r = rng()
    images = {"first.png": noise(r, 10, 6), "second.png": noise(r, 10, 6), "third.png": noise(r, 7, 12)}
    pipelines = {"x.json": [GRAY, scale(0.5)], "y.json": [GRAY, scale(0.5), FLIP_H], "z.json": [FLIP_V]}
    with workspace(images, pipelines, label="三张图用同样的流水线，但每张图都要从自己的像素算起") as ws:
        call = p23("process_images_shared_prefix", 1)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=1)
        check_shared_count(got, ws, ws.describe(call))


@case("Part 3", title="跨进程规则与 Part 2 相同")
def p3_boundary(sol):
    r = rng()
    images = {f"photo{i}.png": noise(r, 160, 120) for i in range(4)}
    pipelines = {"keep.json": [], "mirror.json": [FLIP_H], "mirror_gray.json": [FLIP_H, GRAY]}
    with workspace(images, pipelines, label="每张图 160×120，解码后的 RGB 像素 57600 字节") as ws:
        call = p23("process_images_shared_prefix", 2)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=2)
        shown = ws.describe(call)
        check_workers_used(got, shown, 2)
        check_no_images_crossed(got, shown)
        check_small_messages(got, ws, shown)
        check_shared_count(got, ws, shown)


@case("Part 3", title="max_workers=2：最多 2 个进程同时工作")
def p3_limit_two(sol):
    r = rng()
    images = {f"job{i}.png": noise(r, 8, 8) for i in range(8)}
    with workspace(images, {"g.json": [GRAY], "gh.json": [GRAY, FLIP_H]},
                   label="每个工作进程解码后都会被评测故意拖慢 0.15 秒，好让同时工作的进程数看得出来") as ws:
        call = p23("process_images_shared_prefix", 2)
        got = run(sol.process_images_shared_prefix, ws, call, max_workers=2, slow_decode=0.15)
        check_parallel_limit(got, ws.describe(call), 2)


@case("Part 3", title="随机：输出正确且共享前缀")
def p3_random(sol):
    r = rng()
    for trial, workers in enumerate((1, 2, 3)):
        with workspace(mixed_images(r, 3), ref.random_pipelines(r, 7), label=f"随机第 {trial + 1} 组") as ws:
            call = p23("process_images_shared_prefix", workers)
            got = run(sol.process_images_shared_prefix, ws, call, max_workers=workers)
            shown = ws.describe(call)
            check_no_images_crossed(got, shown)
            check_shared_count(got, ws, shown)
