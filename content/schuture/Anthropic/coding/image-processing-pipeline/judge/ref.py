"""Helpers for tests.py.

- apply_step / run_pipeline: the statement's six steps, written straight from its Pillow calls.
- Workspace: a temporary image_dir / transformation_dir / output_dir with small synthetic images, the pipeline
  files, the expected pixels of every (image, pipeline) output, and a get_output_path closure (a closure on
  purpose: it cannot be pickled, so it can only be called in the process that received it).
- Counters that work across processes. The harness sets the "fork" start method, so patches made here at import
  time are inherited by every worker the answer starts; a worker the answer starts with "spawn" or "forkserver"
  imports `sol` by name, finds sol.py next to this file, and that imports this module again (so the judge's pid
  and the slow-decode setting reach workers through the environment and a file, not through module globals):
  * every real decode of an image file (ImageFile.load while tiles are still pending) and every outermost call of
    convert / transpose / resize / filter / rotate is appended as one line to EVENTS, with the process id;
  * in the main process only, everything multiprocessing pickles to send to a worker or unpickles from one is
    measured, and pickling or unpickling a PIL Image there is counted.
  Recording never changes what a call returns.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field

import multiprocessing.reduction as mp_reduction
from PIL import Image, ImageFile, ImageFilter

# The judge's own process. It imports this module first (from tests.py) and puts its pid in the environment, which
# every worker inherits, also one started with "spawn" or "forkserver" that imports this module again.
MAIN_PID = int(os.environ.setdefault("JUDGE_PIPELINE_MAIN_PID", str(os.getpid())))
EVENTS = os.path.join(tempfile.gettempdir(), "judge_pipeline_events.log")
SLOW = os.path.join(tempfile.gettempdir(), "judge_pipeline_slow_decode")    # see run_answer
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP"}
BIG_PAYLOAD = 16 * 1024     # bytes; a task or result made of paths and pipeline JSON stays far below this


# ---------------------------------------------------------------- the reference

def apply_step(image, step):
    kind = step["type"]
    if kind == "grayscale":
        return image.convert("L")
    if kind == "flip_horizontal":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if kind == "flip_vertical":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if kind == "scale":
        f = step["factor"]
        return image.resize((max(1, round(image.width * f)), max(1, round(image.height * f))),
                            resample=Image.Resampling.BILINEAR)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=step["radius"]))
    if kind == "rotate":
        return image.rotate(step["angle"], resample=Image.Resampling.BILINEAR, expand=True)
    raise ValueError(kind)


def run_pipeline(image, steps):
    for step in steps:
        image = apply_step(image, step)
    return image


def step_key(step):
    return tuple(sorted(step.items()))


def distinct_prefixes(pipelines):
    """Number of distinct non-empty prefixes among one image's pipelines (key order inside a step ignored)."""
    seen = set()
    for steps in pipelines.values():
        prefix = ()
        for step in steps:
            prefix += (step_key(step),)
            seen.add(prefix)
    return len(seen)


def noise_image(r, w, h):
    return Image.frombytes("RGB", (w, h), r.randbytes(w * h * 3))


def random_step(r, plain=False):
    """plain=True: no step that leaves the image unchanged (blur 0, rotate 0 / 360, scale 1.0)."""
    kind = r.choice(["grayscale", "flip_horizontal", "flip_vertical", "scale", "blur", "rotate"])
    if kind == "scale":
        return {"type": "scale", "factor": r.choice([0.5, 0.75, 1.5, 2.0, 0.3] + ([] if plain else [1.0]))}
    if kind == "blur":
        return {"type": "blur", "radius": r.choice([1, 2, 3] + ([] if plain else [0]))}
    if kind == "rotate":
        return {"type": "rotate", "angle": r.choice([45, 90, 180, 270, 33.5, -30] + ([] if plain else [0, 360]))}
    return {"type": kind}


def random_pipelines(r, n, plain=False, max_extra=3):
    """n pipelines, most of them starting with one of a few shared segments, some empty or repeated."""
    segments = [[random_step(r, plain) for _ in range(r.randint(1, 2))] for _ in range(3)]
    out = {}
    for i in range(n):
        steps = [dict(s) for s in r.choice(segments)] if r.random() < 0.7 else []
        steps += [random_step(r, plain) for _ in range(r.randint(0, max_extra))]
        if r.random() < 0.3:            # write some steps with their keys in another order
            steps = [dict(reversed(list(s.items()))) for s in steps]
        out[f"pl{i}.json"] = steps
    return out


# ---------------------------------------------------------------- recording (patched once, at import)

def _record(kind, detail=""):
    line = f"{kind}\t{os.getpid()}\t{time.monotonic():.6f}\t{detail}\n".encode()
    try:
        fd = os.open(EVENTS, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError:
        pass


_depth = threading.local()


def _counted(name, orig):
    def wrapper(self, *args, **kwargs):
        depth = getattr(_depth, "n", 0)
        if depth == 0:
            mode = args[0] if args else kwargs.get("mode")
            if not (name == "convert" and (mode is None or mode == self.mode)):     # a no-op convert is no step
                _record("op", name)
        _depth.n = depth + 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _depth.n = depth
    wrapper.__name__ = name
    wrapper.__wrapped__ = orig
    return wrapper


_orig_load = ImageFile.ImageFile.load


def _slow_decode():
    """Seconds a worker process sleeps after each decode while a test sets them (see run_answer)."""
    try:
        with open(SLOW) as f:
            return float(f.read() or 0)
    except (OSError, ValueError):
        return 0.0


def _load(self):
    if getattr(self, "tile", None):
        _record("decode", os.path.basename(getattr(self, "filename", "") or "") or "?")
        if os.getpid() != MAIN_PID:
            # Read in the worker at decode time, from a file, so that workers started any way see the test's value.
            # The sleep makes every worker's busy period long, so overlapping workers really overlap in the log.
            slow = _slow_decode()
            if slow:
                time.sleep(slow)
    return _orig_load(self)


class _Watch:
    def __init__(self):
        self.active = False
        self.reset()

    def reset(self):
        self.sent, self.received, self.images_sent, self.images_received = [], [], 0, 0


WATCH = _Watch()
_orig_dumps = mp_reduction.ForkingPickler.dumps.__func__
_orig_loads = mp_reduction.ForkingPickler.loads
_orig_getstate = Image.Image.__getstate__
_orig_setstate = Image.Image.__setstate__


def _watching():
    return WATCH.active and os.getpid() == MAIN_PID


def _dumps(cls, obj, protocol=None):
    buf = _orig_dumps(cls, obj, protocol)
    if _watching():
        WATCH.sent.append(memoryview(buf).nbytes)
    return buf


def _loads(data, *args, **kwargs):
    if _watching():
        WATCH.received.append(memoryview(data).nbytes)
    return _orig_loads(data, *args, **kwargs)


def _getstate(self):
    if _watching():
        WATCH.images_sent += 1
    return _orig_getstate(self)


def _setstate(self, state):
    if _watching():
        WATCH.images_received += 1
    return _orig_setstate(self, state)


def _install():
    if getattr(Image.Image, "_judge_patched", False):
        return
    for name in ("convert", "transpose", "resize", "filter", "rotate"):
        setattr(Image.Image, name, _counted(name, getattr(Image.Image, name)))
    ImageFile.ImageFile.load = _load
    mp_reduction.ForkingPickler.dumps = classmethod(_dumps)
    mp_reduction.ForkingPickler.loads = staticmethod(_loads)
    Image.Image.__getstate__ = _getstate
    Image.Image.__setstate__ = _setstate
    Image.Image._judge_patched = True


_install()


# ---------------------------------------------------------------- one run of the answer

@dataclass
class Run:
    events: list = field(default_factory=list)     # (kind, pid, t, detail)
    sent: list = field(default_factory=list)
    received: list = field(default_factory=list)
    images_sent: int = 0
    images_received: int = 0

    def decodes(self):
        out = {}
        for kind, _pid, _t, name in self.events:
            if kind == "decode":
                out[name] = out.get(name, 0) + 1
        return out

    def ops(self):
        return sum(1 for e in self.events if e[0] == "op")

    def worker_pids(self):
        return {pid for _k, pid, _t, _d in self.events if pid != MAIN_PID}

    def main_did_work(self):
        return any(pid == MAIN_PID for _k, pid, _t, _d in self.events)

    def max_parallel_workers(self):
        """Most worker processes that were busy at the same moment (from first to last event of each)."""
        span = {}
        for _k, pid, t, _d in self.events:
            if pid != MAIN_PID:
                lo, hi = span.get(pid, (t, t))
                span[pid] = (min(lo, t), max(hi, t))
        points = sorted([(lo, 1) for lo, _ in span.values()] + [(hi, -1) for _, hi in span.values()],
                        key=lambda p: (p[0], -p[1]))
        best = cur = 0
        for _t, d in points:
            cur += d
            best = max(best, cur)
        return best


def _offset():
    try:
        return os.path.getsize(EVENTS)
    except OSError:
        return 0


def _events_from(offset):
    try:
        with open(EVENTS, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return []
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            out.append((parts[0], int(parts[1]), float(parts[2]), parts[3]))
    return out


def _set_slow_decode(seconds):
    if seconds:
        with open(SLOW, "w") as f:
            f.write(repr(float(seconds)))
    else:
        try:
            os.remove(SLOW)
        except OSError:
            pass


def run_answer(fn, ws, slow_decode=0.0, **kwargs):
    """Call fn(image_dir, transformation_dir, output_dir, get_output_path, **kwargs) and record what happened.
    slow_decode: seconds each worker process sleeps right after each decode."""
    start = _offset()
    _set_slow_decode(slow_decode)
    WATCH.reset()
    WATCH.active = True
    try:
        fn(ws.image_dir, ws.transformation_dir, ws.output_dir, ws.get_output_path, **kwargs)
    finally:
        WATCH.active = False
        _set_slow_decode(0)
    return Run(_events_from(start), list(WATCH.sent), list(WATCH.received), WATCH.images_sent,
               WATCH.images_received)


# ---------------------------------------------------------------- the folders

def pixels(im):
    return im.mode, im.size, im.tobytes()


class Workspace:
    """image_dir with the given images, transformation_dir with one JSON file per pipeline, an empty output_dir.

    images: {image_name: PIL RGB image}; pipelines: {json_name: steps}. The expected output of every pair is
    computed here from the decoded files, before the answer runs."""

    def __init__(self, images, pipelines, label=None):
        self.root = tempfile.mkdtemp(prefix="pipe_")
        self.image_dir = os.path.join(self.root, "images")
        self.transformation_dir = os.path.join(self.root, "transformations")
        self.output_dir = os.path.join(self.root, "out")
        for d in (self.image_dir, self.transformation_dir, self.output_dir):
            os.mkdir(d)
        self.images = {}
        for name, im in images.items():
            path = os.path.join(self.image_dir, name)
            fmt = FORMATS[os.path.splitext(name)[1].lower()]
            im.save(path, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
            with Image.open(path) as src:
                src.load()
                self.images[name] = src.copy()
        self.pipelines = {k: [dict(s) for s in v] for k, v in pipelines.items()}
        for name, steps in self.pipelines.items():
            with open(os.path.join(self.transformation_dir, name), "w") as f:
                json.dump({"transformations": steps}, f)
        self.expected = {}
        for img, src in self.images.items():
            for js, steps in self.pipelines.items():
                self.expected[img, js] = pixels(run_pipeline(src, steps))
        self.label = label
        out = self.output_dir

        def get_output_path(image_name, json_name):
            return os.path.join(out, f"{image_name}__{json_name}.png")

        self.get_output_path = get_output_path

    def out_name(self, img, js):
        return f"{img}__{js}.png"

    def json_bytes(self):
        """Size of every path and pipeline of this workspace written out as JSON: all a worker ever needs."""
        paths = [os.path.join(self.image_dir, i) for i in self.images]
        paths += [self.get_output_path(i, j) for i, j in self.expected]
        return len(json.dumps([paths, self.pipelines]).encode())

    def describe(self, call, limit_images=6, limit_pipes=8):
        lines = [call]
        names = list(self.images)
        if self.label:
            lines.append(f"# {self.label}")
        if not names:
            lines.append("# image_dir：空文件夹")
        elif len(names) <= limit_images:
            lines.append("# image_dir：" + "，".join(
                f"{n}（{self.images[n].width}×{self.images[n].height}）" for n in names))
        else:
            sizes = sorted({im.size for im in self.images.values()})
            lines.append(f"# image_dir：{len(names)} 张图片 {names[0]} … {names[-1]}，尺寸 "
                         + "、".join(f"{w}×{h}" for w, h in sizes[:4]) + ("…" if len(sizes) > 4 else ""))
        if not self.pipelines:
            lines.append("# transformation_dir：空文件夹")
        for k, (js, steps) in enumerate(self.pipelines.items()):
            if k == limit_pipes:
                lines.append(f"#   …（共 {len(self.pipelines)} 个流水线文件）")
                break
            lines.append(f"# transformation_dir/{js}：" + json.dumps({"transformations": steps}, ensure_ascii=False))
        lines.append("# get_output_path(图片名, json名) 返回 output_dir/<图片名>__<json名>.png")
        return "\n".join(lines)

    def check_outputs(self):
        """None when output_dir holds exactly the expected PNG files with the expected pixels, else a Chinese
        sentence about the first problem found."""
        want = {self.out_name(i, j): (i, j) for i, j in self.expected}
        try:
            got = set(os.listdir(self.output_dir))
        except OSError as e:
            return f"读不了 output_dir：{e}"
        missing = sorted(set(want) - got)
        extra = sorted(got - set(want))
        if missing:
            i, j = want[missing[0]]
            return (f"缺少输出：get_output_path({i!r}, {j!r}) 没有写出文件（应当写出 {len(want)} 个，缺了 "
                    f"{len(missing)} 个）")
        if extra:
            return f"output_dir 里多出了不该有的文件 {extra[:3]}：应当恰好写出 {len(want)} 个文件，每个组合一个"
        for name, (img, js) in sorted(want.items()):
            path = os.path.join(self.output_dir, name)
            try:
                with Image.open(path) as im:
                    fmt = im.format
                    im.load()
                    got_px = pixels(im)
            except Exception as e:      # noqa: BLE001 - a broken file is the answer's fault
                return f"{img} 经过 {js} 的输出文件打不开：{type(e).__name__}: {e}"
            if fmt != "PNG":
                return f"{img} 经过 {js} 的输出不是 PNG 格式（是 {fmt}）"
            msg = diff(self.expected[img, js], got_px)
            if msg:
                return f"{img} 经过 {js}（{json.dumps(self.pipelines[js], ensure_ascii=False)}）的输出不对：{msg}"
        return None

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def diff(want, got):
    (wm, ws, wb), (gm, gs, gb) = want, got
    if wm != gm:
        return f"模式应为 {wm!r}，得到 {gm!r}"
    if ws != gs:
        return f"尺寸应为 {ws}，得到 {gs}（(宽, 高)）"
    if wb == gb:
        return None
    import numpy as np      # here, not at the top: a spawned worker importing this module need not load numpy
    bands = len(Image.new(wm, (1, 1)).getbands())
    a = np.frombuffer(wb, np.uint8).reshape(ws[1], ws[0], bands)
    b = np.frombuffer(gb, np.uint8).reshape(ws[1], ws[0], bands)
    bad = np.argwhere((a != b).any(axis=2))
    y, x = (int(v) for v in bad[0])
    show = (lambda v: int(v[0])) if bands == 1 else (lambda v: tuple(int(t) for t in v))
    return (f"模式和尺寸都对，但有 {len(bad)} / {ws[0] * ws[1]} 个像素不同；第一个在 (x={x}, y={y})："
            f"期望 {show(a[y, x])}，得到 {show(b[y, x])}")
