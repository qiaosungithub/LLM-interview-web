"""What a problem's judge/tests.py imports: the @case decorator and the expect* checks.

    from judgelib import case, expect, expect_close, expect_raises, expect_valid, info, rng

    @case("Part 1", sample=True)
    def statement_example(sol):
        "The statement's first example."
        expect(sol.tokenize("abcxbcdy", VOCAB), [3, 0, 5, 0], call='tokenize("abcxbcdy", VOCAB)')

A case receives the answer as a module (`sol`) and fails by raising: every expect* raises Mismatch, and a plain
`assert` works too. Cases run in the order they are defined; each part's sample cases should come first.
Rules for writing them are in tools/JUDGE_BRIEF.md.
"""
from __future__ import annotations

import math
import random
import reprlib
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

SEED = 0                        # set by the harness; vary it to re-draw every rng()
CASES: list["Case"] = []
_current: dict = {"case": None, "info": [], "call": None}


@dataclass
class Case:
    part: str
    name: str
    fn: Callable
    sample: bool = False
    cpu: float | None = None    # CPU seconds for this case; None means judge.yaml's case_cpu
    title: str = ""
    extra: dict = field(default_factory=dict)


def case(part: str, sample: bool = False, cpu: float | None = None, name: str | None = None, title: str | None = None):
    """Register a test. `part` groups the results ("Part 1", "Level 2"); `sample` marks the statement's own
    examples (what 运行样例 runs); `cpu` overrides the per-case CPU limit; `title` is a short Chinese label."""
    def deco(fn):
        doc = (fn.__doc__ or "").strip().splitlines()
        CASES.append(Case(part=part, name=name or fn.__name__, fn=fn, sample=sample, cpu=cpu,
                          title=title or (doc[0] if doc else "")))
        return fn
    return deco


class Mismatch(AssertionError):
    """A wrong answer; carries what was called, what was expected and what came back."""

    def __init__(self, msg: str = "", call: str | None = None, want: Any = None, got: Any = None, show: bool = True,
                 want_none: bool = False):
        """want=None means "nothing to show", unless want_none says None really is the expected value."""
        super().__init__(msg)
        self.msg, self.call = msg, (call if call is not None else _current.get("call"))
        self.want = short(want) if show and (want is not None or want_none) else None
        self.got = short(got) if show else None


# ---------------------------------------------------------------- display

_repr = reprlib.Repr()
_repr.maxstring, _repr.maxother, _repr.maxlist, _repr.maxtuple = 400, 400, 40, 40
_repr.maxdict, _repr.maxset, _repr.maxlevel = 30, 30, 6


def short(x: Any, limit: int = 3000) -> str:
    """repr() cut to a readable size; numpy arrays and torch tensors are shown with their shape."""
    try:
        if _is_tensor(x):
            x = x.detach().cpu().numpy()
        if type(x).__module__ == "numpy" and hasattr(x, "shape"):
            import numpy as np
            body = np.array2string(x, threshold=200, edgeitems=4, precision=6)
            text = f"array(shape={tuple(x.shape)}, dtype={x.dtype})\n{body}" if x.ndim else repr(x)
        else:
            text = _repr.repr(x) if not isinstance(x, (str, int, float, bool, type(None))) else repr(x)
    except Exception as e:          # a user object whose __repr__ fails must not crash the judge
        text = f"<{type(x).__name__} 对象，repr 出错：{type(e).__name__}>"
    return text if len(text) <= limit else text[:limit] + f" …（共 {len(text)} 个字符，已截断）"


def _arg(x: Any, limit: int = 600) -> str:
    """An argument as the user can paste it back: the full repr when it fits in `limit` characters."""
    try:
        if not _is_array(x):
            text = repr(x)
            if len(text) <= limit:
                return text
    except Exception:
        pass
    return short(x, limit)


def show_call(fn: str, *args, **kwargs) -> str:
    """How a call looks in the result, e.g. show_call("tokenize", text, vocab, 0) -> tokenize('ab', {'a': 1}, 0).
    Arguments are shown in full when they fit in 600 characters, so the call can be pasted into 自定义运行."""
    parts = [_arg(a) for a in args] + [f"{k}={_arg(v)}" for k, v in kwargs.items()]
    return f"{fn}({', '.join(parts)})"


def _is_tensor(x) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "detach")


def _as_array(x):
    import numpy as np
    if _is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _is_array(x) -> bool:
    return _is_tensor(x) or (type(x).__module__ == "numpy" and hasattr(x, "shape"))


# ---------------------------------------------------------------- checks

def expect(got: Any, want: Any, call: str | None = None, msg: str = "") -> None:
    """got == want. Arrays and tensors compare by shape and every element."""
    if _is_array(got) or _is_array(want):
        import numpy as np
        a, b = _as_array(got), _as_array(want)
        if a.shape != b.shape:
            raise Mismatch(msg or f"形状不对：期望 {b.shape}，得到 {a.shape}", call, want, got)
        if not np.array_equal(a, b):
            raise Mismatch(msg or "数值不一致", call, want, got)
        return
    try:
        same = got == want
        if not isinstance(same, bool):
            same = bool(same)
    except Exception as e:
        raise Mismatch(msg or f"无法和期望值比较（{type(e).__name__}: {e}）", call, want, got, want_none=True) from None
    if not same:
        raise Mismatch(msg or _type_hint(got, want), call, want, got, want_none=True)


def _type_hint(got, want) -> str:
    if want is None and got is not None:
        return f"应当返回 None，却返回了 {type(got).__name__}"
    if got is None and want is not None:
        return f"返回了 None，期望 {type(want).__name__}"
    if type(got) is not type(want) and not (isinstance(got, (int, float)) and isinstance(want, (int, float))):
        return f"类型不对：期望 {type(want).__name__}，得到 {type(got).__name__}"
    if isinstance(want, (list, tuple, str)) and isinstance(got, (list, tuple, str)):
        for k, (a, b) in enumerate(zip(got, want)):
            try:
                if a != b:
                    return f"从第 {k} 个元素（下标从 0 开始）开始不同：期望 {short(b, 200)}，得到 {short(a, 200)}" + \
                        (f"；长度期望 {len(want)}，得到 {len(got)}" if len(got) != len(want) else "")
            except Exception:
                break
        if len(got) != len(want):
            return f"长度不对：期望 {len(want)}，得到 {len(got)}（前 {min(len(got), len(want))} 个元素相同）"
    return ""


def expect_close(got: Any, want: Any, rtol: float = 1e-6, atol: float = 1e-8, call: str | None = None, msg: str = "") -> None:
    """Numbers, sequences, arrays or tensors equal within |got - want| <= atol + rtol * |want|."""
    import numpy as np
    try:
        a = _as_array(got).astype(float)
    except Exception:
        raise Mismatch(msg or f"返回值不是数值：{type(got).__name__}", call, want, got) from None
    b = _as_array(want).astype(float)
    if a.shape != b.shape:
        raise Mismatch(msg or f"形状不对：期望 {b.shape}，得到 {a.shape}", call, want, got)
    bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not bad.any():
        return

    def at(i) -> str:
        return "" if a.ndim == 0 else f"，位置 {tuple(int(k) for k in np.unravel_index(i, a.shape))}"
    odd = bad & ~np.isfinite(a)             # nan/inf where a number was expected: report it, not a misleading error size
    if odd.any():
        i = int(np.flatnonzero(odd)[0])
        raise Mismatch(msg or f"结果里有 {a.flat[i]}{at(i)}（共 {int(odd.sum())} 处），期望 {b.flat[i]:.6g}", call, want, got)
    err = np.where(bad, np.abs(a - b), 0.0)
    i = int(np.argmax(err))
    raise Mismatch(msg or f"误差太大：最大误差 {float(err.flat[i]):.3g}{at(i)}（允许 atol={atol:g}, rtol={rtol:g}）", call, want, got)


def expect_raises(exc: type | tuple, fn: Callable, *args, call: str | None = None, msg: str = "", **kwargs) -> BaseException:
    """fn(*args, **kwargs) must raise `exc` (or a subclass); returns the exception for further checks."""
    names = " / ".join(e.__name__ for e in (exc if isinstance(exc, tuple) else (exc,)))
    try:
        value = fn(*args, **kwargs)
    except exc as e:
        return e
    except Exception as e:
        raise Mismatch(msg or f"应当抛出 {names}，实际抛出 {type(e).__name__}: {e}", call, None, None, show=False) from None
    raise Mismatch(msg or f"应当抛出 {names}，但正常返回了", call, f"抛出 {names}", value)


def expect_valid(got: Any, check: Callable[[Any], Any], call: str | None = None, want: Any = None) -> None:
    """For answers that are not unique: check(got) returns None/True when acceptable, or a string saying what is
    wrong (False also fails). `want` is only shown, e.g. one correct answer or the optimal cost."""
    try:
        verdict = check(got)
    except Mismatch:
        raise
    except Exception as e:
        raise Mismatch(f"返回值不符合要求（检查时出错：{type(e).__name__}: {e}）", call, want, got) from None
    if verdict is None or verdict is True:
        return
    raise Mismatch(verdict if isinstance(verdict, str) else "返回值不符合要求", call, want, got)


def expect_true(cond: Any, msg: str, call: str | None = None, got: Any = None) -> None:
    if not cond:
        raise Mismatch(msg, call, None, got, show=got is not None)


def close(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


# ---------------------------------------------------------------- extras for special judges

def calling(text: str) -> str:
    """Record what the test is about to call (for a stateful API, the replay of every operation so far). If the
    answer raises or runs out of time inside it, the result shows this text; expect*() without call= uses it too."""
    _current["call"] = text
    return text


def info(text: str) -> None:
    """A line shown under the case even when it passes, e.g. "指令数 12（已知最短 9）"."""
    _current["info"].append(str(text))


def rng(salt: str = "") -> random.Random:
    """A random generator fixed by SEED, the case and `salt`: the same draw every run with the same seed."""
    name = _current["case"].name if _current["case"] else ""
    return random.Random(zlib.crc32(f"{SEED}|{name}|{salt}".encode()))
