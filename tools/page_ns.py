"""Load a topic page's ```python blocks into a namespace, for quick experiments against the reference code.

    import sys; sys.path.insert(0, "/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/tools")
    from page_ns import load, blocks
    ns = load("content/schuture/Anthropic/coding/longest-match-tokenizer/README.md", skip_last=True)
    ns["tokenize"](...)

`skip_last=True` leaves out the final block (normally the page's own runnable checks), so the namespace
holds only the reference solution. Paths may be absolute or relative to the project root.
"""
from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")   # shared 240-core host: keep numeric thread pools small

ROOT = Path(__file__).resolve().parent.parent
FENCE_RE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def blocks(page) -> list[str]:
    p = Path(page)
    if not p.is_absolute():
        p = ROOT / p
    return [b for b in FENCE_RE.findall(p.read_text(encoding="utf-8")) if b.strip()]


def load(page, skip_last: bool = False, upto: int | None = None) -> dict:
    bs = blocks(page)
    if skip_last:
        bs = bs[:-1]
    if upto is not None:
        bs = bs[:upto]
    # A real module registered in sys.modules: @dataclass and typing look the defining module up by name.
    mod = types.ModuleType("__snippets__")
    sys.modules["__snippets__"] = mod
    ns = mod.__dict__
    for i, b in enumerate(bs, 1):
        exec(compile(b, f"{page} block #{i}", "exec"), ns)
    return ns
