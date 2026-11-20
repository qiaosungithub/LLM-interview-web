"""Not a test. The harness runs the answer as an in-memory module named "sol". A worker process the answer starts
with the "spawn" or "forkserver" method is a new interpreter: to unpickle the worker function it imports "sol" by
name, and as it has the judge's sys.path (which starts with /judge) it finds this file. Here the judge's counters
(ref) are installed in the worker and /work/answer.py is run as this module, as `import sol` would do in an
ordinary project. In the judge's own process, and in a forked worker, "sol" is already the harness's module, so
this file is never run there."""
import ref  # noqa: F401 - patches Pillow in this worker, so its decodes and steps are counted


def _run_answer(path="/work/answer.py"):
    with open(path, encoding="utf-8") as f:
        code = compile(f.read(), path, "exec")
    globals()["__file__"] = path
    exec(code, globals())


_run_answer()
