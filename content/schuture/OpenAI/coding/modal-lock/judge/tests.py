"""Tests for ModalLock and FairModalLock.

Every scripted case drives the lock with a Rig like the statement's: one thread per name, arrivals and releases in
an exact order, each step waiting until its effect is visible (a granted acquire() has returned; a blocked one is
counted by waiting_count()). After every step the holders are compared with the statement's rules (ref.FairModel /
ref.ModalModel), and waiting_count() with the number of calls still blocked. Deadlines only guard against a lock
that never reaches the expected state; a correct lock's verdict never depends on timing (the slowed threads of the
p*_slow_threads cases only make a correct lock slower).
"""
import threading

from judgelib import Mismatch, calling, case, info, rng

import ref
from ref import Script, fmt, wait_until

MODES = ["train", "infer", "export"]


def room():
    n = threading.active_count()
    if n > 150:
        raise Mismatch(f"前面失败的测试留下了 {n} 个卡在 acquire 里的线程，线程数接近上限，这个测试没有运行",
                       show=False)


def holds(s, want):
    """A hard-coded check of the holders, for the statement's examples."""
    got = s.holders()
    if got != frozenset(want):
        s.fail(f"持有者应当是 {fmt(want)}，实际是 {fmt(got)}")


def waiting(s, want):
    got = s.lock.waiting_count()
    if got != want:
        s.fail(f"waiting_count() 应当返回 {want}，实际返回 {got!r}")


def mismatch(s, err, text, fn, why):
    """fn() is a release() the statement says must raise ModeMismatchError."""
    try:
        s.call(text, fn)
    except err:
        return
    except Mismatch:
        raise
    except Exception as e:      # noqa: BLE001
        s.fail(f"{text} 应当抛出 ModeMismatchError（{why}），实际抛出 {type(e).__name__}: {e}")
    s.fail(f"{text} 应当抛出 ModeMismatchError（{why}），但它正常返回了")


SLOW = ref.SLOW_LINE
SLOW_NOTE = f"测试让 {{}} 在 acquire 里每执行一行代码前都先停 {SLOW * 1000:g} 毫秒"
LATE_WAIT = ("{} 被放慢后，从决定等待到真正开始等待之间隔了一段时间：如果决定等待和开始等待不在同一次持锁里完成"
             "（比如先在锁里查看状态，放开锁以后才去 wait()），这段时间里发生的释放就唤醒不到它。")
SPLIT_CLAIM = ("{} 被放慢后，如果「看到锁可以拿」和「把锁记成自己的模式」不在同一次持锁里完成，另一个模式的调用就能在"
               "这两步之间也看到锁可以拿。")


def slowed(s, name, mode):
    s.arrive(name, mode, slow=SLOW, note=SLOW_NOTE.format(name))


def drain(s):
    """Release the holders one by one until everybody is done."""
    while s.holders():
        for n in sorted(s.holders()):
            s.release(n)
    s.end()


# ================================================================ Part 1

@case("Part 1", sample=True, title="题目的六步例子")
def p1_example(sol):
    s = Script(sol, "ModalLock")
    s.arrive("a", "train")
    holds(s, "a")
    s.arrive("b", "train")
    holds(s, "ab")
    s.arrive("c", "infer")
    holds(s, "ab")
    s.release("a")
    holds(s, "b")
    s.release("b")
    holds(s, "c")
    s.arrive("d", "infer")
    holds(s, "cd")
    s.release("c")
    holds(s, "d")
    s.release("d")
    holds(s, "")
    s.end()


@case("Part 1", title="同模式的等待者要一起放行")
def p1_wake_same_mode(sol):
    s = Script(sol, "ModalLock")
    s.arrive("a", "train")
    for n in "bcd":
        s.arrive(n, "infer")
    waiting(s, 3)
    s.release("a")          # all three parked 'infer' callers must get the lock, not just one of them
    holds(s, "bcd")
    s.arrive("e", "train")
    s.arrive("f", "train")
    s.release("b")
    s.release("c")
    s.release("d")          # both parked 'train' callers
    holds(s, "ef")
    drain(s)


@case("Part 1", title="不同模式的等待者不能一起放行")
def p1_two_modes_waiting(sol):
    room()
    for waiters in (["b:infer", "c:export"],
                    ["b:infer", "c:export", "d:infer", "e:export", "f:serve"],
                    ["b:infer", "c:infer", "d:export", "e:export", "f:infer", "g:export"]):
        s = Script(sol, "ModalLock")
        s.arrive("a", "train")
        for w in waiters:
            name, mode = w.split(":")
            s.arrive(name, mode)
        s.release("a")      # exactly one mode's waiters get in; the others stay blocked
        drain(s)


@case("Part 1", title="有人等待时同模式仍可直接加入")
def p1_join_while_waiting(sol):
    # Part 1 has no fairness: a 'train' caller joins the running 'train' holders even while 'infer' waits
    s = Script(sol, "ModalLock")
    s.arrive("w0", "train")
    s.arrive("starved", "infer")
    for i in range(1, 8):
        s.arrive(f"w{i}", "train")
        s.release(f"w{i - 1}")
    s.release("w7")
    holds(s, ["starved"])
    s.release("starved")
    s.end()


@case("Part 1", title="release 模式不符时抛出异常")
def p1_mismatch(sol):
    err = sol.ModeMismatchError
    s = Script(sol, "ModalLock")
    lock = s.lock
    mismatch(s, err, "lock.release('train')", lambda: lock.release("train"), "锁没有任何持有者")
    s.arrive("p", "train")
    s.arrive("q", "infer")
    mismatch(s, err, "lock.release('infer')", lambda: lock.release("infer"), "锁正以 'train' 被持有")
    # the refused call must not have touched the lock: p still holds 'train'
    s.arrive("r", "train")
    s.release("p")
    s.release("r")
    holds(s, "q")
    mismatch(s, err, "lock.release('train')", lambda: lock.release("train"), "锁正以 'infer' 被持有")
    s.arrive("t", "infer")
    s.release("q")
    s.release("t")
    mismatch(s, err, "lock.release('infer')", lambda: lock.release("infer"), "锁已经空闲，没有任何持有者")
    s.arrive("u", "export")
    s.release("u")
    s.end()


@case("Part 1", title="空串、大小写、非驻留字符串")
def p1_special_modes(sol):
    # modes are plain strings compared with ==: '' is a mode like any other, 'Train' is not 'train', and two
    # equal strings that are different objects are the same mode
    s = Script(sol, "ModalLock", fresh_modes=True)
    s.arrive("a", "")
    s.arrive("b", "")
    s.arrive("c", "train")
    s.release("a")
    s.release("b")
    holds(s, "c")
    s.arrive("d", "train")
    s.arrive("e", "Train")
    s.arrive("f", "train ")
    s.arrive("g", "推理")
    s.release("c")
    s.release("d")
    drain(s)


@case("Part 1", title="放慢线程：判断和动作不能分开")
def p1_slow_threads(sol):
    room()
    # a caller slowed on its way into its wait: the release that comes meanwhile must still wake it
    s = Script(sol, "ModalLock")
    s.arrive("a", "train")
    slowed(s, "b", "infer")
    s.release("a", hint=LATE_WAIT.format("b"))
    holds(s, "b")
    s.release("b")
    s.end()
    # callers of two modes woken together, both slowed: only one mode may take the lock
    for waiters in (["b:infer", "c:export"], ["b:infer", "c:export", "d:infer"]):
        s = Script(sol, "ModalLock")
        s.arrive("a", "train")
        for w in waiters:
            slowed(s, *w.split(":"))
        s.release("a", hint=SPLIT_CLAIM.format(fmt(w.split(":")[0] for w in waiters)))
        drain(s)


@case("Part 1", title="随机脚本对照题目规则")
def p1_random(sol):
    room()
    r = rng()
    steps = 0
    for _ in range(60):
        s = Script(sol, "ModalLock", fresh_modes=True)
        ref.random_trial(s, r, MODES, n_names=16, n_events=40)
        s.end()
        steps += s.count
    info(f"60 段随机脚本，共 {steps} 步，每一步之后都核对持有者和 waiting_count()")


def stress_case(sol, cls_name, salt):
    r = rng(salt)
    for rnd in range(3):
        plans = [[(r.choice(MODES), 0.0 if r.random() < 0.5 else r.uniform(0, 0.0006)) for _ in range(25)]
                 for _ in range(16)]
        calling(f"lock = {cls_name}()\n# 16 个线程同时运行，每个线程 25 轮：随机选一个模式 acquire(mode)，"
                f"随机停顿不到 1 毫秒，再 release(mode)（第 {rnd + 1} 组）")
        lock = getattr(sol, cls_name)()
        clashes, errors, unfinished = ref.stress(lock, plans, deadline=5.0)
        if errors:
            raise errors[0]
        if clashes:
            got = "、".join(f"{n}:{m!r}" for n, m in clashes[0].items())
            raise Mismatch(f"不同模式的线程同时持有了锁：{got}", show=False)
        if unfinished:
            raise Mismatch(f"{len(unfinished)} 个线程 5 秒后还没有做完自己的 25 轮（正常几十毫秒就能做完）："
                           "可能死锁了，或者有等待中的调用一直没有被唤醒", show=False)


@case("Part 1", title="16 个线程真实并发")
def p1_stress(sol):
    room()
    stress_case(sol, "ModalLock", "p1")


# ================================================================ Part 2

@case("Part 2", sample=True, title="基本公平性")
def p2_basic(sol):
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    s.arrive("b", "infer")
    s.arrive("c", "train")      # may not join a's batch: b is already waiting behind it
    holds(s, "a")
    s.release("a")
    holds(s, "b")
    s.release("b")
    holds(s, "c")
    s.release("c")
    holds(s, "")
    s.end()


@case("Part 2", sample=True, title="交错到达")
def p2_interleaved(sol):
    s = Script(sol, "FairModalLock")
    for w in ("a:train", "b:infer", "c:train", "d:infer", "e:train"):
        name, mode = w.split(":")
        s.arrive(name, mode)
    holds(s, "a")
    for done, nxt in zip("abcde", ["b", "c", "d", "e", ""]):
        s.release(done)
        holds(s, nxt)
    s.end()


@case("Part 2", sample=True, title="同模式扎堆")
def p2_burst(sol):
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    for n in "bcd":
        s.arrive(n, "infer")
    holds(s, "a")
    waiting(s, 3)
    s.release("a")
    holds(s, "bcd")
    waiting(s, 0)
    for n in "bcd":
        s.release(n)
    s.end()


@case("Part 2", sample=True, title="三种模式的例子")
def p2_three_modes(sol):
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    s.arrive("b", "infer")
    s.arrive("c", "export")
    s.arrive("d", "infer")      # the tail is c's 'export' batch: d may not join b
    holds(s, "a")
    s.release("a")
    holds(s, "b")
    s.arrive("e", "infer")      # the tail is d's batch: e joins d, not b
    holds(s, "b")
    s.release("b")
    holds(s, "c")
    s.release("c")
    holds(s, "de")
    s.release("d")
    holds(s, "e")
    s.release("e")
    holds(s, "")
    s.end()


@case("Part 2", title="无人排队时直接加入正在运行的批次")
def p2_join_running(sol):
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    s.arrive("b", "train")
    s.arrive("c", "train")
    holds(s, "abc")
    s.release("a")
    s.arrive("d", "train")
    holds(s, "bcd")
    s.arrive("e", "infer")
    s.arrive("f", "infer")
    s.release("b")
    s.release("c")
    s.release("d")
    holds(s, "ef")
    s.arrive("g", "infer")      # e and f's batch is running and is the tail: g joins it at once
    holds(s, "efg")
    drain(s)


@case("Part 2", title="同模式也不能插到等待者前面")
def p2_no_cut_in(sol):
    # the statement's starvation script for ModalLock: with FairModalLock the 'train' callers queue behind 'infer'
    s = Script(sol, "FairModalLock")
    s.arrive("w0", "train")
    s.arrive("starved", "infer")
    for i in range(1, 8):
        s.arrive(f"w{i}", "train")
    s.release("w0")
    holds(s, ["starved"])
    s.release("starved")
    holds(s, [f"w{i}" for i in range(1, 8)])
    drain(s)


@case("Part 2", title="只能加入队尾的批次")
def p2_tail_only(sol):
    s = Script(sol, "FairModalLock")
    for w in ("a:train", "b:infer", "c:train", "d:infer", "e:infer", "f:train"):
        name, mode = w.split(":")
        s.arrive(name, mode)
    s.release("a")
    s.release("b")
    s.release("c")
    holds(s, "de")
    s.arrive("g", "infer")      # the tail is f's 'train' batch, so g starts a new batch behind it
    s.release("d")
    s.release("e")
    holds(s, "f")
    s.arrive("h", "train")      # f's batch is running and is not the tail (g is): h queues behind g
    s.release("f")
    holds(s, "g")
    drain(s)


@case("Part 2", title="被唤醒还没拿回锁的成员仍在批次里")
def p2_woken_member(sol):
    room()
    for slow, behind in ((["c"], ["d:export"]), (["c1", "c2"], ["d:export", "e:export"])):
        gate = ref.Gate()
        s = Script(sol, "FairModalLock")
        s.arrive("a", "train")
        s.arrive("b", "infer")
        for n in slow:
            s.arrive(n, "infer", gate=gate, note=f"测试让 {n} 被唤醒后先等一下，再去重新拿锁")
        for w in behind:
            name, mode = w.split(":")
            s.arrive(name, mode)
        batch = frozenset(["b", *slow])
        gate.arm()          # from now on the slow members' wake-ups are held back until gate.set()
        s.release("a", settle=False)
        if not wait_until(lambda: "b" in s.holders(), s.bad(batch)):
            s.fail(f"a 释放后批次 {fmt(batch)} 到达队列头部，b 应当获得锁，但 {ref.WAIT:g} 秒后它仍停在 acquire 里")
        s.release("b", settle=False)
        gate.set()
        s.settle(hint=f"{fmt(slow)} 在 a 释放时就被唤醒了，测试只是让它们晚一步才重新拿到锁：在那之前它们仍是"
                      f"头部批次的成员，这个批次不能在它们获得并释放锁之前离开队列。")
        drain(s)


@case("Part 2", title="放慢线程：判断和动作不能分开")
def p2_slow_threads(sol):
    room()
    # a caller slowed on its way into its wait: the release that comes meanwhile must still wake it
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    slowed(s, "b", "infer")
    s.release("a", hint=LATE_WAIT.format("b"))
    holds(s, "b")
    s.release("b")
    s.end()
    # p2_woken_member again, with the woken member slowed instead of held back in wait(): this reaches locks
    # that do not park on a Condition too
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    s.arrive("b", "infer")
    slowed(s, "c", "infer")
    s.arrive("d", "export")
    s.release("a", settle=False)
    if not wait_until(lambda: "b" in s.holders(), s.bad(frozenset("bc"))):
        s.fail(f"a 释放后批次 {{b, c}} 到达队列头部，b 应当获得锁，但 {ref.WAIT:g} 秒后它仍停在 acquire 里")
    s.release("b", settle=False)
    s.settle(hint="c 在 a 释放时就该被放行了，测试只是让它走得慢：在它获得并释放锁之前，它仍是头部批次的成员，"
                  "这个批次不能先离开队列。")
    drain(s)


@case("Part 2", title="批次到达头部时全部成员一起放行")
def p2_wake_batch(sol):
    s = Script(sol, "FairModalLock")
    s.arrive("a", "train")
    for i in range(6):
        s.arrive(f"i{i}", "infer")
    for i in range(4):
        s.arrive(f"x{i}", "export")
    s.release("a")
    holds(s, [f"i{i}" for i in range(6)])
    s.arrive("t0", "train")
    s.arrive("t1", "train")
    for i in range(6):
        s.release(f"i{i}")
    holds(s, [f"x{i}" for i in range(4)])
    drain(s)


@case("Part 2", title="release 模式不符时抛出异常")
def p2_mismatch(sol):
    err = sol.ModeMismatchError
    s = Script(sol, "FairModalLock")
    lock = s.lock
    mismatch(s, err, "lock.release('train')", lambda: lock.release("train"), "锁没有任何持有者")
    s.arrive("a", "train")
    s.arrive("b", "infer")
    mismatch(s, err, "lock.release('infer')", lambda: lock.release("infer"),
             "锁正以 'train' 被持有；'infer' 的 b 还在排队，并没有持有锁")
    s.arrive("c", "train")
    s.release("a")
    holds(s, "b")
    mismatch(s, err, "lock.release('train')", lambda: lock.release("train"),
             "锁正以 'infer' 被持有；'train' 的 c 还在排队")
    s.release("b")
    s.release("c")
    mismatch(s, err, "lock.release('train')", lambda: lock.release("train"), "锁已经空闲，没有任何持有者")
    s.arrive("d", "export")
    s.release("d")
    s.end()


@case("Part 2", title="随机脚本对照题目规则")
def p2_random(sol):
    room()
    r = rng()
    cover = {"join": 0, "queued": 0, "together": 0}
    steps = 0
    for _ in range(60):
        s = Script(sol, "FairModalLock", fresh_modes=True)
        ref.random_trial(s, r, MODES, n_names=16, n_events=40, cover=cover)
        s.end()
        steps += s.count
    info(f"60 段随机脚本，共 {steps} 步；同模式直接加入 {cover['join']} 次，同模式但必须排队 {cover['queued']} 次，"
         f"一次放行 2 个以上 {cover['together']} 次")


@case("Part 2", title="16 个线程真实并发")
def p2_stress(sol):
    room()
    stress_case(sol, "FairModalLock", "p2")
