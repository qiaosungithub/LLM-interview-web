import ast
import json
import time

from judgelib import calling, case, expect, rng
import ref

SHOW = 20           # a replay lists at most this many queries


# ---------------------------------------------------------------- how a query list is shown

def _lit(x):
    return json.dumps(x, ensure_ascii=False)


def _replay(queries, wants, k, mark="# <- 这一步"):
    """queries[:k + 1] as code to paste into 自定义运行, with the expected answers of the earlier ones."""
    lines = [_lit(q) + "," for q in queries[: k + 1]]
    notes = [f"# -> {_lit(w)}" for w in wants[:k]] + [mark]
    skipped = max(0, len(lines) - SHOW)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["queries = ["]
    if skipped:
        head.append(f"    # …（省略了前面 {skipped} 条命令，所以不能直接粘贴重现）")
    body = ["    " + s.ljust(width) + n for s, n in zip(lines, notes)]
    return "\n".join(head + body + ["]", "print(solution(queries)[-1])"])


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _message(got, want, why):
    if type(got) is not type(want):
        hint = f"每条命令都返回字符串：期望 {_lit(want)}，得到 {type(got).__name__} 类型的 {got!r}"
        return f"{why}；{hint}" if why else hint
    return why or "返回值不对"


def _first_bad(got, wants):
    """None if got answers every query as wanted; else (k, problem) for the first query that is not."""
    if not isinstance(got, (list, tuple)):
        return 0, "type"
    for k, want in enumerate(wants):
        if k >= len(got):
            return k, "short"
        if not _same(got[k], want):
            return k, "value"
    if len(got) > len(wants):
        return len(wants) - 1, "long"
    return None


def _report(queries, wants, got, k, problem, why=""):
    if problem == "type":
        expect(type(got).__name__, "list", call=_replay(queries, wants, len(queries) - 1),
               msg="solution 应该返回一个列表，每条命令对应一个字符串")
    if problem == "short":
        expect(len(got), len(queries), call=_replay(queries, wants, k, mark="# <- 这一条没有对应的返回值"),
               msg=f"返回的列表只有 {len(got)} 个元素，但一共有 {len(queries)} 条命令；每条命令（包括失败的）都要对应一个字符串")
    if problem == "long":
        expect(len(got), len(queries), call=_replay(queries, wants, k, mark="# <- 最后一条"),
               msg=f"返回的列表有 {len(got)} 个元素，比命令（{len(queries)} 条）多")
    why = why or f"第 {k + 1} 条命令 {' '.join(queries[k])} 的返回值不对"
    expect(got[k], wants[k], call=_replay(queries, wants, k), msg=_message(got[k], wants[k], why))


def _call(sol, queries):
    return sol.solution([list(q) for q in queries])


def _judge(sol, queries, wants):
    """None if the answer returns wants. Otherwise the first thing that goes wrong, as a dict with the index k
    of the query: a wrong or missing answer ("problem", "got"), or an exception that query k raises ("exc").
    All the queries go to one solution() call; when that call raises, shorter prefixes find the query that
    raises, and an earlier wrong answer, if any, is reported instead."""
    try:
        got = _call(sol, queries)
    except Exception as whole:
        err, k = whole, len(queries) - 1
        for i in range(len(queries)):
            try:
                _call(sol, queries[: i + 1])
            except Exception as e:
                err, k = e, i
                break
        earlier = _judge(sol, queries[:k], wants[:k]) if k else None
        return earlier or {"k": k, "exc": err}
    bad = _first_bad(got, wants)
    if bad is None:
        return None
    return {"k": bad[0], "problem": bad[1], "got": got, "exc": None}


def _show(queries, wants, bad, why=""):
    """Fails the case with what _judge found."""
    if bad["exc"] is not None:
        calling(_replay(queries, wants, bad["k"], mark="# <- 这一步抛出异常"))
        raise bad["exc"]
    _report(queries, wants, bad["got"], bad["k"], bad["problem"], why if bad["problem"] == "value" else "")


# ---------------------------------------------------------------- hand-written query lists

def _parse(script):
    """Lines like  ADD_FILE /a 5 -> "true"   # why  become (query, want, why)."""
    rows = []
    for line in script.strip().splitlines():
        query, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        rows.append((query.split(), ast.literal_eval(expr.strip()), why.strip()))
    return rows


def run(sol, script):
    """Passes all the queries to one solution() call, then checks the answers in order."""
    rows = _parse(script)
    queries = [q for q, _, _ in rows]
    wants = [w for _, w, _ in rows]
    calling(_replay(queries, wants, len(queries) - 1, mark="# <- 整段命令一次传入 solution"))
    bad = _judge(sol, queries, wants)
    if bad is not None:
        _show(queries, wants, bad, rows[bad["k"]][2])


# ---------------------------------------------------------------- random query lists against ref

def _outcome(sol, queries):
    """None if the answer agrees with ref on every query; else what _judge found, with ref's answers."""
    wants = ref.solution([list(q) for q in queries])
    bad = _judge(sol, queries, wants)
    if bad is not None:
        bad["wants"] = wants
    return bad


def _shrink(sol, queries, bad, budget=0.6):
    """Drops queries that the failure does not need, so the replay shown is short."""
    queries = queries[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(queries) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(queries) and time.process_time() < stop:
            cand = queries[:i] + queries[i + size:]
            out = _outcome(sol, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                queries, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return queries, bad


def random_check(sol, level, sequences, length):
    r = rng()
    for _ in range(sequences):
        queries = ref.random_queries(r, level, r.randint(*length))
        bad = _outcome(sol, queries)
        if bad is None:
            continue
        queries, bad = _shrink(sol, queries, bad)
        k = bad["k"]
        _show(queries, bad["wants"][: len(queries)], bad,
              f"随机命令序列的第 {k + 1} 条返回值不对（已自动删去与出错无关的命令）")


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, """
        ADD_FILE /dir1/dir2/file.txt 10     -> "true"
        ADD_FILE /dir1/dir2/file.txt 5      -> "false"   # 同名文件已存在，大小仍是 10
        GET_FILE_SIZE /dir1/dir2/file.txt   -> "10"
        DELETE_FILE /not-existing.file      -> ""
        DELETE_FILE /dir1/dir2/file.txt     -> "10"
        GET_FILE_SIZE /not-existing.file    -> ""
    """)


@case("Level 1", title="删除后名字可以再用")
def l1_delete_frees_name(sol):
    run(sol, """
        ADD_FILE /a 5           -> "true"
        ADD_FILE /b 7           -> "true"
        DELETE_FILE /a          -> "5"       # 返回被删文件的大小
        GET_FILE_SIZE /a        -> ""
        DELETE_FILE /a          -> ""        # 已经删掉了
        GET_FILE_SIZE /b        -> "7"       # 别的文件不受影响
        ADD_FILE /a 9           -> "true"    # 删除之后，这个名字可以再次使用
        GET_FILE_SIZE /a        -> "9"
        DELETE_FILE /a          -> "9"
    """)


@case("Level 1", title="添加失败不改大小")
def l1_failed_add(sol):
    run(sol, """
        ADD_FILE /f 10          -> "true"
        ADD_FILE /f 10          -> "false"   # 大小相同也算同名
        ADD_FILE /f 99          -> "false"
        GET_FILE_SIZE /f        -> "10"      # 失败的命令不改变任何状态
        DELETE_FILE /f          -> "10"
        ADD_FILE /f 99          -> "true"
        GET_FILE_SIZE /f        -> "99"
    """)


@case("Level 1", title="名字区分大小写、互为前缀、很大的大小")
def l1_names(sol):
    run(sol, """
        ADD_FILE /a 1                   -> "true"
        ADD_FILE /A 2                   -> "true"    # 区分大小写：是另一个名字
        ADD_FILE /a/b 3                 -> "true"    # 没有目录，"/a" 和 "/a/b" 只是两个字符串
        ADD_FILE 10 4                   -> "true"    # 名字可以全是数字
        GET_FILE_SIZE /a                -> "1"
        GET_FILE_SIZE /A                -> "2"
        GET_FILE_SIZE 10                -> "4"
        DELETE_FILE /a                  -> "1"
        GET_FILE_SIZE /a/b              -> "3"
        GET_FILE_SIZE /A                -> "2"
        ADD_FILE /big 1000000000        -> "true"
        GET_FILE_SIZE /big              -> "1000000000"
        DELETE_FILE /big                -> "1000000000"
    """)


@case("Level 1", title="随机命令序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=200, length=(5, 40))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        ADD_FILE /dir/file1.txt 5           -> "true"
        ADD_FILE /dir/file2 20              -> "true"
        ADD_FILE /dir/deeper/file3.mov 9    -> "true"
        GET_N_LARGEST /dir 2                -> "/dir/file2(20), /dir/deeper/file3.mov(9)"
        GET_N_LARGEST /dir/file 3           -> "/dir/file2(20), /dir/file1.txt(5)"     # 只有两个匹配
        GET_N_LARGEST /another_dir 2        -> ""
        ADD_FILE /big_file.mp4 20           -> "true"
        GET_N_LARGEST / 2                   -> "/big_file.mp4(20), /dir/file2(20)"     # 大小并列，按名字排
    """)


@case("Level 2", title="大小并列：名字字典序从小到大")
def l2_ties_by_name(sol):
    run(sol, """
        ADD_FILE /d 5           -> "true"
        ADD_FILE /c 5           -> "true"
        ADD_FILE /b 5           -> "true"
        ADD_FILE /a 5           -> "true"
        ADD_FILE /B 5           -> "true"
        ADD_FILE /a1 5          -> "true"
        GET_N_LARGEST / 6       -> "/B(5), /a(5), /a1(5), /b(5), /c(5), /d(5)"    # 普通字符串比较：大写在小写前面，"/a" 在 "/a1" 前面
        GET_N_LARGEST / 2       -> "/B(5), /a(5)"
        ADD_FILE /z 6           -> "true"
        GET_N_LARGEST / 3       -> "/z(6), /B(5), /a(5)"
        GET_N_LARGEST /a 5      -> "/a(5), /a1(5)"
    """)


@case("Level 2", title="普通字符串前缀，和目录无关")
def l2_prefix_plain(sol):
    run(sol, """
        ADD_FILE /dir 1             -> "true"
        ADD_FILE /dir/a 2           -> "true"
        ADD_FILE /dirt 3            -> "true"
        ADD_FILE /x/dir/b 4         -> "true"
        ADD_FILE /dir2/c 5          -> "true"
        GET_N_LARGEST /dir 9        -> "/dir2/c(5), /dirt(3), /dir/a(2), /dir(1)"   # "/x/dir/b" 中间含 "/dir"，但不是以它开头
        GET_N_LARGEST /dir/ 9       -> "/dir/a(2)"       # 名字 "/dir" 本身不以 "/dir/" 开头
        GET_N_LARGEST /dir/a 9      -> "/dir/a(2)"       # 和名字完全相同也算以它开头
        GET_N_LARGEST /dir/ab 9     -> ""                # 比名字还长的前缀
        GET_N_LARGEST dir 9         -> ""                # 前缀要从第一个字符开始匹配
        GET_N_LARGEST /x 9          -> "/x/dir/b(4)"
    """)


@case("Level 2", title="n 的边界；大小按数值比较")
def l2_n_and_sizes(sol):
    run(sol, """
        ADD_FILE /s9 9              -> "true"
        ADD_FILE /s10 10            -> "true"
        ADD_FILE /s100 100          -> "true"
        ADD_FILE /s2 2              -> "true"
        GET_N_LARGEST /s 2          -> "/s100(100), /s10(10)"    # 按数值比较：100 > 10 > 9 > 2
        GET_N_LARGEST /s 4          -> "/s100(100), /s10(10), /s9(9), /s2(2)"    # n 恰好等于匹配数
        GET_N_LARGEST /s 1000000000 -> "/s100(100), /s10(10), /s9(9), /s2(2)"    # 不足 n 个时全部返回
        GET_N_LARGEST /s 1          -> "/s100(100)"
        DELETE_FILE /s100           -> "100"
        GET_N_LARGEST /s 1          -> "/s10(10)"        # 删掉的文件不再出现
        ADD_FILE /s100 1            -> "true"
        GET_N_LARGEST /s100 5       -> "/s100(1)"        # 重新添加后是新的大小
    """)


@case("Level 2", title="名字里的符号和非 ASCII 字符")
def l2_unicode_names(sol):
    run(sol, """
        ADD_FILE /a 1               -> "true"
        ADD_FILE /a~ 2              -> "true"
        ADD_FILE /a~x 3             -> "true"
        ADD_FILE /aé 4              -> "true"
        ADD_FILE /a文件 5           -> "true"
        ADD_FILE /b 6               -> "true"
        GET_N_LARGEST /a 9          -> "/a文件(5), /aé(4), /a~x(3), /a~(2), /a(1)"    # 以 /a 开头的都算，包括含 ~、é、汉字的名字
        GET_N_LARGEST /a~ 9         -> "/a~x(3), /a~(2)"
        GET_N_LARGEST /a文 9        -> "/a文件(5)"
        GET_FILE_SIZE /aé           -> "4"
        DELETE_FILE /a文件          -> "5"
        GET_N_LARGEST /a 2          -> "/aé(4), /a~x(3)"
    """)


@case("Level 2", title="随机命令序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 40))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        ADD_USER user1 200                      -> "true"
        ADD_USER user1 100                      -> "false"   # 已存在，容量仍是 200
        ADD_FILE_BY user1 /dir/file.med 50      -> "150"
        ADD_FILE_BY user1 /big.blob 140         -> "10"
        ADD_FILE_BY user1 /file-small 20        -> ""        # 只剩 10
        ADD_FILE /dir/admin_file 300            -> "true"    # admin，容量不限
        ADD_USER user2 110                      -> "true"
        ADD_FILE_BY user2 /dir/file.med 45      -> ""        # 同名文件已存在（属于 user1）
        ADD_FILE_BY user2 /new_file 50          -> "60"
        MERGE_USER user1 user2                  -> "70"      # 上限 200 + 110 = 310，已用 190 + 50 = 240
    """)


@case("Level 3", title="刚好用完容量是允许的")
def l3_capacity_exact(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_FILE_BY u /a 4                      -> "6"
        ADD_FILE_BY u /b 6                      -> "0"       # 刚好用完，返回 "0"
        ADD_FILE_BY u /c 1                      -> ""
        DELETE_FILE /b                          -> "6"
        ADD_FILE_BY u /c 7                      -> ""        # 只剩 6
        ADD_FILE_BY u /c 6                      -> "0"
        ADD_USER big 1000000000                 -> "true"
        ADD_FILE_BY big /huge 1000000000        -> "0"
        ADD_FILE_BY big /one 1                  -> ""
    """)


@case("Level 3", title="ADD_FILE_BY 的各种失败")
def l3_add_file_by_failures(sol):
    run(sol, """
        ADD_FILE_BY nobody /a 1                 -> ""        # 用户不存在
        ADD_FILE /admin_file 5                  -> "true"
        ADD_USER u 20                           -> "true"
        ADD_USER v 20                           -> "true"
        ADD_FILE_BY u /admin_file 1             -> ""        # 名字被 admin 占用
        ADD_FILE_BY v /v_file 3                 -> "17"
        ADD_FILE_BY u /v_file 1                 -> ""        # 名字被别的用户占用
        ADD_FILE_BY u /u_file 2                 -> "18"
        ADD_FILE_BY u /u_file 2                 -> ""        # 名字被自己占用
        ADD_FILE_BY u /big 19                   -> ""        # 超过剩余容量
        ADD_FILE_BY u /ok 18                    -> "0"       # 失败的命令没有占用容量
        ADD_FILE /u_file 1                      -> "false"   # admin 也不能用别人的名字
        GET_FILE_SIZE /admin_file               -> "5"
        GET_FILE_SIZE /v_file                   -> "3"
        ADD_FILE_BY nobody /new 1               -> ""
        GET_FILE_SIZE /new                      -> ""
    """)


@case("Level 3", title="删除用户的文件会还回容量")
def l3_delete_gives_space_back(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_FILE_BY u /a 6                      -> "4"
        ADD_FILE_BY u /b 4                      -> "0"
        GET_FILE_SIZE /a                        -> "6"       # 照旧能看到用户的文件
        GET_N_LARGEST / 5                       -> "/a(6), /b(4)"
        DELETE_FILE /a                          -> "6"       # 可以删除任何人的文件
        ADD_FILE_BY u /c 5                      -> "1"       # 删掉的 6 还给了 u
        DELETE_FILE /b                          -> "4"
        DELETE_FILE /c                          -> "5"
        ADD_FILE_BY u /d 10                     -> "0"
        ADD_FILE /e 1000000000                  -> "true"    # admin 的文件不占任何用户的容量
        DELETE_FILE /e                          -> "1000000000"
        ADD_FILE_BY u /e 1                      -> ""
    """)


@case("Level 3", title="ADD_USER 失败不改容量")
def l3_add_user(sol):
    run(sol, """
        ADD_USER u 5                            -> "true"
        ADD_USER u 50                           -> "false"
        ADD_FILE_BY u /a 6                      -> ""        # 容量仍是 5
        ADD_FILE_BY u /a 5                      -> "0"
        ADD_USER U 7                            -> "true"    # 区分大小写：是另一个用户
        ADD_FILE_BY U /b 7                      -> "0"
        ADD_USER 10 3                           -> "true"    # 用户名可以全是数字
        ADD_FILE_BY 10 /c 3                     -> "0"
    """)


@case("Level 3", title="合并：上限相加，文件改归 userId1")
def l3_merge(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 20                           -> "true"
        ADD_FILE_BY a /a1 7                     -> "3"
        ADD_FILE_BY b /b1 5                     -> "15"
        ADD_FILE_BY b /b2 6                     -> "9"
        MERGE_USER a b                          -> "12"      # 上限 10 + 20 = 30，已用 7 + 11 = 18
        DELETE_FILE /b1                         -> "5"       # 原来属于 b 的文件现在属于 a
        ADD_FILE_BY a /a2 17                    -> "0"       # 12 + 5 = 17
        DELETE_FILE /b2                         -> "6"
        ADD_FILE_BY a /a3 6                     -> "0"
        GET_N_LARGEST / 9                       -> "/a2(17), /a1(7), /a3(6)"
    """)


@case("Level 3", title="合并失败的情况")
def l3_merge_failures(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 20                           -> "true"
        ADD_FILE_BY a /x 4                      -> "6"
        MERGE_USER a a                          -> ""        # 两个参数相同
        MERGE_USER a c                          -> ""        # userId2 不存在
        MERGE_USER c a                          -> ""        # userId1 不存在
        ADD_FILE_BY a /y 6                      -> "0"       # 失败的合并什么也没改
        ADD_FILE_BY b /z 20                     -> "0"
        MERGE_USER b a                          -> "0"       # 并入 b：上限 30，已用 30
        ADD_FILE_BY b /w 1                      -> ""
    """)


@case("Level 3", title="被合并的用户名可以重新创建")
def l3_merged_id_reusable(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 20                           -> "true"
        ADD_FILE_BY b /b1 5                     -> "15"
        MERGE_USER a b                          -> "25"
        ADD_USER b 3                            -> "true"    # b 已被删除，名字可以再用
        ADD_FILE_BY b /c 3                      -> "0"       # 全新的空用户：容量 3，没有文件
        DELETE_FILE /b1                         -> "5"       # 这个文件属于 a，不属于新的 b
        ADD_FILE_BY b /d 1                      -> ""
        ADD_FILE_BY a /e 30                     -> "0"
        MERGE_USER a b                          -> "0"       # 新的 b 也可以再并入
    """)


@case("Level 3", title="随机命令序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=400, length=(10, 60))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        ADD_USER user 100                       -> "true"
        ADD_FILE_BY user /dir/file1 50          -> "50"
        ADD_FILE_BY user /file2.txt 30          -> "20"
        RESTORE_USER user                       -> "0"       # 没有备份：两个文件都被删除
        ADD_FILE_BY user /file3.mp4 60          -> "40"
        ADD_FILE_BY user /file4.txt 10          -> "30"
        BACKUP_USER user                        -> "2"
        DELETE_FILE /file3.mp4                  -> "60"
        DELETE_FILE /file4.txt                  -> "10"
        ADD_FILE /file3.mp4 140                 -> "true"    # admin 占用了这个名字
        ADD_FILE_BY user /dir/file5.new 20      -> "80"
        RESTORE_USER user                       -> "1"       # 删除 file5.new；file4.txt 恢复；file3.mp4 被跳过
    """)


@case("Level 4", title="用户不存在；空备份")
def l4_unknown_and_empty(sol):
    run(sol, """
        BACKUP_USER u                           -> ""        # 用户不存在
        RESTORE_USER u                          -> ""
        ADD_USER u 10                           -> "true"
        BACKUP_USER u                           -> "0"       # 没有文件：备份是空的
        ADD_FILE_BY u /a 4                      -> "6"
        RESTORE_USER u                          -> "0"       # 恢复空备份：文件全部删除
        GET_FILE_SIZE /a                        -> ""
        ADD_FILE_BY u /b 10                     -> "0"
        BACKUP_USER nobody                      -> ""
    """)


@case("Level 4", title="从没备份过：文件全部删除")
def l4_restore_without_backup(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_USER v 10                           -> "true"
        ADD_FILE_BY u /a 3                      -> "7"
        ADD_FILE_BY u /b 4                      -> "3"
        ADD_FILE_BY v /c 5                      -> "5"
        RESTORE_USER u                          -> "0"
        GET_FILE_SIZE /a                        -> ""
        GET_FILE_SIZE /c                        -> "5"       # 别人的文件不动
        ADD_FILE_BY u /d 10                     -> "0"       # 容量全部空出来了
        BACKUP_USER v                           -> "1"
    """)


@case("Level 4", title="恢复：先删掉用户现在的全部文件")
def l4_restore_deletes_new_files(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_FILE_BY u /a 3                      -> "7"
        BACKUP_USER u                           -> "1"
        ADD_FILE_BY u /b 4                      -> "3"       # 备份之后才添加
        ADD_FILE_BY u /c 2                      -> "1"
        RESTORE_USER u                          -> "1"       # /b、/c 被删除，/a 重建
        GET_FILE_SIZE /b                        -> ""
        GET_FILE_SIZE /c                        -> ""
        GET_FILE_SIZE /a                        -> "3"
        ADD_FILE_BY u /d 7                      -> "0"       # 剩余容量 = 10 − 3
        GET_N_LARGEST / 9                       -> "/d(7), /a(3)"
    """)


@case("Level 4", title="一直属于自己的文件：按备份重建并计数")
def l4_restore_rebuilds_owned(sol):
    run(sol, """
        ADD_USER u 20                           -> "true"
        ADD_FILE_BY u /a 4                      -> "16"
        ADD_FILE_BY u /b 5                      -> "11"
        BACKUP_USER u                           -> "2"
        DELETE_FILE /a                          -> "4"
        ADD_FILE_BY u /a 9                      -> "6"       # 同名，大小变了，仍归 u
        RESTORE_USER u                          -> "2"       # /a 和 /b 都在第 1 步删掉、第 2 步重建，都计数
        GET_FILE_SIZE /a                        -> "4"       # 按备份里的大小
        ADD_FILE_BY u /c 11                     -> "0"       # 20 − 4 − 5
        RESTORE_USER u                          -> "2"
    """)


@case("Level 4", title="名字被别人占用：跳过，别人的文件不动")
def l4_restore_skips_taken(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_USER v 10                           -> "true"
        ADD_FILE_BY u /a 1                      -> "9"
        ADD_FILE_BY u /b 2                      -> "7"
        ADD_FILE_BY u /c 3                      -> "4"
        BACKUP_USER u                           -> "3"
        DELETE_FILE /a                          -> "1"
        DELETE_FILE /b                          -> "2"
        ADD_FILE_BY v /a 7                      -> "3"       # v 占用了 /a
        ADD_FILE /b 99                          -> "true"    # admin 占用了 /b
        RESTORE_USER u                          -> "1"       # 只有 /c 恢复成功
        GET_N_LARGEST / 5                       -> "/b(99), /a(7), /c(3)"
        ADD_FILE_BY u /d 7                      -> "0"       # u 只有 /c
        ADD_FILE_BY v /e 3                      -> "0"       # v 的 /a 仍是 v 的
        DELETE_FILE /a                          -> "7"
        RESTORE_USER u                          -> "2"       # 删掉 /c、/d；/a 空出来了，这次能恢复；/b 仍被占用
        GET_FILE_SIZE /d                        -> ""
        GET_N_LARGEST / 5                       -> "/b(99), /c(3), /e(3), /a(1)"
    """)


@case("Level 4", title="备份是副本；新备份覆盖旧的；可以反复恢复")
def l4_backup_is_copy(sol):
    run(sol, """
        ADD_USER u 10                           -> "true"
        ADD_FILE_BY u /a 4                      -> "6"
        BACKUP_USER u                           -> "1"
        DELETE_FILE /a                          -> "4"
        ADD_FILE_BY u /a 9                      -> "1"       # 之后的操作不改动备份
        RESTORE_USER u                          -> "1"
        GET_FILE_SIZE /a                        -> "4"
        ADD_FILE_BY u /b 2                      -> "4"
        BACKUP_USER u                           -> "2"       # 新备份覆盖旧的
        DELETE_FILE /a                          -> "4"
        DELETE_FILE /b                          -> "2"
        RESTORE_USER u                          -> "2"
        RESTORE_USER u                          -> "2"       # 恢复不消耗备份
        GET_N_LARGEST / 5                       -> "/a(4), /b(2)"
    """)


@case("Level 4", title="合并不改动 userId1 的备份")
def l4_merge_keeps_target_backup(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 10                           -> "true"
        ADD_FILE_BY a /a 1                      -> "9"
        ADD_FILE_BY b /b 2                      -> "8"
        BACKUP_USER a                           -> "1"
        BACKUP_USER b                           -> "1"
        MERGE_USER a b                          -> "17"
        DELETE_FILE /b                          -> "2"
        RESTORE_USER a                          -> "1"       # a 的备份里只有 /a；b 的备份没有并进来
        GET_FILE_SIZE /b                        -> ""
        ADD_FILE_BY a /c 19                     -> "0"
    """)


@case("Level 4", title="合并后 userId1 恢复：接过来的文件被删")
def l4_merge_then_restore(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 10                           -> "true"
        ADD_FILE_BY a /a 1                      -> "9"
        BACKUP_USER a                           -> "1"
        ADD_FILE_BY b /b 2                      -> "8"
        MERGE_USER a b                          -> "17"
        RESTORE_USER a                          -> "1"       # 第 1 步删的是 a 现在拥有的全部文件，包括 /b
        GET_FILE_SIZE /b                        -> ""
        GET_FILE_SIZE /a                        -> "1"
    """)


@case("Level 4", title="userId2 的备份随它一起删除")
def l4_merge_drops_source_backup(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 10                           -> "true"
        ADD_FILE_BY b /b 2                      -> "8"
        BACKUP_USER b                           -> "1"
        MERGE_USER a b                          -> "18"
        DELETE_FILE /b                          -> "2"
        ADD_USER b 5                            -> "true"    # 全新的 b：没有备份
        RESTORE_USER b                          -> "0"
        GET_FILE_SIZE /b                        -> ""
        ADD_FILE_BY b /x 5                      -> "0"
    """)


@case("Level 4", title="合并后的备份与恢复")
def l4_merge_then_backup(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 10                           -> "true"
        ADD_FILE_BY a /a 1                      -> "9"
        ADD_FILE_BY b /b 2                      -> "8"
        BACKUP_USER b                           -> "1"
        MERGE_USER a b                          -> "17"      # a 从没备份过；b 的备份随 b 一起删除，不归 a
        RESTORE_USER a                          -> "0"       # 按空备份处理：/a 和接过来的 /b 都被删除
        GET_FILE_SIZE /b                        -> ""
        GET_FILE_SIZE /a                        -> ""
        ADD_FILE_BY a /c 4                      -> "16"
        ADD_USER c 10                           -> "true"
        ADD_FILE_BY c /d 3                      -> "7"
        MERGE_USER a c                          -> "23"      # 上限 20 + 10 = 30，已用 4 + 3 = 7
        BACKUP_USER a                           -> "2"       # 记下 a 现在拥有的全部文件，包括接过来的 /d
        DELETE_FILE /d                          -> "3"
        RESTORE_USER a                          -> "2"
        GET_FILE_SIZE /d                        -> "3"
        ADD_FILE_BY a /e 23                     -> "0"
    """)


@case("Level 4", title="恢复不改变容量上限")
def l4_restore_keeps_limit(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 20                           -> "true"
        ADD_FILE_BY a /a 6                      -> "4"
        BACKUP_USER a                           -> "1"
        MERGE_USER a b                          -> "24"      # 上限变成 30
        RESTORE_USER a                          -> "1"
        ADD_FILE_BY a /c 24                     -> "0"       # 恢复后上限仍是 30：30 − 6
        ADD_USER u 10                           -> "true"
        ADD_FILE_BY u /u 8                      -> "2"
        BACKUP_USER u                           -> "1"
        DELETE_FILE /u                          -> "8"
        RESTORE_USER u                          -> "1"
        ADD_FILE_BY u /v 2                      -> "0"
    """)


@case("Level 4", title="合并后自己的备份和接过来的文件同名")
def l4_merge_same_name(sol):
    run(sol, """
        ADD_USER a 10                           -> "true"
        ADD_USER b 10                           -> "true"
        ADD_FILE_BY a /f 3                      -> "7"
        BACKUP_USER a                           -> "1"
        DELETE_FILE /f                          -> "3"
        ADD_FILE_BY b /f 8                      -> "2"
        MERGE_USER a b                          -> "12"      # /f（8）现在属于 a
        RESTORE_USER a                          -> "1"       # 先删掉 a 的 /f（8），再按备份重建 /f（3）
        GET_FILE_SIZE /f                        -> "3"
        ADD_FILE_BY a /g 17                     -> "0"
    """)


@case("Level 4", cpu=2, title="随机命令序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=500, length=(10, 70))
