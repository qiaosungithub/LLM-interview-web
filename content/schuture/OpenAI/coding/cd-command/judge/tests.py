from judgelib import calling, case, expect, expect_raises, expect_true, rng, show_call
import ref

HOME = "/home/priya"
EX_LINKS = {"/srv/www/current": "/srv/archive/2025-09"}


def _path_msg(got, want):
    """A hint for a wrong path string; "" leaves expect()'s own message."""
    if not isinstance(got, str) or got == want:
        return ""
    if got.startswith("//"):
        return "结果开头多了一个 /：根目录只写一个 /（cwd 或 home 是 / 时最容易出这个错）"
    if len(got) > 1 and got.endswith("/"):
        return "结果不应以 / 结尾（只有根目录本身是 \"/\"）"
    return ""


def rel(sol, cwd, destination, want, msg=""):
    call = calling(show_call("cd_relative", cwd, destination))
    got = sol.cd_relative(cwd, destination)
    expect(got, want, call=call, msg=msg or _path_msg(got, want))


def ab(sol, cwd, destination, home, want, msg=""):
    call = calling(show_call("cd_absolute", cwd, destination, home))
    got = sol.cd_absolute(cwd, destination, home)
    expect(got, want, call=call, msg=msg or _path_msg(got, want))


def ab_raises(sol, cwd, destination, home):
    call = calling(show_call("cd_absolute", cwd, destination, home))
    expect_raises(ValueError, sol.cd_absolute, cwd, destination, home, call=call)


LOOP = object()     # want: SymlinkLoopError


def cdx(sol, cwd, destination, home, links, want, msg=""):
    """cd(...) must return `want`, or raise SymlinkLoopError when want is LOOP."""
    call = calling(show_call("cd", cwd, destination, home, links))
    if want is LOOP:
        expect_raises(sol.SymlinkLoopError, sol.cd, cwd, destination, home, dict(links), call=call)
        return
    loop_error = getattr(sol, "SymlinkLoopError", None)
    try:
        got = sol.cd(cwd, destination, home, dict(links))
    except Exception as e:
        if loop_error is None or not isinstance(e, loop_error):
            raise
        hops = ref.resolve(cwd, destination, home, links)[1]
        expect_true(False, f"这次调用只需要 {hops} 次展开，没有超过 MAX_SYMLINK_HOPS = 20，不应抛出 SymlinkLoopError"
                           f"（应当返回 {want!r}）", call=call)
    expect(got, want, call=call, msg=msg or _path_msg(got, want))


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="例子 guides")
def p1_ex_guides(sol):
    rel(sol, "/srv/www/docs", "guides", "/srv/www/docs/guides")


@case("Part 1", sample=True, title="例子 ../assets")
def p1_ex_assets(sol):
    rel(sol, "/srv/www/docs", "../assets", "/srv/www/assets")


@case("Part 1", sample=True, title="例子 ../guides/v2/")
def p1_ex_trailing_slash(sol):
    rel(sol, "/srv/www/docs", "../guides/v2/", "/srv/www/guides/v2")


@case("Part 1", sample=True, title="例子 ./guides/")
def p1_ex_dot(sol):
    rel(sol, "/srv/www/docs", "./guides/", "/srv/www/docs/guides")


@case("Part 1", sample=True, title="例子 ../guides/v2/../")
def p1_ex_dotdot_at_end(sol):
    rel(sol, "/srv/www/docs", "../guides/v2/../", "/srv/www/guides")


@case("Part 1", sample=True, title="例子 ../../../static")
def p1_ex_clamp_then_down(sol):
    rel(sol, "/srv/www", "../../../static", "/static")


@case("Part 1", sample=True, title="例子 / 下的 ../..")
def p1_ex_root(sol):
    rel(sol, "/", "../..", "/")


@case("Part 1", sample=True, title="例子 ../../logs/./today//")
def p1_ex_mixed(sol):
    rel(sol, "/srv/www/docs", "../../logs/./today//", "/srv/logs/today")


@case("Part 1", title="cwd 是 / 时向下走")
def p1_from_root(sol):
    rel(sol, "/", "etc", "/etc")
    rel(sol, "/", "etc/nginx/", "/etc/nginx")
    rel(sol, "/", "./var//log", "/var/log")
    rel(sol, "/", "a/../b", "/b")
    rel(sol, "/", "..", "/")


@case("Part 1", title="停在原地：. 和抵消掉的路径")
def p1_stay(sol):
    rel(sol, "/srv/www", ".", "/srv/www")
    rel(sol, "/srv/www", "./", "/srv/www")
    rel(sol, "/srv/www", ".//./", "/srv/www")
    rel(sol, "/srv/www", "logs/..", "/srv/www")
    rel(sol, "/", ".", "/")


@case("Part 1", title="退到根目录")
def p1_back_to_root(sol):
    rel(sol, "/srv", "..", "/")
    rel(sol, "/srv/www/docs", "../../..", "/")
    rel(sol, "/srv/www/docs", "../../../../../", "/")
    rel(sol, "/a", "../../../../b/../c", "/c")


@case("Part 1", title="点开头的普通名字")
def p1_dot_names(sol):
    msg = "只有恰好是 . 或 .. 的分量有特殊含义，...、.hidden、..a 都是普通的目录名"
    rel(sol, "/srv", "...", "/srv/...", msg)
    rel(sol, "/srv", ".hidden/..a/a..", "/srv/.hidden/..a/a..", msg)
    rel(sol, "/srv/www", "../.../..", "/srv", msg)


@case("Part 1", title="重复的斜杠")
def p1_repeated_slashes(sol):
    rel(sol, "/srv", "a//b///c", "/srv/a/b/c")
    rel(sol, "/srv", "a/b///", "/srv/a/b")
    rel(sol, "/srv", ".///..//x", "/x")


@case("Part 1", title="空格和中文名字")
def p1_unicode(sol):
    rel(sol, "/home/用户", "我的 文档/照片 2025/", "/home/用户/我的 文档/照片 2025")
    rel(sol, "/home/用户", "../项目", "/home/项目")


@case("Part 1", title="随机测试（对照参考实现）")
def p1_random(sol):
    r = rng()
    for _ in range(400):
        cwd, destination = ref.random_dir(r), ref.random_rest(r)
        rel(sol, cwd, destination, ref.resolve(cwd, destination)[0])


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="例子：绝对路径")
def p2_ex_absolute(sol):
    ab(sol, "/srv/www/docs", "/opt/tools/bin", "/home/priya", "/opt/tools/bin")


@case("Part 2", sample=True, title="例子 ~")
def p2_ex_tilde(sol):
    ab(sol, "/srv/www/docs", "~", "/home/priya", "/home/priya")


@case("Part 2", sample=True, title="例子 ~/scripts")
def p2_ex_tilde_rest(sol):
    ab(sol, "/srv/www/docs", "~/scripts", "/home/priya", "/home/priya/scripts")


@case("Part 2", sample=True, title="例子：相对路径照旧")
def p2_ex_relative(sol):
    ab(sol, "/tmp/scratch", "../backups", "/home/priya", "/tmp/backups")


@case("Part 2", sample=True, title="例子 a/~b 不展开")
def p2_ex_tilde_inside(sol):
    ab(sol, "/home/priya", "a/~b", "/home/priya", "/home/priya/a/~b")


@case("Part 2", sample=True, title="例子 ~bob 抛出 ValueError")
def p2_ex_tilde_user(sol):
    ab_raises(sol, "/home/priya", "~bob", "/home/priya")


@case("Part 2", title="绝对路径里的 .、.. 和斜杠")
def p2_absolute_forms(sol):
    ab(sol, "/srv/www", "/", HOME, "/")
    ab(sol, "/srv/www", "/..", HOME, "/")
    ab(sol, "/srv/www", "/opt//tools/./bin/", HOME, "/opt/tools/bin")
    ab(sol, "/srv/www", "/a/../../b", HOME, "/b")
    ab(sol, "/", "/etc", HOME, "/etc")


@case("Part 2", title="~/ 后面的 .、.. 和斜杠")
def p2_tilde_forms(sol):
    ab(sol, "/srv", "~/", HOME, "/home/priya")
    ab(sol, "/srv", "~/.", HOME, "/home/priya")
    ab(sol, "/srv", "~/..", HOME, "/home")
    ab(sol, "/srv", "~/../..", HOME, "/")
    ab(sol, "/srv", "~/../../../x", HOME, "/x")
    ab(sol, "/srv", "~//scripts/", HOME, "/home/priya/scripts")
    ab(sol, "/srv", "~/a/../b", HOME, "/home/priya/b")


@case("Part 2", title="家目录就是 /")
def p2_home_is_root(sol):
    ab(sol, "/srv", "~", "/", "/")
    ab(sol, "/srv", "~/x", "/", "/x")
    ab(sol, "/srv", "~/..", "/", "/")


@case("Part 2", title="~ 不在开头时只是普通字符")
def p2_tilde_not_first(sol):
    msg = "~ 只有作为 destination 的第一个字符时才有特殊含义，别的位置上的 ~ 是普通的名字"
    ab(sol, "/home/priya", "x/~/y", HOME, "/home/priya/x/~/y", msg)
    ab(sol, "/srv", "./~", HOME, "/srv/~", msg)
    ab(sol, "/srv/www", "../~", HOME, "/srv/~", msg)
    ab(sol, "/srv", "/~", HOME, "/~", msg)
    ab(sol, "/srv", "a/~/..", HOME, "/srv/a", msg)


@case("Part 2", title="其他 ~ 开头的形式都抛出 ValueError")
def p2_tilde_other_forms(sol):
    for destination in ["~bob/docs", "~root", "~+", "~-", "~~", "~.", "~1", "~ /x"]:
        ab_raises(sol, "/srv", destination, HOME)


@case("Part 2", title="随机测试（对照参考实现）")
def p2_random(sol):
    r = rng()
    for _ in range(400):
        cwd, home, rest = ref.random_dir(r), ref.random_dir(r), ref.random_rest(r)
        destination = r.choice([rest, "/" + rest, "~/" + rest, "~"])
        ab(sol, cwd, destination, home, ref.resolve(cwd, destination, home)[0])


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="例子 current/gallery/../notes")
def p3_example(sol):
    cdx(sol, "/srv/www", "current/gallery/../notes", HOME, EX_LINKS, "/srv/archive/2025-09/notes")


@case("Part 3", title="链接是最后一个分量")
def p3_link_last(sol):
    cdx(sol, "/srv/www", "current", HOME, EX_LINKS, "/srv/archive/2025-09")
    cdx(sol, "/srv/www", "current/", HOME, EX_LINKS, "/srv/archive/2025-09")
    cdx(sol, "/srv/www", "./current/.", HOME, EX_LINKS, "/srv/archive/2025-09")


@case("Part 3", title="链接之后的 ..：回到目标的上一级")
def p3_dotdot_after_link(sol):
    msg = ".. 作用在物理路径上：紧跟在链接之后时，回到链接所指向的目录的上一级，而不是链接所在的目录"
    cdx(sol, "/srv/www", "current/../snapshot", HOME, EX_LINKS, "/srv/archive/snapshot", msg)
    cdx(sol, "/srv/www", "current/..", HOME, EX_LINKS, "/srv/archive", msg)
    cdx(sol, "/srv/www", "current/../../x", HOME, EX_LINKS, "/srv/x", msg)
    cdx(sol, "/srv", "www/current/../2025-10", HOME, EX_LINKS, "/srv/archive/2025-10", msg)


@case("Part 3", title="相对目标：相对于链接所在的目录")
def p3_relative_target(sol):
    msg = "相对目标要相对于链接所在的目录解析（不是相对于 cwd，也不是从链接本身往下走）"
    links = {"/srv/www/current": "../archive/vault"}
    cdx(sol, "/srv/www", "current/notes", HOME, links, "/srv/archive/vault/notes", msg)
    cdx(sol, "/srv", "www/current/notes", HOME, links, "/srv/archive/vault/notes", msg)
    cdx(sol, "/", "srv/www/current", HOME, links, "/srv/archive/vault", msg)
    cdx(sol, "/srv/www/docs", "../current", HOME, links, "/srv/archive/vault", msg)


@case("Part 3", title="目标本身是链接，或经过链接")
def p3_target_chain(sol):
    msg = "目标也要按同样的规则逐个分量地跟随：目标落在另一个链接上、或者中途经过另一个链接时，都要继续展开"
    chain = {"/srv/www/current": "/srv/archive/latest", "/srv/archive/latest": "/srv/archive/2025-09"}
    cdx(sol, "/srv/www", "current/notes", HOME, chain, "/srv/archive/2025-09/notes", msg)
    through = {"/a/l": "/b/m/c", "/b/m": "/d"}
    cdx(sol, "/a", "l/x", HOME, through, "/d/c/x", msg)
    relative = {"/a/l": "m/c", "/a/m": "../e"}
    cdx(sol, "/a", "l", HOME, relative, "/e/c", msg)
    cdx(sol, "/a", "l/../..", HOME, relative, "/", msg)


@case("Part 3", title="目标里的 . 和 ..")
def p3_target_dots(sol):
    cdx(sol, "/a/b", "l/e", HOME, {"/a/b/l": "../../c/./d/.."}, "/c/e")
    cdx(sol, "/a", "l", HOME, {"/a/l": "/b/c/.."}, "/b")
    cdx(sol, "/a", "l/x", HOME, {"/a/l": "./"}, "/a/x")
    cdx(sol, "/a", "l/x", HOME, {"/a/l": "k//m/"}, "/a/k/m/x")


@case("Part 3", title="越过根目录的 .. 停在 /")
def p3_dotdot_at_root(sol):
    cdx(sol, "/a", "l/x", HOME, {"/a/l": "../../../.."}, "/x")
    cdx(sol, "/", "up/..", HOME, {"/up": "/.."}, "/")
    cdx(sol, "/a/b", "l/../x", HOME, {"/a/b/l": "/"}, "/x")


@case("Part 3", title="key 按完整的路径分量匹配")
def p3_whole_component_key(sol):
    msg = "key 是完整的绝对路径，按完整的路径分量匹配：名字相同但所在目录不同、或只是前缀相同的路径都不是链接"
    cdx(sol, "/srv/www", "current-backup/x", HOME, EX_LINKS, "/srv/www/current-backup/x", msg)
    cdx(sol, "/srv/www", "curr", HOME, EX_LINKS, "/srv/www/curr", msg)
    cdx(sol, "/opt", "current", HOME, EX_LINKS, "/opt/current", msg)
    cdx(sol, "/srv", "current/www", HOME, EX_LINKS, "/srv/current/www", msg)


@case("Part 3", title="绝对路径和 ~ 也经过链接")
def p3_absolute_and_home(sol):
    links = {"/srv/www/current": "/srv/archive/2025-09", "/home/priya/proj": "/srv/proj"}
    cdx(sol, "/tmp", "/srv/www/current/notes", HOME, links, "/srv/archive/2025-09/notes")
    cdx(sol, "/tmp", "~/proj/..", HOME, links, "/srv")
    cdx(sol, "/tmp", "~", HOME, links, "/home/priya")
    cdx(sol, "/tmp", "~/../priya/proj", HOME, links, "/srv/proj")
    call = calling(show_call("cd", "/tmp", "~bob", HOME, links))
    expect_raises(ValueError, sol.cd, "/tmp", "~bob", HOME, dict(links), call=call)


@case("Part 3", title="同一个链接走几次不是环")
def p3_repeated_link(sol):
    msg = "同一个链接在一次调用里被展开多次是合法的，只有总展开次数超过 20 才抛出 SymlinkLoopError"
    selfish = {"/srv/shared/self": "."}
    cdx(sol, "/srv/shared", "self/self/self/x", HOME, selfish, "/srv/shared/x", msg)
    cdx(sol, "/srv", "shared/self/../shared/self", HOME, selfish, "/srv/shared", msg)
    cdx(sol, "/srv/www", "current/../../www/current", HOME, EX_LINKS, "/srv/archive/2025-09", msg)


@case("Part 3", title="真正的环抛出 SymlinkLoopError")
def p3_cycle(sol):
    cdx(sol, "/opt/cycle", "a", HOME, {"/opt/cycle/a": "/opt/cycle/b", "/opt/cycle/b": "/opt/cycle/a"}, LOOP)
    cdx(sol, "/l", "a", HOME, {"/l/a": "a"}, LOOP)
    cdx(sol, "/l", "x/../a/y", HOME, {"/l/a": "./a"}, LOOP)
    cdx(sol, "/", "l/a", HOME, {"/l/a": "../l/a"}, LOOP)
    cdx(sol, "/", "c/a", HOME, {"/c/a": "b", "/c/b": "/c/c", "/c/c": "a"}, LOOP)
    cdx(sol, "/home/priya", "~/loop", HOME, {"/home/priya/loop": "loop/x"}, LOOP)


def chain(n):
    """/c/l1 -> l2 -> ... -> ln -> real: n expansions."""
    links = {f"/c/l{k}": f"l{k + 1}" for k in range(1, n)}
    links[f"/c/l{n}"] = "real"
    return links


@case("Part 3", title="上限：20 次可以，21 次抛出")
def p3_hop_cap(sol):
    selfish = {"/srv/shared/self": "."}
    cdx(sol, "/c", "l1", HOME, chain(20), "/c/real")
    cdx(sol, "/c", "l1", HOME, chain(21), LOOP)
    cdx(sol, "/srv/shared", "/".join(["self"] * 20), HOME, selfish, "/srv/shared")
    cdx(sol, "/srv/shared", "/".join(["self"] * 21), HOME, selfish, LOOP)
    two = {"/c/l1": "/c/l2", "/c/l2": "/c/real", "/c/k": "."}         # 2 + 18 = 20, then 2 + 19 = 21
    cdx(sol, "/c", "l1/../" + "k/" * 18 + "x", HOME, two, "/c/x")
    cdx(sol, "/c", "l1/../" + "k/" * 19 + "x", HOME, two, LOOP)


@case("Part 3", title="随机测试（对照参考实现）")
def p3_random(sol):
    r = rng()
    for _ in range(500):
        for _attempt in range(4):           # prefer inputs that go through at least one link
            cwd, destination, home, links = ref.random_cd_input(r)
            want, hops = ref.resolve(cwd, destination, home, links)
            if hops:
                break
        cdx(sol, cwd, destination, home, links, LOOP if want is None else want)
