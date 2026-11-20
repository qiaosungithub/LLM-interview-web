from judgelib import case, expect, rng, show_call
import ref

VOCAB = {"a": 1, "ab": 2, "abc": 3, "b": 4, "bcd": 5}


def check_tokenize(sol, text, vocab, unk):
    want = ref.tokenize(text, vocab, unk)
    expect(sol.tokenize(text, dict(vocab), unk), want, call=show_call("tokenize", text, vocab, unk))


def check_merged(sol, text, vocab, unk):
    want = ref.merge_unk(ref.tokenize(text, vocab, unk), unk)
    expect(sol.tokenize_merged(text, dict(vocab), unk), want, call=show_call("tokenize_merged", text, vocab, unk))


def check_trie(sol, texts, vocab, unk):
    tk = sol.TrieTokenizer(dict(vocab), unk)
    for text in texts:
        want = ref.merge_unk(ref.tokenize(text, vocab, unk), unk)
        expect(tk.tokenize(text), want, call=f"tk = TrieTokenizer({vocab!r:.300}, {unk}); tk.tokenize({text!r:.300})")


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 abcxbcdy")
def p1_example(sol):
    expect(sol.tokenize("abcxbcdy", dict(VOCAB), 0), [3, 0, 5, 0], call='tokenize("abcxbcdy", vocab, 0)')


@case("Part 1", sample=True, title="题目例子 abczzbcd（不合并）")
def p1_example_unmerged(sol):
    expect(sol.tokenize("abczzbcd", dict(VOCAB), 0), [3, 0, 0, 5], call='tokenize("abczzbcd", vocab, 0)')


@case("Part 1", title="空字符串")
def p1_empty_text(sol):
    check_tokenize(sol, "", VOCAB, 0)


@case("Part 1", title="空词表：每个字符都是 unk")
def p1_empty_vocab(sol):
    check_tokenize(sol, "xyz", {}, -1)


@case("Part 1", title="不同词条可以有相同 id")
def p1_shared_ids(sol):
    check_tokenize(sol, "abba", {"a": 7, "b": 7, "bb": 9}, 0)


@case("Part 1", title="贪心最长匹配（不是 token 最少）")
def p1_greedy(sol):
    check_tokenize(sol, "aaaaaa", {"aaaa": 1, "aaa": 2, "a": 3}, -1)


@case("Part 1", title="最长匹配要一直延伸到最长的键")
def p1_extend(sol):
    check_tokenize(sol, "abcdab", {"a": 1, "abcd": 2, "ab": 3}, 0)


@case("Part 1", title="非 ASCII 字符")
def p1_unicode(sol):
    check_tokenize(sol, "héllo世界!", {"hé": 1, "llo": 2, "世": 3, "界": 4}, 99)


@case("Part 1", title="随机测试（对照参考实现）")
def p1_random(sol):
    r = rng()
    for _ in range(150):
        alphabet = r.choice(["ab", "abc", "abcd"])
        vocab = ref.random_vocab(r, r.randint(1, 8), alphabet, 4)
        check_tokenize(sol, ref.random_text(r, vocab, alphabet + "z", r.randint(0, 40)), vocab, -1)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 abczzbcd")
def p2_example(sol):
    expect(sol.tokenize_merged("abczzbcd", dict(VOCAB), 0), [3, 0, 5], call='tokenize_merged("abczzbcd", vocab, 0)')


@case("Part 2", title="相邻的相同 id（不是 unk）不能合并")
def p2_repeated_real_id(sol):
    check_merged(sol, "aab", {"a": 1, "b": 1}, 0)


@case("Part 2", title="开头和结尾的 unk 连续段")
def p2_unk_edges(sol):
    check_merged(sol, "zzzabzz", VOCAB, 0)


@case("Part 2", title="空字符串和空词表")
def p2_empty(sol):
    check_merged(sol, "", VOCAB, 0)
    check_merged(sol, "xyz", {}, -1)


@case("Part 2", title="随机测试（对照参考实现）")
def p2_random(sol):
    r = rng()
    for _ in range(150):
        alphabet = r.choice(["ab", "abc", "abcd"])
        vocab = ref.random_vocab(r, r.randint(1, 8), alphabet, 4)
        check_merged(sol, ref.random_text(r, vocab, alphabet + "z", r.randint(0, 40)), vocab, -1)


@case("Part 2", cpu=3, title="长文本、短词条：每个位置至多试 L 种长度")
def p2_long_text(sol):
    r = rng()
    vocab = {"ab": 1, "abc": 2, "b": 3, "ca": 4}
    text = "".join(r.choice("abcxyz") for _ in range(100_000))
    want = ref.merge_unk(ref.tokenize(text, vocab, 0), 0)
    expect(sol.tokenize_merged(text, dict(vocab), 0), want, call=f"tokenize_merged(<100000 个字符的文本>, {vocab}, 0)")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：同一个对象调用两次")
def p3_example(sol):
    tk = sol.TrieTokenizer(dict(VOCAB), 0)
    expect(tk.tokenize("abcxbcdy"), [3, 0, 5, 0], call='tk = TrieTokenizer(vocab, 0); tk.tokenize("abcxbcdy")')
    expect(tk.tokenize("abczzbcd"), [3, 0, 5], call='tk.tokenize("abczzbcd")（同一个 tk）')


@case("Part 3", title="空字符串和空词表")
def p3_empty(sol):
    check_trie(sol, ["", "ab", ""], VOCAB, 0)
    check_trie(sol, ["xyz", ""], {}, -1)


@case("Part 3", title="匹配走到半路断掉时要退回到最后一个完整的键")
def p3_backtrack(sol):
    check_trie(sol, ["abcab", "abcdx", "ababc"], {"a": 1, "abcd": 2, "b": 3}, 0)


@case("Part 3", title="随机测试：一个对象处理多段文本")
def p3_random(sol):
    r = rng()
    for _ in range(60):
        alphabet = r.choice(["ab", "abc", "abcd"])
        vocab = ref.random_vocab(r, r.randint(1, 8), alphabet, 4)
        check_trie(sol, [ref.random_text(r, vocab, alphabet + "z", r.randint(0, 40)) for _ in range(4)], vocab, -1)


@case("Part 3", cpu=4, title="大词表复用：trie 只能在构造时建一次")
def p3_reuse_big_vocab(sol):
    r = rng()
    vocab = ref.random_vocab(r, 40_000, "abcdefgh", 7)
    texts = [ref.random_text(r, vocab, "abcdefghz", 20) for _ in range(3000)]
    tk = sol.TrieTokenizer(dict(vocab), -1)
    for k, text in enumerate(texts):
        if k % 300 == 0:            # spot-check correctness; the point of this case is speed
            want = ref.merge_unk(ref.tokenize(text, vocab, -1), -1)
            expect(tk.tokenize(text), want, call=f"tk.tokenize({text!r})（40000 个词条的词表）")
        else:
            tk.tokenize(text)


@case("Part 3", cpu=3, title="长文本：每个位置只沿 trie 往下走")
def p3_long_text(sol):
    r = rng()
    vocab = {"ab": 1, "abc": 2, "b": 3, "ca": 4}
    text = "".join(r.choice("abcxyz") for _ in range(100_000))
    tk = sol.TrieTokenizer(dict(vocab), 0)
    want = ref.merge_unk(ref.tokenize(text, vocab, 0), 0)
    expect(tk.tokenize(text), want, call=f"TrieTokenizer({vocab}, 0).tokenize(<100000 个字符的文本>)")
