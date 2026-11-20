#!/usr/bin/env python3
"""Build the study site's data from the topic folders under content/.

    .venv/bin/python build.py            # writes site/data/ and site/vendor/
    .venv/bin/python build.py --check    # parse and render only; exit 1 on any problem, write nothing

A topic folder holds README.zh.md, README.md and meta.yaml (and audit.json once the page has been audited).
Each page is split into statement and reference solution; behavioural rounds and quizzes are split further into
one card per question. Markdown is rendered to HTML here (tools/node/render.mjs), so a formula that does not
parse is reported by the build instead of appearing broken on a page.

Sources, in order:  content/schuture/<Company>/<category>/<slug>/   third-party problem bank, audited
                    content/extra/<Company>/<category>/<slug>/      problems written or imported for this site
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
SITE = ROOT / "site"
NODE_DIR = ROOT / "tools" / "node"

COMPANIES = [
    {"key": "anthropic", "dir": "Anthropic", "name": "Anthropic", "process": "PROCESS",
     "repo": "https://github.com/Schuture/Anthropic-Interview-Notes"},
    {"key": "openai", "dir": "OpenAI", "name": "OpenAI", "process": "INTERVIEW-PROCESS",
     "repo": "https://github.com/Schuture/OpenAI-Interview-Notes"},
    {"key": "deepmind", "dir": "Google-DeepMind", "name": "Google DeepMind", "process": "PROCESS",
     "repo": "https://github.com/Schuture/Google-DeepMind-Interview-Notes"},
]
COMPANY_BY_DIR = {c["dir"]: c for c in COMPANIES}
CATEGORIES = ["coding", "system-design", "take-home", "quiz", "discussion", "behavioral", "safety"]
LANGS = ["zh", "en"]
README = {"zh": "README.zh.md", "en": "README.md"}
STATEMENT_RE = {"zh": r"^## 题目[ \t]*$", "en": r"^## Problem[ \t]*$"}
SOLUTION_RE = {"zh": r"^## 参考解答[ \t]*$", "en": r"^## Reference solution[ \t]*$"}
FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
QUESTION_RE = re.compile(r"^(- )?\*\*Q(\d+)\.\*\*[ \t]*")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# question groups whose guidance sits under a differently named heading in the reference section
GUIDANCE_ALIAS = {"向经理提出的问题": "值得向用人经理提出的问题",
                  "Your questions for the manager": "Questions worth asking the hiring manager"}

problems: list[str] = []          # build problems; any entry makes --check fail


def warn(msg: str) -> None:
    problems.append(msg)


# ---------------------------------------------------------------- markdown splitting

def lines_outside_fences(text: str):
    """Yield (index, line, in_fence) so callers can ignore markdown syntax that sits inside a code fence."""
    fence = None
    for i, line in enumerate(text.split("\n")):
        m = FENCE_RE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * 3
            yield i, line, True
            continue
        if fence is not None:
            if line.startswith(fence) and line.strip(fence[0]).strip() == "":
                fence = None
            yield i, line, True
            continue
        yield i, line, False


def split_page(text: str, lang: str, where: str) -> tuple[str, str]:
    """Return (statement, solution) markdown. The solution loses its outer <details> wrapper."""
    m1 = re.search(STATEMENT_RE[lang], text, re.M)
    m2 = re.search(SOLUTION_RE[lang], text, re.M)
    if not m1 or not m2 or m2.start() < m1.end():
        warn(f"{where}: missing statement/solution headings")
        return text, ""
    statement = text[m1.end():m2.start()].strip("\n")
    solution = text[m2.end():].strip()
    m = re.match(r"<details>\s*<summary>.*?</summary>", solution, re.S)
    if m and solution.endswith("</details>"):
        solution = solution[m.end():-len("</details>")].strip("\n")
    else:
        warn(f"{where}: solution is not wrapped in <details>")
    return statement, solution


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split on '### ' headings outside fences: (text before the first heading, [(title, body), ...])."""
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for _, line, in_fence in lines_outside_fences(text):
        if not in_fence and line.startswith("### "):
            sections.append((line[4:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            intro.append(line)
    return "\n".join(intro).strip("\n"), [(t, "\n".join(b).strip("\n")) for t, b in sections]


def split_bullets(body: str) -> tuple[list[str], str]:
    """Top-level '- ' bullets become separate questions; anything else in the section is returned as a note."""
    questions: list[list[str]] = []
    note: list[str] = []
    current: list[str] | None = None
    for _, line, in_fence in lines_outside_fences(body):
        if not in_fence and line.startswith("- "):
            current = [line[2:]]
            questions.append(current)
        elif current is not None and (line.startswith("  ") or in_fence):
            current.append(line[2:] if line.startswith("  ") else line)
        elif current is not None and line.strip() == "":
            current.append("")
        else:
            current = None
            note.append(line)
    return ["\n".join(q).strip() for q in questions], "\n".join(note).strip("\n")


def split_numbered(text: str) -> tuple[str, list[dict], str]:
    """Split a quiz-style text into (intro, [{n, group, body}], trailing <details> block if any)."""
    intro: list[str] = []
    items: list[dict] = []
    tail: list[str] = []
    group = ""
    current: dict | None = None
    in_tail = False
    for _, line, in_fence in lines_outside_fences(text):
        if in_tail:
            tail.append(line)
            continue
        if not in_fence and line.startswith("<details>"):
            in_tail = True
            tail.append(line)
            continue
        if not in_fence and line.startswith("### "):
            group = line[4:].strip()
            current = None
            continue
        m = None if in_fence else QUESTION_RE.match(line)
        if m:
            # questions come either as paragraphs ("**Q1.** ...") or as list items ("- **Q1.** ..." with
            # continuation lines indented by two spaces)
            current = {"n": int(m.group(2)), "group": group, "lines": [line[m.end():]], "bullet": bool(m.group(1))}
            items.append(current)
        elif current is not None:
            current["lines"].append(line[2:] if current["bullet"] and line.startswith("  ") else line)
        elif not items and not group:
            intro.append(line)
        elif line.strip():
            # prose that follows a group heading before its first question: keep it with the next question's group
            intro.append(line)
    for it in items:
        it["body"] = "\n".join(it.pop("lines")).strip("\n")
        del it["bullet"]
    return "\n".join(intro).strip("\n"), items, "\n".join(tail).strip("\n")


# ---------------------------------------------------------------- links

def rewrite_links(text: str, company: dict, base: str, lang: str) -> str:
    """Point repository-relative links at site routes. `base` is the page's folder relative to the company root."""
    known = {c: True for c in CATEGORIES}

    def target(href: str) -> str:
        if re.match(r"^[a-z][a-z0-9+.-]*:", href) or href.startswith("#") or href.startswith("//"):
            return href
        path, _, frag = href.partition("#")
        full = posixpath.normpath(posixpath.join(base, path)) if path else ""
        parts = full.split("/") if full else []
        if len(parts) >= 2 and parts[0] in known and (len(parts) == 2 or re.fullmatch(r"README(\.zh)?\.md", parts[2])):
            return f"#/{company['key']}/{parts[0]}/{parts[1]}"
        if len(parts) == 1 and re.fullmatch(r"README(\.zh)?\.md", parts[0]):
            return f"#/{company['key']}"
        if len(parts) == 1 and re.fullmatch(r"(INTERVIEW-)?PROCESS(\.zh)?\.md", parts[0]):
            return f"#/{company['key']}/process"
        if full.startswith(".."):
            return href
        return f"{company['repo']}/blob/main/{full}" + (f"#{frag}" if frag else "")

    out = []
    for _, line, in_fence in lines_outside_fences(text):
        out.append(line if in_fence else LINK_RE.sub(lambda m: f"]({target(m.group(1))})", line))
    return "\n".join(out)


# ---------------------------------------------------------------- per-problem labels

STAGE_ZH = {"oa": "在线测评", "phone-screen": "电话面", "onsite": "现场面", "take-home": "Take-home",
            "recruiter-screen": "招聘初筛", "hm-screen": "用人经理面", "skills": "技能面", "final": "终面",
            "team-match": "团队匹配"}


def ai_rule(company: str, category: str, meta: dict, item_id: str, overrides: dict) -> dict:
    """Whether AI tools may be used in the round this problem comes from, and where the round is held.

    level: no | yes | ask | talk   (talk = a conversation round where the question does not arise)
    The per-company rules and their sources are spelled out in content/site/rules.<company>.zh.md; this function
    only picks the short label shown next to each problem. Overrides come from content/site/ai_overrides.yaml.
    """
    stage = meta.get("stage") or []
    if category == "safety":
        rule = {"level": "talk", "label": "复习材料", "env": "不是单独的面试轮次；内容会在文化面和各轮的交谈里被问到",
                "basis": "这是面试前自己复习用的材料，准备时可以随意用 AI；面试里是当面交谈，不涉及工具。"}
    elif category == "discussion":
        rule = {"level": "talk", "label": "对话轮", "env": "视频对话，口头讨论或角色扮演，一般不写代码",
                "basis": "这一轮是和面试官当面讨论，看的是你的思路和提问，不涉及工具；准备阶段可以随意用 AI。"}
    elif category == "behavioral":
        rule = {"level": "talk", "label": "对话轮", "env": "视频对话，不写代码",
                "basis": "这一轮是面对面交谈，不涉及工具；准备阶段可以随意用 AI。"}
    elif company == "anthropic":
        if "oa" in stage:
            rule = {"level": "no", "label": "禁用 AI", "env": "CodeSignal 在线测评：浏览器内写代码，自带单元测试，按关卡解锁",
                    "basis": "有监考的在线测评，不允许任何 AI 辅助；每一关的测试通过后才解锁下一关。"}
        elif category == "take-home":
            rule = {"level": "no", "label": "默认禁用 AI", "env": "在自己的机器上限时完成",
                    "basis": "官方说明：take-home 默认不使用 Claude，除非题目明确允许。"}
        elif category == "system-design":
            rule = {"level": "no", "label": "禁用 AI", "env": "视频会议 + 白板或共享文档，口头讨论，不运行代码",
                    "basis": "官方说明：现场面试全靠自己，除非另行告知。"}
        else:
            rule = {"level": "no", "label": "默认禁用 AI", "env": "视频会议 + 共享编辑器（CodeSignal、Colab 等），代码可以运行；通常不给测试，要自己写",
                    "basis": "官方说明：现场面试全靠自己，除非另行告知。"}
    elif company == "openai":
        if category == "system-design":
            rule = {"level": "no", "label": "禁用 AI", "env": "视频会议 + Excalidraw 之类的白板，口头讨论，不运行代码",
                    "basis": "候选人反馈一致：面试中不允许使用 AI 助手。"}
        else:
            rule = {"level": "no", "label": "禁用 AI", "env": "CoderPad + Google Meet：代码可以运行；题目分段放出，测试有时给、有时要自己写",
                    "basis": "候选人反馈：不允许 AI 编程助手，至少有一轮还要求关掉编辑器的自动补全。"}
    else:
        if category == "quiz":
            rule = {"level": "no", "label": "禁用 AI", "env": "视频通话，口头回答，必要时在纸上或共享文档里写推导",
                    "basis": "官方面试指南：除非另行告知，实时面试中不要使用任何 AI 工具。"}
        elif category == "system-design":
            rule = {"level": "no", "label": "禁用 AI", "env": "视频通话 + 共享文档或白板，口头讨论，不运行代码",
                    "basis": "官方面试指南：除非另行告知，实时面试中不要使用任何 AI 工具。"}
        elif category == "take-home":
            rule = {"level": "no", "label": "默认禁用 AI", "env": "在自己的机器上限时完成；平台、时长和能不能查资料以招聘联系人的说明为准",
                    "basis": "官方面试指南：除非另行告知，面试任务（interview tasks）里也不要使用任何 AI 工具。"}
        else:
            rule = {"level": "no", "label": "默认禁用 AI", "env": "视频通话 + 能运行代码的共享编辑器（如 CoderPad），有时要通过面试官准备的测试用例",
                    "basis": "官方面试指南：除非另行告知，实时面试和面试任务中不要使用任何 AI 工具。"}
    rule = dict(rule)
    rule.update(overrides.get(item_id, {}))
    return rule


def answer_parts(category: str) -> list[dict]:
    if category in ("coding", "take-home"):
        return [{"key": "answer", "file": "answer.py", "label": "代码", "mode": "python"},
                {"key": "notes", "file": "notes.md", "label": "笔记", "mode": "markdown"}]
    if category == "system-design":
        return [{"key": "answer", "file": "answer.md", "label": "我的设计", "mode": "markdown"}]
    return []       # question-by-question rounds: q01.md, q02.md, ... (one file per question)


# ---------------------------------------------------------------- judge folders

JUDGE_KEYS = {"timeout", "case_cpu", "load_cpu", "memory_gb", "nproc", "special", "note"}


def judge_cases(tests: Path, item_id: str) -> list[dict]:
    """The @case(...) functions of judge/tests.py, read without running it: [{name, part, sample}]."""
    try:
        tree = ast.parse(tests.read_text(encoding="utf-8"))
    except SyntaxError as e:
        warn(f"{item_id}: judge/tests.py does not parse ({e})")
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "case":
                if not d.args or not isinstance(d.args[0], ast.Constant) or not isinstance(d.args[0].value, str):
                    warn(f"{item_id}: judge/tests.py {node.name}: the part name must be a string literal")
                    continue
                sample = any(k.arg == "sample" and isinstance(k.value, ast.Constant) and k.value.value is True
                             for k in d.keywords)
                out.append({"name": node.name, "part": d.args[0].value, "sample": sample, "line": node.lineno})
    return sorted(out, key=lambda c: c["line"])


def judge_info(folder: Path, item_id: str) -> tuple[dict | None, str]:
    """What the site needs to know about a problem's judge/ folder, and its starter code."""
    jdir = folder / "judge"
    if not jdir.is_dir():
        return None, ""
    try:
        cfg = yaml.safe_load((jdir / "judge.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        warn(f"{item_id}: judge/judge.yaml is missing or does not parse ({e})")
        cfg = {}
    for key in set(cfg) - JUDGE_KEYS:
        warn(f"{item_id}: judge/judge.yaml has an unknown key '{key}'")
    if not isinstance(cfg.get("timeout", 20), (int, float)) or not 1 <= cfg.get("timeout", 20) <= 120:
        warn(f"{item_id}: judge/judge.yaml timeout must be 1..120 seconds")
    starter = ""
    if (jdir / "starter.py").is_file():
        starter = (jdir / "starter.py").read_text(encoding="utf-8")
        try:
            compile(starter, f"{item_id}/judge/starter.py", "exec")
        except SyntaxError as e:
            warn(f"{item_id}: judge/starter.py does not compile ({e})")
    else:
        warn(f"{item_id}: judge/ has no starter.py")
    cases = judge_cases(jdir / "tests.py", item_id) if (jdir / "tests.py").is_file() else []
    if not cases:
        warn(f"{item_id}: judge/tests.py is missing or defines no @case")
    elif not any(c["sample"] for c in cases):
        warn(f"{item_id}: judge/tests.py marks no case as sample=True")
    if not (jdir / "solution.py").is_file():
        warn(f"{item_id}: judge/ has no solution.py (the reference answer judge_check.py runs)")
    wrong = sorted((jdir / "wrong").glob("*.py")) if (jdir / "wrong").is_dir() else []
    if len(wrong) < 2:
        warn(f"{item_id}: judge/wrong/ needs at least two wrong solutions, has {len(wrong)}")
    parts: dict[str, dict] = {}
    for c in cases:
        part = parts.setdefault(c["part"], {"name": c["part"], "total": 0, "samples": 0})
        part["total"] += 1
        part["samples"] += c["sample"]
    return {"dir": str(jdir.relative_to(ROOT)), "parts": list(parts.values()), "total": len(cases),
            "special": cfg.get("special"), "note": cfg.get("note"), "timeout": cfg.get("timeout", 20)}, starter


# ---------------------------------------------------------------- collecting

class Fragments:
    """Markdown fragments waiting to be rendered; `add` returns the key to look the HTML up under afterwards."""

    def __init__(self) -> None:
        self.md: dict[str, str] = {}

    def add(self, key: str, text: str) -> str:
        self.md[key] = text
        return key


def topic_dirs() -> list[tuple[str, Path, dict]]:
    out = []
    for source in ("schuture", "extra"):
        base = CONTENT / source
        if not base.is_dir():
            continue
        for meta_path in sorted(base.rglob("meta.yaml")):
            folder = meta_path.parent
            rel = folder.relative_to(base).parts
            if len(rel) != 3 or rel[0] not in COMPANY_BY_DIR or rel[1] not in CATEGORIES:
                warn(f"{folder}: unexpected location")
                continue
            if not (folder / README["zh"]).exists():
                # a topic still being written must not show up on the site as an empty page
                warn(f"{source}/{'/'.join(rel)}: no Chinese page yet, left out of the site")
                continue
            out.append((source, folder, COMPANY_BY_DIR[rel[0]]))
    return out


def load_topic(source: str, folder: Path, company: dict, frags: Fragments, overrides: dict) -> tuple[dict, dict]:
    category, slug = folder.parts[-2], folder.parts[-1]
    item_id = f"{company['key']}/{category}/{slug}"
    meta = yaml.safe_load((folder / "meta.yaml").read_text(encoding="utf-8")) or {}
    audit = None
    if (folder / "audit.json").exists():
        try:
            audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            warn(f"{item_id}: audit.json does not parse ({e})")
    qa = category in ("behavioral", "quiz", "discussion", "safety")
    entry = {
        "id": item_id, "company": company["key"], "category": category, "slug": slug,
        "mode": "qa" if qa else "problem", "source": source,
        "title": {"zh": meta.get("title_zh") or meta.get("title"), "en": meta.get("title")},
        "summary": {"zh": meta.get("summary_zh") or meta.get("summary") or "", "en": meta.get("summary") or ""},
        "kind": {"zh": meta.get("kind_zh") or meta.get("kind") or "", "en": meta.get("kind") or ""},
        "format": {"zh": meta.get("format_zh") or meta.get("format") or "", "en": meta.get("format") or ""},
        "roles": meta.get("roles") or [], "topics": meta.get("topics") or [],
        "difficulty": meta.get("difficulty"), "frequency": meta.get("frequency"),
        "stage": meta.get("stage") or [], "order": meta.get("order") or 999,
        "origin": meta.get("origin") or "", "added": bool(meta.get("added")),
        "ai": ai_rule(company["key"], category, meta, item_id, overrides),
        "parts": answer_parts(category),
        "judge": None,
        "audit": ({"verdict": audit.get("verdict", "ok"), "summary": audit.get("summary_zh", "")} if audit else None),
    }
    detail: dict = {"id": item_id, "audit": audit, "langs": {}, "starter": ""}
    if category in ("coding", "take-home"):
        entry["judge"], detail["starter"] = judge_info(folder, item_id)
    elif (folder / "judge").exists():
        warn(f"{item_id}: only coding and take-home problems can have a judge/ folder")
    counts = {}
    for lang in LANGS:
        page = folder / README[lang]
        if not page.exists():
            if lang == "zh":
                warn(f"{item_id}: no Chinese page")
            continue
        where = f"{item_id} [{lang}]"
        statement, solution = split_page(page.read_text(encoding="utf-8"), lang, where)
        base = f"{category}/{slug}"
        statement = rewrite_links(statement, company, base, lang)
        solution = rewrite_links(solution, company, base, lang)
        key = f"{item_id}|{lang}|"
        if not qa:
            detail["langs"][lang] = {"statement": frags.add(key + "statement", statement),
                                     "solution": frags.add(key + "solution", solution)}
            continue
        if category == "behavioral":
            intro, sections = split_sections(statement)
            sol_intro, sol_sections = split_sections(solution)
            guidance = dict(sol_sections)
            for title, alias in GUIDANCE_ALIAS.items():
                if title not in guidance and alias in guidance:
                    guidance[title] = guidance[alias]
            titles = [t for t, _ in sections] + [GUIDANCE_ALIAS.get(t) for t, _ in sections]
            groups, n = [], 0
            for gi, (title, body) in enumerate(sections):
                questions, note = split_bullets(body)
                if not questions:
                    warn(f"{where}: group '{title}' has no questions")
                if title not in guidance:
                    warn(f"{where}: group '{title}' has no guidance in the reference section")
                qs = []
                for q in questions:
                    n += 1
                    qs.append({"n": n, "q": frags.add(f"{key}q{n}", q)})
                groups.append({"title": title, "note": frags.add(f"{key}note{gi}", note) if note else None,
                               "guidance": frags.add(f"{key}guide{gi}", guidance.get(title, "")), "questions": qs})
            extras = [{"title": t, "html": frags.add(f"{key}extra{i}", b)}
                      for i, (t, b) in enumerate(sol_sections) if t not in titles]
            detail["langs"][lang] = {"intro": frags.add(key + "intro", intro),
                                     "solutionIntro": frags.add(key + "solintro", sol_intro),
                                     "groups": groups, "extras": extras, "checks": None}
            counts[lang] = [len(g["questions"]) for g in groups]
        else:
            intro, questions, _ = split_numbered(statement)
            sol_intro, answers, checks = split_numbered(solution)
            by_n = {a["n"]: a for a in answers}
            if [q["n"] for q in questions] != list(range(1, len(questions) + 1)):
                warn(f"{where}: question numbers are not 1..N: {[q['n'] for q in questions]}")
            if sorted(by_n) != [q["n"] for q in questions]:
                warn(f"{where}: answers {sorted(by_n)} do not match questions {[q['n'] for q in questions]}")
            groups = []
            for q in questions:
                if not groups or groups[-1]["title"] != q["group"]:
                    groups.append({"title": q["group"], "note": None, "guidance": None, "questions": []})
                a = by_n.get(q["n"])
                groups[-1]["questions"].append({
                    "n": q["n"], "q": frags.add(f"{key}q{q['n']}", q["body"]),
                    "a": frags.add(f"{key}a{q['n']}", a["body"]) if a else None})
            detail["langs"][lang] = {"intro": frags.add(key + "intro", intro),
                                     "solutionIntro": frags.add(key + "solintro", sol_intro),
                                     "groups": groups, "extras": [],
                                     "checks": frags.add(key + "checks", checks) if checks else None}
            counts[lang] = [len(g["questions"]) for g in groups]
    if qa:
        if len({json.dumps(v) for v in counts.values()}) > 1:
            warn(f"{item_id}: Chinese and English pages split into different questions: {counts}")
        entry["nq"] = sum(counts.get("zh") or next(iter(counts.values()), []))
    return entry, detail


def audit_fragments(detail: dict, frags: Fragments) -> None:
    """Queue the audit's Chinese notes for rendering: they use backticks and formulas like the pages do."""
    audit = detail["audit"]
    if not audit:
        return
    key = f"audit|{detail['id']}|"
    if audit.get("summary_zh"):
        audit["summary_html"] = frags.add(key + "summary", audit["summary_zh"])
    for i, issue in enumerate(audit.get("issues") or []):
        for field in ("problem", "change"):
            if issue.get(field + "_zh"):
                issue[field + "_html"] = frags.add(f"{key}{i}|{field}", issue[field + "_zh"])


def audit_html(detail: dict, html: dict) -> dict | None:
    """The audit record under the site id, with the queued notes replaced by HTML (a lone paragraph loses its <p>)."""
    audit = detail["audit"]
    if not audit:
        return None

    def one(key: str) -> str:
        h = html[key].strip()
        return h[3:-4] if h.startswith("<p>") and h.endswith("</p>") and h.count("<p>") == 1 else h

    out = {**audit, "id": detail["id"]}
    if "summary_html" in out:
        out["summary_html"] = one(out["summary_html"])
    out["issues"] = [{**issue, **{f: one(issue[f]) for f in ("problem_html", "change_html") if f in issue}}
                     for issue in audit.get("issues") or []]
    return out


def resolve(obj, html: dict):
    """Replace fragment keys by rendered HTML throughout a detail structure."""
    if isinstance(obj, dict):
        return {k: (html[v] if k in ("statement", "solution", "intro", "solutionIntro", "note", "guidance", "q", "a",
                                     "html", "checks") and isinstance(v, str) else resolve(v, html))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, html) for v in obj]
    return obj


def render(md: dict[str, str]) -> tuple[dict, list]:
    with tempfile.TemporaryDirectory(prefix="build_") as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(md, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ)
        env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
        proc = subprocess.run(["node", str(NODE_DIR / "render.mjs"), str(src), str(dst)], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"render failed:\n{proc.stdout}\n{proc.stderr}")
        out = json.loads(dst.read_text(encoding="utf-8"))
    return out["html"], out["errors"]


VENDOR = [   # (path under tools/node/node_modules, path under site/vendor)
    ("katex/dist/katex.min.css", "katex/katex.min.css"),
    ("katex/dist/katex.min.js", "katex/katex.min.js"),
    ("katex/dist/fonts", "katex/fonts"),
    ("markdown-it/dist/markdown-it.min.js", "markdown-it.min.js"),
    ("@highlightjs/cdn-assets/highlight.min.js", "highlight.min.js"),
    ("mermaid/dist/mermaid.min.js", "mermaid.min.js"),
    ("codemirror/lib/codemirror.js", "codemirror/codemirror.js"),
    ("codemirror/lib/codemirror.css", "codemirror/codemirror.css"),
    ("codemirror/mode/python/python.js", "codemirror/python.js"),
    ("codemirror/mode/markdown/markdown.js", "codemirror/markdown.js"),
    ("codemirror/mode/xml/xml.js", "codemirror/xml.js"),
    ("codemirror/addon/edit/matchbrackets.js", "codemirror/matchbrackets.js"),
    ("codemirror/addon/edit/closebrackets.js", "codemirror/closebrackets.js"),
    ("codemirror/addon/comment/comment.js", "codemirror/comment.js"),
    ("codemirror/addon/selection/active-line.js", "codemirror/active-line.js"),
    ("codemirror/addon/display/placeholder.js", "codemirror/placeholder.js"),
]


def copy_vendor() -> None:
    for src_rel, dst_rel in VENDOR:
        src, dst = NODE_DIR / "node_modules" / src_rel, SITE / "vendor" / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="parse and render, report problems, write nothing")
    args = ap.parse_args()

    site_cfg = CONTENT / "site"
    overrides_path = site_cfg / "ai_overrides.yaml"
    overrides = (yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}) if overrides_path.exists() else {}

    frags = Fragments()
    entries, details = [], []
    for source, folder, company in topic_dirs():
        entry, detail = load_topic(source, folder, company, frags, overrides)
        audit_fragments(detail, frags)
        entries.append(entry)
        details.append(detail)
    ids = [e["id"] for e in entries]
    for dup in {i for i in ids if ids.count(i) > 1}:
        warn(f"{dup}: defined twice (content/schuture and content/extra)")
    for item_id in overrides:
        if item_id not in ids:
            warn(f"ai_overrides.yaml: unknown problem {item_id}")

    # per-company pages: the interview process (from the problem bank) and the AI/environment rules (written here)
    pages = {}
    for c in COMPANIES:
        for lang in LANGS:
            suffix = ".zh.md" if lang == "zh" else ".md"
            p = CONTENT / "schuture" / c["dir"] / (c["process"] + suffix)
            if p.exists():
                text = re.sub(r"^\[English\]\(.*?\) · \[中文\]\(.*?\)\s*$", "", p.read_text(encoding="utf-8"), flags=re.M)
                pages[f"{c['key']}.process.{lang}"] = frags.add(f"page|{c['key']}|process|{lang}",
                                                              rewrite_links(text, c, "", lang))
        for name in ("rules", "selection"):
            p = site_cfg / f"{name}.{c['key']}.zh.md"
            p_en = site_cfg / f"{name}.{c['key']}.md"
            if p.exists() and p_en.exists():
                # both languages: published as <key>.<name>.zh / .en, which the front end picks by language
                for lang, src in (("zh", p), ("en", p_en)):
                    pages[f"{c['key']}.{name}.{lang}"] = frags.add(f"page|{c['key']}|{name}|{lang}",
                                                                  src.read_text(encoding="utf-8"))
            elif p.exists():
                pages[f"{c['key']}.{name}"] = frags.add(f"page|{c['key']}|{name}", p.read_text(encoding="utf-8"))
    cards_path = site_cfg / "companies.yaml"
    company_cfg = (yaml.safe_load(cards_path.read_text(encoding="utf-8")) or {}) if cards_path.exists() else {}
    for ckey, cfg in company_cfg.items():
        if ckey not in {c["key"] for c in COMPANIES}:
            warn(f"companies.yaml: unknown company {ckey}")
        for i, card in enumerate(cfg.get("cards") or []):
            card["html"] = frags.add(f"card|{ckey}|{i}", card.pop("text"))
            if card.get("text_en"):
                card["html_en"] = frags.add(f"card|{ckey}|{i}|en", card.pop("text_en"))
    for p in sorted(site_cfg.glob("*.zh.md")) if site_cfg.is_dir() else []:
        name = p.name[:-len(".zh.md")]
        if "." not in name:
            pages[name] = frags.add(f"page|{name}", p.read_text(encoding="utf-8"))

    html, errors = render(frags.md)
    for e in errors:
        warn(f"{e['key']}: {e['kind']}: {e['message']}  <<{e['source'][:120]}>>")

    print(f"{len(entries)} topics, {len(frags.md)} fragments, {len(problems)} problem(s)")
    for p in problems:
        print("  PROBLEM", p)
    if args.check:
        return 1 if problems else 0

    data = SITE / "data"
    staging = SITE / "data.new"
    if staging.exists():
        shutil.rmtree(staging)
    for entry, detail in zip(entries, details):
        for lang, body in detail["langs"].items():
            write_json(staging / "items" / f"{entry['id']}.{lang}.json",
                       {"id": entry["id"], "lang": lang, "audit": audit_html(detail, html), "starter": detail["starter"],
                        **resolve(body, html)})
        entry["langs"] = sorted(detail["langs"])
    for name, key in pages.items():
        write_json(staging / "pages" / f"{name}.json", {"html": html[key]})
    audits = [audit_html(d, html) for d in details if d["audit"]]
    write_json(staging / "audit.json", audits)
    write_json(staging / "index.json", {
        "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "companies": [{"key": c["key"], "name": c["name"], "repo": c["repo"],
                       "tagline": company_cfg.get(c["key"], {}).get("tagline", ""),
                       "tagline_en": company_cfg.get(c["key"], {}).get("tagline_en", ""),
                       "cards": [{"title": k["title"], "html": html[k["html"]],
                                  "title_en": k.get("title_en", ""),
                                  "html_en": html[k["html_en"]] if k.get("html_en") else ""}
                                 for k in company_cfg.get(c["key"], {}).get("cards") or []]} for c in COMPANIES],
        "categories": CATEGORIES, "stageLabels": STAGE_ZH, "pages": sorted(pages),
        "items": entries, "problems": problems,
    })
    old = SITE / "data.old"
    if old.exists():
        shutil.rmtree(old)
    if data.exists():
        data.rename(old)
    staging.rename(data)
    if old.exists():
        shutil.rmtree(old)
    copy_vendor()
    print(f"wrote {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
