// Interview-prep site: a small hash-routed app over the JSON written by build.py and the answers API of server.py.
//
// Routes   #/<company>                         problem lists for one company
//          #/<company>/<category>              same page, scrolled to that section
//          #/<company>/process|rules|selection long-form pages
//          #/<company>/<category>/<slug>       a problem (split view) or a question round (overview)
//          #/<company>/<category>/<slug>/q/<n> one question of a round
//          #/audit   #/answers                 what was checked and changed; where the saved answers are
import { createRenderer } from "./md.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const app = $("#app");

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem("prep." + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem("prep." + key, JSON.stringify(value)); } catch { /* private mode */ } },
  del(key) { try { localStorage.removeItem("prep." + key); } catch { /* private mode */ } },
};

const CATS = {
  coding: { name: "编程", lead: "要动手写代码的题：在线测评、电话面和现场编程轮。答案存成 answer.py，可以直接在本机运行。" },
  "system-design": { name: "系统设计", lead: "以口头讨论为主的设计题。把你的设计写成文字提纲，存成 answer.md。" },
  "take-home": { name: "Take-home", lead: "带回去限时完成的作业题。" },
  quiz: { name: "知识测验", lead: "口头问答的知识题，已拆成单题，一题一答，答完再看参考答案。" },
  discussion: { name: "讨论与情景", lead: "没有标准答案的口头轮：研究讨论、排查故障的角色扮演、开放问题。已拆成一步一步的问题，先写下你会怎么说，再看参考思路。" },
  behavioral: { name: "行为与文化", lead: "行为面和文化面，已拆成单个问题。逐题写下你的回答，再对照这一组问题的回答思路。" },
  safety: { name: "安全理念复习", lead: "Anthropic 的安全理念要点，配自测问题。" },
};
const BANDS = [
  { title: "技术题", note: "写代码、做设计、答知识题", cats: ["coding", "system-design", "take-home", "quiz", "discussion"] },
  { title: "文科", note: "行为、文化与价值观", cats: ["behavioral", "safety"] },
];
const PAGES = { process: "面试流程", rules: "AI 规则与面试环境", selection: "选题说明" };
const FREQ = { "very-high": 5, high: 4, medium: 3, low: 2, rare: 1 };
const FREQ_ZH = { "very-high": "极高", high: "高", medium: "中等", low: "低", rare: "罕见" };
const DIFF_ZH = { easy: "简单", medium: "中等", hard: "困难" };
const STATUS = { "": "未标记", done: "已完成", review: "待复习", mastered: "已掌握" };
const VERDICT = { ok: ["没发现问题", ""], fixed: ["已修正", "fixed"], relaxed: ["已放宽要求", "fixed"], dropped: ["有硬伤", "dropped"] };
const ACTION = { fixed: "已修正", relaxed: "已放宽", removed: "已删除", flag: "存疑，未改" };

const S = {
  index: null,
  byId: new Map(),
  state: { answers: {}, status: {}, reviews: {}, judge: {}, root: "" },
  offline: false,
  lang: store.get("lang", "en"),
  nav: 0,                 // incremented on every navigation, so a slow view can tell it has been superseded
  view: "",               // which page is on screen (lets a section link scroll instead of re-rendering)
  hash: "",
  scroll: new Map(),      // hash -> window scroll position, restored when coming back to a list
  filters: {},
  tabs: {},
  timers: {},
  editors: new Set(),
  cleanup: [],
  revealed: new Set(),
  judgeMem: {},           // per problem: the last test result and custom run shown in the judge panel
};

// ------------------------------------------------------------------ data access

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function api(path, body) {
  const opts = body === undefined ? { cache: "no-store" }
    : { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body.data), keepalive: !!body.keepalive };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function refreshState() {
  try {
    S.state = await api("/api/state");
    S.offline = false;
  } catch {
    S.offline = true;
  }
  $("#offline")?.remove();
  if (S.offline) {
    $("#topbar").insertAdjacentHTML("afterend",
      `<div class="banner" id="offline">答案服务没有连上：现在写的内容只会暂存在这个浏览器里。请确认是用 <code>server.py</code> 打开的网站。</div>`);
  }
}

async function loadItem(it) {
  const lang = it.langs.includes(S.lang) ? S.lang : it.langs.includes("zh") ? "zh" : it.langs[0];
  return getJSON(`data/items/${it.id}.${lang}.json`);
}

const company = (key) => S.index.companies.find((c) => c.key === key);
const title = (it) => it.title[S.lang] || it.title.zh || it.title.en;
const partOf = (n) => "q" + String(n).padStart(2, "0");
// Review notes are read in a fixed order; everything else is listed most-asked first.
const byPriority = (a, b) => (a.category === "safety" && b.category === "safety" ? 0 : (FREQ[b.frequency] || 0) - (FREQ[a.frequency] || 0)) || a.order - b.order || a.slug.localeCompare(b.slug);
const itemsOf = (ckey, cat) => S.index.items.filter((it) => it.company === ckey && (!cat || it.category === cat)).sort(byPriority);

function progress(it) {
  const answers = S.state.answers[it.id] || {};
  const count = Object.keys(answers).length;
  if (it.mode === "qa") {
    if (count >= it.nq && it.nq) return { cls: "done", mark: "✓", text: "全部答过", started: true, finished: true, count };
    if (count) return { cls: "draft", mark: "", text: `答了 ${count} / ${it.nq}`, started: true, finished: false, count };
    return { cls: "", mark: "", text: "还没做", started: false, finished: false, count };
  }
  const status = S.state.status[it.id]?.status || "";
  if (status === "mastered") return { cls: "mastered", mark: "✓", text: "已掌握", started: true, finished: true, status };
  if (status === "done") return { cls: "done", mark: "✓", text: "已完成", started: true, finished: true, status };
  if (status === "review") return { cls: "review", mark: "!", text: "待复习", started: true, finished: false, status };
  if (count) return { cls: "draft", mark: "", text: "写了一部分", started: true, finished: false, status };
  return { cls: "", mark: "", text: "还没做", started: false, finished: false, status };
}

function noteAnswer(id, part, mtime, content) {
  const forItem = (S.state.answers[id] ||= {});
  if (content.trim()) forItem[part] = { mtime, size: content.length };
  else delete forItem[part];
  if (!Object.keys(forItem).length) delete S.state.answers[id];
}

async function setStatus(key, value) {
  try {
    await api("/api/status", { data: { key, value } });
    if (value) S.state.status[key] = { status: value, updated: Math.floor(Date.now() / 1000) };
    else delete S.state.status[key];
  } catch {
    toast("状态没有存上，请检查答案服务是否在运行");
  }
}

// ------------------------------------------------------------------ small pieces of markup

function freqBars(freq) {
  const n = FREQ[freq] || 0;
  if (!n) return "";
  const bars = [1, 2, 3, 4, 5].map((i) => `<i class="${i <= n ? "on" : ""}"></i>`).join("");
  return `<span class="freq" title="被问到的频率：${FREQ_ZH[freq]}">${bars}</span>`;
}

const aiTag = (ai) => `<span class="tag ${ai.level === "talk" ? "" : esc(ai.level)}" title="${esc(ai.basis)}">${esc(ai.label)}</span>`;

function auditTag(it) {
  const verdict = it.audit?.verdict;
  if (!verdict || verdict === "ok") return "";
  const [label, cls] = VERDICT[verdict] || [verdict, ""];
  return `<span class="tag ${cls}" title="${esc(it.audit.summary)}">${label}</span>`;
}

const stages = (it) => it.stage.map((s) => S.index.stageLabels[s] || s).join(" · ");
const when = (mtime) => new Date(mtime * 1000).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
const clock = () => new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });

function toast(text) {
  $(".toast")?.remove();
  document.body.insertAdjacentHTML("beforeend", `<div class="toast">${esc(text)}</div>`);
  setTimeout(() => $(".toast")?.remove(), 4000);
}

function loadScript(src) {
  loadScript.cache ||= new Map();
  if (!loadScript.cache.has(src)) {
    loadScript.cache.set(src, new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src;
      el.onload = resolve;
      el.onerror = () => reject(new Error("cannot load " + src));
      document.head.append(el);
    }));
  }
  return loadScript.cache.get(src);
}

let rendererPromise = null;
function mdRenderer() {      // only needed for text that is not pre-rendered by the build (grading notes)
  rendererPromise ||= Promise.all([loadScript("vendor/markdown-it.min.js"), loadScript("vendor/katex/katex.min.js"),
    loadScript("vendor/highlight.min.js")]).then(() => createRenderer({ MarkdownIt: window.markdownit, katex: window.katex, hljs: window.hljs }));
  return rendererPromise;
}

async function renderMermaid(root) {
  const blocks = $$("pre.mermaid-src, .mermaid-svg", root);
  if (!blocks.length) return;
  try {
    await loadScript("vendor/mermaid.min.js");
  } catch {
    return;                                   // the diagram source stays readable as text
  }
  const dark = document.documentElement.dataset.theme === "dark";
  window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "neutral", fontFamily: "inherit" });
  for (const el of blocks) {
    const source = el.dataset.src || el.textContent;
    try {
      const { svg } = await window.mermaid.render("mm" + Math.random().toString(36).slice(2), source);
      const box = document.createElement("div");
      box.className = "mermaid-svg";
      box.dataset.src = source;
      box.innerHTML = svg;
      el.replaceWith(box);
    } catch {
      $$("body > [id^='dmm'], body > svg[id^='mm']").forEach((x) => x.remove());   // mermaid leaves its error box behind
    }
  }
}

function enhance(root) {      // things rendered HTML needs once it is in the page
  for (const pre of $$("pre.code", root)) {
    if (pre.querySelector(".copy")) continue;
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.textContent = "复制";
    btn.addEventListener("click", async () => {
      const text = pre.querySelector("code").textContent;
      try { await navigator.clipboard.writeText(text); btn.textContent = "已复制"; } catch { btn.textContent = "复制失败"; }
      setTimeout(() => { btn.textContent = "复制"; }, 1500);
    });
    pre.append(btn);
  }
  renderMermaid(root);
}

// ------------------------------------------------------------------ answer editor (one file on disk)

const draftKey = (key) => "draft:" + key;

class Editor {
  // onState(kind, text): kind is saved | saving | unsaved
  // starter: shown while nothing is saved yet, and only written to disk once it is edited
  constructor(host, { id, part, mode, placeholder, grow, onState, starter, keys }) {
    this.id = id;
    this.part = part;
    this.key = `${id}#${part}`;
    this.onState = onState || (() => {});
    this.starter = starter || "";
    this.keep = false;          // the next save also keeps the replaced version in answers/.history/
    this.saved = "";
    this.timer = null;
    this.busy = false;
    this.again = false;
    const code = mode === "python";
    const indent = (cm) => (cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection(" ".repeat(cm.getOption("indentUnit")), "end"));
    this.cm = window.CodeMirror(host, {
      value: "", mode: code ? "python" : "markdown", lineNumbers: code, lineWrapping: !code, indentUnit: code ? 4 : 2, tabSize: 4,
      indentWithTabs: false, matchBrackets: code, autoCloseBrackets: code, styleActiveLine: code, placeholder: placeholder || "",
      viewportMargin: grow ? Infinity : 10,
      extraKeys: { Tab: indent, "Shift-Tab": "indentLess", "Ctrl-/": "toggleComment", "Cmd-/": "toggleComment",
        "Ctrl-S": () => this.save(), "Cmd-S": () => this.save(), ...(keys || {}) },
    });
    if (!code) this.cm.getWrapperElement().classList.add("prose");
    this.cm.on("change", (cm, change) => { if (change.origin !== "setValue") this.touch(); });
    S.editors.add(this);
  }

  async load() {
    let text = "";
    try {
      text = (await api(`/api/answer?id=${encodeURIComponent(this.id)}&part=${this.part}`)).content;
    } catch { /* offline: fall back to the browser draft below */ }
    this.saved = text;
    const draft = store.get(draftKey(this.key), null);       // exists only if an earlier save did not reach the disk
    const useDraft = draft !== null && draft !== text;
    const starter = !useDraft && !text && this.starter;
    this.cm.setValue(useDraft ? draft : starter ? this.starter : text);
    this.cm.clearHistory();
    if (useDraft) this.touch();
    else { store.del(draftKey(this.key)); this.onState("saved", text ? "已保存到磁盘" : starter ? "这是初始代码，改动后会自动保存" : "写下的内容会自动保存"); }
  }

  unchanged(content) {     // nothing new to write: the saved text, or the untouched starter of an empty answer
    return content === this.saved || (!this.saved && content === this.starter);
  }

  touch() {
    this.onState("unsaved", "正在输入…");
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.save(), 700);
  }

  async save(unloading = false) {
    clearTimeout(this.timer);
    this.timer = null;
    if (this.busy) { this.again = true; return; }
    const content = this.cm.getValue();
    if (this.unchanged(content)) { this.onState("saved", this.saved ? "已保存到磁盘" : "这是初始代码，改动后会自动保存"); return; }
    this.busy = true;
    this.onState("saving", "保存中…");
    try {
      const keepalive = unloading && content.length < 20000;      // browsers cap keepalive bodies at 64 KB
      const r = await api("/api/answer", { data: { id: this.id, part: this.part, content, keep: this.keep }, keepalive });
      this.saved = content;
      this.keep = false;
      store.del(draftKey(this.key));
      noteAnswer(this.id, this.part, r.mtime, content);
      this.onState("saved", `已保存到磁盘 ${clock()}`);
    } catch {
      store.set(draftKey(this.key), content);
      this.onState("unsaved", "没存到磁盘（已暂存在浏览器里，会自动重试）");
      if (!this.closed) this.timer = setTimeout(() => this.save(), 5000);
    } finally {
      this.busy = false;
      if (this.again) { this.again = false; this.save(); }
    }
  }

  flush(unloading = false) {
    if (!this.unchanged(this.cm.getValue())) {
      if (unloading) store.set(draftKey(this.key), this.cm.getValue());    // survives even if the request is cut off
      return this.save(unloading);
    }
    return undefined;
  }

  close() {
    this.closed = true;
    clearTimeout(this.timer);
    S.editors.delete(this);
  }
}

function saveLabel(el) {
  return (kind, text) => { el.className = kind; el.textContent = text; };
}

window.addEventListener("pagehide", () => S.editors.forEach((e) => e.flush(true)));
document.addEventListener("visibilitychange", () => { if (document.hidden) S.editors.forEach((e) => e.flush(true)); });

// ------------------------------------------------------------------ top bar and search

function renderTopbar() {
  $("#topbar").innerHTML = `
    <a class="brand" href="#/">面试准备</a>
    <nav class="switch" aria-label="公司">${S.index.companies.map((c) => `<a href="#/${c.key}" data-company="${c.key}">${esc(c.name)}</a>`).join("")}</nav>
    <span class="topbar-space"></span>
    <div class="search"><input id="search" type="search" placeholder="搜索题目   /" autocomplete="off" spellcheck="false"><div class="search-results" hidden></div></div>
    <nav class="topnav">
      <a href="#/audit" data-nav="audit">核查报告</a>
      <a href="#/answers" data-nav="answers">我的答案</a>
      <button class="icon-btn" id="lang" title="切换题目语言"></button>
      <button class="icon-btn" id="theme" title="切换浅色 / 深色">◐</button>
    </nav>`;
  $("#lang").addEventListener("click", () => {
    S.lang = S.lang === "zh" ? "en" : "zh";
    store.set("lang", S.lang);
    route(true);
  });
  $("#theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("prep.theme", next); } catch { /* private mode */ }
    renderMermaid(app);
  });

  const input = $("#search");
  const box = $(".search-results");
  let hits = [];
  let active = 0;
  const close = () => { box.hidden = true; };
  const draw = () => {
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return close();
    const scored = [];
    for (const it of S.index.items) {
      const head = `${it.title.zh} ${it.title.en} ${it.slug}`.toLowerCase();
      const rest = `${it.kind.zh} ${it.topics.join(" ")} ${it.summary.zh}`.toLowerCase();
      if (terms.every((t) => head.includes(t) || rest.includes(t))) scored.push([terms.every((t) => head.includes(t)) ? 0 : 1, it]);
    }
    hits = scored.sort((a, b) => a[0] - b[0] || byPriority(a[1], b[1])).slice(0, 12).map((x) => x[1]);
    active = Math.min(active, Math.max(hits.length - 1, 0));
    box.hidden = false;
    box.innerHTML = hits.length ? hits.map((it, i) => `<a href="#/${it.id}" class="${i === active ? "active" : ""}">${esc(title(it))}<small>${esc(company(it.company).name)} · ${CATS[it.category].name}</small></a>`).join("")
      : `<div class="empty">没有找到匹配的题目</div>`;
  };
  input.addEventListener("input", () => { active = 0; draw(); });
  input.addEventListener("focus", draw);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { active = (active + (e.key === "ArrowDown" ? 1 : hits.length - 1)) % Math.max(hits.length, 1); draw(); e.preventDefault(); }
    if (e.key === "Enter" && hits[active]) { location.hash = "#/" + hits[active].id; input.value = ""; close(); input.blur(); }
    if (e.key === "Escape") { input.value = ""; close(); input.blur(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search") || e.target.closest(".search-results a")) close(); });
}

function updateTopbar(ckey, nav) {
  document.body.dataset.company = ckey || store.get("company", "anthropic");
  $$(".switch a").forEach((a) => a.classList.toggle("on", a.dataset.company === ckey));
  $$(".topnav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === nav));
  $("#lang").textContent = S.lang === "zh" ? "中文题面" : "English";
}

// ------------------------------------------------------------------ company page

function rowHTML(it) {
  const p = progress(it);
  const js = it.judge && S.state.judge?.[it.id];
  const judged = !js ? "" : js.ac ? `<span class="tag yes" title="提交 ${js.n} 次，通过 ${js.ac} 次">评测通过</span>`
    : `<span class="tag" title="提交 ${js.n} 次，还没有通过">评测 ${js.best}/${js.total}</span>`;
  const tags = `${judged}${auditTag(it)}${it.added && it.category !== "safety" ? `<span class="tag new" title="${esc(it.origin || "题库里没有具体题目，按题型补写的")}">补写</span>` : ""}`;
  const right = it.mode === "qa"
    ? `<span>${it.nq} 个问题${p.count ? ` · 已答 ${p.count}` : ""}</span>${freqBars(it.frequency)}`
    : `<span class="hide-s">${esc(stages(it))}</span><span class="hide-s">${DIFF_ZH[it.difficulty] || ""}</span>${freqBars(it.frequency)}${aiTag(it.ai)}`;
  return `<a class="row" href="#/${it.id}">
    <span class="dot ${p.cls}" title="${p.text}">${p.mark}</span>
    <div><div class="title">${esc(title(it))}${tags}</div><div class="sub">${esc(it.summary[S.lang] || it.summary.zh)}</div></div>
    <div class="meta">${right}</div></a>`;
}

function viewCompany(ckey, section) {
  const c = company(ckey);
  store.set("company", ckey);
  const all = itemsOf(ckey);
  const f = (S.filters[ckey] ||= { status: "all", hot: false, changed: false, role: "", judge: "" });
  const roles = [...new Set(all.flatMap((it) => it.roles))].filter((r) => r !== "All").sort();
  const cats = (band) => band.cats.filter((cat) => all.some((it) => it.category === cat));
  const pageNames = Object.keys(PAGES).filter((name) => S.index.pages.some((p) => p === `${ckey}.${name}` || p.startsWith(`${ckey}.${name}.`)));

  const started = all.filter((it) => progress(it).started).length;
  const finished = all.filter((it) => progress(it).finished).length;
  const en = S.lang === "en";
  const cards = c.cards.length ? `<div class="rules">${c.cards.map((k) => `<div class="card"><h3>${esc((en && k.title_en) || k.title)}</h3>${(en && k.html_en) || k.html}</div>`).join("")}</div>` : "";

  app.innerHTML = `<div class="page">
    <nav class="side">
      ${BANDS.map((b) => (cats(b).length ? `<h4>${b.title}</h4>${cats(b).map((cat) => `<a href="#/${ckey}/${cat}" data-sec="${cat}">${CATS[cat].name}<span class="count">${all.filter((it) => it.category === cat).length}</span></a>`).join("")}` : "")).join("")}
      ${pageNames.length ? `<h4>资料</h4>${pageNames.map((n) => `<a href="#/${ckey}/${n}">${PAGES[n]}</a>`).join("")}` : ""}
    </nav>
    <div>
      <div class="hero">
        <div><h1>${esc(c.name)}</h1><p>${esc((en && c.tagline_en) || c.tagline || `共 ${all.length} 道题`)}</p></div>
        <div class="progress">已动笔 ${started} / ${all.length} · 已完成 ${finished}<div class="bar"><i style="width:${all.length ? (100 * started) / all.length : 0}%"></i></div></div>
      </div>
      ${cards}
      ${pageNames.length ? `<p class="more">${pageNames.map((n) => `<a href="#/${ckey}/${n}">${PAGES[n]} →</a>`).join("　")}</p>` : ""}
      <div class="filters" id="filters"></div>
      <div id="lists"></div>
    </div></div>`;

  const drawFilters = () => {
    const chip = (key, value, label) => `<button class="chip ${f[key] === value ? "on" : ""}" data-key="${key}" data-value="${value}">${label}</button>`;
    $("#filters").innerHTML = `${chip("status", "all", "全部")}${chip("status", "todo", "还没做")}${chip("status", "started", "已动笔")}${chip("status", "review", "待复习")}
      <span class="sep"></span>
      <button class="chip ${f.hot ? "on" : ""}" data-toggle="hot">只看高频</button>
      <button class="chip ${f.changed ? "on" : ""}" data-toggle="changed">核查时改过的</button>
      ${all.some((it) => it.judge) ? `<span class="sep"></span><button class="chip ${f.judge === "ac" ? "on" : ""}" data-judge="ac">评测已通过</button><button class="chip ${f.judge === "todo" ? "on" : ""}" data-judge="todo">可评测、未通过</button>` : ""}
      ${roles.length > 1 ? `<span class="sep"></span><select class="sel" id="role"><option value="">全部岗位</option>${roles.map((r) => `<option ${f.role === r ? "selected" : ""}>${esc(r)}</option>`).join("")}</select>` : ""}`;
  };
  const pass = (it) => {
    const p = progress(it);
    if (f.status === "todo" && p.started) return false;
    if (f.status === "started" && !p.started) return false;
    if (f.status === "review" && p.status !== "review") return false;
    if (f.hot && (FREQ[it.frequency] || 0) < 4) return false;
    if (f.changed && (!it.audit || it.audit.verdict === "ok")) return false;
    if (f.role && !it.roles.includes(f.role) && !it.roles.includes("All")) return false;
    const passed = !!S.state.judge?.[it.id]?.ac;
    if (f.judge === "ac" && !(it.judge && passed)) return false;
    if (f.judge === "todo" && !(it.judge && !passed)) return false;
    return true;
  };
  const drawLists = () => {
    let html = "";
    for (const band of BANDS) {
      const sections = cats(band).map((cat) => {
        const inCat = all.filter((it) => it.category === cat);
        const shown = inCat.filter(pass);
        if (!shown.length) return "";
        const unit = cat === "safety" ? "个专题" : cat === "behavioral" || cat === "quiz" || cat === "discussion" ? "轮" : "道";
        return `<section class="section" id="sec-${cat}"><h3>${CATS[cat].name}<small>${shown.length === inCat.length ? `${inCat.length} ${unit}` : `显示 ${shown.length} / ${inCat.length} ${unit}`}</small></h3>
          <p class="lead">${CATS[cat].lead}</p><div class="rows">${shown.map(rowHTML).join("")}</div></section>`;
      }).join("");
      if (sections) html += `<div class="band"><h2>${band.title}</h2><span>${band.note}</span></div>${sections}`;
    }
    $("#lists").innerHTML = html || `<div class="rows"><div class="empty-note">没有符合筛选条件的题目。</div></div>`;
  };
  $("#filters").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.key) f[b.dataset.key] = b.dataset.value;
    if (b.dataset.toggle) f[b.dataset.toggle] = !f[b.dataset.toggle];
    if (b.dataset.judge) f.judge = f.judge === b.dataset.judge ? "" : b.dataset.judge;
    drawFilters();
    drawLists();
  });
  $("#filters").addEventListener("change", (e) => { if (e.target.id === "role") { f.role = e.target.value; drawLists(); } });
  drawFilters();
  drawLists();

  // highlight the section in view in the side list
  const spy = () => {
    let current = "";
    for (const sec of $$(".section")) if (sec.getBoundingClientRect().top < 140) current = sec.id.slice(4);
    $$(".side a[data-sec]").forEach((a) => a.classList.toggle("on", a.dataset.sec === current));
  };
  window.addEventListener("scroll", spy, { passive: true });
  S.cleanup.push(() => window.removeEventListener("scroll", spy));
  if (section) scrollToSection(section);
  else window.scrollTo(0, S.scroll.get(location.hash) || 0);
  spy();
}

function scrollToSection(cat) {
  const el = $("#sec-" + cat);
  if (el) el.scrollIntoView({ block: "start" });
}

// ------------------------------------------------------------------ shared item header

function itemHead(it) {
  const facts = [
    it.kind.zh ? `<span class="tag">${esc(it.kind.zh)}</span>` : "",
    it.difficulty ? `<span>${DIFF_ZH[it.difficulty]}</span>` : "",
    it.frequency ? `${freqBars(it.frequency)}<span>频率${FREQ_ZH[it.frequency]}</span>` : "",
    it.stage.length ? `<span>${esc(stages(it))}</span>` : "",
    it.format.zh ? `<span>${esc(it.format.zh)}</span>` : "",
    it.roles.length ? `<span>${esc(it.roles.join(" / "))}</span>` : "",
    it.added && it.category !== "safety" ? '<span class="tag new">补写的题</span>' : "",
  ].filter(Boolean).join("");
  const origin = it.origin ? `<div class="originline"><b>来源：</b>${esc(it.origin)}</div>` : "";
  const judged = it.judge ? `<div class="judgeline" id="judgeline">${judgeLine(it)}</div>` : "";
  return `<div class="item-head"><h1>${esc(title(it))}</h1><div class="facts">${facts}</div>
    <div class="envline">${aiTag(it.ai)} ${esc(it.ai.basis)}<br><b>面试环境：</b>${esc(it.ai.env)}　<a href="#/${it.company}/rules">这家公司的完整规则</a></div>${origin}${judged}</div>`;
}

function auditHTML(a) {
  if (!a) return `<p class="empty-note">这道题还没有核查记录。</p>`;
  const [label, cls] = VERDICT[a.verdict] || [a.verdict, ""];
  const issues = (a.issues || []).map((x) => `<div class="issue"><h4><span class="tag ${x.action === "flag" ? "ask" : "fixed"}">${ACTION[x.action] || esc(x.action)}</span>${esc(x.where)}</h4>
    <dl><dt>原来的问题</dt><dd class="md">${x.problem_html || esc(x.problem_zh)}</dd><dt>${x.action === "flag" ? "为什么没改" : "改成了"}</dt><dd class="md">${x.change_html || esc(x.change_zh)}</dd></dl></div>`).join("");
  const run = a.code_run || {};
  return `<p><span class="tag ${cls}">${label}</span> ${a.summary_html || esc(a.summary_zh) || "把页面上的代码跑了一遍，并尽力找错，没有发现会误导人的问题。"}</p>
    ${issues}
    <h4>核查时实际做了什么</h4>
    <ul class="checked">${(a.checked || []).map((x) => `<li>${esc(x)}</li>`).join("")}
      <li>${run.ran ? (run.passed ? "页面自带的代码全部运行通过" : "页面自带的代码没有全部通过") : "这一页没有可运行的代码"}${run.notes ? `（${esc(run.notes)}）` : ""}</li></ul>`;
}

function gradingHelp(it, file) {
  return `<div class="gate"><p>还没有批改。</p><p>答案写好后，对助手说一句 <b>“批改 ${esc(it.id)}”</b>（或者“批改我新写的答案”）。<br>
    批改意见会存到 <code>reviews/${esc(it.id)}/${file}</code>，再打开这里就能看到。</p></div>`;
}

async function reviewHTML(it) {
  let files = {};
  try { files = (await api(`/api/review?id=${encodeURIComponent(it.id)}`)).files; } catch { /* shown as "no review" */ }
  const names = Object.keys(files);
  if (!names.length) return gradingHelp(it, "review.md");
  const md = await mdRenderer();
  return names.map((n) => `${names.length > 1 ? `<h4 class="review-file">${esc(n)}</h4>` : ""}${md.render(files[n], {})}`).join("");
}

// ------------------------------------------------------------------ judge: tests, custom runs, submissions

const JV = {
  AC: ["通过", "ok"], WA: ["答案错误", "no"], RE: ["运行出错", "no"], TLE: ["超时", "warn"], CE: ["语法错误", "no"],
  JE: ["评测程序出错", "warn"], OK: ["运行完成", "ok"], skip: ["没有运行", "skip"],
};
const jvName = (v) => (JV[v] || [v || "?"])[0];
const jvCls = (v) => (JV[v] || [null, "no"])[1];
const jvMark = (v) => (v === "AC" || v === "OK" ? "✓" : v === "skip" ? "–" : "✗");
const secs = (x) => (x == null ? "" : x < 0.01 ? "<0.01 秒" : `${x < 10 ? x.toFixed(2) : x.toFixed(1)} 秒`);
const percent = (a, b) => (b ? Math.round((100 * a) / b) : 0);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function judgeFetch(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { cache: "no-store" });
  let data = null;
  try { data = await r.json(); } catch { /* e.g. an HTML error page from the tunnel */ }
  if (!r.ok) throw new Error(data?.error || `服务器返回 ${r.status}`);
  return data;
}

async function refreshJudgeStats(it) {
  try {
    const data = await judgeFetch(`/api/submissions?id=${encodeURIComponent(it.id)}`);
    (S.state.judge ||= {})[it.id] = data.stats;
  } catch { /* the header keeps its old numbers */ }
}

function judgeLine(it) {
  const j = it.judge;
  const s = S.state.judge?.[it.id];
  const parts = j.parts.length > 1 ? `（${j.parts.map((p) => `${esc(p.name)} ${p.total} 个`).join("，")}）` : "";
  const subs = s ? `提交 ${s.n} 次，通过 ${s.ac} 次（${percent(s.ac, s.n)}%）` : "还没有提交过";
  return `<b>在线评测：</b>${j.total} 个测试${parts}　·　${subs}`;
}

// Error text with the answer's line numbers made clickable ("File "answer.py", line 12" and "answer.py 第 12 行").
function codeText(text, links) {
  const html = esc(text);
  if (!links) return html;
  return html.replace(/File &quot;answer\.py&quot;, line (\d+)/g, (m, n) => `File &quot;answer.py&quot;, <a class="jline" data-line="${n}">line ${n}</a>`)
    .replace(/answer\.py 第 (\d+) 行/g, (m, n) => `<a class="jline" data-line="${n}">answer.py 第 ${n} 行</a>`);
}

function caseDetail(c, links) {
  const block = (label, text, cls = "") => (text ? `<div class="jlabel">${label}</div><pre class="jpre ${cls}">${codeText(text, links)}</pre>` : "");
  return `${c.msg ? `<div class="jmsg">${codeText(c.msg, links)}</div>` : ""}
    ${block("调用", c.call)}${block("期望", c.want)}${block("你的返回值", c.got)}
    ${(c.info || []).length ? `<ul class="jinfo">${c.info.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${block("print 输出", c.out)}${block("报错", c.trace, "err")}`;
}

function resultHTML(it, r, { links = true, label = "" } = {}) {
  if (r.mode === "scratch") {
    return `<div class="jhead ${jvCls(r.verdict)}"><b>${jvMark(r.verdict)} ${jvName(r.verdict)}</b><span>CPU ${secs(r.cpu)}</span></div>
      ${r.msg ? `<div class="jmsg">${codeText(r.msg, links)}</div>` : ""}
      <div class="jlabel">输出</div>${r.out ? `<pre class="jpre">${esc(r.out)}</pre>` : `<p class="checked">没有输出。用 print(...) 打印想看的值。</p>`}
      ${r.trace ? `<div class="jlabel">报错</div><pre class="jpre err">${codeText(r.trace, links)}</pre>` : ""}`;
  }
  const multi = r.parts.length > 1;
  const first = r.cases.find((c) => c.i === r.first);
  const where = first ? `第 ${first.i} 个测试${multi ? `（${esc(first.part)} 第 ${first.k} 个）` : ""}` : "";
  const headline = r.verdict === "AC" ? (r.mode === "samples" ? `样例全部通过` : `通过全部 ${r.total} 个测试`)
    : r.stage === "load" ? `${jvName(r.verdict)}，测试没有运行` : `${jvName(r.verdict)} · ${where}`;
  const facts = `${label}${r.mode === "samples" ? "只跑了样例 · " : ""}通过 ${r.passed}/${r.total} · CPU ${secs(r.cpu)} · 用时 ${secs(r.wall)}`;
  const parts = multi && r.stage !== "load" ? `<div class="jparts">${r.parts.map((p) => {
    const ok = p.passed === p.total;
    return `<span class="jpart ${ok ? "ok" : jvCls(p.verdict)}">${esc(p.name)} ${ok ? "✓" : "✗"} ${p.passed}/${p.total}</span>`;
  }).join("")}</div>` : "";
  const top = r.stage === "load"
    ? `${r.msg ? `<div class="jmsg">${codeText(r.msg, links)}</div>` : ""}${r.trace ? `<pre class="jpre err">${codeText(r.trace, links)}</pre>` : ""}`
    : r.msg ? `<div class="jmsg">${esc(r.msg)}</div>` : "";
  const loadOut = r.out ? `<div class="jlabel">载入代码时的 print 输出</div><pre class="jpre">${esc(r.out)}</pre>` : "";
  const more = r.mode === "samples" && r.verdict === "AC" ? `<p class="checked">样例只是一小部分。点“提交”跑全部 ${it.judge.total} 个测试。</p>` : "";
  const rows = r.stage === "load" ? "" : r.cases.map((c) => {
    const failed = c.verdict !== "AC" && c.verdict !== "skip";
    const sum = `<span class="jv ${jvCls(c.verdict)}">${jvMark(c.verdict)}</span><span class="jnum">#${c.i}</span>
      <span class="jtitle">${multi ? `<small>${esc(c.part)}</small>` : ""}${esc(c.title || c.name)}${c.sample ? '<span class="tag">样例</span>' : ""}</span>
      <span class="jres ${jvCls(c.verdict)}">${c.verdict === "AC" ? "" : jvName(c.verdict)}</span><span class="jtime">${c.verdict === "skip" ? "" : secs(c.cpu)}</span>`;
    return failed || (c.info || []).length || c.out
      ? `<details class="jcase" ${c.i === r.first ? "open" : ""}><summary>${sum}</summary><div class="jdetail">${caseDetail(c, links)}</div></details>`
      : `<div class="jcase"><div class="jsum">${sum}</div></div>`;
  }).join("");
  return `<div class="jtop"><div class="jhead ${jvCls(r.verdict)}"><b>${jvMark(r.verdict)} ${headline}</b><span>${facts}</span></div>${parts}</div>
    ${top}${loadOut}${more}${rows ? `<div class="jcases">${rows}</div>` : ""}`;
}

const judgePanelHTML = () => `<div class="judge" id="judge">
    <div class="jgrip" title="拖动调整高度"></div>
    <div class="judge-bar">
      <span class="seg" id="jtabs"><button data-jt="result">测试结果</button><button data-jt="custom">自定义运行</button></span>
      <span class="grow"></span>
      <button class="btn small" id="j-samples" title="只跑题目里的例子（Ctrl+Enter）">运行样例</button>
      <button class="btn small primary" id="j-submit" title="跑全部测试，记入提交记录">提交</button>
      <button class="icon-btn" id="j-fold" title="收起或展开测试面板">▾</button>
    </div>
    <div class="judge-body">
      <div id="jresult"></div>
      <div id="jcustom" hidden>
        <div class="jscratch"></div>
        <div class="jcustom-bar"><button class="btn small" id="j-run" title="Ctrl+Enter">运行</button><span>先运行你的 answer.py，再接着运行上面这几行，显示 print 的输出和报错</span></div>
        <div id="jcustom-out"></div>
      </div>
    </div>
  </div>`;

// Wires up the panel under the code editor. `editor` is the answer.py editor; `focusAnswer` brings it to the front.
function mountJudge(it, nav, editor, focusAnswer, onSubmitted) {
  const j = it.judge;
  const root = $("#judge");
  const body = $(".judge-body", root);
  const resultEl = $("#jresult");
  const outEl = $("#jcustom-out");
  const mem = (S.judgeMem[it.id] ||= { result: null, label: "", scratch: null, tab: "result" });
  const buttons = [$("#j-samples"), $("#j-submit"), $("#j-run")];
  let busy = false;
  let marked = null;

  const fn = (editor.starter.match(/^\s*def (\w+)\(/m) || editor.starter.match(/^class (\w+)/m) || [])[1];
  const scratchKey = "scratch:" + it.id;
  const scratch = window.CodeMirror($(".jscratch", root), {
    value: store.get(scratchKey, ""), mode: "python", lineNumbers: true, indentUnit: 4, tabSize: 4, indentWithTabs: false,
    matchBrackets: true, autoCloseBrackets: true, viewportMargin: 10,
    placeholder: `# 写几行调用你的代码，例如\n# print(${fn || "my_function"}(...))`,
    extraKeys: { "Ctrl-Enter": () => run("scratch"), "Cmd-Enter": () => run("scratch"), Tab: (cm) => cm.replaceSelection("    ", "end") },
  });
  scratch.on("change", () => store.set(scratchKey, scratch.getValue()));

  const idle = () => `<div class="jidle">
      <p><b>运行样例</b>只跑题目里的例子；<b>提交</b>跑全部 ${j.total} 个测试${j.parts.length > 1 ? `（${j.parts.map((p) => `${esc(p.name)} ${p.total} 个`).join("，")}）` : ""}，结果记入提交记录。</p>
      <p>评测的是编辑器“代码”一栏的内容，在服务器上的沙箱里运行。没通过的测试会显示调用、期望值、你的返回值和 print 输出；报错里的行号可以点，点了跳到那一行。</p>
      ${j.note ? `<p class="jnote">${esc(j.note)}</p>` : ""}</div>`;
  const running = (mode, st) => `<div class="jrun"><span class="spin"></span>${st?.status === "running"
    ? (mode === "scratch" ? "正在运行…" : `正在评测，最长 ${j.timeout} 秒…`)
    : st?.ahead ? `排队中，前面还有 ${st.ahead} 个…` : "已提交，等待评测…"}</div>`;

  const showTab = (t) => {
    mem.tab = t;
    $$("#jtabs button").forEach((b) => b.classList.toggle("on", b.dataset.jt === t));
    resultEl.hidden = t !== "result";
    $("#jcustom").hidden = t !== "custom";
    if (t === "custom") scratch.refresh();
  };
  const setFolded = (folded) => {
    root.classList.toggle("folded", folded);
    $("#j-fold").textContent = folded ? "▴" : "▾";
    store.set("judgeFolded", folded);
    editor.cm.refresh();
    if (!folded) scratch.refresh();
  };
  const clearMark = () => { if (marked) { editor.cm.removeLineClass(marked, "background", "cm-jump"); marked = null; } };
  const openFirst = () => {         // bring the failing test into view just below the pinned verdict line
    const el = $("details[open]", resultEl);
    const top = $(".jtop", resultEl);
    body.scrollTop = 0;
    if (!el) return;
    const y = el.getBoundingClientRect().top - body.getBoundingClientRect().top;   // where it sits with no scroll
    if (y + Math.min(el.offsetHeight, 160) > body.clientHeight) body.scrollTop = y - (top ? top.offsetHeight : 0);
  };

  async function run(mode) {
    if (busy) return;
    busy = true;
    buttons.forEach((b) => { b.disabled = true; });
    clearMark();
    const target = mode === "scratch" ? outEl : resultEl;
    showTab(mode === "scratch" ? "custom" : "result");
    if (root.classList.contains("folded")) setFolded(false);
    target.innerHTML = running(mode);
    try {
      await editor.flush();
      const { job } = await judgeFetch("/api/judge", { id: it.id, mode, code: editor.cm.getValue(), scratch: mode === "scratch" ? scratch.getValue() : "" });
      let st;
      for (let k = 0; ; k++) {
        await sleep(k < 6 ? 400 : 1000);
        st = await judgeFetch(`/api/judge?job=${job}`);
        if (st.status === "done") break;
        if (nav === S.nav) target.innerHTML = running(mode, st);
      }
      const r = st.result;
      const label = mode === "submit" && r.n ? `提交 #${r.n} · ` : "";
      if (mode === "scratch") mem.scratch = r;
      else Object.assign(mem, { result: r, label });
      if (mode === "submit") await refreshJudgeStats(it);
      if (nav !== S.nav) return;            // left the page meanwhile; a submission is recorded all the same
      target.innerHTML = resultHTML(it, r, { label });
      if (mode !== "scratch") openFirst();
      if (mode === "submit") { $("#judgeline").innerHTML = judgeLine(it); onSubmitted(); }
    } catch (e) {
      if (nav === S.nav) target.innerHTML = `<div class="jhead warn"><b>没能完成评测</b><span>${esc(e.message)}</span></div>`;
    } finally {
      busy = false;
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  $("#jtabs").addEventListener("click", (e) => { if (e.target.dataset.jt) showTab(e.target.dataset.jt); });
  $("#j-samples").addEventListener("click", () => run("samples"));
  $("#j-submit").addEventListener("click", () => run("submit"));
  $("#j-run").addEventListener("click", () => run("scratch"));
  $("#j-fold").addEventListener("click", () => setFolded(!root.classList.contains("folded")));
  root.addEventListener("click", (e) => {
    const a = e.target.closest("a.jline");
    if (!a) return;
    focusAnswer();
    const line = Math.min(Number(a.dataset.line), editor.cm.lineCount()) - 1;
    clearMark();
    marked = editor.cm.addLineClass(line, "background", "cm-jump");
    editor.cm.setCursor({ line, ch: 0 });
    editor.cm.scrollIntoView({ line, ch: 0 }, 80);
    editor.cm.focus();
  });
  editor.cm.on("change", clearMark);

  // drag the top edge to resize the panel; the height is kept as a share of the answer pane
  const grip = $(".jgrip", root);
  grip.addEventListener("pointerdown", (down) => {
    grip.setPointerCapture(down.pointerId);
    grip.classList.add("drag");
    const pane = root.parentElement;
    const move = (e) => {
      const box = pane.getBoundingClientRect();
      const foot = $(".answer-foot", pane).offsetHeight;
      const pct = Math.min(80, Math.max(15, (100 * (box.bottom - foot - e.clientY)) / box.height));
      pane.style.setProperty("--jh", pct + "%");
      store.set("judgeH", Math.round(pct));
    };
    const up = () => {
      grip.classList.remove("drag");
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      editor.cm.refresh();
      scratch.refresh();
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
  });
  root.parentElement.style.setProperty("--jh", store.get("judgeH", 42) + "%");

  resultEl.innerHTML = mem.result ? resultHTML(it, mem.result, { label: mem.label }) : idle();
  if (mem.scratch) outEl.innerHTML = resultHTML(it, mem.scratch);
  showTab(mem.tab);
  setFolded(store.get("judgeFolded", false));
  return { run };
}

async function submissionsHTML(it) {
  let data;
  try {
    data = await judgeFetch(`/api/submissions?id=${encodeURIComponent(it.id)}`);
  } catch (e) {
    return `<p class="empty-note">提交记录读取失败：${esc(e.message)}</p>`;
  }
  const subs = data.items.slice().reverse();
  if (!subs.length) return `<p class="empty-note">还没有提交记录。写好代码后，点编辑器下方的“提交”。</p>`;
  const s = data.stats;
  return `<p class="checked">提交 ${s.n} 次，通过 ${s.ac} 次（${percent(s.ac, s.n)}%）。每次提交的代码都存在 <code>answers/${esc(it.id)}/submissions/</code>；点一行看当时的代码和结果。</p>
    <table class="plain subs"><tr><th>#</th><th>时间</th><th>结果</th><th>通过</th><th class="hide-s">CPU</th></tr>
    ${subs.map((x) => `<tr class="sub" data-n="${x.n}"><td>${x.n}</td><td style="white-space:nowrap">${when(x.t)}</td>
      <td><span class="jres ${jvCls(x.verdict)}">${jvMark(x.verdict)} ${jvName(x.verdict)}</span>${x.verdict !== "AC" && x.first ? `<span class="checked hide-s">　第 ${x.first} 个测试</span>` : ""}</td>
      <td style="font-variant-numeric:tabular-nums">${x.passed}/${x.total}</td><td class="hide-s">${secs(x.cpu)}</td></tr>
      <tr class="subdetail" data-n="${x.n}" hidden><td colspan="5"></td></tr>`).join("")}</table>`;
}

async function toggleSubmission(it, row) {
  const detail = row.nextElementSibling;
  detail.hidden = !detail.hidden;
  const cell = detail.firstElementChild;
  if (detail.hidden || cell.innerHTML) return;
  cell.innerHTML = `<p class="checked">读取中…</p>`;
  try {
    const { code, result } = await judgeFetch(`/api/submission?id=${encodeURIComponent(it.id)}&n=${row.dataset.n}`);
    cell.innerHTML = `<div class="jlabel">提交的代码</div><pre class="code"><code>${esc(code)}</code></pre>
      ${result ? `<div class="jlabel">结果</div>${resultHTML(it, result, { links: false })}` : ""}`;
    enhance(cell);
  } catch (e) {
    cell.innerHTML = `<p class="checked">读取失败：${esc(e.message)}</p>`;
  }
}

// ------------------------------------------------------------------ problem page (statement left, answer right)

async function viewProblem(it, nav) {
  const data = await loadItem(it);
  if (nav !== S.nav) return;
  const siblings = itemsOf(it.company, it.category);
  const at = siblings.findIndex((x) => x.id === it.id);
  const prev = siblings[at - 1];
  const next = siblings[at + 1];
  const audit = data.audit;
  const changed = audit && audit.verdict !== "ok";
  const flagged = audit && (audit.issues || []).some((x) => x.action === "flag");
  const layout = store.get("layout", "split");
  const [vLabel] = VERDICT[audit?.verdict] || [""];

  app.innerHTML = `<div class="crumbs">
      <a href="#/${it.company}">${esc(company(it.company).name)}</a><span>/</span><a href="#/${it.company}/${it.category}">${CATS[it.category].name}</a><span>/</span><b>${esc(title(it))}</b>
      <span class="grow"></span>
      <span class="seg" id="layout"><button data-v="read">只看题</button><button data-v="split">分栏</button><button data-v="write">只作答</button></span>
      ${prev ? `<a class="btn small" href="#/${prev.id}" title="${esc(title(prev))}">← 上一题</a>` : ""}${next ? `<a class="btn small" href="#/${next.id}" title="${esc(title(next))}">下一题 →</a>` : ""}
    </div>
    <div class="work" data-layout="${layout}" style="--left:${store.get("split", 54)}%">
      <div class="read-pane"><div class="read-inner">
        ${itemHead(it)}
        ${changed ? `<div class="callout ${audit.verdict === "dropped" ? "dropped" : ""}"><b>核查时${vLabel}：</b>${audit.summary_html || esc(audit.summary_zh)}<button data-tab="audit">看详情</button></div>` : ""}
        <div class="tabs" id="tabs">
          <button data-tab="statement">题目</button><button data-tab="solution">参考解答</button>
          <button data-tab="audit">核查说明${changed || flagged ? '<span class="pip"></span>' : ""}</button>
          ${it.judge ? `<button data-tab="subs">提交记录</button>` : ""}
          <button data-tab="review">批改${S.state.reviews[it.id] ? '<span class="pip"></span>' : ""}</button>
        </div>
        <div class="md" id="tabbody"></div>
      </div></div>
      <div class="splitter" title="拖动调整宽度"></div>
      <div class="answer-pane${it.judge ? " has-judge" : ""}">
        <div class="answer-bar">
          <div class="filetabs">${it.parts.map((p) => `<button data-part="${p.key}">${esc(p.label)}</button>`).join("")}</div>
          <span class="grow"></span>
          <button class="btn small timer" id="timer" title="计时：点击开始或暂停"></button><button class="icon-btn" id="timer-reset" title="计时清零">↺</button>
          <select class="sel" id="status">${Object.entries(STATUS).map(([v, label]) => `<option value="${v}">${label}</option>`).join("")}</select>
        </div>
        <div class="editor-host">${it.parts.map((p) => `<div class="editor-slot" data-part="${p.key}"></div>`).join("")}</div>
        ${it.judge ? judgePanelHTML() : ""}
        <div class="answer-foot"><span id="savestate"></span>${data.starter ? '<button class="linkish" id="restore" title="把代码换回初始代码；现在的代码另存进 answers/.history/">恢复初始代码</button>' : ""}<span class="grow"></span><code id="filepath"></code></div>
      </div>
    </div>`;

  // --- reading side
  const body = $("#tabbody");
  const showTab = async (tab) => {
    S.tabs[it.id] = tab;
    $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
    if (tab === "statement") body.innerHTML = data.statement;
    else if (tab === "solution") {
      if (S.revealed.has(it.id)) body.innerHTML = data.solution;
      else {
        body.innerHTML = `<div class="gate"><p>先自己做一遍，再看参考解答。</p><button class="btn primary" id="reveal">显示参考解答</button></div>`;
        $("#reveal").addEventListener("click", () => { S.revealed.add(it.id); showTab("solution"); });
      }
    } else if (tab === "audit") body.innerHTML = auditHTML(audit);
    else if (tab === "subs") {
      body.innerHTML = `<p class="empty-note">读取中…</p>`;
      const html = await submissionsHTML(it);
      if (S.tabs[it.id] !== "subs" || nav !== S.nav) return;
      body.innerHTML = html;
    } else {
      body.innerHTML = `<p class="empty-note">读取中…</p>`;
      const html = await reviewHTML(it);
      if (S.tabs[it.id] !== "review" || nav !== S.nav) return;
      body.innerHTML = html;
    }
    enhance(body);
    $(".read-pane").scrollTop = tab === "statement" ? (S.scroll.get("read:" + it.id) || 0) : 0;
  };
  $(".read-inner").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (b) showTab(b.dataset.tab);
    const row = e.target.closest("tr.sub");
    if (row) toggleSubmission(it, row);
  });
  S.cleanup.push(() => { if (S.tabs[it.id] === "statement") S.scroll.set("read:" + it.id, $(".read-pane")?.scrollTop || 0); });
  showTab(S.tabs[it.id] || "statement");
  window.scrollTo(0, 0);

  // --- layout
  const work = $(".work");
  const editors = {};
  const setLayout = (v) => {
    work.dataset.layout = v;
    store.set("layout", v);
    $$("#layout button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
    Object.values(editors).forEach((e) => e.cm.refresh());
  };
  $("#layout").addEventListener("click", (e) => { if (e.target.dataset.v) setLayout(e.target.dataset.v); });
  const splitter = $(".splitter");
  splitter.addEventListener("pointerdown", (down) => {
    splitter.setPointerCapture(down.pointerId);
    splitter.classList.add("drag");
    const move = (e) => {
      const box = work.getBoundingClientRect();
      const pct = Math.min(75, Math.max(25, (100 * (e.clientX - box.left)) / box.width));
      work.style.setProperty("--left", pct + "%");
      store.set("split", Math.round(pct));
    };
    const up = () => {
      splitter.classList.remove("drag");
      splitter.removeEventListener("pointermove", move);
      splitter.removeEventListener("pointerup", up);
      Object.values(editors).forEach((e) => e.cm.refresh());
    };
    splitter.addEventListener("pointermove", move);
    splitter.addEventListener("pointerup", up);
  });

  // --- answer side
  const stateEl = $("#savestate");
  const states = {};
  let current = it.parts[0].key;
  const hints = {
    python: "# 在这里写你的解答（自动保存成下方显示的文件，可以直接在本机运行）",
    markdown: it.category === "system-design" ? "在这里写你的设计：需求与规模估算、接口、数据模型、整体架构、关键难点、取舍……" : "思路、复杂度分析、被追问时的回答要点……",
  };
  let judge = null;
  const runKeys = { "Ctrl-Enter": () => judge?.run("samples"), "Cmd-Enter": () => judge?.run("samples") };
  for (const p of it.parts) {
    const slot = $(`.editor-slot[data-part="${p.key}"]`);
    const code = p.key === "answer" && it.judge;
    editors[p.key] = new Editor(slot, { id: it.id, part: p.key, mode: p.mode, placeholder: hints[p.mode],
      starter: code ? data.starter : "", keys: code ? runKeys : undefined,
      onState: (kind, text) => { states[p.key] = [kind, text]; if (current === p.key) saveLabel(stateEl)(kind, text); } });
    S.cleanup.push(async () => { await editors[p.key].flush(); editors[p.key].close(); });
  }
  const showPart = (key) => {
    current = key;
    const part = it.parts.find((p) => p.key === key);
    $$(".filetabs button").forEach((b) => b.classList.toggle("on", b.dataset.part === key));
    $$(".editor-slot").forEach((s) => { s.hidden = s.dataset.part !== key; });
    $("#filepath").textContent = `answers/${it.id}/${part.file}`;
    if ($("#restore")) $("#restore").hidden = key !== "answer";
    if (states[key]) saveLabel(stateEl)(...states[key]);
    editors[key].cm.refresh();
  };
  $(".filetabs").addEventListener("click", (e) => { if (e.target.dataset.part) showPart(e.target.dataset.part); });
  if (it.parts.length < 2) $(".filetabs").innerHTML = `<button class="on">${esc(it.parts[0].label)}</button>`;
  await Promise.all(Object.values(editors).map((e) => e.load()));
  if (nav !== S.nav) return;
  setLayout(layout);
  showPart(current);
  if (it.judge && editors.answer) {
    judge = mountJudge(it, nav, editors.answer, () => { if (current !== "answer") showPart("answer"); },
      () => { if (S.tabs[it.id] === "subs") showTab("subs"); });
  }
  $("#restore")?.addEventListener("click", () => {
    const ed = editors.answer;
    if (ed.cm.getValue() === data.starter) { toast("已经是初始代码了"); return; }
    if (!window.confirm("把代码换回初始代码吗？现在的代码会另存一份到 answers/.history/，不会丢。")) return;
    ed.keep = true;
    ed.cm.setValue(data.starter);
    ed.touch();
  });

  const statusEl = $("#status");
  statusEl.value = S.state.status[it.id]?.status || "";
  statusEl.addEventListener("change", () => setStatus(it.id, statusEl.value));

  // --- stopwatch (kept while the page stays open, so leaving a problem and coming back does not reset it)
  const t = (S.timers[it.id] ||= { elapsed: 0, since: null });
  const timerEl = $("#timer");
  const paint = () => {
    const sec = Math.floor((t.elapsed + (t.since ? Date.now() - t.since : 0)) / 1000);
    timerEl.textContent = `${t.since ? "❚❚" : "▶"} ${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
  };
  timerEl.addEventListener("click", () => {
    if (t.since) { t.elapsed += Date.now() - t.since; t.since = null; } else t.since = Date.now();
    paint();
  });
  $("#timer-reset").addEventListener("click", () => { t.elapsed = 0; t.since = null; paint(); });
  const tick = setInterval(paint, 1000);
  S.cleanup.push(() => clearInterval(tick));
  paint();
}

// ------------------------------------------------------------------ question rounds (behavioural, quiz, safety review)

const flatQuestions = (data) => data.groups.flatMap((g) => g.questions.map((q) => ({ ...q, group: g })));

async function viewRound(it, nav) {
  const data = await loadItem(it);
  if (nav !== S.nav) return;
  const qs = flatQuestions(data);
  const answered = S.state.answers[it.id] || {};
  const firstOpen = qs.find((q) => !answered[partOf(q.n)]) || qs[0];
  const audit = data.audit;
  const changed = audit && audit.verdict !== "ok";
  const [vLabel] = VERDICT[audit?.verdict] || [""];
  const count = Object.keys(answered).length;
  const fold = (summary, html, open = false) => (html ? `<details class="fold" ${open ? "open" : ""}><summary>${summary}</summary><div class="md">${html}</div></details>` : "");

  app.innerHTML = `<div class="single">
    <div class="qhead"><a href="#/${it.company}/${it.category}">← ${esc(company(it.company).name)} · ${CATS[it.category].name}</a></div>
    ${itemHead(it)}
    ${changed ? `<div class="callout" style="margin-top:1rem"><b>核查时${vLabel}：</b>${audit.summary_html || esc(audit.summary_zh)}</div>` : ""}
    <div class="md" style="margin-top:1.2rem">${data.intro}</div>
    <div class="qactions">
      <a class="btn primary" href="#/${it.id}/q/${firstOpen.n}">${count ? `接着做（第 ${firstOpen.n} 题）` : "从第 1 题开始"}</a>
      <button class="btn" id="random">随机抽一题</button>
      <span class="grow"></span><span class="checked">已答 ${count} / ${qs.length}</span>
    </div>
    ${data.groups.map((g) => `<section class="section">${g.title ? `<h3>${esc(g.title)}<small>${g.questions.length} 个问题</small></h3>` : ""}
      ${g.note ? `<div class="md lead">${g.note}</div>` : ""}
      <div class="rows qlist">${g.questions.map((q) => {
        const st = S.state.status[`${it.id}#${partOf(q.n)}`]?.status;
        const cls = st === "review" ? "review" : answered[partOf(q.n)] ? "done" : "";
        return `<a href="#/${it.id}/q/${q.n}"><span class="dot ${cls}">${cls === "review" ? "!" : cls ? "✓" : ""}</span><span class="num">${q.n}</span><span class="text md">${q.q}</span></a>`;
      }).join("")}</div></section>`).join("")}
    <div class="band"><h2>参考资料</h2><span>建议先自己答，再展开</span></div>
    <div style="margin-top:1rem">
      ${fold(it.category === "safety" ? "出处与核对情况" : "这一轮在考什么、怎么准备", data.solutionIntro)}
      ${data.extras.map((x) => fold(esc(x.title), x.html)).join("")}
      ${fold("验证这些答案的代码", data.checks)}
      ${it.category === "safety" && !audit ? "" : fold("核查说明", auditHTML(audit))}
      ${S.state.reviews[it.id] ? `<details class="fold" id="reviewfold"><summary>批改意见</summary><div class="md"></div></details>` : ""}
    </div></div>`;
  enhance(app);
  $("#random").addEventListener("click", () => {
    const open = qs.filter((q) => !(S.state.answers[it.id] || {})[partOf(q.n)]);
    const pool = open.length ? open : qs;
    location.hash = `#/${it.id}/q/${pool[Math.floor(Math.random() * pool.length)].n}`;
  });
  $("#reviewfold")?.addEventListener("toggle", async (e) => {
    const box = $(".md", e.target);
    if (e.target.open && !box.innerHTML) { box.innerHTML = await reviewHTML(it); enhance(box); }
  });
  window.scrollTo(0, S.scroll.get(location.hash) || 0);
}

async function viewQuestion(it, n, nav) {
  const data = await loadItem(it);
  if (nav !== S.nav) return;
  const qs = flatQuestions(data);
  const at = qs.findIndex((q) => q.n === n);
  if (at < 0) return viewMissing();
  const q = qs[at];
  const part = partOf(q.n);
  const key = `${it.id}#${part}`;
  const isTalk = it.category === "behavioral";
  const answerHTML = isTalk ? q.group.guidance : q.a;
  const answerLabel = isTalk ? `“${esc(q.group.title)}”这一组问题的回答思路` : "参考答案";
  const link = (target, label) => (target ? `<a class="btn" href="#/${it.id}/q/${target.n}">${label}</a>` : `<button class="btn" disabled>${label}</button>`);
  const reviewName = `${part}.md`;
  const hasReview = (S.state.reviews[it.id] || []).includes(reviewName);

  app.innerHTML = `<div class="single">
    <div class="qhead"><a href="#/${it.id}">← ${esc(title(it))}</a><span class="grow"></span><span>第 ${at + 1} / ${qs.length} 题</span>${aiTag(it.ai)}</div>
    <div class="qcard">
      ${q.group.title ? `<div class="group">${esc(q.group.title)}</div>` : ""}
      <div class="qtext md">${q.q}</div>
      ${q.group.note ? `<div class="context md">${q.group.note}</div>` : ""}
    </div>
    <div class="qedit"></div>
    <div class="answer-foot" style="border:0;padding:.4rem .2rem 0"><span id="savestate"></span><span class="grow"></span><code>answers/${it.id}/${part}.md</code></div>
    <div class="qactions">
      <button class="btn primary" id="reveal">${isTalk ? "看回答思路" : "看参考答案"}</button>
      <label class="checked"><input type="checkbox" id="review"> 标记为待复习</label>
      <span class="grow"></span>
      ${link(qs[at - 1], "← 上一题")}${link(qs[at + 1], "下一题 →")}
    </div>
    <div class="reveal" id="answer" hidden><h4>${answerLabel}</h4><div class="md">${answerHTML || "<p>这一题没有单独的参考答案。</p>"}</div></div>
    ${hasReview ? `<div class="reveal" id="reviewbox"><h4>批改意见</h4><div class="md">读取中…</div></div>` : ""}
    <p class="checked" style="margin-top:1.4rem"><kbd>Ctrl</kbd> + <kbd>Enter</kbd> 下一题　<kbd>Esc</kbd> 离开输入框后：<kbd>←</kbd> <kbd>→</kbd> 换题，<kbd>A</kbd> 看答案</p>
  </div>`;
  enhance(app);

  const go = (target) => { if (target) location.hash = `#/${it.id}/q/${target.n}`; };
  const editor = new Editor($(".qedit"), { id: it.id, part, mode: "markdown", grow: true, onState: saveLabel($("#savestate")),
    placeholder: isTalk ? "像面试时那样回答：先一句话给结论，再讲具体经历（当时的情况、你做了什么、结果、学到什么）……" : it.category === "safety" ? "先不看笔记，用自己的话答一遍……" : it.category === "discussion" ? "像面试时那样写：你会怎么说、先问什么、下一步做什么……" : "写下你的答案和推导……" });
  editor.cm.addKeyMap({ "Ctrl-Enter": () => go(qs[at + 1]), "Cmd-Enter": () => go(qs[at + 1]), Esc: (cm) => cm.getInputField().blur() });
  S.cleanup.push(async () => { await editor.flush(); editor.close(); });
  await editor.load();
  if (nav !== S.nav) return;
  editor.cm.refresh();

  const toggleAnswer = () => {
    const box = $("#answer");
    box.hidden = !box.hidden;
    $("#reveal").textContent = box.hidden ? (isTalk ? "看回答思路" : "看参考答案") : "收起";
    if (!box.hidden) box.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };
  $("#reveal").addEventListener("click", toggleAnswer);
  const reviewEl = $("#review");
  reviewEl.checked = S.state.status[key]?.status === "review";
  reviewEl.addEventListener("change", () => setStatus(key, reviewEl.checked ? "review" : ""));
  const onKey = (e) => {
    if (e.target.closest("input, textarea, select, .CodeMirror") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "ArrowLeft") go(qs[at - 1]);
    if (e.key === "ArrowRight") go(qs[at + 1]);
    if (e.key === "a" || e.key === "A") toggleAnswer();
  };
  document.addEventListener("keydown", onKey);
  S.cleanup.push(() => document.removeEventListener("keydown", onKey));
  if (hasReview) {
    const files = (await api(`/api/review?id=${encodeURIComponent(it.id)}`).catch(() => ({ files: {} }))).files;
    const md = await mdRenderer();
    if (nav !== S.nav) return;
    $("#reviewbox .md").innerHTML = md.render(files[reviewName] || "", {});
    enhance($("#reviewbox"));
  }
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ long-form pages, audit report, my answers

async function viewPage(ckey, name, nav) {
  const lang = S.index.pages.includes(`${ckey}.${name}.${S.lang}`) ? S.lang : "zh";
  const file = S.index.pages.includes(`${ckey}.${name}`) ? `${ckey}.${name}` : `${ckey}.${name}.${lang}`;
  if (!S.index.pages.includes(file)) return viewMissing();
  const page = await getJSON(`data/pages/${file}.json`);
  if (nav !== S.nav) return;
  app.innerHTML = `<div class="single"><div class="qhead"><a href="#/${ckey}">← ${esc(company(ckey).name)}</a><span class="grow"></span><span>${PAGES[name]}</span></div><div class="md">${page.html}${name === "selection" ? addedTable(ckey) : ""}</div></div>`;
  enhance(app);
  window.scrollTo(0, S.scroll.get(location.hash) || 0);
}

// The list of problems written for this site, appended to each company's selection page so that it cannot go stale.
function addedTable(ckey) {
  const rows = itemsOf(ckey).filter((it) => it.added && it.category !== "safety");
  if (!rows.length) return "";
  return `<h2 id="added">补写的题目（${rows.length} 道）</h2>
    <p>面经里提到了这类题、却没有留下具体题目的，每一类补写了两道，问答和讨论类是一页里有多个问题；面经只留下一句话题面的，按那句话补全成一道完整的题。“依据”一栏写的是面经原本说了什么；题目本身是这次新写的，不是面试原题。有代码的都实际运行过；每道还另外请人核查了一遍，记录在题目页底部。</p>
    <table class="stack"><tr><th>题目</th><th>类型</th><th>依据</th></tr>
    ${rows.map((it) => `<tr><td><a href="#/${it.id}">${esc(title(it))}</a></td><td style="white-space:nowrap">${CATS[it.category].name}</td><td>${esc(it.origin || "")}</td></tr>`).join("")}</table>`;
}

async function viewAudit(nav) {
  const [audits, intro] = await Promise.all([getJSON("data/audit.json"), S.index.pages.includes("audit") ? getJSON("data/pages/audit.json") : null]);
  if (nav !== S.nav) return;
  const known = audits.filter((a) => S.byId.has(a.id));
  const n = (v) => known.filter((a) => a.verdict === v).length;
  const flags = known.filter((a) => (a.issues || []).some((x) => x.action === "flag")).length;
  const added = S.index.items.filter((it) => it.added).length;
  const block = (a) => {
    const it = S.byId.get(a.id);
    const [label, cls] = VERDICT[a.verdict] || [a.verdict, ""];
    const rows = (a.issues || []).map((x) => `<tr><td style="white-space:nowrap">${esc(x.where)}</td><td class="md">${x.problem_html || esc(x.problem_zh)}</td>
      <td class="md"><span class="tag ${x.action === "flag" ? "ask" : "fixed"}">${ACTION[x.action] || esc(x.action)}</span> ${x.change_html || esc(x.change_zh)}</td></tr>`).join("");
    return `<div class="audit-item"><h4><a href="#/${it.id}">${esc(title(it))}</a><span class="tag ${cls}">${label}</span><span class="tag">${CATS[it.category].name}</span></h4>
      ${a.summary_html ? `<div class="md">${a.summary_html}</div>` : ""}
      ${rows ? `<table class="plain stack"><tr><th>位置</th><th>原来的问题</th><th>处理</th></tr>${rows}</table>` : ""}</div>`;
  };
  const perCompany = S.index.companies.map((c) => {
    const mine = known.filter((a) => S.byId.get(a.id).company === c.key);
    const touched = mine.filter((a) => a.verdict !== "ok" || (a.issues || []).length);
    const clean = mine.filter((a) => a.verdict === "ok" && !(a.issues || []).length);
    const total = itemsOf(c.key).length;
    return `<div class="band"><h2>${esc(c.name)}</h2><span>已核查 ${mine.length} / ${total} 页，其中 ${touched.length} 页有改动或存疑</span></div>
      <div style="margin-top:1rem">${touched.map(block).join("") || '<p class="empty-note">没有需要改动的题。</p>'}
      ${clean.length ? `<details class="fold"><summary>没发现问题的 ${clean.length} 页（展开看每页做了哪些检查）</summary><div><table class="plain stack"><tr><th>题目</th><th>做过的检查</th></tr>
        ${clean.map((a) => `<tr><td style="white-space:nowrap"><a href="#/${a.id}">${esc(title(S.byId.get(a.id)))}</a></td><td>${(a.checked || []).map(esc).join("；")}</td></tr>`).join("")}</table></div></details>` : ""}</div>`;
  }).join("");
  app.innerHTML = `<div class="single wide">
    <div class="hero"><div><h1>核查报告</h1><p>每一页题目都实际运行并尝试找错；只处理会误导人的真问题，用最小的改动修好，或把做不到的要求放宽。</p></div></div>
    <div class="stats">
      <div class="stat"><b>${S.index.items.length}</b><span>页题目（已核查 ${known.length}）</span></div>
      <div class="stat"><b>${n("fixed") + n("relaxed")}</b><span>页有改动（修正 ${n("fixed")} · 放宽 ${n("relaxed")}）</span></div>
      <div class="stat"><b>${n("dropped")}</b><span>页有硬伤</span></div>
      <div class="stat"><b>${flags}</b><span>页留有存疑点${added ? ` · 补写 ${added} 题` : ""}</span></div>
    </div>
    ${intro ? `<div class="md">${intro.html}</div>` : ""}
    ${perCompany}</div>`;
  enhance(app);
  window.scrollTo(0, S.scroll.get(location.hash) || 0);
}

async function viewAnswers(nav) {
  await refreshState();
  if (nav !== S.nav) return;
  const rows = Object.entries(S.state.answers).filter(([id]) => S.byId.has(id)).map(([id, parts]) => {
    const it = S.byId.get(id);
    const last = Math.max(...Object.values(parts).map((p) => p.mtime));
    return { it, parts, last };
  }).sort((a, b) => b.last - a.last);
  const fileName = (it, part) => (it.parts.find((p) => p.key === part)?.file || `${part}.md`);
  app.innerHTML = `<div class="single wide">
    <div class="hero"><div><h1>我的答案</h1><p>所有答案都是你机器上的普通文件，网站不会把它们发到任何地方。</p></div></div>
    <div class="rules">
      <div class="card"><h3>存在哪里</h3><p><code>${esc(S.state.root || "interview_prep")}/answers/</code><br>每道题一个文件夹，代码是 <code>answer.py</code>，文字是 <code>.md</code>。</p></div>
      <div class="card"><h3>怎么批改</h3><p>对助手说 <b>“批改我新写的答案”</b>，或者点名某一题，例如 <b>“批改 anthropic/coding/oa-bank-system”</b>。</p></div>
      <div class="card"><h3>批改结果在哪</h3><p>助手把意见写进 <code>reviews/</code> 下同名文件夹，题目页的“批改”一栏会显示出来。旧版本的答案保留在 <code>answers/.history/</code>。</p></div>
    </div>
    <div class="band"><h2>已经写了的</h2><span>${rows.length} 道题，按最后修改时间排序</span></div>
    <div class="rows" style="margin-top:1rem">${rows.length ? `<table class="plain"><tr><th>题目</th><th>文件</th><th>最后修改</th><th>批改</th></tr>
      ${rows.map(({ it, parts, last }) => `<tr><td><a href="#/${it.id}">${esc(title(it))}</a><br><span class="checked">${esc(company(it.company).name)} · ${CATS[it.category].name}</span></td>
        <td><code>answers/${it.id}/</code><br><span class="checked">${Object.keys(parts).sort().map((p) => fileName(it, p)).join("、")}</span></td>
        <td style="white-space:nowrap">${when(last)}</td><td>${S.state.reviews[it.id] ? '<span class="tag yes">已批改</span>' : '<span class="checked">还没有</span>'}</td></tr>`).join("")}</table>`
      : '<div class="empty-note">还没有写任何答案。打开一道题，在右边写就行，内容会自动保存。</div>'}</div></div>`;
  window.scrollTo(0, 0);
}

function viewMissing() {
  app.innerHTML = `<div class="single"><div class="gate"><p>没有这个页面。</p><a class="btn" href="#/">回到首页</a></div></div>`;
}

// ------------------------------------------------------------------ router

async function route(force = false) {
  const hash = location.hash;
  if (!hash.startsWith("#/")) { location.replace("#/" + store.get("company", "anthropic")); return; }
  const parts = hash.slice(2).split("/").filter(Boolean).map(decodeURIComponent);
  if (!parts.length) { location.replace("#/" + store.get("company", "anthropic")); return; }
  const isCompany = !!company(parts[0]);

  // a section link on the page that is already showing only scrolls
  if (!force && isCompany && parts.length === 2 && CATS[parts[1]] && S.view === "company:" + parts[0]) { scrollToSection(parts[1]); return; }

  const nav = ++S.nav;
  if (S.hash) S.scroll.set(S.hash, window.scrollY);
  const cleanup = S.cleanup;
  S.cleanup = [];
  for (const fn of cleanup) { try { await fn(); } catch (e) { console.error(e); } }
  if (nav !== S.nav) return;
  S.hash = hash;
  S.view = "";
  try {
    if (parts[0] === "audit") { updateTopbar("", "audit"); await viewAudit(nav); }
    else if (parts[0] === "answers") { updateTopbar("", "answers"); await viewAnswers(nav); }
    else if (!isCompany) { updateTopbar("", ""); viewMissing(); }
    else {
      updateTopbar(parts[0], "");
      const it = S.byId.get(parts.slice(0, 3).join("/"));
      if (parts.length === 1 || (parts.length === 2 && CATS[parts[1]])) { viewCompany(parts[0], parts[1]); S.view = "company:" + parts[0]; }
      else if (parts.length === 2 && PAGES[parts[1]]) await viewPage(parts[0], parts[1], nav);
      else if (it && parts.length === 3) await (it.mode === "qa" ? viewRound(it, nav) : viewProblem(it, nav));
      else if (it && it.mode === "qa" && parts.length === 5 && parts[3] === "q") await viewQuestion(it, Number(parts[4]), nav);
      else viewMissing();
    }
  } catch (e) {
    console.error(e);
    if (nav === S.nav) app.innerHTML = `<div class="single"><div class="gate"><p>这个页面加载失败了：${esc(e.message)}</p><a class="btn" href="#/">回到首页</a></div></div>`;
  }
  document.title = app.querySelector("h1")?.textContent ? `${app.querySelector("h1").textContent} · 面试准备` : "面试准备";
}

document.addEventListener("click", (e) => {
  // links written for GitHub ("#some-heading") scroll within the page instead of replacing the route
  const a = e.target.closest("a[href^='#']");
  if (!a || a.getAttribute("href").startsWith("#/")) return;
  e.preventDefault();
  const id = decodeURIComponent(a.getAttribute("href").slice(1));
  (document.getElementById("h-" + id) || document.getElementById(id))?.scrollIntoView({ behavior: "smooth", block: "start" });
});

document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !e.target.closest("input, textarea, select, .CodeMirror")) { e.preventDefault(); $("#search").focus(); }
});

async function boot() {
  try {
    S.index = await getJSON("data/index.json");
  } catch (e) {
    app.innerHTML = `<div class="single"><div class="gate"><p>找不到题库数据（${esc(e.message)}）。请先运行 <code>.venv/bin/python build.py</code>。</p></div></div>`;
    return;
  }
  S.byId = new Map(S.index.items.map((it) => [it.id, it]));
  renderTopbar();
  await refreshState();
  window.addEventListener("hashchange", () => route());
  window.addEventListener("focus", () => { if (S.view) refreshState(); });
  route();
}

boot();
