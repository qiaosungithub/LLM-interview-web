// Markdown renderer shared by the build (Node, tools/node/render.mjs) and the browser (reviews, previews):
// markdown-it + KaTeX ($inline$, $$display$$, ```math) + highlight.js, GitHub-style heading ids.
// The libraries are passed in, so this file imports nothing and runs unchanged in both places.

const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");


// ---- math: $inline$, $$display$$ (inline or as a block), and ```math fences (handled in the fence renderer)
function mathInline(state, silent) {
  const src = state.src;
  const start = state.pos;
  if (src.charCodeAt(start) !== 0x24 /* $ */) return false;
  const display = src.charCodeAt(start + 1) === 0x24;
  const open = display ? 2 : 1;
  const first = src.charCodeAt(start + open);
  if (!display && (first === 0x20 || first === 0x0a || first === 0x09 || Number.isNaN(first))) return false;
  let pos = start + open;
  let end = -1;
  while (pos < state.posMax) {
    const ch = src.charCodeAt(pos);
    if (ch === 0x5c /* \ */) { pos += 2; continue; }
    if (ch === 0x24) {
      if (display) {
        if (src.charCodeAt(pos + 1) === 0x24) { end = pos; break; }
      } else {
        const prev = src.charCodeAt(pos - 1);
        const next = src.charCodeAt(pos + 1);
        const nextIsDigit = next >= 0x30 && next <= 0x39;
        if (prev !== 0x20 && prev !== 0x0a && prev !== 0x09 && !nextIsDigit) { end = pos; break; }
      }
    }
    pos += 1;
  }
  if (end < 0 || end === start + open) return false;
  if (!silent) {
    const token = state.push(display ? "math_display_inline" : "math_inline", "math", 0);
    token.content = src.slice(start + open, end);
  }
  state.pos = end + open;
  return true;
}

function mathBlock(state, startLine, endLine, silent) {
  let pos = state.bMarks[startLine] + state.tShift[startLine];
  let max = state.eMarks[startLine];
  if (state.sCount[startLine] - state.blkIndent >= 4) return false;
  if (state.src.slice(pos, pos + 2) !== "$$") return false;
  const firstLine = state.src.slice(pos + 2, max);
  let content;
  let nextLine = startLine;
  const closeAt = firstLine.indexOf("$$");
  if (closeAt >= 0) {
    if (firstLine.slice(closeAt + 2).trim() !== "") return false;      // "$$a$$ trailing text" is inline, not a block
    content = firstLine.slice(0, closeAt);
  } else {
    const lines = [firstLine];
    let found = false;
    for (nextLine = startLine + 1; nextLine < endLine; nextLine++) {
      pos = state.bMarks[nextLine] + state.tShift[nextLine];
      max = state.eMarks[nextLine];
      const line = state.src.slice(pos, max);
      const at = line.indexOf("$$");
      if (at >= 0 && line.slice(at + 2).trim() === "") { lines.push(line.slice(0, at)); found = true; break; }
      lines.push(line);
    }
    if (!found) return false;
    content = lines.join("\n");
  }
  if (silent) return true;
  const token = state.push("math_block", "math", 0);
  token.block = true;
  token.content = content;
  token.map = [startLine, nextLine + 1];
  state.line = nextLine + 1;
  return true;
}

// ---- GitHub-style heading slugs, so in-page anchors written for GitHub keep working
function slugify(text) {
  return text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s/g, "-");
}

export function createRenderer({ MarkdownIt, katex, hljs, onError }) {
  function renderMath(tex, displayMode) {
    try {
      return katex.renderToString(tex, { displayMode, throwOnError: true, output: "html", strict: "ignore" });
    } catch (e) {
      if (onError) onError({ kind: "katex", message: String(e.message || e), source: tex.slice(0, 300) });
      return `<code class="math-error" title="${escapeHtml(String(e.message || e))}">${escapeHtml(tex)}</code>`;
    }
  }
  const md = new MarkdownIt({ html: true, linkify: false, typographer: false, breaks: false });
  md.inline.ruler.before("escape", "math_inline", mathInline);
  md.block.ruler.before("fence", "math_block", mathBlock, { alt: ["paragraph", "reference", "blockquote", "list"] });
  md.renderer.rules.math_inline = (tokens, idx) => renderMath(tokens[idx].content, false);
  md.renderer.rules.math_display_inline = (tokens, idx) => renderMath(tokens[idx].content, true);
  md.renderer.rules.math_block = (tokens, idx) => `<div class="math-block">${renderMath(tokens[idx].content, true)}</div>\n`;

  // A source line break inside a paragraph shows as a space. Chinese is written without spaces, so a break
  // between two Chinese characters (or Chinese and a quote or dash next to it) must not leave one.
  const han = /[⺀-鿿豈-﫿＀-￯]/;
  const hanOrMark = /[⺀-鿿豈-﫿＀-￯—‘’“”…·]/;
  const neighbour = (tokens, idx, step) => {
    let i = idx + step;
    while (tokens[i] && /_(open|close)$/.test(tokens[i].type)) i += step;      // look through **bold**, links and so on
    const t = tokens[i];
    if (!t || t.type !== "text" || !t.content) return "";
    return step < 0 ? t.content.slice(-1) : t.content.charAt(0);
  };
  md.renderer.rules.softbreak = (tokens, idx) => {
    const a = neighbour(tokens, idx, -1);
    const b = neighbour(tokens, idx, 1);
    return hanOrMark.test(a) && hanOrMark.test(b) && (han.test(a) || han.test(b)) ? "" : "\n";
  };

  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const info = (token.info || "").trim().split(/\s+/)[0];
    const code = token.content;
    if (info === "mermaid") return `<pre class="mermaid-src">${escapeHtml(code)}</pre>\n`;
    if (info === "math") return `<div class="math-block">${renderMath(code, true)}</div>\n`;
    const lang = { py: "python", sh: "bash", shell: "bash", yml: "yaml" }[info] || info;
    let body;
    if (hljs && lang && lang !== "text" && hljs.getLanguage(lang)) {
      body = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
    } else {
      body = escapeHtml(code);
    }
    const label = lang && lang !== "text" ? ` data-lang="${escapeHtml(lang)}"` : "";
    return `<pre class="code"${label}><code>${body}</code></pre>\n`;
  };

  const defaultLinkOpen = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = tokens[idx].attrGet("href") || "";
    if (/^https?:\/\//.test(href)) {
      tokens[idx].attrSet("target", "_blank");
      tokens[idx].attrSet("rel", "noopener noreferrer");
    }
    return defaultLinkOpen(tokens, idx, options, env, self);
  };

  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n';
  md.renderer.rules.table_close = () => "</table></div>\n";

  md.renderer.rules.heading_open = (tokens, idx, options, env, self) => {
    const inline = tokens[idx + 1];
    const text = inline && inline.children ? inline.children.filter((t) => t.type === "text" || t.type === "code_inline").map((t) => t.content).join("") : "";
    const seen = (env.slugs ||= new Map());
    let slug = slugify(text);
    const n = seen.get(slug) || 0;
    seen.set(slug, n + 1);
    if (n) slug = `${slug}-${n}`;
    if (slug) tokens[idx].attrSet("id", `h-${slug}`);
    return self.renderToken(tokens, idx, options);
  };
  return md;
}
