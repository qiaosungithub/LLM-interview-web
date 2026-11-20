// Render markdown strings to HTML at build time with the renderer the site also uses in the browser.
//   node render.mjs <in.json> <out.json>
// in.json:  {"<key>": "<markdown>", ...}
// out.json: {"html": {"<key>": "<html>"}, "errors": [{"key", "kind", "message", "source"}]}
// Math errors are collected, not thrown, so one bad formula shows up in the build report instead of on a page.
import fs from "node:fs";
import MarkdownIt from "markdown-it";
import katex from "katex";
import hljs from "highlight.js";
import { createRenderer } from "../../site/assets/md.js";

const errors = [];
let currentKey = "";
const md = createRenderer({ MarkdownIt, katex, hljs, onError: (e) => errors.push({ key: currentKey, ...e }) });

const [inPath, outPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inPath, "utf8"));
const html = {};
for (const [key, source] of Object.entries(input)) {
  currentKey = key;
  html[key] = md.render(source, {});
}
fs.writeFileSync(outPath, JSON.stringify({ html, errors }));
console.log(`rendered ${Object.keys(html).length} fragments, ${errors.length} error(s)`);
