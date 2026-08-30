"""The review page: one self-contained HTML file, built from run records.

``mm-eval review`` serves it and the page PUTs every change to the run's
``decisions/decisions.json``.

The page still falls back to ``localStorage`` when its fetch to the backend
fails, so a page left open through a server restart keeps the reviewer's work
instead of silently discarding it. That fallback is a safety net, not a second
way to run this: browser storage is per-origin and does not merge back into the
decisions file, which is a bad way to lose an afternoon of judgement.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from meeplemate.eval.eval_gen_layout import EvalGenLayout

PAGE_TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">

<style>
:root{
  --paper:#FAF9FC; --surface:#FFFFFF; --surface-2:#F3F1F8; --line:#E1DEEC;
  --ink:#191830; --ink-2:#413E5C; --muted:#6C6884;
  --accent:#5B4FCF; --accent-soft:#EDEAFB;
  --keep:#17795A; --keep-soft:#E2F1EB;
  --review:#A8720F; --review-soft:#F7EEDC;
  --drop:#A93B36; --drop-soft:#F8E7E5;
  --shadow:0 1px 2px rgba(25,24,48,.06),0 8px 24px -12px rgba(25,24,48,.18);
  --rail:300px; --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --display:"Bricolage Grotesque",var(--sans); --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#14131F; --surface:#1D1B2C; --surface-2:#252336; --line:#332F4A;
  --ink:#EAE8F4; --ink-2:#BDB9D0; --muted:#8C88A6;
  --accent:#9A8FF5; --accent-soft:#2A2547;
  --keep:#4FC49B; --keep-soft:#183328;
  --review:#DFA23C; --review-soft:#382A14;
  --drop:#E8746C; --drop-soft:#3A1F1D;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}}
:root[data-theme="dark"]{
  --paper:#14131F; --surface:#1D1B2C; --surface-2:#252336; --line:#332F4A;
  --ink:#EAE8F4; --ink-2:#BDB9D0; --muted:#8C88A6;
  --accent:#9A8FF5; --accent-soft:#2A2547;
  --keep:#4FC49B; --keep-soft:#183328;
  --review:#DFA23C; --review-soft:#382A14;
  --drop:#E8746C; --drop-soft:#3A1F1D;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---------- header ---------- */
header{position:sticky;top:0;z-index:20;background:var(--surface);
  border-bottom:1px solid var(--line);padding:14px 22px;
  display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.brand{display:flex;flex-direction:column;gap:1px;margin-right:auto}
h1{font-family:var(--display);font-size:19px;font-weight:700;margin:0;letter-spacing:-.02em}
.sub{font-size:12px;color:var(--muted)}
.meter{display:flex;align-items:center;gap:11px}
.bar{width:190px;height:7px;border-radius:99px;background:var(--surface-2);overflow:hidden;display:flex}
.bar i{display:block;height:100%}
.bar .b-keep{background:var(--keep)} .bar .b-drop{background:var(--drop)} .bar .b-skip{background:var(--muted)}
.count{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;color:var(--ink-2)}
.hbtn{background:var(--surface-2);border:1px solid var(--line);border-radius:7px;
  padding:7px 13px;font-size:13px;font-weight:500;cursor:pointer}
.hbtn:hover{border-color:var(--accent);color:var(--accent)}
.status{font-size:12px;color:var(--muted);min-width:112px;text-align:right}
.status[data-kind="saved"]{color:var(--keep)}
.status[data-kind="error"]{color:var(--drop)}
.status[data-kind="local"]{color:var(--review)}

/* ---------- shell ---------- */
.shell{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);
  height:calc(100vh - 63px);min-height:0}
@media(max-width:900px){.shell{grid-template-columns:1fr;height:auto}
  .rail{height:auto;max-height:52vh} }

/* ---------- rail ---------- */
.rail{border-right:1px solid var(--line);background:var(--surface);
  display:flex;flex-direction:column;min-height:0}
.filters{padding:11px 13px;border-bottom:1px solid var(--line);display:flex;
  flex-direction:column;gap:9px}
.search{width:100%;padding:7px 10px;border:1px solid var(--line);border-radius:7px;
  background:var(--paper);color:var(--ink);font:inherit;font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{border:1px solid var(--line);background:transparent;border-radius:99px;
  padding:3px 10px;font-size:11.5px;font-weight:500;cursor:pointer;color:var(--muted)}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.list{overflow-y:auto;flex:1;min-height:0}
.row{display:grid;grid-template-columns:4px 1fr;gap:0;width:100%;text-align:left;
  background:none;border:0;border-bottom:1px solid var(--line);cursor:pointer;padding:0}
.row:hover .row-in{background:var(--surface-2)}
.row[aria-current="true"] .row-in{background:var(--accent-soft)}
.stripe{background:var(--muted)}
.stripe.keep{background:var(--keep)} .stripe.review{background:var(--review)} .stripe.drop{background:var(--drop)}
.row-in{padding:9px 12px;display:flex;flex-direction:column;gap:4px;min-width:0}
.row-q{font-size:13px;line-height:1.4;color:var(--ink);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.row-meta{display:flex;align-items:center;gap:7px;font-family:var(--mono);
  font-size:10.5px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--line)}
.dot.yes{background:var(--keep)} .dot.no{background:var(--drop)} .dot.later{background:var(--review)}

/* ---------- detail ---------- */
.detail{overflow-y:auto;min-height:0;padding:26px 30px 130px}
.wrap{max-width:760px;margin:0 auto}
.eyebrow{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
  font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-bottom:13px}
.pill{border-radius:99px;padding:2px 9px;font-size:11px;font-weight:600;
  font-family:var(--sans);letter-spacing:.02em}
.pill.keep{background:var(--keep-soft);color:var(--keep)}
.pill.review{background:var(--review-soft);color:var(--review)}
.pill.drop{background:var(--drop-soft);color:var(--drop)}
.question{font-family:var(--display);font-size:26px;line-height:1.28;font-weight:500;
  letter-spacing:-.02em;margin:0 0 20px;text-wrap:balance}
.signals{display:flex;gap:0;flex-wrap:wrap;border:1px solid var(--line);
  border-radius:10px;overflow:hidden;margin-bottom:22px;background:var(--surface)}
.sig{padding:9px 15px;border-right:1px solid var(--line);flex:1;min-width:104px}
.sig:last-child{border-right:0}
.sig b{display:block;font-family:var(--mono);font-size:17px;font-weight:500;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.sig span{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
h2.sec{font-family:var(--display);font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin:26px 0 10px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:16px 19px;box-shadow:var(--shadow)}
.answer p{margin:0 0 12px} .answer p:last-child{margin-bottom:0}
.answer blockquote{margin:12px 0;padding:10px 15px;border-left:3px solid var(--line);
  background:var(--surface-2);border-radius:0 7px 7px 0;color:var(--ink-2);font-size:14px}
.answer blockquote.verified{border-left-color:var(--keep);background:var(--keep-soft)}
.vtag{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);
  font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--keep);margin-bottom:5px}
.reasons{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px}
.reasons li{font-size:13.5px;color:var(--ink-2);padding-left:16px;position:relative}
.reasons li::before{content:"";position:absolute;left:0;top:8px;width:6px;height:6px;
  border-radius:50%;background:var(--accent)}
.rules{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}
.rule{background:var(--accent-soft);color:var(--accent);border-radius:6px;
  padding:3px 9px;font-size:12px;font-weight:500}
.ctx{font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.7}

/* ---------- verdict bar ---------- */
.bar-fixed{position:fixed;left:var(--rail);right:0;bottom:0;z-index:15;
  background:var(--surface);border-top:1px solid var(--line);
  padding:12px 30px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
@media(max-width:900px){.bar-fixed{left:0;position:sticky}}
.vbtn{border:1px solid var(--line);background:var(--surface-2);border-radius:8px;
  padding:9px 17px;font-size:13.5px;font-weight:600;cursor:pointer;
  display:inline-flex;align-items:center;gap:8px}
.vbtn kbd{font-family:var(--mono);font-size:10.5px;background:var(--paper);
  border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-weight:400}
.vbtn.yes[aria-pressed="true"]{background:var(--keep);border-color:var(--keep);color:#fff}
.vbtn.later[aria-pressed="true"]{background:var(--review);border-color:var(--review);color:#fff}
.vbtn.no[aria-pressed="true"]{background:var(--drop);border-color:var(--drop);color:#fff}
.vbtn[aria-pressed="true"] kbd{background:rgba(255,255,255,.18);border-color:transparent;color:inherit}
.vbtn.flag[aria-pressed="true"]{background:var(--review-soft);border-color:var(--review);color:var(--review)}
.sep{width:1px;height:26px;background:var(--line);margin:0 4px}
.notes{width:100%;min-height:74px;margin-top:9px;padding:10px 12px;border:1px solid var(--line);
  border-radius:8px;background:var(--paper);color:var(--ink);font:inherit;font-size:13.5px;
  line-height:1.5;resize:vertical}
.notes::placeholder{color:var(--muted)}
.flagged-note{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--review);
  font-weight:500;margin-bottom:9px}
.answer-editor{border:1px solid var(--line);border-radius:10px;background:var(--surface);
  padding:16px 19px;box-shadow:var(--shadow)}
.answer-editor.filled{border-color:var(--keep)}
.ae-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:9px}
.ae-head .hbtn{padding:4px 10px;font-size:12px}
.ae-status{font-size:12px;color:var(--muted);margin-left:auto}
.ae-status.done{color:var(--keep)}
.ref{width:100%;min-height:150px;padding:11px 13px;border:1px solid var(--line);
  border-radius:8px;background:var(--paper);color:var(--ink);font-family:var(--mono);
  font-size:13px;line-height:1.6;resize:vertical}
.ref::placeholder{color:var(--muted);font-family:var(--sans)}
.ae-hint{font-size:12px;color:var(--muted);margin:8px 0 0}
.answer-mark{color:var(--keep);font-size:11px}
.near{border:1px solid var(--line);border-left:3px solid var(--review);border-radius:0 10px 10px 0;
  background:var(--surface);padding:13px 17px;margin:0 0 22px}
.near.high{border-left-color:var(--drop)}
.near-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:7px;
  font-size:12px;color:var(--muted)}
.near-sim{font-family:var(--mono);font-weight:500;color:var(--review)}
.near.high .near-sim{color:var(--drop)}
.near-verdict{border-radius:99px;padding:1px 8px;font-size:11px;font-weight:600}
.near-verdict.yes{background:var(--keep-soft);color:var(--keep)}
.near-verdict.no{background:var(--drop-soft);color:var(--drop)}
.near-verdict.later{background:var(--review-soft);color:var(--review)}
.near-q{font-size:14px;color:var(--ink);line-height:1.45}
.near-go{margin-top:9px}
.near-go .hbtn{padding:4px 11px;font-size:12px}
.near-mark{color:var(--review);font-size:10px;font-family:var(--mono)}
.flag-mark{color:var(--review);font-size:11px}
.nav{margin-left:auto;display:flex;gap:8px;align-items:center}
.note{font-size:12px;color:var(--muted)}
.empty{padding:60px 20px;text-align:center;color:var(--muted)}
</style>

<header>
  <div class="brand">
    <h1>Candidate Rules Questions</h1>
    <div class="sub">__SUBTITLE__ · <span id="hcount"></span></div>
  </div>
  <div class="meter">
    <div class="bar" id="bar" role="img" aria-label="Review progress"></div>
    <span class="count" id="progress"></span>
  </div>
  <span class="status" id="status"></span>
  <button class="hbtn" id="export">Export decisions</button>
</header>

<div class="shell">
  <aside class="rail">
    <div class="filters">
      <input class="search" id="search" type="search" placeholder="Search questions…" aria-label="Search questions">
      <div class="chips" id="f-game" role="group" aria-label="Filter by game"></div>
      <div class="chips" id="f-rec" role="group" aria-label="Filter by recommendation"></div>
      <div class="chips" id="f-mine" role="group" aria-label="Filter by your verdict"></div>
    </div>
    <div class="list" id="list"></div>
  </aside>
  <main class="detail" id="detail"></main>
</div>

<div class="bar-fixed">
  <button class="vbtn yes"   data-v="yes"><kbd>1</kbd> Promote</button>
  <button class="vbtn later" data-v="later"><kbd>2</kbd> Maybe</button>
  <button class="vbtn no"    data-v="no"><kbd>3</kbd> Reject</button>
  <button class="hbtn" data-v="">Clear</button>
  <span class="sep"></span>
  <button class="vbtn flag" id="flag"><kbd>f</kbd> Answer looks wrong</button>
  <div class="nav">
    <span class="note">j / k move · 1 2 3 decide · f flag answer</span>
    <button class="hbtn" id="prev">← Prev</button>
    <button class="hbtn" id="next">Next →</button>
  </div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const KEY = "__KEY__";
const RUN_ID = "__RUNID__";
let verdicts = {}, flags = {}, notes = {}, answers = {};
/* Relative on purpose: served by `mm-eval review` this resolves to the backend,
   and opened as a file:// page the fetch simply fails and we fall back to
   localStorage. One page, both ways of running it. */
const API = "api/decisions";
let backend = "local", saveTimer = null;
/* `baseVersion` is the `updated_at` this page last read or wrote, sent back as
   the precondition on every PUT. `dirty` is the ids edited in this session --
   the only ones this page may impose on the file when it turns out someone
   else wrote in the meantime. */
let baseVersion = null;
const dirty = new Set();

function exportDoc() {
  const rows = DATA.filter(d => verdicts[d.id] || flags[d.id] || notes[d.id] || answers[d.id])
    .map(d => ({
      id: d.id, game: d.game, question: d.q,
      filter_recommendation: d.rec, filter_verdict: d.verdict,
      reviewer: verdicts[d.id] || null,
      answer_suspect: !!flags[d.id],
      // The verified reference answer, when the reviewer wrote one. This is
      // what gets promoted into test_cases.yaml -- `note` is for observations
      // about the case and is not promotable.
      corrected_answer: answers[d.id] || null,
      note: notes[d.id] || null
    }));
  return {
    run_id: RUN_ID, reviewed: rows.length, total: DATA.length,
    answers_flagged: DATA.filter(d => flags[d.id]).length,
    answers_corrected: DATA.filter(d => answers[d.id]).length,
    updated_at: new Date().toISOString(),
    decisions: rows
  };
}

/* Accepts the canonical export document, and also the two shapes earlier
   versions of this page left in localStorage, so an in-progress review is not
   thrown away by an upgrade. */
function applyDoc(doc) {
  verdicts = {}; flags = {}; notes = {}; answers = {};
  if (!doc || typeof doc !== "object") return;
  if (Array.isArray(doc.decisions)) {
    for (const r of doc.decisions) {
      if (!r || !r.id) continue;
      if (r.reviewer) verdicts[r.id] = r.reviewer;
      if (r.answer_suspect) flags[r.id] = true;
      if (r.note) notes[r.id] = r.note;
      if (r.corrected_answer) answers[r.id] = r.corrected_answer;
    }
    return;
  }
  if (doc.verdicts) {
    verdicts = doc.verdicts || {}; flags = doc.flags || {};
    notes = doc.notes || {}; answers = doc.answers || {};
    return;
  }
  verdicts = doc;  // oldest shape: a bare {id: verdict} map
}

function setStatus(kind) {
  const el = document.getElementById("status");
  if (!el) return;
  const text = { saving: "Saving…", saved: "Saved", local: "Saved in this browser", error: "Not saved" };
  el.textContent = text[kind] || "";
  el.dataset.kind = kind;
}

async function hydrate() {
  try {
    const r = await fetch(API, { cache: "no-store" });
    if (r.ok) {
      const doc = await r.json();
      applyDoc(doc); baseVersion = doc.updated_at || null;
      backend = "server"; setStatus("saved"); return;
    }
  } catch (e) { /* no backend -- local file, or server down */ }
  try { applyDoc(JSON.parse(localStorage.getItem(KEY) || "null")); } catch (e) {}
  setStatus("local");
}

/* Rebase this session's edits onto a document written by someone else.

   The file wins for every candidate this page has not touched, so another
   reviewer's work -- or a correction written straight into the file -- is
   preserved rather than reverted. Only the ids in `dirty` are re-imposed. */
function rebase(current) {
  const mine = {};
  for (const id of dirty) {
    mine[id] = { v: verdicts[id], f: flags[id], n: notes[id], a: answers[id] };
  }
  applyDoc(current);
  for (const [id, s] of Object.entries(mine)) {
    if (s.v) verdicts[id] = s.v; else delete verdicts[id];
    if (s.f) flags[id] = s.f; else delete flags[id];
    if (s.n) notes[id] = s.n; else delete notes[id];
    if (s.a) answers[id] = s.a; else delete answers[id];
  }
}

const put = doc => fetch(API, {
  method: "PUT", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...doc, base_updated_at: baseVersion })
});

async function flush() {
  let doc = exportDoc();
  if (backend === "server") {
    try {
      let r = await put(doc);
      if (r.status === 409) {
        // The file moved on since we read it. Take what is there, replay this
        // session's own edits on top, and try once. A second conflict means a
        // genuine race rather than a stale page, so let it fall through to the
        // local copy instead of looping.
        const body = await r.json().catch(() => ({}));
        rebase(body.current);
        baseVersion = (body.current || {}).updated_at || null;
        render();
        doc = exportDoc();
        r = await put(doc);
      }
      if (r.ok) {
        const body = await r.json().catch(() => ({}));
        baseVersion = body.updated_at || baseVersion;
        dirty.clear();
        setStatus("saved");
        return;
      }
    } catch (e) { /* fall through to the local copy */ }
    // Keep a local copy when the backend is unreachable, so a dropped
    // connection costs nothing; the banner says it did not reach the file.
    try { localStorage.setItem(KEY, JSON.stringify(doc)); } catch (e) {}
    setStatus("error");
    return;
  }
  try { localStorage.setItem(KEY, JSON.stringify(doc)); setStatus("local"); }
  catch (e) { setStatus("error"); }
}

/* Debounced: a verdict is one keystroke, but a note is dozens. `id` is the
   candidate just edited, recorded so a rebase knows what is ours to keep. */
const save = id => {
  if (id) dirty.add(id);
  setStatus("saving"); clearTimeout(saveTimer); saveTimer = setTimeout(flush, 400);
};
addEventListener("beforeunload", () => { if (saveTimer) { clearTimeout(saveTimer); flush(); } });

const GAMES = [...new Set(DATA.map(d => d.game))];
const RECS = ["keep", "review", "drop"];
const MINE = [["", "All"], ["undecided", "Undecided"], ["yes", "Promoted"], ["later", "Maybe"],
              ["no", "Rejected"], ["flagged", "Answer flagged"],
              ["needs_answer", "Needs an answer"], ["near", "Has a near twin"]];
const state = { game: "", rec: "", mine: "", q: "", i: 0 };

const esc = s => (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = g => g.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

/* The agent's answer is markdown carrying <div data-quote-status="verified"></div>
   immediately before each blockquote that verified against the corpus. Pull those
   markers out before escaping, so a verified quote can be styled as verified
   rather than rendered as literal markup. */
function renderAnswer(src) {
  const MARK = '<div data-quote-status="verified"></div>';
  const lines = (src || "").split("\n");
  let out = "", para = [], quote = [], verified = false;
  const inline = t => esc(t)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  const flushP = () => { if (para.length) { out += "<p>" + inline(para.join(" ")) + "</p>"; para = []; } };
  const flushQ = () => {
    if (!quote.length) return;
    out += '<blockquote class="' + (verified ? "verified" : "") + '">' +
      (verified ? '<span class="vtag">✓ verified quote</span>' : "") +
      quote.map(l => "<p>" + inline(l) + "</p>").join("") + "</blockquote>";
    quote = []; verified = false;
  };
  for (let raw of lines) {
    const line = raw.trim();
    if (line === MARK) { flushP(); flushQ(); verified = true; continue; }
    if (line.startsWith(">")) { flushP(); const t = line.replace(/^>\s?/, ""); if (t) quote.push(t); continue; }
    if (!line) { flushP(); flushQ(); continue; }
    flushQ(); para.push(line);
  }
  flushP(); flushQ();
  return out || "<p class='note'>No answer recorded.</p>";
}

/* Below this the neighbour is almost always a different question about the same
   rule, and showing it is noise. Above it, judge by eye -- 0.914 was the same
   question reworded, 0.890 was Race vs Class ability and both were worth having. */
const NEAR_FLOOR = 0.85;

function nearBlock(d) {
  const n = d.near;
  if (!n || !n.similarity || n.similarity < NEAR_FLOOR) return "";
  const inSet = DATA.some(x => x.id === n.ref);
  const v = inSet ? verdicts[n.ref] : null;
  const label = { yes: "you promoted this", no: "you rejected this", later: "you marked maybe" }[v];
  const high = n.similarity >= 0.90 ? " high" : "";
  return `<div class="near${high}">
    <div class="near-head">
      <span class="near-sim">${n.similarity.toFixed(3)} similar</span>
      <span>${n.kind === "existing" ? "to an existing test case" : "to another candidate"}</span>
      <span>${esc(n.ref || "")}</span>
      ${label ? `<span class="near-verdict ${v}">${label}</span>` : ""}
    </div>
    <div class="near-q">${esc(n.question || "")}</div>
    ${inSet ? `<div class="near-go"><button class="hbtn" data-goto="${esc(n.ref)}">Go to it</button></div>` : ""}
  </div>`;
}

function filtered() {
  const q = state.q.toLowerCase();
  return DATA.filter(d => {
    if (state.game && d.game !== state.game) return false;
    if (state.rec && d.rec !== state.rec) return false;
    if (state.mine === "undecided" && verdicts[d.id]) return false;
    if (state.mine === "flagged" && !flags[d.id]) return false;
    if (state.mine === "near" && !(d.near && d.near.similarity >= NEAR_FLOOR)) return false;
    // Outstanding work: the draft was judged wrong, the question is going in,
    // and nobody has written the replacement yet.
    if (state.mine === "needs_answer" &&
        !(flags[d.id] && verdicts[d.id] === "yes" && !answers[d.id])) return false;
    if (state.mine && !["undecided", "flagged", "needs_answer", "near"].includes(state.mine)
        && verdicts[d.id] !== state.mine) return false;
    if (q && !(d.q.toLowerCase().includes(q) || d.id.includes(q))) return false;
    return true;
  });
}

function chips(el, opts, key) {
  el.innerHTML = opts.map(([v, label]) =>
    `<button class="chip" data-v="${esc(v)}" aria-pressed="${state[key] === v}">${esc(label)}</button>`).join("");
  el.onclick = e => {
    const b = e.target.closest(".chip"); if (!b) return;
    state[key] = b.dataset.v; state.i = 0; render();
  };
}

function render() {
  const rows = filtered();
  if (state.i >= rows.length) state.i = Math.max(0, rows.length - 1);

  chips(document.getElementById("f-game"), [["", "Both games"], ...GAMES.map(g => [g, pretty(g)])], "game");
  chips(document.getElementById("f-rec"), [["", "All"], ...RECS.map(r => [r, r])], "rec");
  chips(document.getElementById("f-mine"), MINE, "mine");

  document.getElementById("list").innerHTML = rows.length ? rows.map((d, i) => `
    <button class="row" aria-current="${i === state.i}" data-i="${i}">
      <span class="stripe ${d.rec}"></span>
      <span class="row-in">
        <span class="row-q">${esc(d.q)}</span>
        <span class="row-meta">
          <span class="dot ${verdicts[d.id] || ""}"></span>
          <span>${esc(d.id)}</span><span>·</span><span>${esc(d.verdict)}</span>
          ${flags[d.id] ? '<span class="flag-mark">&#9873;</span>' : ""}
          ${answers[d.id] ? '<span class="answer-mark">&#9998;</span>' : ""}
          ${d.near && d.near.similarity >= NEAR_FLOOR
            ? `<span class="near-mark">~${d.near.similarity.toFixed(2)}</span>` : ""}
        </span>
      </span>
    </button>`).join("") : '<div class="empty">No candidates match these filters.</div>';

  const done = DATA.filter(d => verdicts[d.id]).length;
  const y = DATA.filter(d => verdicts[d.id] === "yes").length;
  const l = DATA.filter(d => verdicts[d.id] === "later").length;
  const n = DATA.filter(d => verdicts[d.id] === "no").length;
  const pc = v => (v / DATA.length * 100).toFixed(2) + "%";
  document.getElementById("bar").innerHTML =
    `<i class="b-keep" style="width:${pc(y)}"></i><i class="b-skip" style="width:${pc(l)}"></i><i class="b-drop" style="width:${pc(n)}"></i>`;
  document.getElementById("progress").textContent = `${done}/${DATA.length}`;
  const fl = DATA.filter(d => flags[d.id]).length;
  const need = DATA.filter(d => flags[d.id] && verdicts[d.id] === "yes" && !answers[d.id]).length;
  document.getElementById("hcount").textContent =
    `${rows.length} shown · ${y} promoted`
    + (fl ? ` · ${fl} answers flagged` : "")
    + (need ? ` · ${need} awaiting a written answer` : "");

  const d = rows[state.i];
  const det = document.getElementById("detail");
  if (!d) { det.innerHTML = '<div class="empty">Nothing to review here.</div>'; return; }
  det.innerHTML = `<div class="wrap">
    <div class="eyebrow">
      <span class="pill ${d.rec}">${esc(d.rec)}</span>
      <span>${esc(d.id)}</span><span>·</span><span>${esc(pretty(d.game))}</span>
      <span>·</span><span>${esc(d.verdict)}</span>
      ${d.dupe !== "unique" ? `<span>·</span><span>${esc(d.dupe)}</span>` : ""}
    </div>
    <h2 class="question">${esc(d.q)}</h2>
    <div class="signals">
      <div class="sig"><b>${d.chunks}</b><span>passages cited</span></div>
      <div class="sig"><b>${d.quotes}</b><span>verified quotes</span></div>
      <div class="sig"><b>${d.w}</b><span>words</span></div>
      <div class="sig"><b>${d.books.length}</b><span>rulebooks</span></div>
    </div>
    ${nearBlock(d)}
    <h2 class="sec">Draft answer — verify this before promoting</h2>
    ${flags[d.id] ? '<div class="flagged-note">&#9873; You marked this answer as wrong or doubtful</div>' : ""}
    <div class="card answer">${renderAnswer(d.resp)}</div>
    <textarea class="notes" id="notes" placeholder="Note — an observation about this case. Not promoted; use the reference answer below for the answer itself."></textarea>

    <h2 class="sec">Reference answer</h2>
    <div class="answer-editor" id="ae">
      <div class="ae-head">
        <button class="hbtn" id="seed-answer">Start from the draft</button>
        <button class="hbtn" id="clear-answer">Clear</button>
        <span class="ae-status" id="ae-status"></span>
      </div>
      <textarea class="ref" id="ref" placeholder="The answer you have verified. This is what goes into test_cases.yaml — leave it empty to promote the draft above as-is."></textarea>
      <p class="ae-hint">Keep the rulebook quotes: they become the case's <code>evidence:</code>,
        which is what lets <code>mm-eval retrieval</code> score it.</p>
    </div>
    <h2 class="sec">Why the filter said “${esc(d.rec)}”</h2>
    <div class="card">
      <ul class="reasons">${(d.reasons || []).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      ${d.judge ? `<p style="margin:12px 0 0;font-size:13.5px;color:var(--ink-2)">${esc(d.judge)}</p>` : ""}
      ${(d.rules || []).length ? `<div class="rules">${d.rules.map(r => `<span class="rule">${esc(r)}</span>`).join("")}</div>` : ""}
    </div>
    <h2 class="sec">Provenance</h2>
    <div class="card ctx">
      seed &nbsp;${esc(d.seed)}<br>
      concepts &nbsp;${esc((d.concepts || []).join(", ") || "—")}<br>
      rulebooks &nbsp;${esc((d.books || []).join(", ") || "—")}
    </div>
  </div>`;

  document.querySelectorAll(".vbtn[data-v]").forEach(b =>
    b.setAttribute("aria-pressed", verdicts[d.id] === b.dataset.v));
  document.getElementById("flag").setAttribute("aria-pressed", !!flags[d.id]);
  const nt = document.getElementById("notes");
  if (nt) {
    nt.value = notes[d.id] || "";
    nt.oninput = () => {
      if (nt.value.trim()) notes[d.id] = nt.value; else delete notes[d.id];
      save(d.id);
    };
  }
  const ref = document.getElementById("ref");
  if (ref) {
    ref.value = answers[d.id] || "";
    const paint = () => {
      const has = !!answers[d.id];
      document.getElementById("ae").classList.toggle("filled", has);
      const st = document.getElementById("ae-status");
      st.textContent = has ? "Will be promoted instead of the draft"
                           : (flags[d.id] ? "Draft flagged wrong — needs one" : "");
      st.classList.toggle("done", has);
    };
    ref.oninput = () => {
      if (ref.value.trim()) answers[d.id] = ref.value; else delete answers[d.id];
      paint(); save(d.id);
    };
    // Most corrections are edits, not rewrites: the draft is usually mostly
    // right, and retyping the verified quotes by hand invites transcription
    // errors in the text that becomes the case's evidence.
    document.getElementById("seed-answer").onclick = () => {
      ref.value = d.resp.replace(/<div data-quote-status="verified"><\/div>/g, "").trim();
      answers[d.id] = ref.value; paint(); save(d.id); ref.focus();
    };
    document.getElementById("clear-answer").onclick = () => {
      ref.value = ""; delete answers[d.id]; paint(); save(d.id);
    };
    paint();
  }
  const cur = document.querySelector('.row[aria-current="true"]');
  if (cur) cur.scrollIntoView({ block: "nearest" });
  det.scrollTop = 0;
}

function setVerdict(v) {
  const rows = filtered(); const d = rows[state.i]; if (!d) return;
  if (v) verdicts[d.id] = v; else delete verdicts[d.id];
  save(d.id);
  // Advance only when the list is not filtered to the thing just changed --
  // otherwise the row vanishes underneath and the index lands somewhere random.
  const sticky = state.mine === "";
  if (v && sticky && state.i < rows.length - 1) state.i++;
  render();
}
function toggleFlag() {
  const d = filtered()[state.i]; if (!d) return;
  if (flags[d.id]) delete flags[d.id]; else flags[d.id] = true;
  save(d.id); render();
}
const move = n => { const r = filtered(); state.i = Math.min(Math.max(0, state.i + n), Math.max(0, r.length - 1)); render(); };

document.getElementById("detail").addEventListener("click", e => {
  const b = e.target.closest("[data-goto]"); if (!b) return;
  // Clear the verdict filter first: the neighbour is usually in a different
  // bucket, and jumping into a filtered list that excludes it does nothing.
  const target = b.dataset.goto;
  state.mine = ""; state.rec = ""; state.q = "";
  document.getElementById("search").value = "";
  const idx = filtered().findIndex(x => x.id === target);
  if (idx >= 0) { state.i = idx; render(); }
});

document.getElementById("list").onclick = e => {
  const b = e.target.closest(".row"); if (!b) return;
  state.i = +b.dataset.i; render();
};
document.querySelectorAll("[data-v]").forEach(b => {
  if (b.classList.contains("chip")) return;
  b.onclick = () => setVerdict(b.dataset.v);
});
document.getElementById("flag").onclick = toggleFlag;
document.getElementById("prev").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);
document.getElementById("search").oninput = e => { state.q = e.target.value; state.i = 0; render(); };

addEventListener("keydown", e => {
  if (e.target.matches("input,textarea") || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if (k === "j" || k === "arrowdown") { e.preventDefault(); move(1); }
  else if (k === "k" || k === "arrowup") { e.preventDefault(); move(-1); }
  else if (k === "1") setVerdict("yes");
  else if (k === "2") setVerdict("later");
  else if (k === "3") setVerdict("no");
  else if (k === "f") { e.preventDefault(); toggleFlag(); }
  else if (k === "u" || k === "0") setVerdict("");
});

document.getElementById("export").onclick = async () => {
  const body = JSON.stringify(exportDoc(), null, 2);
  const btn = document.getElementById("export");
  const downloads = (typeof claude !== "undefined" && claude.use)
    ? await claude.use("downloads") : null;
  if (downloads) {
    try {
      await downloads.save({ filename: "review-decisions-__RUNID__.json", data: body });
      btn.textContent = "Exported"; setTimeout(() => btn.textContent = "Export decisions", 1800);
      return;
    } catch (e) { /* viewer declined -- fall through to clipboard */ }
  }
  try { await navigator.clipboard.writeText(body); btn.textContent = "Copied to clipboard"; }
  catch (e) { btn.textContent = "Export unavailable"; }
  setTimeout(() => btn.textContent = "Export decisions", 2200);
};

render();
hydrate().then(render);
</script>
"""


def load_candidates(layout: "EvalGenLayout", game_id: str) -> list[dict[str, Any]]:
    """Answered candidates for one game, joined across the three step outputs.

    Mining, dedupe and answers are separate files now, so this is a real join on
    candidate id. Candidates with no answer, or whose answer errored, are
    skipped: there is nothing to review, and a card with no answer reads as a
    bug rather than as a gap.
    """
    from meeplemate.eval.eval_gen_store import (
        iter_candidates, load_json, read_answers,
    )

    dedupe = (load_json(layout.dedupe_file(game_id), {}) or {}).get("candidates", {})
    answers = read_answers(layout, game_id)

    out: list[dict[str, Any]] = []
    for cand, seed_rec in iter_candidates(layout, game_id):
        ans = (answers.get(cand["id"]) or {}).get("answer")
        if not isinstance(ans, Mapping) or ans.get("status") == "error":
            continue
        seed = seed_rec.get("seed") or {}
        page = seed.get("page_num") or seed.get("page_ordinal", "?")
        out.append({
            "id": cand["id"], "game": game_id, "q": cand["question"],
            "w": cand.get("word_count") or len(cand["question"].split()),
            "rec": ans["recommendation"], "verdict": ans.get("verdict", ""),
            "reasons": list(ans.get("reasons") or []),
            "judge": ans.get("judge_reasoning", ""),
            "rules": list(ans.get("rules_involved") or []),
            "quotes": ans.get("verified_quotes", 0),
            "chunks": ans.get("evidence_chunks", 0),
            "books": list(ans.get("rulebooks") or []),
            "resp": ans.get("response", ""),
            "seed": f"{seed.get('rulebook', '?')} p.{page}",
            "concepts": list(seed_rec.get("concepts") or []),
            "dupe": (dedupe.get(cand["id"]) or {}).get("status", "unique"),
            # The nearest neighbour is recorded for every candidate, not only
            # flagged ones. No cosine threshold separates "same question
            # reworded" from "distinct question about one rule" -- surfacing the
            # neighbour turns that into a one-second human judgement instead.
            "near": (dedupe.get(cand["id"]) or {}).get("nearest") or None,
        })
    return out


def load_run(layout: "EvalGenLayout", games: Iterable[str]) -> list[dict[str, Any]]:
    """Every answered candidate across the named games in one group run.

    A game with no mined seeds raises: an empty review page looks identical to a
    run where nothing was recommended, and the two need different responses.
    """
    rows: list[dict[str, Any]] = []
    for game in games:
        if not layout.seeds_dir(game).exists():
            raise FileNotFoundError(
                f"No mined seeds for {game!r} at {layout.seeds_dir(game)}")
        rows.extend(load_candidates(layout, game))
    return rows


def empty_decisions(run_id: str, total: int = 0) -> dict[str, Any]:
    return {"run_id": run_id, "reviewed": 0, "total": total,
            "answers_flagged": 0, "decisions": []}


def build_page(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    games: Iterable[str],
    title: str = "Candidate Rules Questions",
) -> str:
    """Inline the candidates into the template and return the whole document."""
    pretty = ", ".join(g.replace("_", " ").title() for g in games)
    # Escaping "<" stops a "</script>" inside any answer body from closing the
    # data block early; JSON parses the \u003c escape back transparently.
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    return (PAGE_TEMPLATE
            .replace("__DATA__", payload)
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", f"Mined {run_id} &middot; {pretty}")
            .replace("__RUNID__", run_id)
            .replace("__KEY__", f"boardbarian-review-{run_id}"))
