#!/usr/bin/env python
"""The single-file explorer's markup, styling and behaviour.

Kept as one template with two substitution points -- the Plotly bundle and the
payload -- rather than assembled by string concatenation, which ADR 0005 asks
for so the JavaScript stays readable in a diff.

**What this file decides, and it is the only interesting thing about it:** how a
map that should not be read is drawn differently from one that should. Three
mechanisms, matching the three levels `payload.space_verdict` produces:

* a panel whose space is `unreadable` gets a red rule, a headline, and its
  points drawn hollow -- you can see where they are, and you cannot mistake the
  picture for a result;
* an individual protein whose position is not faithful is drawn hollow in every
  panel, even inside an otherwise trustworthy space;
* a withheld cross-space number renders as the word *withheld* with its reason,
  never as a blank cell, because a blank reads as zero.

The alternative -- a tooltip, or a collapsed "diagnostics" pane -- was rejected
for the reason ADR 0014 gives for running diagnostics unconditionally: a caveat
you have to open is read by exactly the people who already suspected it.
"""

from __future__ import annotations

__all__ = ["render"]

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__ — multi-space explorer</title>
<script type="text/javascript">__PLOTLY__</script>
<style>
:root {
  --ink: #14171a; --muted: #5b6570; --line: #d8dee4; --bg: #ffffff;
  --panel: #f7f9fa; --ok: #2f7d4f; --caution: #9a6a00;
  --bad: #b3261e;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
header { padding: 18px 22px 12px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 2px; font-size: 19px; font-weight: 600; }
.sub { color: var(--muted); font-size: 13px; }
.controls {
  display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end;
  padding: 12px 22px; border-bottom: 1px solid var(--line); background: var(--panel);
}
.control { display: flex; flex-direction: column; gap: 3px; }
.control label { font-size: 11px; text-transform: uppercase;
                 letter-spacing: .04em; color: var(--muted); }
select, button {
  font: inherit; padding: 5px 8px; border: 1px solid var(--line);
  border-radius: 4px; background: #fff; color: var(--ink);
}
button { cursor: pointer; }
button.on { background: var(--ink); color: #fff; border-color: var(--ink); }
#grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 14px; padding: 16px 22px;
}
.panel { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: #fff; }
.panel.level-caution { border-color: var(--caution); }
.panel.level-unreadable { border-color: var(--bad); }
.panel h2 { margin: 0; padding: 9px 12px 8px; font-size: 14px; font-weight: 600;
            border-bottom: 1px solid var(--line); }
.verdict { padding: 7px 12px; font-size: 12px; border-bottom: 1px solid var(--line); }
.verdict.level-ok { color: var(--ok); background: #f2f9f5; }
.verdict.level-caution { color: var(--caution); background: #fdf7e8; }
.verdict.level-unreadable { color: var(--bad); background: #fdf1f0; font-weight: 600; }
.verdict ul { margin: 5px 0 0; padding-left: 17px; font-weight: 400; }
.shares { padding: 6px 12px; font-size: 12px; color: #333;
  border-bottom: 1px solid var(--line); background: #fbfbfc; }
.shares .drift { color: var(--caution); }
.column-warn { border-top: 0; background: #fdf9ef; color: #7a4e00;
  border-left: 3px solid var(--caution); }
.column-warn b { color: var(--caution); }
.plot { height: 330px; }
/* The similarity matrix needs to be square and readable; 330px makes a 367-row
   heatmap four cells tall per cluster. */
.plot-tall { height: 460px; }
.sheets { display: flex; flex-wrap: wrap; gap: 2px; padding: 8px 22px 0; }
.sheets button { border-radius: 4px 4px 0 0; border-bottom-color: transparent; }
.sheets button.on { background: var(--ink); color: #fff; border-color: var(--ink); }
/* The source tags every section real / mixed / synthetic / method (4, 5 notes).
   A reader must never have to guess whether a panel shows measurements or a
   drawing of an argument, so the tag is on the panel, not in a caption. */
.colour-key { display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  padding: 8px 22px 2px; font-size: 12px; color: var(--muted); }
.key-chip { display: inline-flex; align-items: center; gap: 5px; }
.key-chip i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.key-ramp { display: inline-block; width: 110px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }
.key-note { font-style: italic; }
/* Floated, the tag dropped out of the header and landed on the divider once a
   title wrapped. The header is a flex row so the tag stays in it at any width. */
.panel h2 { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.panel h2 .tag { flex: none; }
.tag { font-size: 10px; text-transform: uppercase; letter-spacing: .05em;
  padding: 2px 6px; border-radius: 3px; font-weight: 600; }
.tag-real { background: #eaf4ee; color: var(--ok); }
.tag-mixed { background: #fdf7e8; color: var(--caution); }
.tag-synthetic { background: #f1f0f6; color: #5a4f8a; }
.tag-method { background: #eef1f4; color: var(--muted); }
.question { padding: 7px 12px; font-size: 12px; color: var(--muted);
  border-bottom: 1px solid var(--line); }
.awaiting .why { display: block; margin-top: 6px; font-size: 11.5px; }
/* 11px at #7a838c fails WCAG AA on this background. Darkened to meet it -- a
   caption naming the work that fills a panel is the last thing to make faint. */
.awaiting .fills { display: block; margin-top: 4px; font-size: 11px; color: #4a535c; }
/* A panel with nothing to draw says so, in the panel's own footprint. The
   source document's rule (7.03 E2) is that refusals are shown as refusals and
   never as blanks; the same reasoning applies to a panel awaiting an input. */
.awaiting { height: 330px; display: flex; align-items: center; justify-content: center;
  text-align: center; padding: 0 22px; color: var(--muted); font-size: 12.5px;
  background: repeating-linear-gradient(45deg, #fcfcfd, #fcfcfd 9px, #f6f7f9 9px, #f6f7f9 18px); }
/* The table must scroll inside its own box. Letting it size the page made the
   whole document scroll sideways on a narrow window. */
/* A panel catalogued on one sheet and drawn on another says where it went.
   Deliberately NOT the `.awaiting` hatch: nothing is missing here, so the card
   must not be given the visual language of a refusal. */
.elsewhere { padding: 12px 14px; font-size: 12px; color: var(--muted); }
/* The line that decided a count, under the column header that prints it. Not
   bold and not the label's size -- it qualifies the heading, it is not a second
   heading. */
.cutoff { display: block; font-weight: 400; font-size: 10.5px; color: var(--muted);
  text-transform: none; letter-spacing: 0; margin-top: 2px; }
/* The sweep under the cluster count. Small and muted: it qualifies the number
   above it and must not compete with it for the reader's eye. */
.sweep { display: block; margin-top: 3px; font-size: 10.5px; color: var(--muted);
  line-height: 1.5; }
.sweep b { color: var(--ink); }
.scroll-x { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: .04em; }
td.withheld { color: var(--muted); font-style: italic; }
span.withheld { color: var(--muted); font-style: italic; }
/* Selection egress. A textarea rather than a copy button: clipboard.writeText
   does exist over file:// but is gated on document focus and rejects with
   NotAllowedError without it, so a button would work sometimes and fail
   silently otherwise. Selecting text always works. */
.egress-label { margin: 8px 0 2px; color: var(--muted); }
textarea.egress {
  width: 100%; box-sizing: border-box; font: 12px/1.45 ui-monospace, monospace;
  resize: vertical; overflow-y: auto; padding: 6px;
  border: 1px solid var(--line); border-radius: 4px;
  background: var(--bg); color: inherit;
}
ul.egress-links { margin: 8px 0 0; padding-left: 18px; max-height: 220px; overflow-y: auto; }
ul.egress-links li { margin: 2px 0; }
section { padding: 6px 22px 20px; }
section h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em;
             color: var(--muted); margin: 18px 0 7px; }
footer { border-top: 1px solid var(--line); background: var(--panel);
         padding: 14px 22px 26px; font-size: 12px; color: var(--muted); }
footer code { font-size: 11.5px; }
.legend { font-size: 12px; color: var(--muted); padding: 0 22px 4px; }
.legend b { color: var(--ink); font-weight: 600; }
/* The one phrase on the page that says "do not trust this point". It is worth
   the colour: a reader who skims the legend still has to see which marker the
   sentence is about. */
.legend .flagged-key { color: var(--bad); }
.legend .flagged-key::before { content: ""; display: inline-block; width: 9px; height: 9px;
  border-radius: 50%; border: 2.6px solid var(--bad); margin-right: 5px;
  vertical-align: baseline; }
/* Beside the disagreement ramp, where high-is-interesting and do-not-read are
   most easily run together. */
.key-caveat { flex-basis: 100%; margin-top: 4px; font-size: 11.5px; color: #4a535c;
  border-left: 3px solid var(--caution); background: #fdf9ef; padding: 5px 8px; }
.key-caveat b { color: var(--caution); }
/* The fold-out. Collapsed by default, because a verdict a reader has to open
   is read only by people who already suspected it -- which is exactly why the
   verdict banner above is NOT in here. This holds the slower material: what is
   on the axes, what the numbers are, and which file computed them. */
.explain { border-bottom: 1px solid var(--line); background: #fcfdfd; }
.explain > summary { cursor: pointer; padding: 6px 12px; font-size: 11.5px;
  color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.explain > summary:hover { color: var(--ink); }
.explain .body { padding: 2px 12px 10px; font-size: 12.5px; }
.explain p { margin: 0 0 7px; }
.explain code { background: #eef1f4; padding: 0 3px; border-radius: 3px; font-size: 11.5px; }
/* A hazard set in body text is a hazard the reader skims. */
.explain .hazard { border-left: 3px solid var(--caution); padding-left: 8px;
  background: #fdf9ef; padding-top: 4px; padding-bottom: 4px; }
.explain .hazard b { color: var(--caution); }
.explain .src { color: var(--muted); font-size: 11px; margin-top: 8px; }
/* The diagnostics' own sentences. A section label, then that section's lines --
   no hazard bar, because these are not this page's caveats about a panel, they
   are the run's own output about a space. */
.explain.wrote .wrote-label { font-size: 11px; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--muted); margin: 8px 0 3px; }
.explain.wrote ul { margin: 0 0 4px; padding-left: 18px; }
.explain.wrote li { margin-bottom: 5px; }
/* An all-clear is not a warning and must not be read as one. */
.explain.wrote li.cleared { color: var(--ok); }
/* A catalogue panel's own body. Bounded and scrolled inside the card, so a
   367-row table cannot set the height of the whole sheet. */
.tablebox { max-height: 330px; overflow: auto; }
.tablebox thead th { position: sticky; top: 0; background: #fff; z-index: 1;
  box-shadow: inset 0 -1px 0 var(--line); }
.panel-bar { display: flex; gap: 10px; align-items: center; padding: 7px 12px;
  border-bottom: 1px solid var(--line); background: #fbfbfc;
  font-size: 11.5px; color: var(--muted); }
.panel-bar input { font: inherit; font-size: 12px; padding: 4px 7px; flex: 1;
  border: 1px solid var(--line); border-radius: 4px; }
td.mono, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
/* The diagnostics report. A column of sections, each of which either carries
   its numbers or refuses in place -- so `.refused` is styled like a panel's
   `awaiting` block but WITHOUT its fixed height: a 330px hole between two
   filled sections reads as a rendering failure rather than as a refusal. */
.report-section { border-top: 1px solid var(--line); }
.report-section:first-child { border-top: 0; }
.report-section h4 { margin: 0; padding: 9px 12px 1px; font-size: 11px;
  text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.report-section .lead { padding: 0 12px 7px; font-size: 12px; color: var(--muted); }
.refused { margin: 0 12px 11px; padding: 7px 10px; font-size: 12px; color: #4a535c;
  border-left: 3px solid var(--caution); background: #fdf9ef; }
.refused b { color: var(--caution); }
.refused .fills { display: block; margin-top: 4px; font-size: 11px; }
/* A table panel's own caption: where its rows were read from. Above the table
   rather than under it, because a reader who does not know the rows are read
   out of the enforcing guard has no reason to trust them. */
.table-note { padding: 7px 12px; font-size: 12px; color: var(--muted);
  border-bottom: 1px solid var(--line); background: #fbfbfc; }
/* The 1.02 option plates. A grid inside the card, so six plates do not become
   six full-width blocks a reader has to scroll past to compare. */
.plates { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px; padding: 11px 12px; }
.plate { border: 1px solid var(--line); border-radius: 5px; padding: 9px 10px;
  background: #fcfdfd; font-size: 12px; }
.plate h3 { margin: 5px 0 3px; font-size: 12.5px; }
.plate .formula { margin: 0 0 3px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: var(--muted); }
.plate .plate-tag { font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); margin-bottom: 5px; }
.plate dl { margin: 0; }
.plate .field { margin-bottom: 5px; }
.plate dt { font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); }
.plate dd { margin: 1px 0 0; }
/* What the plate cannot answer, and how it fails, in the caution colour. These
   are the half of a plate that stops a reader adopting a geometry for a
   question it cannot answer. */
.plate .field.bad dd { color: #7a4e00; }
.plate .field.quiet dd { color: var(--muted); }
.plate-state { float: right; font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .04em; padding: 2px 5px; border-radius: 3px; font-weight: 600;
  background: #eef1f4; color: var(--muted); }
/* Three states and never two: built here, buildable and not asked for, and not
   implemented at all are three different facts. */
.plate-state.state-built { background: #eaf4ee; color: var(--ok); }
.plate-state.state-available { background: #fdf7e8; color: var(--caution); }
.plate-state.state-not { background: #eef1f4; color: var(--muted); }
.plate-hazard { margin-top: 7px; border-left: 3px solid var(--caution);
  background: #fdf9ef; padding: 5px 8px; font-size: 11.5px; }
.plate-hazard b { color: var(--caution); }
/* The perturbation grid. Chips rather than numbers: no scan has been run, and a
   cell carrying a number here would be read as a measurement of this cohort. */
.gridtable td.cell { font-size: 10.5px; text-transform: uppercase;
  letter-spacing: .03em; white-space: nowrap; }
.gridtable th .cost { display: block; font-size: 10px; font-weight: 400;
  text-transform: none; letter-spacing: 0; color: var(--muted); }
/* Four states and they must stay four: a cell that works, a cell flat by
   construction, a control that is not a measurement, and one the source simply
   does not annotate -- which is NOT the source calling it useless. */
.kind-informative { color: var(--ok); }
.kind-empty { color: var(--bad); }
.kind-control { color: #5a4f8a; }
.kind-unannotated { color: var(--muted); }
.gridnotes { padding: 4px 12px 11px; font-size: 12px; }
.gridnotes p { margin: 0 0 7px; }
.gridnotes .kind { font-size: 9.5px; text-transform: uppercase; letter-spacing: .04em; }
/* The pipeline diagram's boxes. CSS boxes and not SVG: 7.03 E4 bans a build
   step, and a box with a caption under it is what 2.01 actually asks for. */
.flow { display: flex; flex-wrap: wrap; gap: 8px; padding: 11px 12px; }
.flowbox { border: 1px solid var(--line); border-radius: 5px; padding: 7px 9px;
  background: #fcfdfd; font-size: 11.5px; min-width: 150px; flex: 1 1 150px; }
.flowbox b { display: block; font-size: 12px; }
.flowbox span { display: block; color: var(--muted); }
.flowbox .out { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
/* The block whose tensor is the similarity matrix. Marked, because it is the
   step readers of this pipeline most often get wrong. */
.flowbox-note { border-color: var(--caution); background: #fdf9ef; }
.flowbox .surprise { margin-top: 5px; color: #7a4e00; font-size: 11px; }
td .out { display: block; color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
/* The verdict word inside a report row, where the banner's full-width tint
   would fight the table. Colour only. */
.v-ok { color: var(--ok); }
.v-caution { color: var(--caution); }
.v-unreadable { color: var(--bad); font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub" id="subtitle"></div>
</header>

<div class="controls">
  <div class="control" id="cohort-control" style="display:none">
    <label for="cohort">Example</label>
    <select id="cohort"></select>
  </div>
  <div class="control">
    <label for="overlay">Colour by</label>
    <select id="overlay"></select>
  </div>
  <div class="control">
    <label for="reducer">Layout</label>
    <select id="reducer"></select>
  </div>
  <div class="control">
    <label>&nbsp;</label>
    <button id="disagreement"
      title="Colour each panel by how much its neighbourhoods disagree with the other spaces"
      >Disagreement mode</button>
  </div>
  <div class="control">
    <label>&nbsp;</label>
    <button id="clear">Clear selection</button>
  </div>
</div>

<div id="sheets" class="sheets"></div>

<div id="colour-key" class="colour-key" style="display:none"></div>

<div class="legend" id="legend"></div>

<div id="grid"></div>

<section id="sheet-body"></section>

<section id="maps-extra">
  <h3>Cross-space comparison</h3>
  <div id="comparisons" class="scroll-x"></div>
  <h3>Selection</h3>
  <div id="inspector">Nothing selected.</div>
</section>

<footer id="provenance"></footer>

<script type="text/javascript">
const PAYLOAD = __PAYLOAD__;

// The judgement lines, read from the payload rather than retyped here. A
// threshold typed into the template can drift from the one the diagnostics
// enforce, and then the picture and the verdict disagree.
const THRESHOLDS = PAYLOAD.thresholds || {};
const COIN_FLIP = THRESHOLDS.coin_flip;

// The cutoff that decides a position is not readable, in words, for the header
// of every count it decided. Read from the payload and never retyped:
// `payload._thresholds()` ships `distorted` for exactly this, and until now it
// reached the browser and nothing read it -- the page printed "positions not
// readable: 306 of 367" with the line that produced 306 stated nowhere.
//
// Returns "" rather than a partial sentence when the payload has no thresholds,
// so a page built by an older writer loses the note instead of gaining
// "at or below undefined".
function distortedCutoffNote() {
  return THRESHOLDS.distorted === undefined
    ? ""
    : `trustworthiness or continuity at or below ${THRESHOLDS.distorted}`;
}

const state = { overlay: "__none__", reducer: null, disagreement: false,
                selected: new Set(), sheet: "maps" };

const el = (id) => document.getElementById(id);

// One page can carry several cohorts. A page built without `--also-cohort` has
// no `cohorts` key, so it becomes a single-cohort list here and every path
// below is identical to what it was -- the selector simply never appears.
const COHORTS = PAYLOAD.cohorts || [Object.assign({}, PAYLOAD, {
  cohort_name: PAYLOAD.cohort_name || PAYLOAD.analysis_name,
})];
let active = COHORTS[0];
let spaces = active.spaces;

// Every reducer any space produced. A space missing the chosen one is drawn
// with whatever it has, and says so, rather than vanishing from the grid.
let reducers = [];

// Everything that depends on WHICH cohort is showing. Called once at startup and
// again on every switch, so the two paths cannot drift -- a switch that rebuilt
// the panels but not the dropdowns would leave a control pointing at a feature
// the new cohort does not have.
function applyCohort(index) {
  active = COHORTS[index];
  spaces = active.spaces;
  reducers = [...new Set(spaces.flatMap((s) => Object.keys(s.embeddings)))].sort();
  state.reducer = reducers[0] || null;
  state.selected = new Set();
  state.overlay = "__none__";

  // The title is the cohort's, not the page's. It was a static substitution, so
  // switching cohorts left the previous cohort's name above the new cohort's
  // numbers -- a header and a subtitle disagreeing about which data is shown.
  document.title = `${active.cohort_name} — multi-space explorer`;
  document.querySelector("h1").textContent = active.cohort_name;
  el("subtitle").textContent =
    `${spaces.length} space(s) over ${active.provenance.n_proteins} proteins` +
    ` · k=${active.provenance.diagnostics_k}` +
    ` · cohort rule: ${active.provenance.cohort_rule || "n/a"}`;

  const overlaySelect = el("overlay");
  overlaySelect.textContent = "";
  overlaySelect.append(new Option("nothing (uniform)", "__none__"));
  const clusterGroup = document.createElement("optgroup");
  clusterGroup.label = "this space's own clusters";
  clusterGroup.append(new Option("cluster", "__cluster__"));
  overlaySelect.append(clusterGroup);
  const featureNames = Object.keys(active.overlays).sort();
  if (featureNames.length) {
    const group = document.createElement("optgroup");
    group.label = "features";
    featureNames.forEach((name) => group.append(new Option(name, name)));
    overlaySelect.append(group);
  }
  // ADR 0005 item 5: diagnostics live in the same dropdown so they are one click
  // away, and in their own group so a QC number is never mistaken for biology.
  const diagGroup = document.createElement("optgroup");
  diagGroup.label = "diagnostics — not biological findings";
  diagGroup.append(new Option("position is readable", "__readable__"));
  overlaySelect.append(diagGroup);

  const reducerSelect = el("reducer");
  reducerSelect.textContent = "";
  reducers.forEach((r) => reducerSelect.append(new Option(r, r)));

  // The grid caches its DOM behind this flag. Switching cohorts changes how many
  // panels there are and what they are called, so the cache has to go with it --
  // otherwise the new cohort draws into the old cohort's divs.
  const grid = el("grid");
  grid.textContent = "";
  delete grid.dataset.built;
  rebuildDisagreement();
}

const overlaySelect = el("overlay");
overlaySelect.onchange = (e) => { state.overlay = e.target.value; draw(); };
el("reducer").onchange = (e) => { state.reducer = e.target.value; draw(); };
el("clear").onclick = () => { state.selected = new Set(); draw(); renderInspector(); };
el("disagreement").onclick = () => {
  state.disagreement = !state.disagreement;
  el("disagreement").classList.toggle("on", state.disagreement);
  draw();
};

// --- disagreement -----------------------------------------------------------

// Per protein, the mean neighbourhood Jaccard against every other space, taken
// from the per-pair tables the pipeline already wrote. Low means this protein's
// neighbours differ between spaces, which is the signal co-registration exists
// to surface -- so it is one button, not a menu (ADR 0005 item 4).
let disagreementByProtid = new Map();

function rebuildDisagreement() {
  const sums = new Map(), counts = new Map();
  for (const row of active.comparisons) {
    if (!row.per_protein) continue;
    for (const [protid, value] of Object.entries(row.per_protein)) {
      if (value === null) continue;
      sums.set(protid, (sums.get(protid) || 0) + value);
      counts.set(protid, (counts.get(protid) || 0) + 1);
    }
  }
  const out = new Map();
  for (const [protid, total] of sums) out.set(protid, total / counts.get(protid));
  disagreementByProtid = out;
}

// --- drawing -----------------------------------------------------------------

// The colour key has to mean the same thing in every panel of one figure.
// Computing levels per panel -- which is what this did -- gave the same colour
// opposite meanings side by side: in a space where every protein is readable
// the single level "readable" took palette slot 0, the same slot "do not read"
// took in a space that had both. A reader scanning the row saw one blue meaning
// two things. So the domain is computed ONCE across all spaces, before any
// panel is drawn, and the legend is rendered from that same domain.
let colourDomain = { kind: "none", label: "", levels: [], min: null, max: null };

function rebuildColourDomain() {
  const label = colourFor(spaces[0] || { protids: [], readable: [], clusters: {} }).label;
  const all = [];
  let kind = "none";
  for (const space of spaces) {
    const c = colourFor(space);
    kind = c.kind;
    for (const v of c.values) if (v !== null && v !== undefined) all.push(v);
  }
  if (kind === "categorical") {
    // Numeric-looking labels sort numerically; LC10 must not land before LC2.
    const levels = [...new Set(all)].sort((a, b) => {
      const na = Number(String(a).replace(/^\D+/, ""));
      const nb = Number(String(b).replace(/^\D+/, ""));
      if (Number.isFinite(na) && Number.isFinite(nb) && na !== nb) return na - nb;
      return String(a).localeCompare(String(b));
    });
    colourDomain = { kind, label, levels, min: null, max: null };
  } else if (kind === "continuous") {
    const nums = all.filter((v) => typeof v === "number");
    colourDomain = {
      kind, label, levels: [],
      min: nums.length ? Math.min(...nums) : 0,
      max: nums.length ? Math.max(...nums) : 1,
    };
  } else {
    colourDomain = { kind: "none", label: "", levels: [], min: null, max: null };
  }
}

// Without this the reader has colour and no key. The source is explicit that a
// map ships with what it needs to be read; a recoloured scatter with no legend
// is the same failure as a diagnostic nobody prints.
// The legend, counted rather than asserted. "Some points are hollow" leaves a
// reader scanning 367 markers for something they cannot count; the number, and
// which panel carries it, turns it into something they can check.
function renderLegend() {
  const node = el("legend");
  const flagged = new Set();
  const perSpace = [];
  spaces.forEach((space) => {
    const bad = space.protids.filter((protid, i) => space.readable[i] === false);
    bad.forEach((protid) => flagged.add(protid));
    if (bad.length) perSpace.push(`${space.space_id} ${bad.length}`);
  });
  const total = spaces.length ? spaces[0].protids.length : 0;
  if (!flagged.size) {
    node.innerHTML =
      "<b>No protein is flagged in any panel.</b> The diagnostics found every " +
      "2-D position on this page worth reading. Selecting in any panel " +
      "highlights the same proteins in all of them.";
    return;
  }
  node.innerHTML =
    `<b class="flagged-key">Red-ringed points should not be trusted</b> — the ` +
    "diagnostics say their 2-D position is not faithful, so their placement, " +
    "their neighbours and their cluster are not evidence. " +
    `<b>${flagged.size} of ${total}</b> proteins are flagged in at least one ` +
    `panel (${perSpace.join(", ")}). ` +
    "Everything else on this page still applies to them: they are real " +
    "proteins with real measurements, and it is only the picture's placement " +
    "that is unreliable. " +
    "Selecting in any panel highlights the same proteins in all of them.";
}

function renderColourKey() {
  const box = el("colour-key");
  box.textContent = "";
  if (colourDomain.kind === "none") { box.style.display = "none"; return; }
  box.style.display = "";
  const label = document.createElement("b");
  label.textContent = colourDomain.label + ": ";
  box.append(label);
  if (colourDomain.kind === "categorical") {
    colourDomain.levels.forEach((level, i) => {
      const chip = document.createElement("span");
      chip.className = "key-chip";
      const dot = document.createElement("i");
      dot.style.background = PALETTE[i % PALETTE.length];
      chip.append(dot, document.createTextNode(String(level)));
      box.append(chip);
    });
  } else {
    const ramp = document.createElement("span");
    ramp.className = "key-ramp";
    // Both ends named. A bare pair of numbers leaves the reader to guess which
    // end is the interesting one, and on the disagreement overlay that guess is
    // usually wrong in the direction that matters.
    box.append(
      document.createTextNode(fmtValue(colourDomain.min) + " "), ramp,
      document.createTextNode(" " + fmtValue(colourDomain.max))
    );
    if (state.disagreement) {
      const ends = document.createElement("span");
      ends.className = "key-note";
      ends.innerHTML =
        "&nbsp;— left: this protein keeps the same neighbours in every space. " +
        "right: its neighbours change most between spaces.";
      box.append(ends);
    }
  }
  const note = document.createElement("span");
  note.className = "key-note";
  note.textContent = "one key for every panel";
  box.append(note);
  // The distinction this page most needs to keep sharp, said where the two are
  // most easily confused -- beside the ramp itself.
  if (state.disagreement) {
    const warn = document.createElement("div");
    warn.className = "key-caveat";
    warn.innerHTML =
      "<b>High disagreement is not the same as untrustworthy.</b> It is the " +
      "source's discovery surface: a protein whose structural neighbours are " +
      "not its chemical ones is the interesting case, not a broken one. The " +
      "proteins you should <b>not</b> trust are the red-ringed ones, and they " +
      "are flagged by the diagnostics rather than by this colour.";
    box.append(warn);
  }
}

function colourFor(space) {
  const n = space.protids.length;
  if (state.disagreement) {
    const values = space.protids.map((p) =>
      disagreementByProtid.has(p) ? 1 - disagreementByProtid.get(p) : null
    );
    return { values, kind: "continuous", label: "disagreement (1 − mean Jaccard)" };
  }
  if (state.overlay === "__cluster__") {
    return {
      values: space.protids.map((p) => space.clusters[p] ?? null),
      kind: "categorical",
      label: "cluster",
    };
  }
  if (state.overlay === "__readable__") {
    return {
      values: space.readable.map((r) => (r ? "readable" : "do not read")),
      kind: "categorical",
      label: "position is readable",
    };
  }
  const overlay = active.overlays[state.overlay];
  if (!overlay) return { values: new Array(n).fill(null), kind: "none", label: "" };
  return { values: overlay.values, kind: overlay.kind, label: state.overlay };
}

// Entry PAGES, never a guessed file url. `fetch_accession.py` records that a
// constructed AF-{acc}-F1 model url 404s for isoform-backed entries, so the
// only honest link is the one the databases route themselves.
const UNIPROT_ENTRY = "https://www.uniprot.org/uniprotkb/{acc}/entry";
const AFDB_ENTRY = "https://alphafold.ebi.ac.uk/entry/{acc}";

// Port of domain_utils.UNIPROT_ACCESSION, split at the SAME point its Python
// source splits so the two read side by side. A test pins this against that
// pattern, because two copies of a regex in two languages drift silently -- and
// the drift would surface as proteins quietly losing their links, not as an
// error. `\\d` here is one backslash reaching JS: this template is a Python raw
// string.
const UNIPROT_ACCESSION = new RegExp(
  "^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]" +
  "|A0A[A-Z0-9]{7})(?:-\\d+)?$"
);

const PALETTE = ["#4269d0","#efb118","#ff725c","#6cc5b0","#3ca951","#ff8ab7",
                 "#a463f2","#97bbf5","#9c6b4e","#9498a0"];

function traceFor(space) {
  const coords = space.embeddings[state.reducer] || Object.values(space.embeddings)[0];
  const colour = colourFor(space);
  const selected = state.selected;
  const anySelected = selected.size > 0;

  // Readable and unreadable proteins go into two traces so the marker can
  // differ. The distinction has to survive being glanced at, and an open circle
  // did not: with a Viridis fill the ring takes the overlay's own colour, so a
  // flagged point in the middle of the ramp looked like an ordinary one. The
  // flagged trace now keeps the overlay colour as its FILL -- it is still a
  // real protein with a real value -- and adds a red ring that no overlay can
  // produce, so "the diagnostics flagged this" is never a shade of the palette.
  const groups = [
    { readable: true, symbol: "circle", opacity: 0.92, ring: null, ringWidth: 1.4 },
    { readable: false, symbol: "circle", opacity: 0.95, ring: "#b3261e", ringWidth: 2.6 },
  ];
  return groups.map(({ readable, symbol, opacity, ring, ringWidth }) => {
    const idx = space.protids
      .map((_, i) => i)
      .filter((i) => space.readable[i] === readable);
    const marker = { symbol, size: 9, line: { width: ringWidth } };
    if (ring) marker.line.color = ring;
    if (colour.kind === "continuous") {
      marker.color = idx.map((i) => colour.values[i]);
      marker.colorscale = "Viridis";
      marker.showscale = false;
      // Pin the scale to the domain shared by every panel. Without cmin/cmax
      // Plotly rescales per trace, so the same value is a different colour in
      // each panel -- which defeats the only thing a cross-panel colour is for.
      if (colourDomain.kind === "continuous" && colourDomain.max > colourDomain.min) {
        marker.cmin = colourDomain.min;
        marker.cmax = colourDomain.max;
      }
    } else if (colour.kind === "categorical") {
      marker.color = idx.map((i) => {
        const at = colourDomain.levels.indexOf(colour.values[i]);
        return at < 0 ? "#c9ced4" : PALETTE[at % PALETTE.length];
      });
    } else {
      marker.color = "#7a8794";
    }
    marker.opacity = idx.map((i) =>
      anySelected ? (selected.has(space.protids[i]) ? 1 : 0.12) : opacity
    );
    return {
      type: "scattergl",
      mode: "markers",
      name: readable ? "readable" : "do not read",
      showlegend: false,
      x: idx.map((i) => coords[i][0]),
      y: idx.map((i) => coords[i][1]),
      text: idx.map((i) => {
        const p = space.protids[i];
        const extra = (active.hover && active.hover[p]) || {};
        const bits = [`<b>${escapeHtml(p)}</b>`];
        // Species first: it is the thing a biologist recognises, and the
        // accession above it is not.
        if (extra.species) bits.push(escapeHtml(extra.species));
        if (extra.leiden !== undefined && extra.leiden !== null) {
          bits.push(`Leiden cluster ${escapeHtml(String(extra.leiden))} (whole-cohort)`);
        }
        // Attributed to its space on purpose. Each space partitions
        // independently, so a bare "cluster 3" invites reading two unrelated
        // partitions as the same grouping.
        if (space.clusters[p]) {
          bits.push(`${escapeHtml(space.space_id)} cluster ${escapeHtml(space.clusters[p])}`);
        }
        if (colour.label && colour.values[i] !== null && colour.values[i] !== undefined) {
          bits.push(`${escapeHtml(colour.label)}: ${escapeHtml(fmtValue(colour.values[i]))}`);
        }
        if (!readable) bits.push("position not faithful — do not read");
        return bits.join("<br>");
      }),
      customdata: idx.map((i) => space.protids[i]),
      hoverinfo: "text",
      marker,
    };
  });
}

// --- panel registry -----------------------------------------------------------
// A panel kind is {build(space, panel), render(space)}. `draw()` dispatches on
// `space.panel_type`, which the payload defaults to "scatter", so a page built
// before this registry existed renders identically. Splitting build from render
// keeps the original build-once / react-every-time split that `grid.dataset.built`
// encodes; putting them in one object is what makes a second kind possible.
const PANELS = {};

//: Panel kinds drawn by a renderer that predates the catalogue, so the sheet
//: body must not also draw them as cards.
const BUILT_ELSEWHERE = new Set(["space_grid", "comparisons", "contributions"]);

//: Where a panel the line above filters out is ACTUALLY drawn, keyed on
//: panel_id. `contributions` is catalogued on the mechanics sheet, filtered out
//: of that sheet's body, and drawn only inside the maps sheet -- so a reader on
//: mechanics was told the panel exists (it counts toward the sheet's panel
//: total) and never told where it went. It is excluded from the `(N empty)`
//: counter and stays excluded: nothing is missing, so it must not be counted as
//: blank.
//:
//: Keyed on panel_id and NOT on panel_type, deliberately. A future panel reusing
//: the type would otherwise inherit a pointer to a place it is not drawn.
//: `space_grid` and `comparisons` need no entry -- each is drawn on the sheet
//: the reader is already looking at.
const DRAWN_AT = {
  contributions:
    "Drawn on the maps sheet, as the contribution: strip under every fused " +
    "map — each block's realized share is shown beside the map it apportions, " +
    "rather than in a table away from it.",
};

// --- the fold-out --------------------------------------------------------
// What a panel is plotting, in words, from `explorer/descriptions.py`. The
// SAME renderer for a space panel and for a catalogue card, so the two cannot
// grow different ideas of how a hazard is displayed.
//
// Collapsed on purpose, and the verdict banner is deliberately NOT in here: a
// caveat you have to open is read by the people who already suspected it (ADR
// 0014). What belongs behind a disclosure is the slower material -- units,
// provenance, the file that computed the number.

// Backticks become <code> and **…** becomes bold, applied AFTER escaping, so a
// description can be written as plain text and no string in it can inject
// markup.
function inlineMarkup(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

//: The diagnostic sections that write prose, in reading order, with the label
//: each gets on the page. Fixed here rather than taken from `Object.keys`, so
//: the order does not follow JSON insertion order and a section nobody named
//: cannot appear under its raw key.
const DIAGNOSTIC_SECTIONS = [
  ["faithfulness", "layout faithfulness"],
  ["stability", "neighbourhood stability"],
  ["redundancy", "block redundancy"],
  ["resolution_sweep", "resolution sweep"],
  ["partition", "partition"],
  ["negative_controls", "negative controls"],
];

//: `embedding.warnings()` appends this when NOTHING fired. It is an all-clear,
//: not a warning, and it is present on 4 of the 11 space/cohort combinations
//: shipped -- so a disclosure calling its contents warnings would contradict
//: itself on more than a third of the page. Two consequences, both deliberate:
//: the disclosure is titled by what it CONTAINS, and an all-clear is marked so
//: it is not styled as a hazard.
const ALL_CLEAR = "the layout is faithful at k=";

// Every sentence the run's diagnostics wrote about ONE space, in one collapsed
// disclosure under that space's panel.
//
// Read from the space handed in -- which is always a space of the ACTIVE cohort
// -- and never by walking the payload. Cohort 0's spaces are duplicated at the
// payload top level on purpose (build_explorer.py), so a walk would draw the
// first cohort's sixteen sentences twice.
//
// Collapsed, per ADR 0014: a caveat you have to open is read by the people who
// already suspected it, which is right for a per-space detail and wrong for a
// banner. The banner above it is unchanged and stays open.
function diagnosticsBlock(space) {
  const diagnostics = space.diagnostics || {};
  const groups = [];
  let total = 0;
  let cleared = 0;
  DIAGNOSTIC_SECTIONS.forEach(([key, label]) => {
    const value = diagnostics[key];
    if (!value) return;
    const entries = Array.isArray(value) ? value : [value];
    const lines = [];
    entries.forEach((entry) => {
      if (!entry || typeof entry !== "object") return;
      (entry.warnings || []).forEach((text) => lines.push(String(text)));
    });
    if (!lines.length) return;
    total += lines.length;
    cleared += lines.filter((t) => t.indexOf(ALL_CLEAR) !== -1).length;
    groups.push({ label, lines });
  });
  if (!total) return null;
  const box = document.createElement("details");
  box.className = "explain wrote";
  const summary = document.createElement("summary");
  // Named by what is in it. "Warnings" would be false on the all-clears and
  // "notes" would be false on "their individual positions should not be read".
  summary.textContent =
    `What the diagnostics wrote about this space (${total} sentence(s)` +
    (cleared ? `, ${cleared} of them an all-clear)` : ")");
  const body = document.createElement("div");
  body.className = "body";
  groups.forEach(({ label, lines }) => {
    const head = document.createElement("p");
    head.className = "wrote-label";
    head.textContent = label;
    body.append(head);
    const list = document.createElement("ul");
    lines.forEach((text) => {
      const item = document.createElement("li");
      if (text.indexOf(ALL_CLEAR) !== -1) item.className = "cleared";
      item.textContent = text;
      list.append(item);
    });
    body.append(list);
  });
  box.append(summary, body);
  return box;
}

function explainBlock(description) {
  const d = description || {};
  const paragraphs = d.paragraphs || [];
  const hazards = d.hazards || [];
  const sources = d.sources || [];
  // Nothing written for this panel yet. Render nothing rather than an empty
  // disclosure triangle, which promises content and then has none.
  if (!paragraphs.length && !hazards.length) return null;
  const box = document.createElement("details");
  box.className = "explain";
  const summary = document.createElement("summary");
  summary.textContent = hazards.length
    ? `What this shows, and ${hazards.length} thing(s) it cannot be read for`
    : "What this shows";
  const body = document.createElement("div");
  body.className = "body";
  // The axes sentence gets its own node, tagged with the reducer that produced
  // it, because it is the one paragraph here that is NOT a property of the
  // space -- switching layout changes it. Identified by matching the composed
  // text rather than by position, so reordering the paragraphs cannot silently
  // mislabel one. No behaviour change yet: phase 2 updates this node on a
  // layout switch.
  const byReducer = d.axes_by_reducer || {};
  const axesOf = (text) =>
    Object.keys(byReducer).find((reducer) => byReducer[reducer] === text) || "";
  body.innerHTML =
    paragraphs
      .map((p) => {
        const reducer = axesOf(p);
        return reducer
          ? `<p class="axes" data-reducer="${escapeHtml(reducer)}">${inlineMarkup(p)}</p>`
          : `<p>${inlineMarkup(p)}</p>`;
      })
      .join("") +
    hazards.map((h) => `<p class="hazard"><b>Hazard.</b> ${inlineMarkup(h)}</p>`).join("") +
    (sources.length
      ? `<p class="src">Computed by: ${sources.map(escapeHtml).join(" · ")}</p>`
      : "");
  box.append(summary, body);
  return box;
}

// The chrome every panel kind shares, whatever it draws: level class, title and
// the verdict banner. A panel that skipped this could render a map with no
// verdict beside it, which is the single thing the diagnostics exist to prevent.
function panelShell(space) {
  const panel = document.createElement("div");
  panel.className = `panel level-${space.verdict.level}`;
  const title = document.createElement("h2");
  title.textContent = space.space_id;
  const verdict = document.createElement("div");
  verdict.className = `verdict level-${space.verdict.level}`;
  verdict.innerHTML =
    space.verdict.headline +
    (space.verdict.reasons.length
      ? "<ul>" + space.verdict.reasons
          .map((r) => `<li>${escapeHtml(r)}</li>`).join("") + "</ul>"
      : "");
  panel.append(title, verdict);
  // ADR 0002: a fused map does not render without its contribution shares
  // visible. Both numbers, because they differ -- `early` concatenates
  // features, so an even request over blocks of unequal width realizes
  // unevenly, and showing only the request would misreport the map.
  if (space.contributions && space.contributions.length) {
    const shares = document.createElement("div");
    shares.className = "shares";
    shares.innerHTML =
      "contribution: " +
      space.contributions
        .map((c) => {
          const asked = c.share == null ? "?" : (100 * c.share).toFixed(0);
          const got = c.realized_share == null ? "?" : (100 * c.realized_share).toFixed(0);
          const drift = asked !== got ? ` <span class="drift">(asked ${asked}%)</span>` : "";
          return `<b>${escapeHtml(String(c.block_id))}</b> ${got}%${drift}`;
        })
        .join(" · ");
    panel.append(shares);
  }
  // The within-block twin of the line above. A fused space is apportioned
  // between its blocks; a one-block feature space is apportioned between its
  // own columns, and it is the same question one level down: what is this
  // picture actually made of. Euclidean distance on raw columns is a sum of
  // per-column squared differences, so a column's share of the variance IS its
  // share of the squared distance -- this is the quantity, not a proxy for it.
  if ((space.column_shares || []).length) {
    const columns = space.column_shares.slice().sort((a, b) => b.share - a.share);
    const top = columns[0];
    const even = top.share_if_standardized;
    const bar = document.createElement("div");
    bar.className = "shares";
    bar.innerHTML =
      "made of: " +
      columns
        .map((c) => `<b>${escapeHtml(String(c.column))}</b> ${(100 * c.share).toFixed(1)}%`)
        .join(" · ");
    panel.append(bar);
    // A single column carrying almost the whole distance is not a fact about
    // the proteins, it is a fact about the units. Said here rather than in the
    // fold-out, because a reader who does not open the fold-out will otherwise
    // read this map as a map of physicochemistry.
    if (even && top.share > 2 * even) {
      const warn = document.createElement("div");
      warn.className = "shares column-warn";
      warn.innerHTML =
        `<b>This map is mostly ${escapeHtml(String(top.column))}.</b> ` +
        `Its ${(100 * top.share).toFixed(1)}% is a fact about the units, not about the ` +
        "proteins: these columns enter the distance raw, on incomparable scales. " +
        (top.declared_normalization
          ? `The block declares <code>${escapeHtml(String(top.declared_normalization))}</code>, ` +
            "which would give every column " + (100 * even).toFixed(0) + "% — but nothing " +
            "reads that field (FOLLOWUPS #32). "
          : "") +
        "Colour by each descriptor in turn to see which structure is which.";
      panel.append(warn);
    }
  }
  const explain = explainBlock(space.description);
  if (explain) panel.append(explain);
  const wrote = diagnosticsBlock(space);
  if (wrote) panel.append(wrote);
  return panel;
}

PANELS.scatter = {
  build(space, panel) {
    const plot = document.createElement("div");
    plot.className = "plot";
    plot.id = `plot-${space.space_id}`;
    panel.append(plot);
  },
  render(space) {
    const node = el(`plot-${space.space_id}`);
    Plotly.react(node, traceFor(space), {
      margin: { l: 26, r: 12, t: 8, b: 26 },
      xaxis: { zeroline: false, showticklabels: false },
      yaxis: { zeroline: false, showticklabels: false },
      dragmode: "lasso",
      hovermode: "closest",
      plot_bgcolor: "#fff",
    }, { displayModeBar: false, responsive: true });
    node.removeAllListeners?.("plotly_selected");
    node.on("plotly_selected", (event) => {
      state.selected = new Set((event?.points || []).map((p) => p.customdata));
      draw();
      renderInspector();
    });
  },
};

// A payload naming a kind this template does not have is a version mismatch
// between the two halves. Draw the mismatch rather than nothing: a panel that
// silently fails to appear cannot be told apart from one that was never asked
// for, and the reader has no way to know a space is missing.
PANELS.__unknown__ = {
  build(space, panel) {
    const note = document.createElement("div");
    note.className = "awaiting";
    note.textContent =
      `no renderer for panel_type "${space.panel_type}" — this page's template ` +
      `is older than the payload that produced it`;
    panel.append(note);
  },
  render() {},
};

function panelFor(space) {
  return PANELS[space.panel_type || "scatter"] || PANELS.__unknown__;
}

function draw() {
  rebuildColourDomain();
  renderColourKey();
  renderLegend();
  const grid = el("grid");
  if (!grid.dataset.built) {
    spaces.forEach((space) => {
      const panel = panelShell(space);
      panelFor(space).build(space, panel);
      grid.append(panel);
    });
    grid.dataset.built = "1";
  }
  spaces.forEach((space) => panelFor(space).render(space));
}

// --- sheets ---------------------------------------------------------------
// The source organises its argument into eight sheets and the panel catalogue
// carries each panel's sheet. Only the `maps` sheet has the live grid; the rest
// render from the cohort's panels, which already say whether each has data.
// That decision is made once, in Python, so a panel cannot be judged drawable
// in one language and blank in the other.

function panelCard(panel) {
  const card = document.createElement("div");
  card.className = "panel";
  const title = document.createElement("h2");
  title.textContent = panel.title;
  const tag = document.createElement("span");
  tag.className = `tag tag-${panel.provenance}`;
  tag.textContent = panel.provenance;
  tag.title = `section ${panel.section} of the source analysis`;
  title.append(tag);
  card.append(title);
  if (panel.question) {
    const q = document.createElement("div");
    q.className = "question";
    q.textContent = panel.question;
    card.append(q);
  }
  const explain = explainBlock(panel.description);
  if (explain) card.append(explain);
  return card;
}

// A panel with an unmet requirement prints the requirement. Naming the missing
// input is what makes it a question someone can answer; "no data" is not.
function awaitingBlock(panel) {
  const box = document.createElement("div");
  box.className = "awaiting";
  const inner = document.createElement("div");
  const head = document.createElement("b");
  head.textContent = "awaiting " + (panel.missing.join(", ") || "an input");
  inner.append(head);
  if (panel.requires) {
    const why = document.createElement("span");
    why.className = "why";
    why.textContent = panel.requires;
    inner.append(why);
  }
  if (panel.fills_in) {
    const fills = document.createElement("span");
    fills.className = "fills";
    fills.textContent = "filled in by: " + panel.fills_in;
    inner.append(fills);
  }
  box.append(inner);
  return box;
}

// --- the catalogue panel renderers -------------------------------------------
// A SECOND registry, deliberately, and not entries in `PANELS`.
//
// `PANELS` renders a SPACE: its entries take `(space, panel)` and reach for
// `space.embeddings`. A catalogue card is a different object entirely, and two
// catalogue panels (`identity_vs_tm`, `tree_space`) already declare
// `panel_type: "scatter"` -- a kind `PANELS` also defines. One shared map would
// hand a catalogue panel to the space scatter renderer the day either of them
// becomes drawable, and the failure would be a thrown exception inside a
// forEach that takes the rest of the sheet down with it.
//
// Each entry is `render(panel) -> Node`, reading whatever it needs off `active`.
const SHEET_PANELS = {};

// Draws that can only happen once the node is IN the document. A renderer
// returns its node before `renderSheet` appends it, so anything that measures
// layout -- Plotly does -- would size itself against a detached element and
// draw into a zero-width box. Queue it here and flush after the append.
//
// Deliberately a queue and NOT requestAnimationFrame. rAF works in a browser
// and is invisible to any check that reads the DOM at a fixed moment, so the
// panel's correctness would depend on when someone happened to look.
const PENDING_DRAWS = [];

function flushPendingDraws() {
  while (PENDING_DRAWS.length) {
    PENDING_DRAWS.shift()();
  }
}

function sheetTable(columns, rows) {
  const box = document.createElement("div");
  box.className = "tablebox";
  const head = columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("");
  const body = rows
    .map((row) => "<tr>" + columns.map((c) => c.cell(row)).join("") + "</tr>")
    .join("");
  box.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  return box;
}

// Every protein on the maps, by name. Every other panel is an aggregate, and a
// map with no way to ask "which protein is that point" is a picture.
SHEET_PANELS.records = {
  render() {
    const rows = active.records || [];
    const wrap = document.createElement("div");
    const bar = document.createElement("div");
    bar.className = "panel-bar";
    const filter = document.createElement("input");
    filter.type = "search";
    filter.placeholder = "filter by accession, protein or organism";
    const count = document.createElement("span");
    bar.append(filter, count);
    const columns = [
      { label: "accession", cell: (r) => `<td class="mono">${escapeHtml(r.accession)}</td>` },
      { label: "protein", cell: (r) => `<td>${escapeHtml(r.protein)}</td>` },
      { label: "organism", cell: (r) => `<td>${escapeHtml(r.organism)}</td>` },
      { label: "length", cell: (r) => `<td>${escapeHtml(r.length)}</td>` },
    ];
    let table = sheetTable(columns, rows);
    // The count is of the FILTERED rows against the total, so a filter that
    // matches nothing says so rather than showing an empty table that looks
    // like a cohort with no proteins in it.
    const paint = () => {
      const needle = filter.value.trim().toLowerCase();
      const shown = needle
        ? rows.filter((r) =>
            `${r.accession} ${r.protein} ${r.organism}`.toLowerCase().includes(needle))
        : rows;
      count.textContent = `${shown.length} of ${rows.length} protein(s)`;
      const next = sheetTable(columns, shown);
      table.replaceWith(next);
      table = next;
    };
    filter.addEventListener("input", paint);
    wrap.append(bar, table);
    paint();
    return wrap;
  },
};

// What the per-query cap removed. The RATE is the least interesting number
// here: what tells a cap from ordinary sparsity is that the rows pile up at one
// partner count while the columns stay free, so both are reported and the
// verdict is the payload's rather than the reader's arithmetic.
SHEET_PANELS.censoring = {
  isEmpty() {
    return !((active.censoring || {}).summary);
  },
  render() {
    const censoring = active.censoring || {};
    const summary = censoring.summary || {};
    const wrap = document.createElement("div");
    const note = document.createElement("div");
    note.className = "table-note";
    // Three states, and the middle one is the one a rate alone would hide.
    if (summary.cap_detected) {
      note.innerHTML =
        `<b>A per-query cap is detected.</b> ${fmtValue(summary.rows_at_max)} of ` +
        `${fmtValue(summary.n_rows)} rows report exactly ${fmtValue(summary.inferred_cap)} ` +
        "partners while the columns do not pile up, which is what a cap on the " +
        "query side looks like and what ordinary sparsity does not.";
    } else if (summary.n_censored > 0) {
      note.innerHTML =
        "<b>Cells are missing, but not in a per-query cap's pattern.</b> The rows " +
        "do not pile up at one partner count, or the columns pile up too — either " +
        "way the missingness is a property of the whole matrix rather than of " +
        "each query, and calling it a cap would be the wrong explanation.";
    } else {
      note.innerHTML =
        "<b>Nothing was censored: this cohort's matrix is exhaustive.</b> Every " +
        "pair was measured. Worth stating plainly, because the default pipeline " +
        "caps each query and a reader who assumed that would misread the " +
        "coverage of every other panel here.";
    }
    wrap.append(note);
    const rows = [
      ["cells", `${fmtValue(summary.n_censored)} censored of ${fmtValue(summary.n_cells)}`,
        "matrix_io.summarize_censoring"],
      ["censoring rate", fmtValue(summary.censoring_rate), "of all cells"],
      ["partners measured per row",
        `min ${fmtValue((summary.measured_per_row || {}).min)} · ` +
        `median ${fmtValue((summary.measured_per_row || {}).median)} · ` +
        `max ${fmtValue((summary.measured_per_row || {}).max)}`,
        "the query side, which a cap bounds"],
      ["partners measured per column",
        `min ${fmtValue((summary.measured_per_col || {}).min)} · ` +
        `median ${fmtValue((summary.measured_per_col || {}).median)} · ` +
        `max ${fmtValue((summary.measured_per_col || {}).max)}`,
        "the target side, which it does not"],
      ["rows at the maximum",
        summary.rows_at_max_fraction === undefined
          ? reportMissing("no row summary in this payload")
          : `${(100 * summary.rows_at_max_fraction).toFixed(1)}%`,
        "a cap's fingerprint"],
      ["columns at the maximum",
        summary.cols_at_max_fraction === undefined
          ? reportMissing("no column summary in this payload")
          : `${(100 * summary.cols_at_max_fraction).toFixed(1)}%`,
        "must NOT match the rows, or it is not a cap"],
      ["measured zeros",
        fmtValue(summary.measured_zero_count),
        "a real 0.000 rather than a fill token; nonzero here breaks the mask"],
    ];
    if (summary.censoring_predicted_by_cap !== undefined) {
      rows.push([
        "rate the cap alone predicts",
        fmtValue(summary.censoring_predicted_by_cap),
        "compare against the observed rate above",
      ]);
    }
    wrap.append(reportRows(rows));
    // The per-protein distribution, sorted worst first. A single rate hides
    // that censoring is uneven across proteins, and it is the unevenness that
    // makes it a per-protein diagnostic rather than a property of the run.
    const rates = censoring.rates || [];
    if (rates.length) {
      const plot = document.createElement("div");
      plot.className = "plot";
      wrap.append(plot);
      PENDING_DRAWS.push(() => {
        Plotly.react(
          plot,
          [{
            type: "bar",
            y: rates.map((r) => r.rate),
            text: rates.map((r) => r.protid),
            hovertemplate: "%{text}<br>%{y:.3f} of its row censored<extra></extra>",
            marker: { color: "#3b528b" },
          }],
          {
            margin: { l: 40, r: 12, t: 8, b: 24 },
            xaxis: { title: "proteins, most censored first", showticklabels: false },
            yaxis: { title: "row censored", range: [0, 1] },
            bargap: 0,
          },
          { displayModeBar: false, responsive: true }
        );
      });
    }
    if (censoring.comparison) wrap.append(renderCensoringComparison(censoring.comparison));
    return wrap;
  },
};

// Would you get the same neighbours again. One trace per space, because
// stability is measured on each SPACE's own distances -- a single shared
// overlay would colour every panel by one space's answer, which is a category
// error rather than a convenience (FOLLOWUPS #62).
//
// This judges the SPACE, not the layout. A protein can have perfectly
// determinate neighbours in the high-dimensional space and still be drawn in
// the wrong place by the reducer; that second question is faithfulness, and it
// is the one the panel banners answer. The two are kept apart here in words.
SHEET_PANELS.stability = {
  isEmpty() {
    return !spaces.some((space) => space.stability && Object.keys(space.stability).length);
  },
  render() {
    const wrap = document.createElement("div");
    const note = document.createElement("div");
    note.className = "table-note";
    note.innerHTML =
      "Each protein's neighbourhood is recomputed on repeated subsamples of the " +
      "cohort and compared with its neighbourhood in the full one; the value is " +
      "the mean overlap. <b>At or below " + fmtValue(COIN_FLIP) + " a protein's " +
      "neighbours are a coin flip</b> and nothing about who it sits near is " +
      "evidence. This judges the <b>space</b>, not the drawing of it — a protein " +
      "with determinate neighbours can still be placed badly by the reducer, " +
      "which is what the panel banners above answer.";
    wrap.append(note);
    const rows = spaces
      .filter((space) => space.stability && Object.keys(space.stability).length)
      .map((space) => {
        const values = space.protids
          .map((protid) => space.stability[protid])
          .filter((value) => value !== undefined);
        const sorted = values.slice().sort((a, b) => a - b);
        const flips = sorted.filter((value) => value <= COIN_FLIP).length;
        return {
          space_id: space.space_id,
          sorted,
          flips,
          median: sorted.length ? sorted[Math.floor(sorted.length / 2)] : null,
        };
      });
    wrap.append(
      reportRows(
        rows.map((row) => [
          row.space_id,
          `${fmtValue(row.flips)} of ${fmtValue(row.sorted.length)} are coin flips` +
            ` · median ${fmtValue(row.median)}`,
          row.flips === row.sorted.length
            ? "every protein: nothing in this space's neighbourhoods is evidence"
            : "",
        ])
      )
    );
    const plot = document.createElement("div");
    plot.className = "plot";
    wrap.append(plot);
    PENDING_DRAWS.push(() => {
      Plotly.react(
        plot,
        rows.map((row) => ({
          type: "scattergl",
          mode: "lines",
          name: row.space_id,
          x: row.sorted.map((_, i) => i / Math.max(1, row.sorted.length - 1)),
          y: row.sorted,
          hovertemplate: `${row.space_id}<br>%{y:.3f}<extra></extra>`,
        })),
        {
          margin: { l: 44, r: 12, t: 8, b: 34 },
          xaxis: { title: "proteins, least stable first", range: [0, 1], tickformat: ".0%" },
          yaxis: { title: "neighbours kept", range: [0, 1] },
          // The threshold as a line, so a reader sees which curves clear it
          // without comparing a number in the table against a curve.
          shapes: [{
            type: "line", x0: 0, x1: 1, y0: COIN_FLIP, y1: COIN_FLIP,
            line: { color: "#b3261e", width: 1, dash: "dot" },
          }],
          legend: { orientation: "h", y: -0.22, font: { size: 10 } },
        },
        { displayModeBar: false, responsive: true }
      );
    });
    return wrap;
  },
};

// What censoring did to a map, measured against its uncensored twin.
//
// THE DRAG LINE IS THE OBVIOUS PANEL AND IT IS THE WRONG ONE. Joining a
// protein's censored position to its uncensored one reads as "this is how far
// the cap moved it", and that reading needs the two layouts to share a frame.
// So the page checks whether they do, refuses with the measured number when
// they do not, and draws neighbourhood retention instead -- which is invariant
// to rotation, reflection and scale and so survives whatever the frames do.
function renderCensoringComparison(comparison) {
  const box = document.createElement("div");
  const head = document.createElement("div");
  head.className = "table-note";
  head.innerHTML =
    `<b>What the cap did to the map.</b> <code>${escapeHtml(comparison.censored_space)}</code> ` +
    `and <code>${escapeHtml(comparison.reference_space)}</code> are the same proteins ` +
    "reduced the same way; the only difference is which pairs were measured.";
  box.append(head);
  box.append(
    reportRows([
      [
        `neighbours kept (k=${fmtValue(comparison.k)})`,
        fmtValue(comparison.jaccard_mean),
        "mean neighbourhood Jaccard — frame-independent",
      ],
      [
        "layouts superimposable",
        `${fmtValue(comparison.procrustes_disparity)} ` +
        (comparison.superimposable
          ? "— below the threshold, so the two frames share a shape"
          : `— above ${fmtValue(comparison.threshold)}, so they do not`),
        "Procrustes disparity, 0 is identical",
      ],
      [
        "cluster agreement",
        comparison.cluster_ari === null || comparison.cluster_ari === undefined
          ? reportMissing("no partition was compared for this pair")
          : fmtValue(comparison.cluster_ari),
        "adjusted Rand index",
      ],
    ])
  );
  // The refusal, with the number as its reason.
  if (!comparison.superimposable) {
    const refused = document.createElement("div");
    refused.className = "refused";
    refused.innerHTML =
      "<b>refused: the two positions are not drawn as a displacement.</b> A line " +
      "from a protein's censored position to its uncensored one would be read as " +
      "how far the cap moved it, and that reading needs the two layouts to share " +
      `a frame. At a Procrustes disparity of ${fmtValue(comparison.procrustes_disparity)} ` +
      "they share almost none — this pair is less superimposable than two " +
      "entirely different modalities are, so the length of such a line would be " +
      "an artifact of two unrelated frames rather than a measurement. What is " +
      "drawn below instead does not depend on the frames at all.";
    box.append(refused);
  }
  // Neighbourhood retention per protein, worst first. This is the honest
  // version of "how far did this protein move".
  const retained = comparison.retained || [];
  if (retained.length) {
    const plot = document.createElement("div");
    plot.className = "plot";
    box.append(plot);
    PENDING_DRAWS.push(() => {
      Plotly.react(
        plot,
        [{
          type: "bar",
          y: retained.map((row) => row.jaccard),
          text: retained.map((row) => row.protid),
          hovertemplate: "%{text}<br>keeps %{y:.3f} of its neighbours<extra></extra>",
          marker: { color: "#2f7d4f" },
        }],
        {
          margin: { l: 44, r: 12, t: 8, b: 26 },
          xaxis: { title: "proteins, least retained first", showticklabels: false },
          yaxis: { title: "neighbours kept", range: [0, 1] },
          bargap: 0,
        },
        { displayModeBar: false, responsive: true }
      );
    });
  }
  // Context, because a disparity means nothing on its own.
  const context = comparison.context || [];
  if (context.length) {
    const rows = context
      .slice()
      .sort((a, b) => (a.procrustes_disparity || 0) - (b.procrustes_disparity || 0))
      .map((row) => [
        `${row.space_a} vs ${row.space_b}`,
        fmtValue(row.procrustes_disparity),
        `neighbours kept ${fmtValue(row.jaccard_mean)}`,
      ]);
    const note = document.createElement("div");
    note.className = "table-note";
    note.innerHTML =
      "Every other pair this run compared, for scale. A disparity is only " +
      "interpretable beside pairs whose difference is already understood — these " +
      "are different modalities, where a large number is expected.";
    box.append(note, reportRows(rows));
  }
  return box;
}

// The similarity matrix itself, cluster-sorted. 2.02's point is that protein i
// has no feature vector of its own: its feature vector IS its row here, so this
// is the tensor the structure map is a reduction of, not an illustration of one.
//
// The WHOLE matrix is drawn, not a triangle, because these matrices are not
// symmetric -- the payload carries the measured asymmetry and the panel prints
// it. A triangle would silently pick one of two real values for a fifth of the
// pairs.
//
// Decoded from base64 into a Uint8Array and handed to Plotly, which is already
// inlined. No new dependency, no build step (7.03 E4).
function decodeMatrix(encoded, n) {
  const binary = atob(encoded);
  if (binary.length !== n * n) return null;
  const flat = new Uint8Array(n * n);
  for (let i = 0; i < binary.length; i++) flat[i] = binary.charCodeAt(i);
  const rows = new Array(n);
  for (let r = 0; r < n; r++) rows[r] = Array.from(flat.subarray(r * n, r * n + n));
  return rows;
}

SHEET_PANELS.heatmap = {
  isEmpty() {
    const matrix = active.tm_matrix || {};
    return !matrix.values || !matrix.n;
  },
  render() {
    const matrix = active.tm_matrix || {};
    const wrap = document.createElement("div");
    const n = matrix.n;
    const rows = decodeMatrix(matrix.values, n);
    if (!rows) {
      const box = document.createElement("div");
      box.className = "refused";
      box.innerHTML =
        "<b>the encoded matrix is not n × n</b> — the payload's byte count and " +
        "its stated size disagree, so nothing is drawn rather than something wrong";
      wrap.append(box);
      return wrap;
    }
    const span = matrix.high - matrix.low;
    // The colour axis carries the REAL values, so the reader never sees a 0-255
    // code. The cells stay quantised; only the labels are restored.
    const note = document.createElement("div");
    note.className = "table-note";
    const asymmetry = matrix.asymmetry || {};
    const share = asymmetry.n_pairs
      ? (100 * asymmetry.n_asymmetric) / asymmetry.n_pairs
      : 0;
    note.innerHTML =
      `Value: <b>${escapeHtml(matrix.value_label || "")}</b>. Sorted by the ` +
      `<code>${escapeHtml(matrix.sorted_by || "")}</code> space's clusters, then by ` +
      "accession. " +
      `<b>Not symmetric:</b> ${fmtValue(asymmetry.n_asymmetric)} of ` +
      `${fmtValue(asymmetry.n_pairs)} pairs (${share.toFixed(1)}%) disagree with their ` +
      `transpose, ${fmtValue(asymmetry.n_above_0_05)} of them by more than 0.05 and the ` +
      `largest by ${fmtValue(asymmetry.max_gap)} — which is why the whole square is ` +
      "drawn and not one triangle. " +
      `Colours are quantised to ${fmtValue(matrix.levels)} levels, so a cell is worth ` +
      `±${fmtValue(matrix.max_error)}: read it as a colour, never as a measurement.`;
    wrap.append(note);
    const plot = document.createElement("div");
    plot.className = "plot plot-tall";
    wrap.append(plot);
    // Queued rather than drawn here: the node is not in the document yet.
    PENDING_DRAWS.push(() => {
      Plotly.react(
        plot,
        [
          {
            type: "heatmap",
            z: rows,
            zmin: 0,
            zmax: matrix.levels,
            colorscale: "Viridis",
            showscale: true,
            colorbar: {
              thickness: 9,
              tickvals: [0, matrix.levels / 2, matrix.levels],
              ticktext: [
                fmtValue(matrix.low),
                fmtValue(matrix.low + span / 2),
                fmtValue(matrix.high),
              ],
            },
            hoverinfo: "text",
            text: rows.map((row, r) =>
              row.map((code, c) =>
                `${matrix.protids[r]} → ${matrix.protids[c]}\n` +
                `≈ ${(matrix.low + (code / matrix.levels) * span).toFixed(3)}`)),
          },
        ],
        {
          margin: { l: 8, r: 8, t: 6, b: 8 },
          // Ranges pinned to the data rather than left to autorange. An
          // autoranged axis is what let a stray shape coordinate rescale the
          // whole panel; a fixed range cannot be moved by anything but the data.
          xaxis: {
            showticklabels: false, ticks: "", range: [-0.5, n - 0.5],
            scaleanchor: "y", constrain: "domain",
          },
          yaxis: { showticklabels: false, ticks: "", range: [n - 0.5, -0.5] },
          shapes: clusterBandShapes(matrix.bands || [], n),
        },
        { displayModeBar: false, responsive: true }
      );
    });
    return wrap;
  },
};

// One line at each cluster boundary, so the blocks on the diagonal can be told
// from the sort that produced them. Drawn from the run-length bands the payload
// carries rather than from a label per row.
//
// EVERY COORDINATE IS INSIDE THE DATA RANGE, and that is the whole lesson of
// this function. The first version ran each line to `+1e6` to mean "the far
// edge", which is not what a shape on a data axis means: it stretched the axis
// to a million, and 367 cells of heat map became a speck in the corner while
// the colour bar and the grid lines drew perfectly. The DOM said 367x367 and
// the panel was blank. Only the picture showed it.
function clusterBandShapes(bands, n) {
  const shapes = [];
  const lo = -0.5;
  const hi = n - 0.5;
  let at = 0;
  bands.forEach((band, index) => {
    at += band.count;
    if (index === bands.length - 1) return;
    const edge = at - 0.5;
    shapes.push({
      type: "line", x0: edge, x1: edge, y0: lo, y1: hi,
      line: { color: "rgba(255,255,255,0.55)", width: 1 },
    });
    shapes.push({
      type: "line", x0: lo, x1: hi, y0: edge, y1: edge,
      line: { color: "rgba(255,255,255,0.55)", width: 1 },
    });
  });
  return shapes;
}

// The resolved pipeline, as the source's 2.01 asks for it: each box a tensor,
// each arrow an operation. Derived per run in `payload.py`, so this is THIS
// cohort's pipeline and not a drawing of the general case.
//
// Two rows of the diagram exist because two things surprise most readers, and
// both are numbers here rather than sentences:
//   * a `profile` block's tensor IS the similarity matrix -- a protein's
//     feature vector is its own row of it, so PCA's input is the matrix.
//   * Leiden clusters on its OWN kNN graph. Both k values are printed side by
//     side, because on these cohorts they are 37 and 15 and nothing on the page
//     otherwise says so.
SHEET_PANELS.pipeline = {
  isEmpty(panel) {
    const pipeline = active.pipeline || {};
    return !(pipeline.blocks || []).length && !(pipeline.rows || []).length;
  },
  render() {
    const pipeline = active.pipeline || {};
    const wrap = document.createElement("div");
    const blocks = document.createElement("div");
    blocks.className = "flow";
    (pipeline.blocks || []).forEach((block) => {
      const box = document.createElement("div");
      box.className = "flowbox" + (block.is_profile ? " flowbox-note" : "");
      box.innerHTML =
        `<b>${escapeHtml(block.block_id)}</b>` +
        `<span>${escapeHtml(block.provider)}</span>` +
        `<span class="out">${escapeHtml(block.representation)}` +
        `${block.metric ? " · " + escapeHtml(block.metric) : ""}</span>` +
        (block.is_profile
          ? '<span class="surprise">its tensor is the similarity matrix itself: a ' +
            "protein's feature vector is its own row of it, so PCA's input is " +
            "the matrix and not a set of descriptors</span>"
          : "");
      blocks.append(box);
    });
    if ((pipeline.blocks || []).length) wrap.append(blocks);
    const columns = [
      { label: "space", cell: (r) => `<td>${escapeHtml(r.space_id)}</td>` },
      {
        label: "blocks in, and how they join",
        cell: (r) => `<td>${escapeHtml((r.blocks || []).join(" + "))}` +
          `<span class="out">strategy: ${escapeHtml(r.strategy)}</span></td>`,
      },
      {
        label: "the map's graph",
        cell: (r) => `<td>${escapeHtml(r.reducer)}` +
          `<span class="out">k = ${r.map_n_neighbors === null || r.map_n_neighbors === undefined
            ? withheldWord("reducer default", "this run's config named no n_neighbors, so " +
                "the reducer's own default applied; printing a number here would " +
                "claim the config said something it did not")
            : fmtValue(r.map_n_neighbors)}</span></td>`,
      },
      {
        label: "the clustering's graph",
        cell: (r) => `<td>${r.cluster_n_neighbors === null || r.cluster_n_neighbors === undefined
          ? reportMissing("no partition section was written for this space")
          : `k = ${fmtValue(r.cluster_n_neighbors)} over ${fmtValue(r.n_pcs)} PCs`}` +
          `<span class="out">resolution ${fmtValue(r.resolution)} · ` +
          `${fmtValue(r.n_clusters)} clusters · ${escapeHtml(r.cluster_source || "")}</span></td>`,
      },
    ];
    if ((pipeline.rows || []).length) {
      const note = document.createElement("div");
      note.className = "table-note";
      note.innerHTML =
        "The last two columns are two <b>different</b> graphs. The map is drawn " +
        "from one and the clusters come from another, so a cluster boundary and " +
        "a gap on the map are not the same statement.";
      wrap.append(note);
      wrap.append(sheetTable(columns, pipeline.rows));
    }
    return wrap;
  },
};

// The 5x5 perturbation grid, whose EMPTY CELLS are the content: the
// combination people reach for first -- one alanine against TM-score -- is the
// flattest cell in it, and a grid that did not say so would be a menu.
//
// The chips are the grid and the notes are listed under it. Twenty-five cells
// of prose in a table makes a wall nobody compares across; twenty-five chips
// can be read in one look, and the nine annotated cells keep their words.
SHEET_PANELS.grid = {
  isEmpty(panel) {
    return !(((panel.content || {}).cells) || null);
  },
  render(panel) {
    const content = panel.content || {};
    const perturbations = content.perturbations || [];
    const observables = content.observables || [];
    const cells = content.cells || {};
    const wrap = document.createElement("div");
    if (content.note) {
      const note = document.createElement("div");
      note.className = "table-note";
      note.innerHTML = inlineMarkup(content.note);
      wrap.append(note);
    }
    const head =
      "<th></th>" +
      observables
        .map((obs) => `<th title="${escapeHtml(obs.sub || "")}">${escapeHtml(obs.name)}</th>`)
        .join("");
    const body = perturbations
      .map((pert) => {
        const row = observables
          .map((obs) => {
            const cell = cells[`${pert.key}|${obs.key}`] || {};
            const kind = cell.kind || "unannotated";
            return `<td class="cell kind-${escapeHtml(kind.split(" ")[0])}" ` +
              `title="${escapeHtml(cell.note || "")}">${escapeHtml(kind)}</td>`;
          })
          .join("");
        return `<tr><th title="${escapeHtml(pert.sub || "")}">${escapeHtml(pert.name)}` +
          `<span class="cost">${escapeHtml(pert.cost || "")}</span></th>${row}</tr>`;
      })
      .join("");
    const table = document.createElement("div");
    table.className = "scroll-x";
    table.innerHTML =
      `<table class="gridtable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    wrap.append(table);
    // The annotated cells, in the grid's own order, with the source's words.
    // A tooltip is not a place to put an argument.
    const notes = document.createElement("div");
    notes.className = "gridnotes";
    perturbations.forEach((pert) => {
      observables.forEach((obs) => {
        const cell = cells[`${pert.key}|${obs.key}`] || {};
        if (!cell.kind || cell.kind === "unannotated") return;
        const item = document.createElement("p");
        item.innerHTML =
          `<b>${escapeHtml(pert.name)} × ${escapeHtml(obs.name)}</b> ` +
          `<span class="kind kind-${escapeHtml(cell.kind.split(" ")[0])}">` +
          `${escapeHtml(cell.kind)}</span><br>${inlineMarkup(cell.note || "")}`;
        notes.append(item);
      });
    });
    wrap.append(notes);
    return wrap;
  },
};

// The six option plates of the source's 1.02, as plates. Four prose fields per
// flavour and one status, and the status is the only part this page computes:
// whether THIS RUN builds that flavour, which the payload answers.
//
// `cannot answer` and `failure mode` are styled apart from the other two on
// purpose. They are the half of each plate that stops a reader adopting a
// geometry for a question it cannot answer, and set in the same grey as the
// cost line they read as small print.
SHEET_PANELS.cards = {
  isEmpty(panel) {
    return !(((panel.content || {}).plates) || []).length;
  },
  render(panel) {
    const content = panel.content || {};
    const wrap = document.createElement("div");
    if (content.note) {
      const note = document.createElement("div");
      note.className = "table-note";
      note.innerHTML = inlineMarkup(content.note);
      wrap.append(note);
    }
    const holder = document.createElement("div");
    holder.className = "plates";
    (content.plates || []).forEach((plate) => {
      const box = document.createElement("article");
      box.className = "plate";
      const fields = [
        ["answers", plate.answers, ""],
        ["cannot answer", plate.cannot, "bad"],
        ["cost", plate.cost, "quiet"],
        ["failure mode", plate.failure, "bad"],
      ];
      const rows = fields
        .filter(([, value]) => value)
        .map(([label, value, kind]) =>
          `<div class="field ${kind}"><dt>${escapeHtml(label)}</dt>` +
          `<dd>${inlineMarkup(value)}</dd></div>`)
        .join("");
      const state = plate.state || "";
      box.innerHTML =
        `<span class="plate-state state-${escapeHtml(state.split(" ")[0])}" ` +
        `title="${escapeHtml(plate.state_note || "")}">${escapeHtml(state)}</span>` +
        `<h3>${escapeHtml(plate.plate_id)}. ${escapeHtml(plate.name)}</h3>` +
        `<p class="formula">${escapeHtml(plate.formula || "")}</p>` +
        `<div class="plate-tag">${escapeHtml(plate.tag || "")}</div>` +
        `<dl>${rows}</dl>`;
      // The plate whose stated failure mode is live in this pipeline says so on
      // the plate. A hazard in a fold-out is a hazard read by the people who
      // already suspected it.
      if (plate.hazard) {
        const hazard = document.createElement("div");
        hazard.className = "plate-hazard";
        hazard.innerHTML = `<b>In this pipeline.</b> ${inlineMarkup(plate.hazard)}`;
        box.append(hazard);
      }
      holder.append(box);
    });
    wrap.append(holder);
    return wrap;
  },
};

// A panel whose content IS a table: columns and rows, both from the payload.
// Two catalogue panels declare `panel_type: "table"` -- the overlay-only signal
// inventory and the seven ways two trees disagree -- and neither is a plot, a
// count or a measurement of this cohort. One renderer serves both because the
// difference between them is entirely in their rows.
//
// The columns come from the payload too, rather than from a per-panel branch
// here. A renderer that knew the signal inventory's three column names would
// have to grow a branch for every table panel after it, and each branch is a
// place for the page to disagree with the data it was handed.
SHEET_PANELS.table = {
  // Declared and empty is empty, whatever the registry says. Without this the
  // tab bar counts a table panel with no rows as filled.
  isEmpty(panel) {
    return !(((panel.content || {}).rows) || []).length;
  },
  render(panel) {
    const content = panel.content || {};
    const columns = content.columns || [];
    const rows = content.rows || [];
    const wrap = document.createElement("div");
    if (!columns.length || !rows.length) {
      // Declared as a table and handed no table. Same rule as everywhere else
      // on this page: say which of the two halves is missing.
      const box = document.createElement("div");
      box.className = "refused";
      box.innerHTML =
        `<b>this table's ${columns.length ? "rows" : "columns"} are empty</b> — ` +
        "the payload declares the panel and does not carry its content";
      wrap.append(box);
      return wrap;
    }
    if (content.note) {
      const note = document.createElement("div");
      note.className = "table-note";
      note.innerHTML = inlineMarkup(content.note);
      wrap.append(note);
    }
    wrap.append(
      sheetTable(
        columns.map((column) => ({
          label: column.label,
          cell: (row) => `<td>${inlineMarkup(String(row[column.key] ?? ""))}</td>`,
        })),
        rows
      )
    );
    return wrap;
  },
};


// --- the diagnostics report ---------------------------------------------------
// 7.03 E2 fixes the order -- cohort and provenance, retrieval coverage,
// geometry health, rate fit, and only THEN the map -- and the order is the
// argument, not a layout preference.
//
// So the ORDER LIVES IN THE PAYLOAD. This renderer walks `panel.content.sections`
// in the order Python put them in and never looks a section up by name. A
// template free to look them up is a template free to draw the map first, which
// is the single thing the source says a report must not do.
//
// Each filler takes the section and returns a Node, or null when the payload
// does not carry what that section needs -- in which case the section's own
// `refused` text is drawn instead of nothing.
const REPORT_FILLERS = {};

// label / value / note, where `value` is trusted markup the filler built and
// `label` and `note` are escaped here. Two columns of numbers with a third for
// where the number came from -- a diagnostic whose provenance is not beside it
// gets quoted without it.
function reportRows(rows) {
  const box = document.createElement("div");
  box.className = "scroll-x";
  box.innerHTML =
    "<table><tbody>" +
    rows
      .map(([label, value, note]) =>
        `<tr><th>${escapeHtml(label)}</th><td>${value}</td>` +
        `<td class="mono">${note ? escapeHtml(note) : ""}</td></tr>`)
      .join("") +
    "</tbody></table>";
  return box;
}

// A number the payload does not carry. Never a blank and never a zero: this is
// the same rule as the withheld cell in the comparisons table, for the same
// reason -- a blank reads as a measurement of nothing and a zero reads as a
// measurement of zero.
function reportMissing(why) {
  return `<span class="withheld" title="${escapeHtml(why)}">not in this payload</span>`;
}

// One column per thing a reader has to check before believing a map, and each
// cell answers with a number or with WHY it has none. A space missing a whole
// diagnostics section is the common case here -- redundancy exists only for a
// fused space, and a negative control only where one was configured.
const GEOMETRY_COLUMNS = [
  { label: "space", cell: (s) => escapeHtml(s.space_id) },
  {
    label: "verdict",
    cell: (s) => `<span class="v-${escapeHtml(s.verdict.level)}">` +
      `${escapeHtml(s.verdict.level)}</span>`,
  },
  {
    label: "stability (mean)",
    cell: (s) => {
      const stability = firstOf(s, "stability");
      if (!stability) return reportMissing("this space's diagnostics carry no stability section");
      // A perfect score that could not have been anything else is not a score.
      if (stability.informative === false) {
        return withheldWord(
          "not informative",
          "k is most of the cohort, so every protein's neighbours are all the " +
            "others and the score is 1.0 by construction"
        );
      }
      return fmtValue(stability.stability_mean);
    },
  },
  {
    label: "coin flips",
    cell: (s) => {
      const stability = firstOf(s, "stability");
      if (!stability) return reportMissing("this space's diagnostics carry no stability section");
      return `${fmtValue(stability.n_coin_flips)} of ${fmtValue(stability.n_measured)}`;
    },
  },
  { label: "trustworthiness", cell: (s) => faithCell(s, "trustworthiness_mean") },
  { label: "continuity", cell: (s) => faithCell(s, "continuity_mean") },
  {
    label: "positions not readable",
    note: distortedCutoffNote(),
    cell: (s) => {
      const readable = s.readable || [];
      const bad = readable.filter((r) => r === false).length;
      return `${fmtValue(bad)} of ${fmtValue(readable.length)}`;
    },
  },
  {
    label: "clusters",
    cell: (s) => {
      const partition = (s.diagnostics || {}).partition || {};
      if (partition.n_clusters === undefined) {
        return reportMissing("this space's diagnostics carry no partition section");
      }
      return `${fmtValue(partition.n_clusters)} at resolution ` +
        `${fmtValue(partition.resolution)}` + sweepNote(s);
    },
  },
  {
    label: "control margin",
    cell: (s) => {
      const controls = (s.diagnostics || {}).negative_controls || {};
      const margins = Object.values(controls.margins || {});
      // The WORST margin, not the mean of them: one control the clusters fail
      // to beat is the finding, and averaging it against a control they beat
      // would hide exactly that.
      if (!margins.length) return reportMissing("no negative control was run for this space");
      return fmtValue(Math.min.apply(null, margins));
    },
  },
  {
    label: "block redundancy",
    cell: (s) => {
      const redundancy = (s.diagnostics || {}).redundancy;
      if (!redundancy) {
        return withheldWord(
          "one block",
          "redundancy compares a space's blocks against each other, so a " +
            "one-block space has no pair to compare"
        );
      }
      return (redundancy.pairs || [])
        .map((pair) => `${escapeHtml(pair.block_a)}/${escapeHtml(pair.block_b)} ` +
          `${fmtValue(pair.spearman)}`)
        .join("<br>");
    },
  },
];

// The diagnostics arrive as one-element lists keyed by reducer, so reaching for
// [0] is right for stability and WRONG for faithfulness -- a space with two
// layouts has two faithfulness rows and reporting the first would hide the
// worse one. Hence two helpers rather than one.
function firstOf(space, key) {
  return ((space.diagnostics || {})[key] || [])[0] || null;
}

// The cluster count printed beside this, and what it was at the other
// resolutions tried.
//
// `partition.n_clusters` reads like a property of the cohort. It is not: it is
// the count at ONE resolution, and the payload has carried the whole sweep --
// the count at every resolution tried, the ARI between adjacent ones, and the
// plateau -- with no template hit at all. On chymo_A1's fused_late the count
// goes 4, 6, 10, 13 across resolutions 0.25 to 2.0 while the table prints "10
// at resolution 1" alone.
//
// This is the page-facing form of the rule that partitions may never be
// compared at unmatched cluster count: the sweep is only meaningful beside the
// number it qualifies, so it is drawn under that number rather than in a panel
// of its own. The resolution the page actually drew is marked, so a reader can
// see which row of the sweep they are looking at.
//
// Numbers only, no verdict. Whether any two adjacent resolutions agree is a
// judgement the diagnostics already wrote in words, and it is one click away in
// the fold-out on the space's own panel; restating it here would be the same
// sentence at two places with nothing keeping them in step.
function sweepNote(space) {
  const sweep = (space.diagnostics || {}).resolution_sweep || {};
  const steps = sweep.steps || [];
  if (!steps.length) return "";
  const drawn = ((space.diagnostics || {}).partition || {}).resolution;
  const walk = steps
    .map((step) => {
      const text = `${fmtValue(step.n_clusters)}@${fmtValue(step.resolution)}`;
      return step.resolution === drawn ? `<b>${escapeHtml(text)}</b>` : escapeHtml(text);
    })
    .join(" · ");
  // `fmtValue` and nothing else, the same formatter the rest of this table
  // uses. Rounding here first was pointless -- fmtValue already prints three
  // decimals -- and a second rounding rule inside one table is how two numbers
  // of the same kind end up printed differently.
  const ari = (sweep.adjacent_ari || [])
    .map((pair) => escapeHtml(fmtValue(pair.ari)))
    .join(" · ");
  return (
    `<span class="sweep">swept: ${walk}` +
    (ari ? `<br>adjacent ARI: ${ari}` : "") +
    "</span>"
  );
}

function faithCell(space, field) {
  const entries = (space.diagnostics || {}).faithfulness || [];
  if (!entries.length) {
    return reportMissing("this space's diagnostics carry no faithfulness section");
  }
  return entries
    .map((entry) => `${escapeHtml(entry.reducer)} ${fmtValue(entry[field])}`)
    .join("<br>");
}

// A word standing in for a number that CANNOT exist here, as against one the
// payload merely does not carry. Both are said out loud; they are different
// facts and a reader who cannot tell them apart cannot act on either.
function withheldWord(word, why) {
  return `<span class="withheld" title="${escapeHtml(why)}">${escapeHtml(word)}</span>`;
}

REPORT_FILLERS.cohort = () => {
  const p = active.provenance || {};
  const rows = [
    ["proteins on the maps", fmtValue(p.n_proteins), "provenance.n_proteins"],
    ["spaces drawn", fmtValue(p.n_spaces), "one panel each"],
    ["cohort selection rule", escapeHtml(p.cohort_rule || "n/a"), "config.cohort.selection"],
    ["diagnostics neighbourhood k", fmtValue(p.diagnostics_k), "config.diagnostics.k"],
    [
      "named records",
      (active.records || []).length
        ? `${(active.records || []).length} of ${fmtValue(p.n_proteins)}`
        : reportMissing("no uniprot_features.tsv was read for this run"),
      "protein_features/uniprot_features.tsv",
    ],
  ];
  Object.entries(p.manifests || {}).forEach(([space, m]) => {
    rows.push([
      space + " cache key",
      `<code>${escapeHtml(m.cache_key || "none")}</code>`,
      (m.versions && m.versions["umap-learn"]) ? "umap-learn " + m.versions["umap-learn"] : "",
    ]);
  });
  return reportRows(rows);
};

REPORT_FILLERS.coverage = () => {
  // The section this fills used to REFUSE, on the grounds that "no payload key
  // carries it". `active.censoring.summary` has carried it since the censoring
  // panel landed, so the refusal was false on a page that drew the number two
  // sheets away. The verdict wording below is deliberately the censoring
  // panel's own three states rather than a second phrasing: two sheets
  // disagreeing about whether a cohort was capped is the defect this section
  // exists to prevent.
  const summary = (active.censoring || {}).summary || {};
  if (!summary.n_cells) {
    return reportUnknownSection({ section_id: "coverage" });
  }
  const measured = summary.n_cells - summary.n_censored;
  let verdict;
  if (summary.cap_detected) {
    verdict =
      `a per-query cap is detected: ${fmtValue(summary.rows_at_max)} of ` +
      `${fmtValue(summary.n_rows)} rows report exactly ` +
      `${fmtValue(summary.inferred_cap)} partners while the columns do not pile up`;
  } else if (summary.n_censored > 0) {
    verdict =
      "cells are missing, but not in a per-query cap's pattern — the missingness " +
      "is a property of the whole matrix rather than of each query";
  } else {
    verdict = "nothing was censored: this cohort's matrix is exhaustive";
  }
  const rows = [
    [
      "cells measured",
      `${fmtValue(measured)} of ${fmtValue(summary.n_cells)}`,
      "matrix_io.summarize_censoring",
    ],
    [
      "censoring rate",
      `${(100 * summary.censoring_rate).toFixed(1)}%`,
      "n_censored / n_cells",
    ],
    ["cap verdict", escapeHtml(verdict), "ADR 0009"],
    [
      "per-space retention at each k",
      reportMissing(
        "the per-space censoring sections are not plumbed into the payload yet, " +
        "so this row reports the cohort's matrix and not each space's own retention"
      ),
      "PC-003 phase 2",
    ],
  ];
  return reportRows(rows);
};

REPORT_FILLERS.geometry = () => {
  const box = document.createElement("div");
  box.className = "scroll-x";
  // A column may carry the cutoff that produced its numbers. Rendered under the
  // label rather than beside it: this table already has ten columns, and a
  // header that widens one of them squeezes the rest.
  const head = GEOMETRY_COLUMNS.map(
    (c) => `<th>${escapeHtml(c.label)}` +
      (c.note ? `<span class="cutoff">${escapeHtml(c.note)}</span>` : "") +
      "</th>"
  ).join("");
  const body = spaces
    .map((space) => {
      const cells = GEOMETRY_COLUMNS.map((c) => `<td>${c.cell(space)}</td>`);
      return "<tr>" + cells.join("") + "</tr>";
    })
    .join("");
  box.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  return box;
};

REPORT_FILLERS.map = () => {
  const box = document.createElement("div");
  box.className = "scroll-x";
  const rows = spaces.map((space) => {
    const reasons = (space.verdict.reasons || []).length
      ? "<ul>" + space.verdict.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("") + "</ul>"
      : "";
    const level = escapeHtml(space.verdict.level);
    return `<tr><th>${escapeHtml(space.space_id)}</th><td class="v-${level}">` +
      `${escapeHtml(space.verdict.headline)}${reasons}</td></tr>`;
  });
  box.innerHTML = `<table><tbody>${rows.join("")}</tbody></table>`;
  return box;
};

// A section the payload declares and this template cannot fill. Same reasoning
// as `PANELS.__unknown__`: a page older than its payload must say so, because a
// section that silently vanishes cannot be told from one nobody wrote.
function reportUnknownSection(section) {
  const box = document.createElement("div");
  box.className = "refused";
  box.innerHTML =
    `<b>no filler for report section "${escapeHtml(section.section_id)}"</b> — ` +
    "this page's template is older than the payload that produced it";
  return box;
}

function refusedBlock(section) {
  const box = document.createElement("div");
  box.className = "refused";
  box.innerHTML = `<b>refused:</b> ${inlineMarkup(section.refused)}`;
  if (section.fills_in) {
    const fills = document.createElement("span");
    fills.className = "fills";
    fills.textContent = "filled in by: " + section.fills_in;
    box.append(fills);
  }
  return box;
}

SHEET_PANELS.report = {
  render(panel) {
    const wrap = document.createElement("div");
    const sections = ((panel.content || {}).sections) || [];
    if (!sections.length) {
      wrap.append(reportUnknownSection({ section_id: "none declared" }));
      return wrap;
    }
    sections.forEach((section) => {
      const box = document.createElement("div");
      box.className = "report-section";
      const head = document.createElement("h4");
      head.textContent = section.title;
      box.append(head);
      if (section.lead) {
        const lead = document.createElement("div");
        lead.className = "lead";
        lead.textContent = section.lead;
        box.append(lead);
      }
      // A section that declares a refusal prints it and does not try to fill
      // itself. The refusal is the content: 7.03 E2's rule is that a report
      // shows what it cannot say, and a half-filled section would be worse
      // than either half.
      if (section.refused) box.append(refusedBlock(section));
      else {
        const filler = REPORT_FILLERS[section.section_id];
        box.append(filler ? filler(section) : reportUnknownSection(section));
      }
      wrap.append(box);
    });
    return wrap;
  },
};

function renderSheet(sheetId) {
  state.sheet = sheetId;
  Array.from(el("sheets").children).forEach((b) => {
    b.classList.toggle("on", b.dataset.sheet === sheetId);
  });
  // The live grid and its two companions belong to `maps` only.
  const isMaps = sheetId === "maps";
  el("grid").style.display = isMaps ? "" : "none";
  el("maps-extra").style.display = isMaps ? "" : "none";
  document.querySelector(".legend").style.display = isMaps ? "" : "none";
  el("colour-key").style.display = isMaps && colourDomain.kind !== "none" ? "" : "none";
  // These drive the scatter grid and nothing else. Left enabled on a sheet
  // where the grid is hidden they read as controls for the cards, and a click
  // on "Disagreement mode" silently recoloured five maps the reader could not
  // see. Disabling is better than hiding: the row keeps its height, so the
  // page does not jump on every tab change.
  document.querySelectorAll(".controls select, .controls button").forEach((node) => {
    if (node.id === "cohort") return;   // the cohort applies to every sheet
    node.disabled = !isMaps;
  });
  document.querySelector(".controls").style.opacity = isMaps ? "" : "0.45";
  document.querySelector(".controls").title = isMaps
    ? "" : "these control the maps, which this sheet does not show";

  const body = el("sheet-body");
  body.textContent = "";
  // Two kinds already have dedicated renderers older than this catalogue --
  // the live grid and the comparisons table. Listing them again as cards would
  // put "renderer not built" directly above the thing that is built.
  const wanted = (active.panels || []).filter(
    (p) => p.sheet === sheetId && !BUILT_ELSEWHERE.has(p.panel_type)
  );
  // A panel drawn on another sheet still belongs in this one's catalogue --
  // pointed at, not drawn twice. Only when it is drawable: a pointer to a strip
  // no cohort renders would be a false statement, and the panel then stays
  // hidden exactly as it was before this existed.
  const elsewhere = (active.panels || []).filter(
    (p) => p.sheet === sheetId && p.drawable && DRAWN_AT[p.panel_id]
  );
  if (!wanted.length && !elsewhere.length) {
    body.style.display = "none";
    return;
  }
  body.style.display = "";
  const holder = document.createElement("div");
  holder.style.display = "grid";
  holder.style.gridTemplateColumns = "repeat(auto-fit, minmax(360px, 1fr))";
  holder.style.gap = "14px";
  wanted.forEach((panel) => {
    const card = panelCard(panel);
    const renderer = SHEET_PANELS[panel.panel_type];
    if (!panel.drawable) card.append(awaitingBlock(panel));
    else if (renderer) card.append(renderer.render(panel));
    else card.append(drawablePlaceholder(panel));
    holder.append(card);
  });
  elsewhere.forEach((panel) => {
    const card = panelCard(panel);
    const note = document.createElement("div");
    note.className = "elsewhere";
    note.textContent = DRAWN_AT[panel.panel_id];
    card.append(note);
    holder.append(card);
  });
  body.append(holder);
  flushPendingDraws();
}

// A panel whose inputs ARE present but whose renderer is not built yet. Said
// plainly and separately from `awaiting`, because the two are different states
// and collapsing them would hide which panels are one step from working.
function drawablePlaceholder(panel) {
  const box = document.createElement("div");
  box.className = "awaiting";
  const inner = document.createElement("div");
  // The card header already carries the question. Repeating it here printed
  // the same sentence twice in every placeholder.
  const head = document.createElement("b");
  head.textContent = "inputs present, renderer not built";
  inner.append(head);
  box.append(inner);
  return box;
}

// Whether a panel draws NOTHING, which is not the same question as whether it
// has a renderer. One renderer serves every `table` panel, so the day it was
// added the tab bar stopped counting a table panel with no rows as empty --
// while that panel still rendered a refusal and nothing else. A renderer may
// therefore answer for itself, and a renderer that does not is assumed to draw
// something once its inputs are present.
function panelIsBlank(panel) {
  if (!panel.drawable) return true;
  const renderer = SHEET_PANELS[panel.panel_type];
  if (!renderer) return true;
  return renderer.isEmpty ? renderer.isEmpty(panel) : false;
}

function renderSheets() {
  const bar = el("sheets");
  bar.textContent = "";
  (active.sheets || []).forEach((sheet) => {
    const count = (active.panels || []).filter((p) => p.sheet === sheet.sheet).length;
    if (!count) return;
    const button = document.createElement("button");
    button.dataset.sheet = sheet.sheet;
    // Count panels that DRAW NOTHING, not panels that lack data. A panel whose
    // inputs are present but whose renderer is unbuilt is just as empty to a
    // reader, and counting only the first understated two sheets to zero while
    // they rendered no content at all.
    const sheetPanels = (active.panels || []).filter(
      (p) => p.sheet === sheet.sheet && !BUILT_ELSEWHERE.has(p.panel_type)
    );
    const blank = sheetPanels.filter(panelIsBlank).length;
    button.textContent = blank ? `${sheet.title} (${blank} empty)` : sheet.title;
    button.addEventListener("click", () => renderSheet(sheet.sheet));
    bar.append(button);
  });
  renderSheet("maps");
}

// `toFixed(3)` printed a chain length as "367.000" and a probability as
// "0.000". Formatting by magnitude is the difference between a number a reader
// can use and one they have to decode.
function fmtValue(v) {
  if (v === null || v === undefined) return "";
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.001) return v.toFixed(3);
  return v.toExponential(1);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- comparisons --------------------------------------------------------------

function renderComparisons() {
  if (!active.comparisons.length) {
    el("comparisons").textContent = "No pairs were co-registered in this run.";
    return;
  }
  const columns = ["space_a", "space_b", "jaccard_mean", "rank_correlation_mean",
                   "procrustes_disparity", "cluster_ari"];
  const rows = active.comparisons.map((row) => {
    const cells = columns.map((c) => {
      const v = row[c];
      if (v === null || v === undefined) {
        // A blank cell reads as zero. Say the word.
        const why = c === "cluster_ari"
          ? "withheld — a partition with one cluster has no meaningful ARI"
          : "not computed for this pair";
        return `<td class="withheld" title="${escapeHtml(why)}">withheld</td>`;
      }
      return `<td>${typeof v === "number" ? v.toFixed(3) : escapeHtml(v)}</td>`;
    });
    return `<tr>${cells.join("")}</tr>`;
  });
  el("comparisons").innerHTML =
    `<table><thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.join("")}</tbody></table>`;
}

function renderInspector() {
  const node = el("inspector");
  if (!state.selected.size) { node.textContent = "Nothing selected."; return; }
  const chosen = [...state.selected].sort();
  const rows = spaces.map((space) => {
    const clusters = {};
    chosen.forEach((p) => {
      const c = space.clusters[p];
      if (c) clusters[c] = (clusters[c] || 0) + 1;
    });
    const unreadable = chosen.filter(
      (p) => space.readable[space.protids.indexOf(p)] === false
    ).length;
    const spread = Object.entries(clusters)
      .sort((a, b) => b[1] - a[1])
      .map(([c, n]) => `${c}×${n}`)
      .join(", ") || "no partition";
    return `<tr><td>${escapeHtml(space.space_id)}</td><td>${spread}</td>` +
           `<td>${unreadable} of ${chosen.length}</td></tr>`;
  });
  // The code line above truncates at 12 so it stays readable. The textarea
  // carries the WHOLE selection, and its label repeats the full count so the
  // two cannot be read as disagreeing.
  //
  // A textarea rather than a copy button, and the reason is measured rather
  // than assumed. file:// IS a secure context in Chrome and
  // navigator.clipboard.writeText exists there -- but it is gated on document
  // focus and rejects with NotAllowedError when that is not met. A button would
  // therefore work sometimes and fail silently other times, which is worse than
  // not having one. Selecting text needs no permission, no focus and no secure
  // context, and it shows the reader exactly what they are being handed.
  const egress =
    `<p class="egress-label">All ${chosen.length}, one per line — click to select all:</p>` +
    `<textarea class="egress" readonly rows="6" onfocus="this.select()">` +
    `${escapeHtml(chosen.join("\n"))}</textarea>`;

  // One row per protein. A protid the guard rejects gets its reason rather than
  // a broken link or a blank -- the page's standing rule that a refusal is
  // shown as a refusal. Domain-suffixed ids (`P60709__d01`) land here by design.
  const links = chosen.map((protid) => {
    const safe = escapeHtml(protid);
    if (!UNIPROT_ACCESSION.test(protid)) {
      return `<li><code>${safe}</code> <span class="withheld">not a UniProt ` +
             `accession, so it has no entry page to link to</span></li>`;
    }
    const acc = encodeURIComponent(protid);
    const uni = UNIPROT_ENTRY.replace("{acc}", acc);
    const afdb = AFDB_ENTRY.replace("{acc}", acc);
    return `<li><code>${safe}</code> ` +
           `<a href="${uni}" target="_blank" rel="noopener">UniProt</a> · ` +
           `<a href="${afdb}" target="_blank" rel="noopener">AlphaFold</a></li>`;
  });

  node.innerHTML =
    `<p>${chosen.length} protein(s): <code>${chosen.slice(0, 12).map(escapeHtml).join(", ")}` +
    `${chosen.length > 12 ? ", …" : ""}</code></p>` +
    egress +
    `<table><thead><tr><th>space</th><th>clusters they fall in</th>` +
    `<th>positions not readable<span class="cutoff">${escapeHtml(distortedCutoffNote())}` +
    `</span></th></tr></thead><tbody>${rows.join("")}</tbody></table>` +
    `<ul class="egress-links">${links.join("")}</ul>`;
}

function renderProvenance() {
  const p = active.provenance;
  const manifests = Object.entries(p.manifests || {})
    .map(([space, m]) =>
      `<li><b>${escapeHtml(space)}</b> — ` +
      `<code>${escapeHtml(m.cache_key || "no cache key")}</code></li>`)
    .join("");
  el("provenance").innerHTML =
    `<b>Provenance.</b> ${p.n_spaces} space(s), ${p.n_proteins} proteins, ` +
    `neighbourhood k=${p.diagnostics_k}, cohort rule ` +
    `<code>${escapeHtml(p.cohort_rule || "n/a")}</code>. ` +
    `This file carries no generation timestamp on purpose: two runs of the same inputs ` +
    `must produce the same bytes. ` +
    `<ul>${manifests}</ul>` +
    `<p>Read <code>docs/INTERPRETING.md</code> before treating anything here as a finding.</p>`;
}

function showCohort(index) {
  applyCohort(index);
  draw();
  renderComparisons();
  renderInspector();
  renderProvenance();
  renderSheets();
}

if (COHORTS.length > 1) {
  const control = el("cohort-control");
  control.style.display = "";
  const select = el("cohort");
  COHORTS.forEach((cohort, i) => select.append(new Option(cohort.cohort_name, String(i))));
  select.onchange = (e) => showCohort(Number(e.target.value));
}

showCohort(0);
</script>
</body>
</html>
"""


def render(payload: dict, plotly_js: str, title: str) -> str:
    """The finished HTML.

    Substitution rather than `str.format` or an f-string: the template is mostly
    CSS and JavaScript, both of which are full of braces, and every one of them
    would have to be doubled. The same reasoning as the Snakefile's ban on
    f-string wildcard paths, met in a different syntax.
    """
    import json

    return (
        _TEMPLATE.replace("__PLOTLY__", plotly_js)
        .replace("__PAYLOAD__", json.dumps(payload, sort_keys=True))
        .replace("__TITLE__", title)
    )
