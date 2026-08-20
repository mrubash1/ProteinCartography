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
.plot { height: 330px; }
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
.scroll-x { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: .04em; }
td.withheld { color: var(--muted); font-style: italic; }
section { padding: 6px 22px 20px; }
section h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em;
             color: var(--muted); margin: 18px 0 7px; }
footer { border-top: 1px solid var(--line); background: var(--panel);
         padding: 14px 22px 26px; font-size: 12px; color: var(--muted); }
footer code { font-size: 11.5px; }
.legend { font-size: 12px; color: var(--muted); padding: 0 22px 4px; }
.legend b { color: var(--ink); font-weight: 600; }
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

<div class="legend">
  <b>Hollow points</b> are proteins whose 2-D position the diagnostics say should not be read.
  Selecting in any panel highlights the same proteins in all of them.
</div>

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
    box.append(
      document.createTextNode(fmtValue(colourDomain.min) + " "), ramp,
      document.createTextNode(" " + fmtValue(colourDomain.max))
    );
  }
  const note = document.createElement("span");
  note.className = "key-note";
  note.textContent = "one key for every panel";
  box.append(note);
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

const PALETTE = ["#4269d0","#efb118","#ff725c","#6cc5b0","#3ca951","#ff8ab7",
                 "#a463f2","#97bbf5","#9c6b4e","#9498a0"];

function traceFor(space) {
  const coords = space.embeddings[state.reducer] || Object.values(space.embeddings)[0];
  const colour = colourFor(space);
  const selected = state.selected;
  const anySelected = selected.size > 0;

  // Readable and unreadable proteins go into two traces so the marker outline
  // can differ. Plotly cannot vary `symbol` fill per point in a way that reads
  // clearly at this size, and the distinction has to survive being glanced at.
  const groups = [
    { readable: true, symbol: "circle", opacity: 0.92 },
    { readable: false, symbol: "circle-open", opacity: 0.85 },
  ];
  return groups.map(({ readable, symbol, opacity }) => {
    const idx = space.protids
      .map((_, i) => i)
      .filter((i) => space.readable[i] === readable);
    const marker = { symbol, size: 9, line: { width: 1.4 } };
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
  body.innerHTML =
    paragraphs.map((p) => `<p>${inlineMarkup(p)}</p>`).join("") +
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
  const explain = explainBlock(space.description);
  if (explain) panel.append(explain);
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
        `${fmtValue(partition.resolution)}`;
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

REPORT_FILLERS.geometry = () => {
  const box = document.createElement("div");
  box.className = "scroll-x";
  const head = GEOMETRY_COLUMNS.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("");
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
  if (!wanted.length) {
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
  body.append(holder);
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
  node.innerHTML =
    `<p>${chosen.length} protein(s): <code>${chosen.slice(0, 12).map(escapeHtml).join(", ")}` +
    `${chosen.length > 12 ? ", …" : ""}</code></p>` +
    `<table><thead><tr><th>space</th><th>clusters they fall in</th>` +
    `<th>positions not readable</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
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
