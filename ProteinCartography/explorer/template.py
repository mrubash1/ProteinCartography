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

_TEMPLATE = """<!doctype html>
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
/* A panel with nothing to draw says so, in the panel's own footprint. The
   source document's rule (7.03 E2) is that refusals are shown as refusals and
   never as blanks; the same reasoning applies to a panel awaiting an input. */
.awaiting { height: 330px; display: flex; align-items: center; justify-content: center;
  text-align: center; padding: 0 22px; color: var(--muted); font-size: 12.5px;
  background: repeating-linear-gradient(45deg, #fcfcfd, #fcfcfd 9px, #f6f7f9 9px, #f6f7f9 18px); }
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
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub" id="subtitle"></div>
</header>

<div class="controls">
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

<div class="legend">
  <b>Hollow points</b> are proteins whose 2-D position the diagnostics say should not be read.
  Selecting in any panel highlights the same proteins in all of them.
</div>

<div id="grid"></div>

<section>
  <h3>Cross-space comparison</h3>
  <div id="comparisons"></div>
  <h3>Selection</h3>
  <div id="inspector">Nothing selected.</div>
</section>

<footer id="provenance"></footer>

<script type="text/javascript">
const PAYLOAD = __PAYLOAD__;

const state = { overlay: "__none__", reducer: null, disagreement: false, selected: new Set() };

const el = (id) => document.getElementById(id);
const spaces = PAYLOAD.spaces;

// Every reducer any space produced. A space missing the chosen one is drawn
// with whatever it has, and says so, rather than vanishing from the grid.
const reducers = [...new Set(spaces.flatMap((s) => Object.keys(s.embeddings)))].sort();
state.reducer = reducers[0] || null;

el("subtitle").textContent =
  `${spaces.length} space(s) over ${PAYLOAD.provenance.n_proteins} proteins` +
  ` · k=${PAYLOAD.provenance.diagnostics_k}` +
  ` · cohort rule: ${PAYLOAD.provenance.cohort_rule || "n/a"}`;

// --- controls ---------------------------------------------------------------

const overlaySelect = el("overlay");
overlaySelect.append(new Option("nothing (uniform)", "__none__"));
const clusterGroup = document.createElement("optgroup");
clusterGroup.label = "this space's own clusters";
clusterGroup.append(new Option("cluster", "__cluster__"));
overlaySelect.append(clusterGroup);
const featureNames = Object.keys(PAYLOAD.overlays).sort();
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

reducers.forEach((r) => el("reducer").append(new Option(r, r)));

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
const disagreementByProtid = (() => {
  const sums = new Map(), counts = new Map();
  for (const row of PAYLOAD.comparisons) {
    if (!row.per_protein) continue;
    for (const [protid, value] of Object.entries(row.per_protein)) {
      if (value === null) continue;
      sums.set(protid, (sums.get(protid) || 0) + value);
      counts.set(protid, (counts.get(protid) || 0) + 1);
    }
  }
  const out = new Map();
  for (const [protid, total] of sums) out.set(protid, total / counts.get(protid));
  return out;
})();

// --- drawing -----------------------------------------------------------------

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
  const overlay = PAYLOAD.overlays[state.overlay];
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
    } else if (colour.kind === "categorical") {
      const levels = [...new Set(colour.values.filter((v) => v !== null))].sort();
      marker.color = idx.map((i) => {
        const at = levels.indexOf(colour.values[i]);
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
        const bits = [p];
        if (space.clusters[p]) bits.push(`cluster ${space.clusters[p]}`);
        if (colour.label && colour.values[i] !== null && colour.values[i] !== undefined) {
          const v = colour.values[i];
          bits.push(`${colour.label}: ${typeof v === "number" ? v.toFixed(3) : v}`);
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

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- comparisons --------------------------------------------------------------

function renderComparisons() {
  if (!PAYLOAD.comparisons.length) {
    el("comparisons").textContent = "No pairs were co-registered in this run.";
    return;
  }
  const columns = ["space_a", "space_b", "jaccard_mean", "rank_correlation_mean",
                   "procrustes_disparity", "cluster_ari"];
  const rows = PAYLOAD.comparisons.map((row) => {
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
  const p = PAYLOAD.provenance;
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

draw();
renderComparisons();
renderInspector();
renderProvenance();
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
