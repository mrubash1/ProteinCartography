#!/usr/bin/env python
"""What the explorer refuses to draw, which is the only interesting thing to test.

Rendering is not tested here -- there is no browser, and asserting on markup
would pin the CSS rather than the behaviour. What *is* tested is the payload:
every judgement about whether something may be read is made in
``explorer/payload.py`` and travels to the page as a flag, precisely so it can
be checked without parsing HTML.

The load-bearing tests are the three refusals. Group 8c produced nine
diagnostics and several exist to say *do not read this*; an explorer that
renders an unreadable space identically to a trustworthy one undoes that group,
and nothing in a shape assertion would notice.
"""

from __future__ import annotations
import json
import os
import re

import pytest
from explorer.payload import space_verdict

pytest.importorskip("pandas")


def _embedded_payload(html: str) -> dict:
    """The JSON object the rendered page carries, parsed rather than grepped.

    `render` substitutes the payload into a `const PAYLOAD = ...;` line, so the
    value is recoverable exactly. Recovering it lets a test assert on the number
    the page received rather than on how `json.dumps` chose to space it.
    """
    match = re.search(r"^const PAYLOAD = (.*);$", html, re.MULTILINE)
    assert match, "the rendered page carries no `const PAYLOAD = ...;` line"
    return json.loads(match.group(1))


# --- the space-level verdict --------------------------------------------------


def test_a_space_with_no_diagnostics_is_not_silently_trusted():
    verdict = space_verdict({}, 11)
    assert verdict["level"] == "caution"
    assert "nothing here is qualified" in verdict["reasons"][0]


def test_an_uninformative_stability_makes_the_space_unreadable():
    """The demo's case, and the reason this field exists at all.

    At eleven proteins `k` clamps to 8 of the 8 candidates a replicate offers,
    so every protein's neighbours are all the others and the Jaccard is 1.0 by
    construction. A perfect score that cannot be anything else must not render
    as a good map.
    """
    verdict = space_verdict(
        {"stability": [{"informative": False, "k": 8, "subsample_size": 9}]}, 11
    )
    assert verdict["level"] == "unreadable"
    assert any("not informative" in r for r in verdict["reasons"])


def test_a_stability_at_the_coin_flip_level_is_unreadable():
    verdict = space_verdict(
        {"stability": [{"informative": True, "stability_mean": 0.22, "subsample_size": 200}]}, 240
    )
    assert verdict["level"] == "unreadable"
    assert any("coin-flip" in r for r in verdict["reasons"])


def test_a_distorted_layout_earns_caution_not_silence():
    verdict = space_verdict(
        {
            "stability": [{"informative": True, "stability_mean": 0.9, "subsample_size": 200}],
            "faithfulness": [
                {"reducer": "pca_umap", "trustworthiness_mean": 0.52, "continuity_mean": 0.61}
            ],
        },
        240,
    )
    assert verdict["level"] == "caution"
    assert any("substantially distorted" in r for r in verdict["reasons"])


def test_a_borrowed_partition_is_surfaced_verbatim():
    """`partition.caveat` says a space's clusters are really another space's.
    It is shown as written rather than paraphrased, because the wording is the
    diagnostic."""
    caveat = "this space was not clustered in its own right"
    verdict = space_verdict(
        {
            "stability": [{"informative": True, "stability_mean": 0.9, "subsample_size": 200}],
            "partition": {"caveat": caveat},
        },
        240,
    )
    assert verdict["level"] == "caution"
    assert caveat in verdict["reasons"]


def test_a_partition_that_loses_to_its_own_noise_control_is_unreadable():
    """Item 8's headline. A clustering that a random matrix matches is not a
    finding however good the picture looks."""
    verdict = space_verdict(
        {
            "stability": [{"informative": True, "stability_mean": 0.9, "subsample_size": 200}],
            "negative_controls": {"margins": {"random_distances": -0.02}},
        },
        240,
    )
    assert verdict["level"] == "unreadable"
    assert any("no better separated" in r for r in verdict["reasons"])


def test_a_healthy_space_is_not_warned_about():
    """The other half of the guard: a diagnostic that always fires is noise."""
    verdict = space_verdict(
        {
            "stability": [{"informative": True, "stability_mean": 0.88, "subsample_size": 200}],
            "faithfulness": [
                {"reducer": "pca_umap", "trustworthiness_mean": 0.95, "continuity_mean": 0.93}
            ],
            "negative_controls": {"margins": {"shuffled_labels": 0.6}},
        },
        240,
    )
    assert verdict["level"] == "ok"
    assert verdict["reasons"] == []


def test_the_bands_come_from_the_modules_that_compute_them():
    """Not re-declared here. If `embedding` or `stability` moves a threshold,
    the explorer moves with it and `docs/INTERPRETING.md` stays true."""
    import inspect

    from explorer import payload

    source = inspect.getsource(payload.space_verdict)
    assert "DISTORTED_THRESHOLD" in source
    assert "COIN_FLIP_THRESHOLD" in source


# --- the per-protein mask -----------------------------------------------------


def _write_faithfulness(directory, reducer, rows):
    path = directory / f"faithfulness_{reducer}.tsv"
    path.write_text(
        "protid\ttrustworthiness\tcontinuity\n" + "".join(f"{p}\t{t}\t{c}\n" for p, t, c in rows)
    )
    return path


def test_an_unfaithful_protein_is_marked_unreadable(tmp_path):
    """The mask is read from `faithfulness_{reducer}.tsv`, not from
    `diagnostics.json`.

    The first draft read per-protein arrays out of the JSON. Those keys do not
    exist -- `to_dict` carries only the means -- so every protein came back
    readable and the mechanism was silently inert while passing every shape
    check. Running it against the demo is what exposed it.
    """
    from explorer.payload import _readable_mask

    protids = ["A", "B", "C"]
    _write_faithfulness(tmp_path, "pca", [("A", 0.95, 0.94), ("B", 0.10, 0.20), ("C", 0.9, 0.9)])
    mask = _readable_mask(str(tmp_path), {"faithfulness": [{"reducer": "pca"}]}, protids)
    assert mask == [True, False, True]


def test_either_direction_alone_makes_a_protein_unreadable(tmp_path):
    """Trustworthiness and continuity fail in opposite directions and are never
    averaged. A point in the wrong neighbourhood is as unreadable as one torn
    away from its own."""
    from explorer.payload import _readable_mask

    _write_faithfulness(tmp_path, "pca", [("A", 0.99, 0.10), ("B", 0.10, 0.99)])
    mask = _readable_mask(str(tmp_path), {"faithfulness": [{"reducer": "pca"}]}, ["A", "B"])
    assert mask == [False, False]


def test_a_protein_unfaithful_in_any_reducer_is_unreadable(tmp_path):
    from explorer.payload import _readable_mask

    _write_faithfulness(tmp_path, "pca", [("A", 0.99, 0.99), ("B", 0.99, 0.99)])
    _write_faithfulness(tmp_path, "umap", [("A", 0.99, 0.99), ("B", 0.05, 0.05)])
    mask = _readable_mask(
        str(tmp_path),
        {"faithfulness": [{"reducer": "pca"}, {"reducer": "umap"}]},
        ["A", "B"],
    )
    assert mask == [True, False]


def test_a_missing_faithfulness_table_marks_everything_unreadable(tmp_path):
    """The section claims the space was scored and the evidence is absent.
    Drawing every point as trustworthy on the strength of a missing file is the
    failure this whole module exists to prevent."""
    from explorer.payload import _readable_mask

    mask = _readable_mask(str(tmp_path), {"faithfulness": [{"reducer": "pca"}]}, ["A", "B"])
    assert mask == [False, False]


def test_no_faithfulness_section_leaves_everything_readable(tmp_path):
    """Distinct from the case above: nothing claimed to have scored this space,
    so there is no missing evidence, and the space-level verdict handles it."""
    from explorer.payload import _readable_mask

    assert _readable_mask(str(tmp_path), {}, ["A", "B"]) == [True, True]


# --- the whole payload, through the real entry point --------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A real run directory, driven through `build_payload`."""
    from config_schema import from_legacy
    from explorer.payload import build_payload
    from fusion_cohort import NARROW_BLOCK, WIDE_BLOCK, fusion_cohort, write_fusion_cohort

    root = tmp_path_factory.mktemp("explorer")
    output = root / "output"
    cohort = fusion_cohort(n=60)
    write_fusion_cohort(output, cohort, [WIDE_BLOCK, NARROW_BLOCK])

    for space_id in ("a", "b"):
        directory = output / "spaces" / space_id
        directory.mkdir(parents=True, exist_ok=True)
        rows = ["protid\tUMAP1\tUMAP2"]
        for i, protid in enumerate(cohort.protids):
            rows.append(f"{protid}\t{float(i)}\t{float(i % 7)}")
        (directory / "embedding_pca.tsv").write_text("\n".join(rows) + "\n")
        (directory / "clusters.tsv").write_text(
            "protid\tcluster\n" + "".join(f"{p}\tLC{i % 3}\n" for i, p in enumerate(cohort.protids))
        )
        _write_faithfulness(
            directory,
            "pca",
            [(p, 0.2 if i == 0 else 0.95, 0.95) for i, p in enumerate(cohort.protids)],
        )
        (directory / "diagnostics.json").write_text(
            json.dumps(
                {
                    "space_id": space_id,
                    "faithfulness": [
                        {
                            "reducer": "pca",
                            "trustworthiness_mean": 0.93,
                            "continuity_mean": 0.94,
                        }
                    ],
                    "stability": [
                        {"informative": True, "stability_mean": 0.85, "subsample_size": 48}
                    ],
                }
            )
        )

    config = from_legacy(
        {
            "blocks": {b: {"provider": "tmscore"} for b in (WIDE_BLOCK, NARROW_BLOCK)},
            "spaces": {
                "a": {"blocks": [WIDE_BLOCK], "strategy": "none", "reducers": ["pca"]},
                "b": {"blocks": [NARROW_BLOCK], "strategy": "none", "reducers": ["pca"]},
            },
        }
    )
    return build_payload(config, str(output), analysis_name="test")


def test_the_payload_has_one_entry_per_space_with_an_embedding(built):
    assert [space.space_id for space in built.spaces] == ["a", "b"]


def test_a_space_without_an_embedding_is_absent_rather_than_empty(tmp_path):
    """A space skipped for a missing provider has nothing to draw, and an empty
    panel would suggest the run covered more than it did (ADR 0006)."""
    from config_schema import from_legacy
    from explorer.payload import build_payload

    config = from_legacy(
        {
            "blocks": {"t": {"provider": "tmscore"}},
            "spaces": {"ghost": {"blocks": ["t"], "strategy": "none", "reducers": ["pca"]}},
        }
    )
    assert build_payload(config, str(tmp_path)).spaces == []


def test_the_provenance_carries_no_timestamp(built):
    """Two runs of the same inputs must produce the same bytes, and a
    generation time is the one field that would guarantee they never do."""
    assert "generated" not in built.provenance
    assert "timestamp" not in json.dumps(built.provenance).lower()


def test_the_payload_is_json_serializable(built):
    """It is embedded in a `<script>` block; anything numpy would break the page
    at load with no error a reader could act on."""
    assert json.loads(json.dumps(built.to_dict()))["analysis_name"] == "test"


def test_the_per_protein_mask_reaches_the_payload(built):
    space = built.spaces[0]
    assert space.readable[0] is False
    assert all(space.readable[1:])


# ==========================================================================
# GE.5 -- the provenance footer read a filename the store never writes, and
# ADR 0002's "rendered on the panel" was the one verb that was not true
# ==========================================================================


def _write_space_manifest(directory, reducer, contributions=None):
    """A manifest under the name the store actually uses."""
    extra = {"fusion": {"contributions": contributions}} if contributions else {}
    (directory / f"manifest_{reducer}.json").write_text(
        json.dumps(
            {
                "cache_key": "deadbeef",
                "provider": "fusion",
                "params": {"strategy": "early"},
                "versions": {"numpy": "1.23.5"},
                "extra": extra,
            }
        )
    )


def test_the_provenance_footer_finds_a_manifest_under_its_real_name(tmp_path):
    """`manifest.json` is never written; `manifest_{reducer}.json` is.

    The footer collected `{}` for every space and rendered an empty list, and
    nothing failed -- the only symptom was a blank section in a 3.7 MB file.
    """
    from explorer.payload import _space_manifest

    _write_space_manifest(tmp_path, "pca_umap")
    assert _space_manifest(str(tmp_path), ["pca_umap"])["cache_key"] == "deadbeef"
    # the name the code used to look for must not start working by accident
    assert not (tmp_path / "manifest.json").exists()


def test_a_manifest_under_the_old_name_is_not_what_is_read(tmp_path):
    from explorer.payload import _space_manifest

    (tmp_path / "manifest.json").write_text(json.dumps({"cache_key": "wrong"}))
    assert _space_manifest(str(tmp_path), ["pca_umap"]) == {}


def test_the_diagnostics_manifest_is_the_fallback(tmp_path):
    from explorer.payload import _space_manifest

    _write_space_manifest(tmp_path, "diagnostics")
    assert _space_manifest(str(tmp_path), ["pca_umap"])["cache_key"] == "deadbeef"


def test_a_fused_space_carries_both_the_asked_and_the_realized_share(tmp_path):
    """ADR 0002 requires the share to be displayed, not merely recorded.

    Both numbers, because `early` concatenates features: an even request over
    blocks of unequal width realizes unevenly, and the demo's `fused_early`
    asks 50/50 and lands at 73/27.
    """
    from explorer.payload import _contributions

    _write_space_manifest(
        tmp_path,
        "pca_umap",
        [
            {"block_id": "tmscore", "weight": 1.0, "share": 0.5, "realized_share": 0.733},
            {"block_id": "biophys", "weight": 1.0, "share": 0.5, "realized_share": 0.267},
        ],
    )
    rows = _contributions(str(tmp_path), ["pca_umap"])
    assert [r["block_id"] for r in rows] == ["tmscore", "biophys"]
    assert [r["share"] for r in rows] == [0.5, 0.5]
    assert [r["realized_share"] for r in rows] == [0.733, 0.267]


def test_a_single_block_space_apportions_nothing(tmp_path):
    """ "tmscore 100%" on every unfused panel is noise, not provenance."""
    from explorer.payload import _contributions

    _write_space_manifest(
        tmp_path,
        "pca_umap",
        [{"block_id": "tmscore", "weight": 1.0, "share": 1.0, "realized_share": 1.0}],
    )
    assert _contributions(str(tmp_path), ["pca_umap"]) == []


def test_the_rendered_page_both_carries_and_draws_the_share():
    """The last verb in ADR 0002. Carrying the number is not displaying it.

    Two assertions, because they fail for different reasons: the payload can be
    right while the panel ignores it, which is the state this fixed.
    """
    from explorer.template import render

    html = render(
        {
            "spaces": [
                {
                    "space_id": "fused",
                    "contributions": [
                        {"block_id": "tmscore", "share": 0.5, "realized_share": 0.733},
                        {"block_id": "biophys", "share": 0.5, "realized_share": 0.267},
                    ],
                }
            ]
        },
        plotly_js="",
        title="t",
    )
    # Parse the payload the page embeds rather than grepping the serialization.
    # `'"realized_share": 0.733' in html` pinned `json.dumps`' separator
    # defaults: `separators=(",", ":")`, or a float repr change, would fail it
    # while the number reaching the page was still exactly right.
    payload = _embedded_payload(html)
    shares = [c["realized_share"] for c in payload["spaces"][0]["contributions"]]
    assert shares == [0.733, 0.267], "the share did not reach the page"
    assert "space.contributions" in html, (
        "the panel must read `space.contributions`; ADR 0002 requires the share "
        "to be visible on a fused map, and recording it was already true."
    )
    # The panel must build a DOM node for the shares, not merely receive them.
    # Matched whitespace-insensitively: the exact spacing of `className =
    # "shares"` is the JS formatter's business, not this test's.
    draws_shares = re.search(r'className\s*=\s*"shares"', html)
    assert draws_shares, "the panel does not create an element for the shares"


# ==========================================================================
# GE.17 -- disagreement mode read a key the payload never wrote
# ==========================================================================


def _write_coregistration(directory, pairs):
    """`summary.tsv` plus one per-pair table, as `coregister.py` writes them."""
    directory.mkdir(parents=True, exist_ok=True)
    header = "space_a\tspace_b\tjaccard_mean\n"
    rows = "".join(f"{a}\t{b}\t0.5\n" for a, b, _ in pairs)
    (directory / "summary.tsv").write_text(header + rows)
    for a, b, per_protein in pairs:
        body = "protid\tneighborhood_jaccard\trank_correlation\n"
        body += "".join(f"{p}\t{v}\t0.1\n" for p, v in per_protein)
        (directory / f"{a}__vs__{b}.tsv").write_text(body)


def test_each_comparison_row_carries_its_per_protein_detail(tmp_path):
    """The defect: `summary.tsv` is aggregate, and nothing read the pair tables.

    The template loops over these rows with `if (!row.per_protein) continue`.
    No row had ever carried the key, so the map it built was empty and every
    protein coloured `null` -- a headline feature (ADR 0005 item 4) that
    toggled a button and conveyed nothing.
    """
    from explorer.payload import _read_comparisons

    directory = tmp_path / "coregistration"
    _write_coregistration(directory, [("a", "b", [("P1", 0.25), ("P2", 0.75)])])
    rows = _read_comparisons(str(directory / "summary.tsv"))
    assert len(rows) == 1
    assert rows[0]["per_protein"] == {"P1": 0.25, "P2": 0.75}


def test_a_protein_with_no_measurement_is_absent_rather_than_zero(tmp_path):
    """NaN must not become 0.0.

    The template averages each protein's Jaccard across the pairs it appears
    in. A missing measurement carried through as 0.0 would report *maximal*
    disagreement for a pair that was never measured, which is the
    substituted-zero defect this codebase already has once (FOLLOWUPS #34).
    """
    from explorer.payload import _read_comparisons

    directory = tmp_path / "coregistration"
    (directory).mkdir(parents=True)
    (directory / "summary.tsv").write_text("space_a\tspace_b\n a\tb\n".replace(" ", ""))
    (directory / "a__vs__b.tsv").write_text("protid\tneighborhood_jaccard\nP1\t0.4\nP2\t\n")
    rows = _read_comparisons(str(directory / "summary.tsv"))
    assert rows[0]["per_protein"] == {"P1": 0.4}
    assert "P2" not in rows[0]["per_protein"]


def test_a_missing_pair_table_is_empty_rather_than_an_error(tmp_path):
    from explorer.payload import _read_comparisons

    directory = tmp_path / "coregistration"
    directory.mkdir(parents=True)
    (directory / "summary.tsv").write_text("space_a\tspace_b\na\tb\n")
    assert _read_comparisons(str(directory / "summary.tsv"))[0]["per_protein"] == {}


def test_the_template_reads_the_key_the_payload_writes():
    """The cross-check that would have caught this without a browser.

    Both halves were individually reasonable: the payload wrote what
    `summary.tsv` contained, and the template read `per_protein`. Nothing
    compared the two, which is the manifest-versus-honored pattern across a
    language boundary.
    """
    from explorer.template import render

    html = render({"comparisons": [{"per_protein": {"P1": 0.5}}], "spaces": []}, "", "t")
    assert "row.per_protein" in html, "the template no longer reads per_protein"
    assert '"per_protein": {"P1": 0.5}' in html, "the payload no longer writes per_protein"


# ==========================================================================
# The panel registry. A payload naming a panel kind must reach a renderer,
# and a payload naming one the template lacks must SAY SO on the page.
# ==========================================================================


def test_panel_type_defaults_to_scatter_so_an_old_payload_is_unchanged():
    """The registry may not alter a page that never asked for it.

    Every space built before `panel_type` existed rendered as a scatter. The
    default keeps that true without the caller restating it, which is what
    makes the field additive rather than a migration.
    """
    from explorer.payload import SpacePayload

    space = SpacePayload(
        space_id="structure",
        protids=["a"],
        embeddings={"pca_umap": [[0.0, 0.0]]},
        clusters={"a": "0"},
        readable=[True],
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
    )
    assert space.panel_type == "scatter"
    assert space.to_dict()["panel_type"] == "scatter", (
        "the field must travel to the page; a default the template cannot see "
        "is not a default, it is a comment"
    )


def test_the_template_dispatches_on_panel_type_rather_than_assuming_scatter():
    """The registry has to be reachable, not merely present.

    Asserted on the source the page carries, because there is no browser here.
    `draw()` must look the kind up; a `draw()` that still hardcodes the scatter
    body would pass every payload test above and render one kind forever.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "PANELS.scatter" in html, "the scatter kind is not registered"
    assert re.search(
        r"space\.panel_type", html
    ), "`draw()` never reads `panel_type`, so the registry cannot be reached"
    assert re.search(r"panelFor\(space\)\.build", html) and re.search(
        r"panelFor\(space\)\.render", html
    ), "draw() must dispatch both halves through the registry, not just one"


def test_an_unknown_panel_type_is_drawn_as_a_mismatch_not_dropped():
    """A missing panel and a broken panel look identical to a reader.

    This is the same rule the source document states for refusals (7.03 E2):
    show them, never blank them. A template older than its payload is exactly
    when a space would vanish silently, so the fallback has to be visible.
    """
    from explorer.template import render

    html = render(
        {"spaces": [{"space_id": "s", "panel_type": "tanglegram"}]},
        plotly_js="",
        title="t",
    )
    payload = _embedded_payload(html)
    assert payload["spaces"][0]["panel_type"] == "tanglegram"
    assert "PANELS.__unknown__" in html, "there is no fallback renderer"
    assert re.search(
        r"no renderer for panel_type", html
    ), "the fallback must put the mismatch on the page, not only in a console"


# ==========================================================================
# The panel catalogue and the sheets. The point of the catalogue is that a
# panel nobody can draw still appears, saying what it is waiting for.
# ==========================================================================


def test_every_catalogue_panel_names_a_sheet_that_exists():
    """A panel filed under an unknown sheet would never be reachable.

    The tab bar is built from `sheets`, so a typo in `sheet` does not error --
    it silently removes the panel from the page, which is the failure this
    whole catalogue exists to prevent.
    """
    from explorer.panels import CATALOGUE, SHEETS

    known = {s for s, _ in SHEETS}
    for spec in CATALOGUE:
        assert spec.sheet in known, f"{spec.panel_id} is filed under unknown sheet {spec.sheet!r}"


def test_every_panel_that_needs_data_says_what_it_needs():
    """`requires` is what the page prints, not a comment.

    "Refusals are shown as refusals rather than as blanks" is the source's own
    rule (7.03 E2). A panel with `needs` and no `requires` would render an
    empty box, which is the thing being ruled out.
    """
    from explorer.panels import CATALOGUE

    for spec in CATALOGUE:
        if spec.needs:
            assert spec.requires, f"{spec.panel_id} can be unmet but names no requirement"
            assert spec.fills_in, f"{spec.panel_id} names no route to filling it in"


def test_every_panel_carries_a_provenance_tag_and_a_section():
    """The source tags every section real / mixed / synthetic / method.

    A reader must not have to guess whether a panel shows measurements or a
    drawing of an argument, and the section number is what makes any claim in
    the catalogue checkable against the document.
    """
    from explorer.panels import CATALOGUE, PROVENANCE

    for spec in CATALOGUE:
        assert spec.provenance in PROVENANCE, f"{spec.panel_id}: {spec.provenance!r}"
        assert spec.section, f"{spec.panel_id} cites no source section"


def test_catalogue_marks_panels_drawable_only_when_their_inputs_are_present():
    """The drawable decision is made once, in Python, and travels as a flag.

    Splitting it across two languages is how a panel ends up judged drawable
    here and blank in the browser.
    """
    from explorer.panels import catalogue_for

    with_nothing = {p["panel_id"]: p for p in catalogue_for(set())}
    assert with_nothing["comparisons"]["drawable"] is False
    assert with_nothing["comparisons"]["missing"] == ["comparisons"]
    # A panel needing nothing draws whatever the payload holds.
    assert with_nothing["perturbation_grid"]["drawable"] is True

    with_comparisons = {p["panel_id"]: p for p in catalogue_for({"comparisons"})}
    assert with_comparisons["comparisons"]["drawable"] is True
    assert with_comparisons["comparisons"]["missing"] == []


def test_six_panels_are_blocked_on_one_missing_input_not_six():
    """The catalogue should make the shape of the gap visible.

    Most of what cannot be drawn is blocked on a phylogeny. If that ever stops
    being true the plan in POST-PLAN is wrong, and this is where it shows.
    """
    from explorer.panels import CATALOGUE

    tree_blocked = [s.panel_id for s in CATALOGUE if "reconciled gene/species tree" in s.requires]
    assert (
        len(tree_blocked) >= 5
    ), f"expected the phylogeny to block most empty panels, blocks {tree_blocked}"


def test_the_page_builds_a_sheet_bar_and_can_render_an_awaiting_panel():
    """Both halves have to exist: the catalogue, and something that draws it.

    Asserted on the source the page carries, because there is no browser here
    -- which is exactly why this is the weakest test in the file and why the
    browser pass is not optional.
    """
    from explorer.panels import catalogue_for, sheet_titles
    from explorer.template import render

    html = render(
        {"spaces": [], "panels": catalogue_for(set()), "sheets": sheet_titles()},
        plotly_js="",
        title="t",
    )
    payload = _embedded_payload(html)
    assert payload["sheets"], "the page carries no sheets"
    assert any(not p["drawable"] for p in payload["panels"]), "no panel is awaiting anything"
    assert (
        "function renderSheets" in html and "function awaitingBlock" in html
    ), "the page carries the catalogue but nothing that draws it"


# ==========================================================================
# Several cohorts in one page. The rule that matters is that a page built
# without the option is byte-for-byte the page that existed before it.
# ==========================================================================


def test_a_single_cohort_page_carries_no_cohorts_key():
    """Backward compatibility, asserted rather than assumed.

    The Snakefile rule and the demo both call the single-cohort form. If that
    grew a `cohorts` key the payload would change for every existing run, and
    the template's single-cohort path would stop being the one that is exercised.
    """
    from explorer.template import render

    html = render({"spaces": [], "analysis_name": "one"}, plotly_js="", title="t")
    payload = _embedded_payload(html)
    assert "cohorts" not in payload, "a single-cohort page must not gain the key"
    assert "COHORTS" in html, "the template still needs its single-cohort fallback"


def test_the_template_falls_back_to_one_cohort_when_the_key_is_absent():
    """The fallback is what keeps the two code paths from drifting.

    Without it, every existing page would need rebuilding to render at all.
    """
    from explorer.template import render

    html = render({"spaces": [], "analysis_name": "one"}, plotly_js="", title="t")
    assert (
        "PAYLOAD.cohorts ||" in html
    ), "the template must treat a page with no `cohorts` key as a single cohort"


def test_a_multi_cohort_page_names_every_cohort():
    """The selector's labels come from the payload, so they have to be in it."""
    from explorer.template import render

    document = {
        "spaces": [],
        "analysis_name": "first",
        "cohorts": [
            {"spaces": [], "cohort_name": "chymotrypsin", "comparisons": [], "overlays": {}},
            {"spaces": [], "cohort_name": "actin", "comparisons": [], "overlays": {}},
        ],
    }
    html = render(document, plotly_js="", title="t")
    payload = _embedded_payload(html)
    names = [c["cohort_name"] for c in payload["cohorts"]]
    assert names == ["chymotrypsin", "actin"]
    assert "showCohort" in html, "the page carries cohorts but nothing that switches them"


def test_switching_cohorts_clears_the_cached_grid():
    """The grid caches its DOM behind `dataset.built`.

    A switch that rebuilt the payload but not the cache would draw the new
    cohort's points into the old cohort's divs, which renders as a map that is
    subtly wrong rather than one that is visibly broken -- the worst failure
    mode this page has.
    """
    from explorer.template import render

    html = render({"spaces": [], "analysis_name": "one"}, plotly_js="", title="t")
    assert re.search(
        r"delete\s+grid\.dataset\.built", html
    ), "applyCohort must drop the grid's build cache"


def test_parse_cohort_rejects_a_spec_it_cannot_split():
    """A malformed --also-cohort must fail loudly, not build half a page."""
    import pytest as _pytest
    from build_explorer import parse_cohort

    assert parse_cohort("actin=/tmp/c.json:/tmp/out") == ("actin", "/tmp/c.json", "/tmp/out")
    for bad in ("no-equals", "name=", "=/tmp/c.json:/tmp/out", "name=/tmp/c.json"):
        with _pytest.raises(SystemExit):
            parse_cohort(bad)


# ==========================================================================
# The cursor, and one colour key for the whole figure. Both came out of the
# adversarial browser pass, which is the only thing here that renders a DOM.
# ==========================================================================


def test_hover_fields_come_from_files_every_run_writes():
    """Species and Leiden cluster must not depend on the overlay table.

    Overlays are harvested from `final_results/*_aggregated_features.tsv`, which
    a run only has if `aggregate_features` ran. Neither N7 cohort has one, so
    sourcing the cursor from overlays would leave it showing an accession and
    nothing else -- which is what it did.
    """
    import textwrap

    import pytest as _pytest

    _pytest.importorskip("pandas")
    import tempfile

    from explorer.payload import _hover_fields

    with tempfile.TemporaryDirectory() as out:
        os.makedirs(os.path.join(out, "protein_features"))
        os.makedirs(os.path.join(out, "foldseek_clustering_results"))
        with open(os.path.join(out, "protein_features", "uniprot_features.tsv"), "w") as fh:
            fh.write(
                textwrap.dedent(
                    """\
                protid\tOrganism\tLength
                P1\tHomo sapiens\t100
                P2\tMus musculus\t120
                """
                )
            )
        with open(
            os.path.join(out, "foldseek_clustering_results", "leiden_features.tsv"), "w"
        ) as fh:
            fh.write("protid\tLeidenCluster\nP1\tLC3\nP2\tLC0\n")
        got = _hover_fields(out, {"P1", "P2"})

    assert got["P1"]["species"] == "Homo sapiens"
    assert got["P1"]["leiden"] == "LC3"
    assert got["P2"]["leiden"] == "LC0"


def test_hover_fields_survive_both_files_being_absent():
    """A run without these files must still build a page.

    Optional inputs stay optional -- the cursor loses two lines, not the page.
    """
    import tempfile

    import pytest as _pytest

    _pytest.importorskip("pandas")
    from explorer.payload import _hover_fields

    with tempfile.TemporaryDirectory() as out:
        assert _hover_fields(out, {"P1"}) == {}


def test_the_colour_domain_is_shared_by_every_panel():
    """The defect this replaced gave one colour two meanings side by side.

    Levels were computed per space, so in a space where every protein was
    readable the single level took palette slot 0 -- the slot "do not read"
    occupied in the space next to it. Asserted on the source, because the bug
    lived in how the domain was scoped, not in any value the payload carries.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function rebuildColourDomain" in html, "there is no shared domain"
    assert (
        "colourDomain.levels.indexOf" in html
    ), "traceFor must index the shared domain, not a per-space level list"
    assert (
        "marker.cmin" in html and "marker.cmax" in html
    ), "a continuous scale must be pinned across panels or Plotly rescales each"
    assert "function renderColourKey" in html, "colour with no key is the other half of this bug"


def test_numbers_in_the_cursor_are_formatted_by_magnitude():
    """`toFixed(3)` rendered a chain length as 367.000."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function fmtValue" in html
    # Assert the CALL SITE, not the absence of a substring. The first version of
    # this test searched for "toFixed(3)" and matched the comment that explains
    # the bug -- it failed on its own documentation.
    assert (
        "fmtValue(colour.values[i])" in html
    ), "the tooltip does not route its overlay value through fmtValue"
    assert "fmtValue(colourDomain.min)" in html, "the colour key does not format its bounds"


def test_the_title_follows_the_cohort_selector():
    """A header naming one cohort above another cohort's numbers is a lie.

    `__TITLE__` is a static substitution, so before this the h1 kept the first
    cohort's name after a switch while the subtitle updated -- header and
    subtitle disagreeing about which data was on screen.
    """
    from explorer.template import render

    html = render({"spaces": [], "analysis_name": "one"}, plotly_js="", title="t")
    assert (
        'querySelector("h1").textContent = active.cohort_name' in html
    ), "applyCohort does not update the title"
