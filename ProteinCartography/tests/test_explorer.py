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


def _matrix_fixture_dir():
    """A scratch directory for the matrix fixtures below.

    A plain `tmp_path` fixture cannot be reached from the helper that builds the
    config, and threading it through four tests obscured what each one is about.
    """
    import pathlib
    import tempfile

    return pathlib.Path(tempfile.mkdtemp(prefix="pc-matrix-"))


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


#: Documents that exist only on the machine this branch was built on. A reader
#: holding the PR has none of them, so a panel citing one renders, on the page,
#: an instruction the reader cannot act on. Kept as a literal tuple rather than
#: a regex because FOLLOWUPS #33's original check WAS a regex and could not
#: match `POST-PLAN` or a bare `PLAN Phase 9` -- it read as a passing check for
#: forty-two commits while eleven such strings were being rendered.
NON_SHIPPING_DOCUMENTS = (
    "POST-PLAN",
    "GEOMETRY_DIGEST",
    "PLAN Phase",
    "PLAN.md",
    "REVIEW_LOG",
    "EXPLORATION.md",
    "PR_NARRATIVE",
    "CLAUDE.md",
    "protein-map-geometry-options",
)


def _document_offences(specs):
    """Every (panel, field, document) where page text cites something local-only."""
    return [
        (spec.panel_id, field, doc)
        for spec in specs
        for field, value in (("requires", spec.requires), ("fills_in", spec.fills_in))
        for doc in NON_SHIPPING_DOCUMENTS
        if doc in value
    ]


def test_no_panel_tells_the_reader_to_consult_a_document_the_pr_does_not_ship():
    """`requires` and `fills_in` are both page text, not comments.

    `awaitingBlock` renders "filled in by: " + fills_in into the DOM, so a value
    like "PLAN Phase 9 item 3" reaches a reader who has no PLAN.md and can only
    be read as a dead end. This is the guard FOLLOWUPS #33 claimed to have and
    did not: its evidence sentence asserted zero surviving references while
    eleven were on the built page.
    """
    from explorer.panels import CATALOGUE

    offences = _document_offences(CATALOGUE)
    assert not offences, f"panels cite documents the PR does not ship: {offences}"


def test_the_guard_above_would_actually_catch_one():
    """A check that has never been seen to fail is not a check.

    This branch has already shipped a gated test that was correct and never
    executed, and a formatter stage no per-commit run invoked (FOLLOWUPS #64).
    So the detector -- the same function, not a re-typed copy of its condition
    -- is run against a panel known to be bad.
    """
    from explorer.panels import PanelSpec

    planted = PanelSpec(
        panel_id="planted",
        title="planted",
        sheet="stability",
        panel_type="stability",
        provenance="real",
        section="0.00",
        needs=("nothing",),
        requires="a thing",
        fills_in="PLAN Phase 9 item 3",
    )
    assert _document_offences([planted]) == [("planted", "fills_in", "PLAN Phase")]


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

    Most of what cannot be drawn is blocked on a phylogeny, and the catalogue
    says so by citing one shared input rather than six different ones. If that
    ever stops being true, this is where it shows.
    """
    from explorer.panels import CATALOGUE

    tree_blocked = {s.panel_id for s in CATALOGUE if "reconciled gene/species tree" in s.requires}
    # An exact set, not `len >= 5`. A threshold cannot notice a panel quietly
    # joining or leaving the group, which is the only thing this test exists to
    # see. The substring stays the selector so a reword of `_NO_TREE` still
    # reddens this rather than silently emptying the set.
    assert tree_blocked == {
        "divergence",
        "innovation_clades",
        "innovation_map",
        "ancestral_path",
        "tanglegram",
        "phylomorphospace",
    }, f"the tree-blocked set has moved: {sorted(tree_blocked)}"


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


# ==========================================================================
# The fold-outs. These strings ship to biologists, so the tests here are
# about honesty rather than shape: a description that states a cohort's
# number as a constant is wrong on the other cohort and looks like a
# measurement while it does it.
# ==========================================================================


def test_a_provider_with_no_description_says_so_rather_than_inventing_one():
    """The rule the whole module exists for.

    A confident sentence about a provider nobody described is worse than no
    sentence: a reader can act on "the code does not say" and cannot act on
    someone's inference dressed as documentation.
    """
    from explorer.descriptions import NOT_DETERMINABLE, describe_block

    text = " ".join(describe_block("some_future_provider")["paragraphs"])
    assert NOT_DETERMINABLE in text
    assert "some_future_provider" in text


def test_the_biophys_hazard_names_the_unapplied_normalization():
    """FOLLOWUPS #32, on the page rather than in a follow-up file.

    The block asks for `zscore_within`, nothing reads `spec.normalization`, and
    the consequence is that the euclidean distance is mostly isoelectric point.
    A physicochemistry map read as "chemistry" rather than "pI" is the
    misreading this sentence exists to stop.
    """
    from explorer.descriptions import describe_block

    hazards = " ".join(describe_block("biophys")["hazards"])
    assert "zscore_within" in hazards
    assert "isoelectric point" in hazards


def test_the_domains_hazards_name_both_the_blank_row_and_the_ties():
    from explorer.descriptions import describe_block

    hazards = " ".join(describe_block("domains")["hazards"])
    assert "nobody has annotated one" in hazards
    assert "PF00022" in hazards, "the tie degeneracy is why this space is unreadable"


def test_the_tmscore_description_says_the_row_is_the_feature_vector():
    """2.02's point, and the one most often skipped: distance in this space is
    between similarity PROFILES, not between two proteins' scores."""
    from explorer.descriptions import describe_block

    described = describe_block("tmscore", params={"representation": "profile"})
    text = " ".join(described["paragraphs"])
    assert "whole row" in text
    assert "second-order" in text
    assert "#60" in " ".join(described["hazards"]), "the 3Di+AA provenance is not stated"


def test_a_cohort_dependent_number_is_read_from_the_cohort_not_typed_in():
    """The load-bearing test of this module.

    The 3Di vocabulary is 4982 columns wide for one cohort on the shipped page
    and 4594 for the other. A constant here would be a measurement-shaped
    sentence that is wrong on one of the two panels it renders into, so the
    number has to come from that cohort's own block manifest.
    """
    from explorer.descriptions import describe_block

    wide = " ".join(describe_block("threedi", facts={"n_kmers": 4982})["paragraphs"])
    narrow = " ".join(describe_block("threedi", facts={"n_kmers": 4594})["paragraphs"])
    assert "4982 columns for this cohort" in wide
    assert "4594 columns for this cohort" in narrow


def test_the_fused_description_quotes_this_cohort_s_own_realized_shares():
    """Same rule, for the number this page is most likely to be quoted on.

    An equal request realizes 0.4405/0.5595 on the chymotrypsin cohort and
    0.4208/0.5792 on the actin one, from the same config. Either typed in would
    be wrong on the other panel.
    """
    from explorer.descriptions import describe_late

    text = " ".join(
        describe_late(
            [
                {"block_id": "tmscore", "share": 0.5, "realized_share": 0.4404504},
                {"block_id": "biophys", "share": 0.5, "realized_share": 0.5595495},
            ]
        )["paragraphs"]
    )
    assert "tmscore 44.0 %" in text and "biophys 56.0 %" in text
    assert "requested 50 %" in text


def test_a_fused_space_with_no_measured_shares_states_none():
    """Absent is absent. Falling back to the nominal weights would print a
    request as if it were a measurement, which is the thing #29/#32 are about."""
    from explorer.descriptions import describe_late

    text = " ".join(describe_late([])["paragraphs"])
    assert "realized shares are" not in text


def test_every_space_description_ends_with_what_the_axes_are_not():
    """The single most common misreading of a UMAP costs one sentence to
    forestall, so it is appended to every space rather than left to the reader
    to remember from another panel."""
    from explorer.descriptions import describe_space

    text = " ".join(describe_space("structure", blocks=[{"provider": "tmscore"}])["paragraphs"])
    assert "no units" in text
    assert "not interpretable" in text


def test_the_payload_carries_a_description_for_every_space(built):
    """A description computed and not shipped is a comment (#29/#32)."""
    for space in built.spaces:
        assert space.to_dict()["description"]["paragraphs"], space.space_id


def test_a_space_payload_written_without_a_description_still_renders():
    """Additive, asserted rather than assumed: the field defaults to empty and
    the template draws nothing for an empty one."""
    from explorer.payload import SpacePayload

    space = SpacePayload(
        space_id="structure",
        protids=["a"],
        embeddings={"pca_umap": [[0.0, 0.0]]},
        clusters={},
        readable=[True],
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
    )
    assert space.to_dict()["description"] == {}


def test_the_description_is_built_from_the_block_manifest_on_disk(tmp_path):
    """The fold-out has to read the cohort, not the config.

    `n_kmers` exists only in the block manifest -- the config says `k: 3` and
    nothing about how wide that made the matrix. If the payload stopped reading
    the manifest the sentence would quietly become generic, which no shape
    assertion would notice.
    """
    from config_schema import from_legacy
    from explorer.payload import _space_blocks

    blocks_dir = tmp_path / "blocks" / "td"
    blocks_dir.mkdir(parents=True)
    (blocks_dir / "manifest.json").write_text(json.dumps({"extra": {"n_kmers": 77}}))
    config = from_legacy(
        {
            "blocks": {"td": {"provider": "threedi", "k": 3}},
            "spaces": {"s": {"blocks": ["td"], "strategy": "none", "reducers": ["pca"]}},
        }
    )
    found = _space_blocks(str(tmp_path), config, config.spaces["s"])
    assert found == [
        {
            "block_id": "td",
            "provider": "threedi",
            "params": {"k": 3, "metric": "euclidean"},
            "facts": {"n_kmers": 77},
        }
    ]


def test_the_template_renders_the_fold_out_in_both_places():
    """One renderer for a space panel and for a catalogue card.

    Two renderers is how a hazard ends up styled as body text in one of them,
    and the hazards are the half of this that matters.
    """
    from explorer.panels import catalogue_for, sheet_titles
    from explorer.template import render

    html = render(
        {"spaces": [], "panels": catalogue_for(set()), "sheets": sheet_titles()},
        plotly_js="",
        title="t",
    )
    assert "function explainBlock" in html
    assert "explainBlock(space.description)" in html, "the maps grid has no fold-out"
    assert "explainBlock(panel.description)" in html, "the panel cards have no fold-out"
    payload = _embedded_payload(html)
    described = [p for p in payload["panels"] if p["description"]["paragraphs"]]
    assert len(described) > 15, "most catalogue panels carry no description"


def test_the_fold_out_escapes_before_it_marks_up():
    """The description is plain text with backticks, and it is inserted as
    innerHTML. Escaping after the markup pass would let a `<` in a description
    become live markup."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    marker = html.index("function inlineMarkup")
    body = html[marker : marker + 400]
    assert body.index("escapeHtml(text)") < body.index("replace(/`([^`]+)`/g")


# ==========================================================================
# The records panel. Its payload key shipped with no test, which is the
# failure #29 and #32 are about: a value written into the payload is not
# thereby read by anything. These assert that it is.
# ==========================================================================


def test_the_records_key_is_restricted_to_the_proteins_on_the_maps(tmp_path):
    """A row for a protein no panel plots is a row the reader cannot find.

    It would also make the table's count disagree with the footer's, and the
    two counts being the same number is what tells a reader the table is of
    this cohort rather than of the feature file.
    """
    from explorer.payload import _records

    features = tmp_path / "protein_features"
    features.mkdir()
    (features / "uniprot_features.tsv").write_text(
        "protid\tProtein names\tOrganism\tLength\n"
        "P1\tactin\tHomo sapiens\t375\n"
        "P2\tprofilin\tMus musculus\t140\n"
        "NOT_PLOTTED\tsomething else\tEscherichia coli\t99\n"
    )
    rows = _records(str(tmp_path), ["P1", "P2"])
    assert [row["accession"] for row in rows] == ["P1", "P2"]
    assert rows[0] == {
        "accession": "P1",
        "protein": "actin",
        "organism": "Homo sapiens",
        "length": "375",
    }


def test_a_run_with_no_uniprot_feature_table_advertises_no_records_panel(tmp_path):
    """`available` must not claim `records` when the file is absent.

    The catalogue marks the panel drawable from that set, so a run without the
    table has to leave the panel in its awaiting state naming the file -- which
    is a question someone can answer -- rather than draw an empty table, which
    reads as a cohort with no proteins in it.
    """
    from explorer.panels import catalogue_for
    from explorer.payload import _records

    assert _records(str(tmp_path), ["P1"]) == []
    entry = next(p for p in catalogue_for(set()) if p["panel_id"] == "records")
    assert entry["drawable"] is False
    assert entry["missing"] == ["records"]
    assert "uniprot_features.tsv" in entry["requires"]


def test_the_template_reads_the_records_key_it_is_sent():
    """The registry entry has to exist and has to reach `active.records`.

    Asserted on the page's own source: there is no browser here, and a
    `records` key nothing reads is exactly the shape of #29 and #32.
    """
    from explorer.template import render

    html = render({"spaces": [], "records": []}, plotly_js="", title="t")
    assert "SHEET_PANELS.records" in html, "the records kind is not registered"
    assert "active.records" in html, "nothing on the page reads the records key"
    assert re.search(
        r"of \$\{rows\.length\} protein\(s\)", html
    ), "the count must be of the filtered rows against the total"


# ==========================================================================
# The diagnostics report. Its section ORDER is the content -- 7.03 E2 puts
# the map last on purpose -- so the order is asserted, not just the sections.
# ==========================================================================


def test_the_report_sections_are_in_the_source_s_fixed_order():
    """Cohort, coverage, geometry, rate fit, and only THEN the map.

    Written out rather than derived so that reordering the tuple has to
    reorder this list too. The map being fifth is the argument the section
    exists to make: a map with no diagnostics beside it is the artifact the
    source document is against, and a report that opened with the map would
    reproduce it while looking complete.
    """
    from explorer.panels import REPORT_SECTIONS

    assert [s["section_id"] for s in REPORT_SECTIONS] == [
        "cohort",
        "coverage",
        "geometry",
        "rate",
        "map",
    ]
    assert REPORT_SECTIONS[-1]["section_id"] == "map", "the map must be last"


def test_the_report_panel_carries_its_sections_into_the_payload():
    """`content` was an unused field until this panel. A section list that
    does not travel is a section list the page cannot render, which is the
    same failure as #29: a value written where nothing reads it."""
    from explorer.panels import REPORT_SECTIONS, catalogue_for

    entry = next(p for p in catalogue_for(set()) if p["panel_id"] == "report")
    assert entry["drawable"] is True, "the report needs no input it does not have"
    assert [s["section_id"] for s in entry["content"]["sections"]] == [
        s["section_id"] for s in REPORT_SECTIONS
    ]


def test_every_refused_report_section_names_what_would_fill_it():
    """A refusal a reader cannot act on is a blank with extra words.

    ONE of the five sections is a refusal: this pipeline builds no phylogeny.
    `coverage` used to be the second and should never have been -- its refusal
    said "no payload key carries it" while `active.censoring.summary` carried
    it, and the censoring panel drew the number two sheets away.
    """
    from explorer.panels import REPORT_SECTIONS

    refused = [s for s in REPORT_SECTIONS if s.get("refused")]
    assert {s["section_id"] for s in refused} == {"rate"}
    for section in refused:
        assert section[
            "fills_in"
        ], f"{section['section_id']} refuses without saying what would fill it"
        assert len(section["refused"]) > 80, "a one-line refusal names no input"


def test_no_report_section_refuses_on_a_key_the_payload_actually_supplies():
    """The `coverage` defect as an invariant rather than as one fixed instance.

    A refusal is a claim about what this run does not have. When the payload
    does have it, the claim is false ON THE PAGE, and no test that pins a set
    of section ids would notice -- the set stays the same shape while the
    sentence inside it rots. So this asserts the relationship instead: nothing
    may refuse while naming a payload key that `build_payload` advertises.
    """
    from explorer.panels import REPORT_SECTIONS

    # section_id -> the payload key that would fill it. Kept here rather than on
    # the section so the mapping is a test's opinion about the pipeline, not
    # something the shipped catalogue asserts about itself.
    FILLED_BY = {"coverage": "censoring", "geometry": "diagnostics", "map": "spaces"}
    supplied = {"censoring", "diagnostics", "spaces", "matrix", "records"}
    for section in REPORT_SECTIONS:
        key = FILLED_BY.get(section["section_id"])
        if key and key in supplied:
            assert not section.get("refused"), (
                f"{section['section_id']} refuses, but build_payload supplies "
                f"{key!r} -- the refusal is false on any page that has it"
            )


def test_the_template_renders_the_report_in_payload_order_not_by_lookup():
    """The order has to be data, not the template's opinion.

    `forEach` over the payload's own list is what makes the Python-side order
    the rendered order. A template that indexed `REPORT_FILLERS` and rendered
    its keys would be free to put the map first, and no payload test would
    notice.
    """
    from explorer.panels import catalogue_for, sheet_titles
    from explorer.template import render

    html = render(
        {"spaces": [], "panels": catalogue_for(set()), "sheets": sheet_titles()},
        plotly_js="",
        title="t",
    )
    assert "SHEET_PANELS.report" in html, "the report kind is not registered"
    assert re.search(
        r"sections\.forEach", html
    ), "the template does not walk the payload's section list"
    body = html[html.index("SHEET_PANELS.report") :]
    assert (
        "Object.keys(REPORT_FILLERS)" not in body
    ), "rendering the filler keys would make the template's order authoritative"


def test_every_fillable_report_section_has_a_filler_and_every_filler_a_section():
    """A section with no filler renders as a version mismatch, which is right
    for an old page and wrong for this one. Both directions are checked: an
    orphan filler is dead code that looks like coverage."""
    from explorer.panels import REPORT_SECTIONS
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    fillable = [s["section_id"] for s in REPORT_SECTIONS if not s.get("refused")]
    for section_id in fillable:
        assert f"REPORT_FILLERS.{section_id} =" in html, f"no filler for {section_id}"
    declared = set(re.findall(r"REPORT_FILLERS\.(\w+) =", html))
    assert declared == set(fillable), f"fillers and sections disagree: {declared}"


def test_a_report_section_the_template_cannot_fill_says_so():
    """Same rule as the unknown panel_type, and for the same reason: a
    section that silently vanishes cannot be told apart from one nobody
    wrote."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "reportUnknownSection" in html
    assert "no filler for report section" in html


def test_the_report_never_prints_a_bare_blank_for_a_number_it_lacks():
    """A blank cell reads as a measurement of nothing and a zero reads as a
    measurement of zero. The comparisons table already learned this; the
    report has more places to forget it -- a space with no negative control,
    a one-block space with no redundancy pair, an uninformative stability."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function reportMissing" in html
    assert "not in this payload" in html
    body = html[html.index("const GEOMETRY_COLUMNS") : html.index("REPORT_FILLERS.cohort")]
    for absent in ("negative control", "one block", "not informative"):
        assert absent in body, f"the geometry row does not say why {absent!r} is empty"


# ==========================================================================
# The signal inventory, and the generic table renderer it shares with the
# discordance panel. The inventory's whole value is that it is READ from the
# guard rather than retyped, so that is what is tested.
# ==========================================================================


def test_the_signal_inventory_is_read_from_the_guard_that_enforces_it():
    """A retyped list is free to drift from what the pipeline refuses.

    A panel showing eight signals while the guard enforced nine would be worse
    than no panel: it would look like a check. So every signal in
    `NOT_FUSABLE_PROVIDERS` has to appear, and nothing else may.
    """
    from config_schema import NOT_FUSABLE_PROVIDERS
    from explorer.panels import CATALOGUE

    entry = next(p for p in CATALOGUE if p.panel_id == "signal_inventory")
    shown = {row["signal"] for row in entry.content["rows"]}
    assert shown == set(NOT_FUSABLE_PROVIDERS.values())


def test_the_inventory_groups_by_signal_so_four_providers_of_one_circularity_are_one_row():
    """The guard is keyed on the provider deliberately -- `patristic`,
    `iqtree`, `noveltree` and `phylogeny` are four ways to produce one
    circularity. Four rows repeating one sentence would hide that they are the
    same argument, so the row is per signal and names its providers."""
    from explorer.panels import CATALOGUE

    entry = next(p for p in CATALOGUE if p.panel_id == "signal_inventory")
    row = next(r for r in entry.content["rows"] if r["signal"] == "phylogeny")
    for provider in ("iqtree", "noveltree", "patristic", "phylogeny"):
        assert provider in row["providers"]


def test_every_signal_the_guard_names_carries_its_reason():
    """A signal with no recorded reason is a hole in the argument, and the
    panel prints `not determinable from the code` for it rather than an empty
    cell. This asserts there are none today, so adding one to the guard
    without its reason fails here rather than shipping a blank cell."""
    from explorer.descriptions import NOT_DETERMINABLE
    from explorer.panels import CATALOGUE

    entry = next(p for p in CATALOGUE if p.panel_id == "signal_inventory")
    holes = [r["signal"] for r in entry.content["rows"] if NOT_DETERMINABLE in r["reason"]]
    assert holes == [], f"these signals are enforced with no reason recorded: {holes}"


def test_one_table_renderer_serves_every_table_panel():
    """Two catalogue panels declare `panel_type: "table"`. A renderer that
    knew one panel's column names would need a branch per panel after it, and
    each branch is a place for the page to disagree with its payload -- so the
    columns come from the payload."""
    from explorer.panels import CATALOGUE
    from explorer.template import render

    tables = [p.panel_id for p in CATALOGUE if p.panel_type == "table"]
    assert len(tables) > 1, "the shared renderer is only interesting if it is shared"
    html = render({"spaces": []}, plotly_js="", title="t")
    assert "SHEET_PANELS.table" in html, "the table kind is not registered"
    assert "content.columns" in html, "the renderer does not read the payload's columns"
    table_body = html[html.index("SHEET_PANELS.table") :][:1500]
    assert "signal_inventory" not in table_body, "the shared renderer branches on a specific panel"


def test_a_table_panel_with_no_rows_says_which_half_is_missing():
    """Declared as a table and handed no table. The two halves are different
    facts -- no columns is a template-side mistake and no rows is an empty
    source -- and collapsing them hides which."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("SHEET_PANELS.table") :][:1500]
    assert "the payload declares the panel and does not carry its content" in body
    assert 'columns.length ? "rows" : "columns"' in body


def test_the_tab_count_asks_the_renderer_whether_a_panel_is_blank():
    """A shared renderer broke this count the day it was added.

    The bar counted a panel as filled as soon as *a* renderer existed for its
    kind, so the discordance panel -- a `table` with no rows yet -- stopped
    being counted as empty while it still drew a refusal and nothing else. The
    two states have to stay distinguishable, so a renderer may answer for
    itself.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function panelIsBlank" in html
    assert "renderer.isEmpty" in html, "the count cannot ask the renderer"
    assert re.search(
        r"sheetPanels\.filter\(panelIsBlank\)", html
    ), "renderSheets still decides blankness itself"


# ==========================================================================
# The six option plates. The prose is the source's; the only thing this page
# computes is which flavour the run in front of the reader actually built.
# ==========================================================================


def test_all_six_flavours_are_present_and_in_the_source_s_order():
    """Ordered by how much each disturbs the existing geometry, which is the
    source's own ordering and the reason the list is worth showing at all.
    Dropping one would leave a reader thinking the taxonomy has five."""
    from explorer.panels import FLAVOUR_PLATES

    assert [p["plate_id"] for p in FLAVOUR_PLATES] == ["A", "B", "C", "D", "E", "F"]
    for plate in FLAVOUR_PLATES:
        for field in ("name", "formula", "answers", "cannot", "cost", "failure"):
            assert plate[field], f"plate {plate['plate_id']} has no {field}"


def test_a_plate_is_either_realized_by_a_payload_key_or_named_unimplemented():
    """Never both and never neither.

    A flavour with no `realized_by` and no `unimplemented` would silently take
    the "available, not used" branch and tell a reader this pipeline could
    build something it cannot.
    """
    from explorer.panels import FLAVOUR_PLATES

    for plate in FLAVOUR_PLATES:
        realized = plate.get("realized_by")
        unimplemented = plate.get("unimplemented")
        assert bool(realized) != bool(
            unimplemented
        ), f"plate {plate['plate_id']} claims both or neither"


def test_the_three_unimplemented_flavours_have_no_provider_to_build_them():
    """The claim is checkable, so it is checked.

    B, D and E need a protein-language-model, a function-call and a
    developability block, and this pipeline registers four providers, none of
    them those. The day one is added, this fails here rather than leaving a
    stale "not implemented" on the page.
    """
    from compute_block import _register_builtins
    from explorer.panels import FLAVOUR_PLATES
    from spaces.registry import BLOCK_GROUP, available_providers

    _register_builtins()
    registered = {info.name for info in available_providers(BLOCK_GROUP)}
    # A subset assertion, not equality: the registry is process-global and other
    # tests in this suite register probe providers into it. Equality here failed
    # only when the whole file ran, which is the worst kind of test.
    assert {"tmscore", "threedi", "biophys", "domains"} <= registered
    for absent in ("esm", "plm", "clean", "deepfri", "deeploc", "developab", "mhc"):
        assert not any(
            absent in name for name in registered
        ), f"a {absent!r} provider is registered, so a plate claiming otherwise is stale"
    unimplemented = {p["plate_id"] for p in FLAVOUR_PLATES if p.get("unimplemented")}
    assert unimplemented == {"B", "D", "E"}


def test_a_plate_reports_what_this_run_did_rather_than_a_general_claim():
    """The status is per run, in three states that stay distinct.

    "this pipeline offers C as a named fusion" is a sentence about the
    pipeline; "this run built it" is a fact about the cohort the reader is
    looking at, and only the second one can be wrong in a way they would
    notice.
    """
    from explorer.panels import PLATE_STATES, catalogue_for

    def plate(available, plate_id):
        entry = next(p for p in catalogue_for(available) if p["panel_id"] == "flavours")
        return next(x for x in entry["content"]["plates"] if x["plate_id"] == plate_id)

    assert plate({"overlays", "comparisons", "fused_spaces"}, "A")["state"] == "built by this run"
    assert plate(set(), "A")["state"] == "available, not used by this run"
    assert plate({"comparisons"}, "F")["state"] == "built by this run"
    assert plate({"comparisons"}, "C")["state"] == "available, not used by this run"
    assert plate(set(), "B")["state"] == "not implemented"
    for available in (set(), {"overlays", "comparisons", "fused_spaces"}):
        for spec in catalogue_for(available):
            for entry in (spec["content"] or {}).get("plates", []):
                assert entry["state"] in PLATE_STATES
                assert entry["state_note"], "a state with no note cannot be acted on"


def test_the_overlay_key_reaches_the_available_set():
    """`overlays` is new in `available` and the flavours panel is what reads
    it. A key nothing reads is #29 again."""
    from explorer.panels import catalogue_for

    without = next(p for p in catalogue_for(set()) if p["panel_id"] == "flavours")
    with_it = next(p for p in catalogue_for({"overlays"}) if p["panel_id"] == "flavours")
    states = [
        next(x["state"] for x in entry["content"]["plates"] if x["plate_id"] == "A")
        for entry in (without, with_it)
    ]
    assert states == ["available, not used by this run", "built by this run"]


def test_the_c_plate_carries_the_hazard_its_own_failure_mode_describes():
    """The source's failure mode for C is "charge in raw units swamps GRAVY in
    raw units. Standardize within the block first" -- and in this pipeline the
    biophysical block declares `zscore_within` while nothing reads the field
    (FOLLOWUPS #32). A plate quoting the warning next to a geometry that does
    not implement it would be worse than not quoting it."""
    from explorer.panels import FLAVOUR_PLATES

    plate = next(p for p in FLAVOUR_PLATES if p["plate_id"] == "C")
    assert "zscore_within" in plate["hazard"]
    assert "read by nothing" in plate["hazard"]
    others = [p["plate_id"] for p in FLAVOUR_PLATES if p.get("hazard")]
    assert others == ["C"], "only the plate this pipeline actually builds carries one"


def test_the_cards_renderer_reads_the_plates_and_answers_the_blank_count():
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "SHEET_PANELS.cards" in html, "the cards kind is not registered"
    body = html[html.index("SHEET_PANELS.cards") :][:2600]
    assert "content.plates" in body
    assert "isEmpty(panel)" in body, "a plateless cards panel would count as filled"
    assert "plate.hazard" in body, "the per-pipeline hazard is not rendered"


def test_all_seven_discordance_patterns_carry_a_test_that_separates_them():
    """The test column is the panel.

    Two patterns -- sequence saturation and structural noise -- resolve in
    OPPOSITE directions on which tree to trust, and both present as "the two
    trees disagree". A cause listed without the test that distinguishes it
    lets a reader pick whichever conclusion they were hoping for, which is
    worse than an empty panel.
    """
    from explorer.panels import DISCORDANCE_PATTERNS

    assert len(DISCORDANCE_PATTERNS) == 7
    for seen, cause, test, effect in DISCORDANCE_PATTERNS:
        assert seen and cause and test and effect
        assert len(test) > 40, f"{cause}: the test is too short to separate anything"
    opposed = [row for row in DISCORDANCE_PATTERNS if "Trust the" in row[3]]
    assert len(opposed) == 2, "the two opposite resolutions are the reason for the test column"
    assert {row[3].split(".")[0] for row in opposed} == {
        "Trust the structure tree here",
        "Trust the gene tree here",
    }


def test_the_discordance_panel_reuses_the_shared_table_renderer():
    """No new renderer, which was the point of building the table kind
    generically: the second table panel is content only."""
    from explorer.panels import CATALOGUE

    entry = next(p for p in CATALOGUE if p.panel_id == "discordance")
    assert entry.panel_type == "table"
    assert [c["key"] for c in entry.content["columns"]] == ["seen", "cause", "test", "effect"]
    assert len(entry.content["rows"]) == 7
    assert (
        "gene tree" in entry.content["note"]
    ), "the panel must say this pipeline cannot detect any of these"


# ==========================================================================
# The perturbation grid, whose empty cells are its content.
# ==========================================================================


def test_the_grid_is_complete_and_every_cell_carries_one_of_four_kinds():
    """Five perturbations against five observables is twenty-five cells, and a
    grid with holes in it would be indistinguishable from a grid whose holes
    are the finding."""
    from explorer.panels import CELL_KINDS, OBSERVABLES, PERTURBATIONS, _perturbation_grid

    content = _perturbation_grid()
    assert len(content["perturbations"]) == len(PERTURBATIONS) == 5
    assert len(content["observables"]) == len(OBSERVABLES) == 5
    assert len(content["cells"]) == 25
    for key, cell in content["cells"].items():
        assert cell["kind"] in CELL_KINDS, key
        assert cell["note"], key


def test_the_cell_people_run_first_is_marked_flat_and_says_what_it_costs():
    """The source's whole point about this grid. One alanine against TM-score
    is the emptiest cell and the one most people pay for first, so the panel
    has to carry both halves: that it is flat, and that it is 126 folds."""
    from explorer.panels import _perturbation_grid

    content = _perturbation_grid()
    cell = content["cells"]["ala1|tm"]
    assert cell["kind"] == "empty by construction"
    assert "flat line" in cell["note"]
    single = next(p for p in content["perturbations"] if p["key"] == "ala1")
    assert single["cost"] == "126 folds"


def test_an_unannotated_cell_is_not_reported_as_an_empty_one():
    """The source not commenting on a combination is not the source calling it
    useless. Sixteen of the twenty-five are unannotated, and inventing the
    difference would be the exact failure this panel exists to point at."""
    from explorer.panels import GRID_CELLS, _perturbation_grid

    cells = _perturbation_grid()["cells"]
    unannotated = [key for key, cell in cells.items() if cell["kind"] == "unannotated"]
    assert len(unannotated) == 25 - len(GRID_CELLS) == 16
    assert cells["polyA|flip"]["note"] == "The source does not annotate this combination."


def test_a_control_cell_is_not_classified_as_a_measurement():
    """A shuffled control tells you the floor of the score. Classified as
    informative it would be quoted as a result, which is how a noise floor
    ends up in a figure legend as a finding."""
    from explorer.panels import _perturbation_grid

    cells = _perturbation_grid()["cells"]
    assert cells["shuffle|tm"]["kind"] == "control"
    assert cells["shuffle|esm"]["kind"] == "control"
    informative = [key for key, cell in cells.items() if cell["kind"] == "informative"]
    assert not any(key.startswith("shuffle") for key in informative)


def test_the_grid_renderer_draws_no_numbers():
    """No scan has been run. A number in a cell here would be read as a
    measurement of this cohort, which is the same misreading the synthetic
    provenance tag exists to prevent -- so the cells carry kinds and the notes
    carry the argument."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "SHEET_PANELS.grid" in html, "the grid kind is not registered"
    body = html[html.index("SHEET_PANELS.grid") :][:3000]
    assert "cell.kind" in body, "the cells do not render their kind"
    assert "fmtValue" not in body, "the grid formats a number, and it has none to format"
    assert "gridnotes" in body, "the source's notes are only in tooltips"


# ==========================================================================
# The pipeline diagram. Derived from the resolved config, so it is this
# cohort's pipeline and not a drawing of the general case.
# ==========================================================================


def test_the_diagram_marks_the_block_whose_tensor_is_the_similarity_matrix():
    """2.01's first surprise, and it is per block rather than per pipeline.

    A `profile` block's feature vector for a protein IS its own row of the
    matrix, so PCA's input is the matrix. A `frequency` block's is not, and
    marking both would make the note meaningless.
    """
    from explorer.payload import _pipeline

    class Block:
        def __init__(self, provider, representation):
            self.provider = provider
            self.representation = representation
            self.metric = "euclidean"

    class Config:
        blocks = {
            "tmscore": Block("tmscore", "profile"),
            "threedi": Block("threedi", None),
        }
        spaces: dict = {}

    pipeline = _pipeline(Config(), [])
    by_id = {b["block_id"]: b for b in pipeline["blocks"]}
    assert by_id["tmscore"]["is_profile"] is True
    assert by_id["threedi"]["is_profile"] is False
    assert by_id["threedi"]["representation"] == "features"


def test_the_diagram_carries_both_graphs_because_they_are_different_graphs():
    """2.01's second surprise. Leiden builds its own kNN graph and the map is
    drawn from another, so a cluster boundary and a gap on the map are not the
    same statement. On the shipped cohorts these k values are 37 and 15, and
    nothing else on the page says so."""
    from explorer.payload import SpacePayload, _pipeline

    class Space:
        blocks = ("tmscore",)
        strategy = "none"
        reducer_params = {"pca_umap": {"n_neighbors": 15}}

    class Config:
        blocks: dict = {}
        spaces = {"structure": Space()}

    space = SpacePayload(
        space_id="structure",
        protids=["a"],
        embeddings={"pca_umap": [[0.0, 0.0]]},
        clusters={"a": "LC0"},
        readable=[True],
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
        diagnostics={"partition": {"n_neighbors": 37, "n_pcs": 30, "resolution": 1.0}},
    )
    row = _pipeline(Config(), [space])["rows"][0]
    assert row["map_n_neighbors"] == 15
    assert row["cluster_n_neighbors"] == 37
    assert row["n_pcs"] == 30


def test_a_reducer_default_is_not_reported_as_a_chosen_value():
    """`reducer_params` empty means the reducer's own default applied, which is
    not the same fact as a value somebody chose. Printing 15 here would claim
    the config said something it did not -- the same rule that keeps cohort
    numbers out of `descriptions.py`."""
    from explorer.payload import SpacePayload, _pipeline
    from explorer.template import render

    class Space:
        blocks = ("tmscore",)
        strategy = "none"
        reducer_params: dict = {}

    class Config:
        blocks: dict = {}
        spaces = {"structure": Space()}

    space = SpacePayload(
        space_id="structure",
        protids=["a"],
        embeddings={"pca_umap": [[0.0, 0.0]]},
        clusters={},
        readable=[True],
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
    )
    assert _pipeline(Config(), [space])["rows"][0]["map_n_neighbors"] is None
    html = render({"spaces": []}, plotly_js="", title="t")
    assert "reducer default" in html, "the page prints nothing where the config said nothing"


def test_a_config_that_resolves_to_nothing_draws_no_diagram():
    """Empty rather than a guessed diagram, so the panel says which input it is
    waiting for -- the same distinction the whole catalogue rests on."""
    from explorer.panels import catalogue_for
    from explorer.payload import _pipeline

    class Config:
        blocks: dict = {}
        spaces: dict = {}

    assert _pipeline(Config(), []) == {}
    entry = next(p for p in catalogue_for(set()) if p["panel_id"] == "pipeline_diagram")
    assert entry["drawable"] is False and entry["missing"] == ["pipeline"]
    drawable = next(p for p in catalogue_for({"pipeline"}) if p["panel_id"] == "pipeline_diagram")
    assert drawable["drawable"] is True


def test_the_pipeline_renderer_reads_the_payload_key():
    from explorer.template import render

    html = render({"spaces": [], "pipeline": {}}, plotly_js="", title="t")
    assert "SHEET_PANELS.pipeline" in html, "the pipeline kind is not registered"
    assert "active.pipeline" in html, "nothing on the page reads the pipeline key"
    body = html[html.index("SHEET_PANELS.pipeline") :][:3000]
    assert "different" in body, "the page does not say the two graphs are different"


# ==========================================================================
# The similarity matrix. The interesting decisions are that the whole square
# ships rather than a triangle, and that the quantisation is declared.
# ==========================================================================


def _tiny_matrix(tmp_path, values, protids):
    """An n x n labelled TSV of `values`, written the way the pipeline writes one.

    A zero is written as the exact `CENSORED_FILL_TOKEN`, because that string is
    how an unmeasured cell is told from a measured 0.000 -- writing "0.0000"
    here produced a fixture with no censoring in it at all, and the cap test
    passed vacuously until it was asserted on.
    """
    from matrix_io import CENSORED_FILL_TOKEN

    header = "protid\t" + "\t".join(protids)
    lines = [header]
    for protid, row in zip(protids, values):
        lines.append(
            protid + "\t" + "\t".join(CENSORED_FILL_TOKEN if v == 0 else f"{v:.4f}" for v in row)
        )
    path = tmp_path / "matrix.tsv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _matrix_fixture(tmp_path, values, clusters):
    from explorer.payload import SpacePayload

    protids = list(clusters)
    path = _tiny_matrix(tmp_path, values, protids)

    class Block:
        provider = "tmscore"
        params = {"matrix_path": path}

    class Space:
        blocks = ("tmscore",)

    class Config:
        blocks = {"tmscore": Block()}
        # The panels resolve the matrix through the SPACE, so a fixture without
        # one exercises nothing.
        spaces = {"structure": Space()}

    space = SpacePayload(
        space_id="structure",
        protids=protids,
        embeddings={"pca_umap": [[0.0, 0.0]] * len(protids)},
        clusters=clusters,
        readable=[True] * len(protids),
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
    )
    return Config(), [space]


def test_the_whole_square_ships_because_the_matrix_is_not_symmetric():
    """The measurement that decides the shape of the payload.

    A triangle would silently choose one of two real values wherever
    ``a_ij != a_ji``, and on the shipped cohorts that is about a fifth of all
    pairs. The asymmetry is measured and travels with the matrix so the panel
    can say so rather than imply symmetry by drawing half.
    """
    import base64

    pytest.importorskip("numpy")
    from explorer.payload import _tm_matrix

    values = [[1.0, 0.9, 0.2], [0.3, 1.0, 0.5], [0.2, 0.5, 1.0]]
    config, spaces = _matrix_fixture(
        _matrix_fixture_dir(), values, {"a": "LC0", "b": "LC0", "c": "LC1"}
    )
    matrix = _tm_matrix(config, spaces, "structure")
    assert matrix["n"] == 3
    assert len(base64.b64decode(matrix["values"])) == 9, "a triangle was shipped"
    # a_01 = 0.9 against a_10 = 0.3 is the only asymmetric pair here.
    assert matrix["asymmetry"]["n_asymmetric"] == 1
    assert matrix["asymmetry"]["n_pairs"] == 3
    assert matrix["asymmetry"]["max_gap"] == pytest.approx(0.6, abs=1e-6)


def test_the_quantisation_error_is_measured_and_shipped_not_assumed():
    """255 levels is coarser than the four significant figures foldseek emits.

    That is invisible in a heat map and fatal in a number, so the panel prints
    the error and calls a cell a colour. A payload that quantised without
    saying by how much would let a reader quote a cell.
    """
    pytest.importorskip("numpy")
    from explorer.payload import _tm_matrix

    values = [[1.0, 0.5], [0.5, 1.0]]
    config, spaces = _matrix_fixture(_matrix_fixture_dir(), values, {"a": "LC0", "b": "LC0"})
    matrix = _tm_matrix(config, spaces, "structure")
    assert matrix["levels"] == 255
    assert 0.0 <= matrix["max_error"] < 0.01
    assert matrix["low"] == pytest.approx(0.5)
    assert matrix["high"] == pytest.approx(1.0)


def test_the_matrix_is_sorted_by_cluster_and_then_by_accession():
    """Deterministic, because two runs of the same inputs must produce the same
    bytes -- the property the provenance footer refuses a timestamp for. A sort
    on cluster alone leaves ties to dict order."""
    pytest.importorskip("numpy")
    from explorer.payload import _tm_matrix

    values = [[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]]
    config, spaces = _matrix_fixture(
        _matrix_fixture_dir(), values, {"z": "LC0", "a": "LC1", "b": "LC0"}
    )
    matrix = _tm_matrix(config, spaces, "structure")
    assert matrix["protids"] == ["b", "z", "a"]
    assert matrix["bands"] == [{"cluster": "LC0", "count": 2}, {"cluster": "LC1", "count": 1}]


def test_a_run_with_no_matrix_keeps_the_panel_awaiting_its_input():
    pytest.importorskip("numpy")
    from explorer.panels import catalogue_for
    from explorer.payload import _tm_matrix

    class Block:
        provider = "tmscore"
        params = {"matrix_path": "/nonexistent/matrix.tsv"}

    class Space:
        blocks = ("tmscore",)

    class Config:
        blocks = {"tmscore": Block()}
        spaces = {"structure": Space()}

    assert _tm_matrix(Config(), []) == {}
    entry = next(p for p in catalogue_for(set()) if p["panel_id"] == "tm_matrix")
    assert entry["drawable"] is False and entry["missing"] == ["matrix"]
    assert next(p for p in catalogue_for({"matrix"}) if p["panel_id"] == "tm_matrix")["drawable"]


def test_the_heatmap_labels_the_value_as_3di_derived_not_tm_align():
    """FOLLOWUPS #60. The score is a foldseek 3Di+AA alignment score, and the
    panel is the most likely place on the page for someone to read it as
    TM-align output -- it is the only panel that shows the number itself."""
    pytest.importorskip("numpy")
    from explorer.payload import _tm_matrix

    values = [[1.0, 0.5], [0.5, 1.0]]
    config, spaces = _matrix_fixture(_matrix_fixture_dir(), values, {"a": "LC0", "b": "LC0"})
    assert "3Di+AA" in _tm_matrix(config, spaces, "structure")["value_label"]
    assert "not TM-align" in _tm_matrix(config, spaces, "structure")["value_label"]


def test_the_heatmap_renderer_decodes_the_matrix_and_refuses_a_wrong_size():
    """A byte count that disagrees with the stated n means the two halves of
    the payload disagree. Drawing something anyway would put a picture on the
    page that is wrong rather than absent."""
    from explorer.template import render

    html = render({"spaces": [], "tm_matrix": {}}, plotly_js="", title="t")
    assert "SHEET_PANELS.heatmap" in html, "the heatmap kind is not registered"
    assert "function decodeMatrix" in html
    assert "active.tm_matrix" in html, "nothing on the page reads the matrix key"
    body = html[html.index("SHEET_PANELS.heatmap") :][:3500]
    assert "not n × n" in body, "a size mismatch is drawn rather than refused"
    assert "never as a measurement" in body, "the quantisation is not declared to the reader"


def test_a_panel_that_measures_layout_draws_after_it_is_attached():
    """The heatmap drew blank in the browser and no test could have seen it.

    A renderer returns its node before `renderSheet` appends it, so Plotly --
    which measures layout -- sized itself against a detached element. The first
    fix was `requestAnimationFrame`, which works in a browser and makes the
    panel's correctness depend on when someone happens to look at the DOM. A
    queue flushed straight after the append is deterministic instead.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "const PENDING_DRAWS" in html
    assert "flushPendingDraws()" in html
    heatmap = html[html.index("SHEET_PANELS.heatmap") :][:3500]
    assert "PENDING_DRAWS.push" in heatmap, "the heatmap draws before it is attached"
    assert (
        "requestAnimationFrame" not in heatmap
    ), "a timing-dependent draw is one no check can see fail"


def test_no_heatmap_shape_reaches_outside_the_data_range():
    """The bug that made the panel blank while every DOM check passed.

    A shape on a data axis is in DATA coordinates, so running a cluster
    boundary line to `+1e6` to mean "the far edge" stretched the axis to a
    million and shrank 367 cells of heat map to a speck. The colour bar and the
    grid lines drew perfectly, the DOM reported a 367x367 trace, and the
    picture was empty. Nothing but a screenshot could see it, so this test
    pins the one property that was wrong.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    shapes = html[html.index("function clusterBandShapes") :]
    shapes = shapes[: shapes.index("\n}")]
    assert "1e6" not in shapes, "a shape coordinate escapes the data range again"
    assert "n - 0.5" in shapes, "the lines do not span the matrix"
    heatmap = html[html.index("SHEET_PANELS.heatmap") :][:4000]
    assert "range: [-0.5, n - 0.5]" in heatmap, "the x axis is left to autorange"
    assert "range: [n - 0.5, -0.5]" in heatmap, "the y axis is left to autorange"


# ==========================================================================
# What the per-query cap removed. The rate is the least interesting number;
# the row/column split is what tells a cap from ordinary sparsity.
# ==========================================================================


def test_an_uncensored_matrix_reports_its_zero_rather_than_nothing():
    """Both shipped cohorts were built from exhaustive matrices, and nothing
    else on the page says so. A reader who assumed the default capped pipeline
    would misread the coverage of every other panel, so "nothing was censored"
    has to be a statement and not an absence."""
    pytest.importorskip("numpy")
    from explorer.payload import _censoring

    values = [[1.0, 0.4], [0.4, 1.0]]
    config, spaces = _matrix_fixture(_matrix_fixture_dir(), values, {"a": "LC0", "b": "LC0"})
    censoring = _censoring(config, spaces, "structure")
    assert censoring["summary"]["n_censored"] == 0
    assert censoring["summary"]["censoring_rate"] == 0.0
    assert censoring["summary"]["cap_detected"] is False
    assert censoring != {}, "an uncensored cohort must still fill the panel"


def test_a_per_query_cap_is_reported_as_one_and_even_sparsity_is_not():
    """ADR 0009's rule, carried onto the page.

    A cap bounds how many partners each QUERY reports, so the rows pile up at
    one count while the columns stay free. A matrix that is merely uniform puts
    both sides on the same count, and calling that a per-query cap would be
    wrong -- so the panel prints both fractions and takes the payload's verdict
    rather than inviting the reader to divide.
    """
    pytest.importorskip("numpy")
    from explorer.payload import _censoring

    # Each row measures itself and one partner; the columns stay uneven.
    capped = [
        [1.0, 0.4, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.7, 0.0, 0.0, 1.0],
    ]
    clusters = {"a": "LC0", "b": "LC0", "c": "LC1", "d": "LC1"}
    config, spaces = _matrix_fixture(_matrix_fixture_dir(), capped, clusters)
    summary = _censoring(config, spaces, "structure")["summary"]
    assert summary["n_censored"] > 0
    assert summary["rows_at_max_fraction"] == 1.0
    assert "cols_at_max_fraction" in summary


def test_the_per_protein_rates_are_restricted_to_the_plotted_proteins():
    """Same rule as the records panel: a row for a protein no panel plots is a
    row the reader cannot find."""
    pytest.importorskip("numpy")
    from explorer.payload import _censoring

    values = [[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]]
    config, spaces = _matrix_fixture(
        _matrix_fixture_dir(), values, {"a": "LC0", "b": "LC0", "c": "LC1"}
    )
    censoring = _censoring(config, spaces, "structure")
    assert {row["protid"] for row in censoring["rates"]} == {"a", "b", "c"}
    assert censoring["n_proteins"] == 3
    rates = [row["rate"] for row in censoring["rates"]]
    assert rates == sorted(rates, reverse=True), "worst-censored first"


def test_the_censoring_panel_prints_both_sides_and_never_only_the_rate():
    """A rate alone cannot distinguish a cap from sparsity, which is the one
    thing this panel exists to say."""
    from explorer.template import render

    html = render({"spaces": [], "censoring": {}}, plotly_js="", title="t")
    assert "SHEET_PANELS.censoring" in html, "the censoring kind is not registered"
    assert "active.censoring" in html, "nothing on the page reads the censoring key"
    body = html[html.index("SHEET_PANELS.censoring") :][:4200]
    assert "cap_detected" in body
    assert "rows_at_max_fraction" in body and "cols_at_max_fraction" in body
    assert "exhaustive" in body, "an uncensored cohort is not told it is uncensored"
    assert "measured_zero_count" in body, "the mask's own assumption is not reported"


# ==========================================================================
# Which points a reader must not trust. Two different things live near each
# other on this page and they must not be run together.
# ==========================================================================


def test_the_flagged_points_are_marked_by_something_no_overlay_can_produce():
    """An open circle was not enough.

    With a continuous overlay the ring takes the palette's own colour, so a
    flagged protein sitting mid-ramp looked like an ordinary one. The flagged
    trace now carries a red ring, which no overlay value can generate, and
    keeps the overlay colour as its fill -- it is still a real protein with a
    real measurement, and only its position is untrustworthy.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    trace = html[html.index("function traceFor") :]
    trace = trace[: trace.index("\n}")]
    assert "circle-open" not in trace, "the flag is still a missing fill"
    assert "#b3261e" in trace, "the flagged trace has no ring colour of its own"
    assert "marker.line.color = ring" in trace


def test_the_legend_counts_the_flagged_proteins_rather_than_asserting_them():
    """ "Some points are hollow" leaves a reader scanning 367 markers for
    something they cannot count. The number, and which panel carries it, is
    checkable."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function renderLegend" in html
    assert "renderLegend()" in html, "the legend is never rendered"
    legend = html[html.index("function renderLegend") :][:1800]
    assert "space.readable[i] === false" in legend, "the count is not read from the mask"
    assert "of at least one" not in legend
    assert "No protein is flagged" in legend, "a clean cohort is not told it is clean"


def test_the_page_says_high_disagreement_is_not_untrustworthy():
    """The distinction this page most needs to keep sharp.

    1.06 calls the disagreement between spaces the actual discovery surface: a
    protein whose structural neighbours are not its chemical ones is the
    interesting case, not a broken one. The do-not-read flag is a different
    judgement from a different source, and a reader who merges them will
    discard their best candidates.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "High disagreement is not the same as untrustworthy" in html
    assert "discovery surface" in html
    key = html[html.index("function renderColourKey") :][:2600]
    assert "left:" in key and "right:" in key, "the ramp's ends are unlabelled"


# ==========================================================================
# What censoring did to a map. The obvious panel here -- a drag line from
# each protein's censored position to its uncensored one -- is the wrong
# one, and the page has to refuse it with a number rather than an argument.
# ==========================================================================


def test_the_superimposable_threshold_sits_far_above_the_reducer_s_own_noise():
    """The threshold has to be a measurement, not a taste.

    Measured on actin_B: the same matrix reduced with the same parameters and
    only a different seed gives a Procrustes disparity of 0.010-0.031, so the
    reducer is stable to three decimals on this data. Two genuinely different
    modalities give 0.862. A threshold of 0.5 is therefore nowhere near reducer
    noise and already past "as different as two different modalities".
    """
    from explorer.payload import SUPERIMPOSABLE_THRESHOLD

    assert 0.031 < SUPERIMPOSABLE_THRESHOLD < 0.862


def test_an_unsuperimposable_pair_refuses_the_displacement():
    """A line between two positions is read as distance travelled, and that
    reading needs a shared frame. Above the threshold the page must say so and
    fall back to something frame-independent."""
    from explorer.template import render

    html = render({"spaces": [], "censoring": {}}, plotly_js="", title="t")
    assert "function renderCensoringComparison" in html
    body = html[html.index("function renderCensoringComparison") :][:4200]
    assert "not drawn as a displacement" in body
    assert "comparison.superimposable" in body, "the refusal is not conditional on the number"
    assert "neighbours kept" in body, "no frame-independent measure is offered instead"


def test_the_comparison_needs_two_named_spaces_and_will_not_guess():
    """Nothing in a space's manifest says "this is the capped twin of that one",
    and a convention over space ids would attach the label to whatever happened
    to match. So it is declared, validated against the defined spaces, and
    absent when not declared."""
    import pytest as _pytest
    from config_schema import ConfigError, DiagnosticsConfig

    assert DiagnosticsConfig().censoring_comparison == ()
    with _pytest.raises(ConfigError) as one:
        DiagnosticsConfig(censoring_comparison=("only_one",))
    assert "exactly two space ids" in str(one.value)
    with _pytest.raises(ConfigError) as same:
        DiagnosticsConfig(censoring_comparison=("structure", "structure"))
    assert "measures the reducer, not the censoring" in str(same.value)


def test_a_censoring_comparison_naming_an_undefined_space_is_rejected():
    """The same rule `coregistration.compare` already follows. A typo here
    would silently produce a panel with no comparison rather than an error."""
    import pytest as _pytest
    from config_schema import ConfigError, from_legacy

    config = {
        "blocks": {"b": {"provider": "biophys"}},
        "spaces": {"s": {"blocks": ["b"]}},
        "diagnostics": {"censoring_comparison": ["s", "nope"]},
    }
    with _pytest.raises(ConfigError) as error:
        from_legacy(config)
    assert "censoring_comparison[1]" in str(error.value)
    assert "not a defined space" in str(error.value)


def test_the_matrix_panels_read_the_named_space_s_own_block_not_the_first_one():
    """A run can carry more than one tmscore block.

    The censoring comparison needs a capped twin alongside the shipped block,
    and both panels used to scan `config.blocks` for the first tmscore provider
    -- which is dict insertion order, a property of how the config was typed
    rather than of what the panel describes. It would let the matrix panel draw
    one matrix under another's caption.
    """
    from explorer.payload import _matrix_path_for

    class Block:
        def __init__(self, path):
            self.provider = "tmscore"
            self.params = {"matrix_path": path}

    class Space:
        def __init__(self, blocks):
            self.blocks = blocks

    class Config:
        # `tmscore_capped` deliberately first, which is what a scan would find.
        blocks = {"tmscore_capped": Block("/capped.tsv"), "tmscore": Block("/full.tsv")}
        spaces = {
            "structure": Space(("tmscore",)),
            "structure_capped": Space(("tmscore_capped",)),
        }

    assert _matrix_path_for(Config(), "structure") == "/full.tsv"
    assert _matrix_path_for(Config(), "structure_capped") == "/capped.tsv"
    assert _matrix_path_for(Config(), "nonexistent") == ""


# ==========================================================================
# The per-protein stability series. `to_frame` computed it all along and
# nothing wrote it, so the panel that needs the ramp had only the summary.
# ==========================================================================


def test_the_stability_series_is_written_beside_the_summary(tmp_path):
    """Faithfulness already writes its per-protein file; stability did not.

    `to_dict` keeps the mean, the min and the coin-flip list -- which answers
    "is this space stable" and discards the ramp that answers "is THIS protein
    stable". The source calls the per-protein overlay not optional (3.01).
    """
    pytest.importorskip("numpy")
    import numpy as np
    from diagnostics.stability import neighborhood_stability
    from spaces import layout

    rng = np.random.default_rng(0)
    protids = [f"p{i}" for i in range(24)]
    points = rng.normal(size=(24, 5))
    # Square distances over the protids, which is what the measurement takes.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    result = neighborhood_stability("s", distances, protids, k=4, replicates=4)
    frame = result.to_frame()
    assert list(frame.index) == protids
    assert "stability" in frame.columns
    assert layout.stability_filename() == "stability.tsv"
    # No reducer in the name, and that asymmetry is the point.
    assert "pca_umap" not in layout.stability_filename()


def test_diagnose_space_writes_the_series_next_to_its_summary():
    """Asserted on the source, because running the whole entry point here needs
    the reducer stack. The write has to sit inside the `bootstrap_replicates`
    branch: with replicates at 0 nothing was measured and a file of NaNs would
    claim otherwise."""
    import inspect

    import diagnose_space

    source = inspect.getsource(diagnose_space)
    assert "layout.stability_filename()" in source
    branch = source[source.index("if config.diagnostics.bootstrap_replicates:") :][:1200]
    assert "stability.to_frame().to_csv" in branch


def test_the_stability_series_is_per_space_and_travels_per_space(tmp_path):
    """Stability is measured on a SPACE's own distances, so one shared overlay
    would colour every panel by one space's answer. It is carried on the space,
    like the readable mask, rather than in the page-wide overlay dict."""
    pytest.importorskip("pandas")
    from explorer.payload import SpacePayload, _stability_series

    (tmp_path / "stability.tsv").write_text(
        "protid\tstability\treplicates_seen\np1\t0.9\t20\np2\t0.1\t20\nother\t0.5\t20\n"
    )
    series = _stability_series(str(tmp_path), ["p1", "p2"])
    assert series == {"p1": 0.9, "p2": 0.1}, "a protein off the maps must not appear"
    space = SpacePayload(
        space_id="s",
        protids=["p1", "p2"],
        embeddings={"pca_umap": [[0.0, 0.0], [1.0, 1.0]]},
        clusters={},
        readable=[True, True],
        verdict={"level": "ok", "reasons": [], "headline": "fine"},
        stability=series,
    )
    assert space.to_dict()["stability"] == series


def test_the_coin_flip_line_is_read_from_the_module_that_enforces_it():
    """A threshold retyped into the template can drift from the one the
    diagnostics apply, and then the picture and the verdict disagree."""
    from diagnostics.stability import COIN_FLIP_THRESHOLD
    from explorer.payload import _thresholds
    from explorer.template import render

    assert _thresholds()["coin_flip"] == COIN_FLIP_THRESHOLD
    html = render({"spaces": [], "thresholds": _thresholds()}, plotly_js="", title="t")
    assert "const COIN_FLIP = THRESHOLDS.coin_flip" in html
    body = html[html.index("SHEET_PANELS.stability") :][:3000]
    assert "0.30" not in body and "0.3;" not in body, "the threshold is retyped in the template"


def test_the_stability_panel_says_it_judges_the_space_not_the_layout():
    """FOLLOWUPS #62. A protein with determinate neighbours can still be drawn
    in the wrong place, and that second question is faithfulness -- which the
    panel banners answer. Merging them is the misreading this panel is most
    likely to cause."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "SHEET_PANELS.stability" in html, "the stability kind is not registered"
    body = html[html.index("SHEET_PANELS.stability") :][:3000]
    assert "space.stability" in body
    assert "not the drawing of it" in body or "not the layout" in body


# ==========================================================================
# Per-descriptor overlays, and each descriptor's share of the distance.
# The only overlays on the page that colour a map by its own input.
# ==========================================================================


def _biophys_block(tmp_path, names, values):
    """A feature block on disk: features.npy, protids.txt and a manifest."""
    import numpy as np

    directory = tmp_path / "blocks" / "biophys"
    directory.mkdir(parents=True)
    array = np.asarray(values, dtype=float)
    np.save(directory / "features.npy", array)
    protids = [f"p{i}" for i in range(array.shape[0])]
    (directory / "protids.txt").write_text("\n".join(protids) + "\n")
    (directory / "manifest.json").write_text(json.dumps({"extra": {"descriptors": list(names)}}))
    return protids


def test_a_named_feature_block_becomes_one_overlay_per_column(tmp_path):
    """These are the only overlays that colour a map by the thing it was built
    from. Every other overlay comes from the features table and is, by
    construction, something the geometry never saw."""
    pytest.importorskip("numpy")
    from explorer.payload import _block_column_overlays

    class Config:
        blocks = {"biophys": object()}

    protids = _biophys_block(tmp_path, ["gravy", "pi"], [[0.1, 7.0], [0.2, 5.0]])
    overlays = _block_column_overlays(str(tmp_path), Config(), protids)
    assert sorted(overlays) == ["biophys:gravy", "biophys:pi"]
    assert overlays["biophys:pi"]["values"] == [7.0, 5.0]
    assert overlays["biophys:gravy"]["kind"] == "continuous"


def test_a_block_with_thousands_of_columns_is_not_offered_as_overlays(tmp_path):
    """The 3Di block has 4,982 3-mer frequency columns. Putting those in a
    dropdown would be absurd rather than merely long, so a block only exposes
    its columns when it has few enough of them to name."""
    pytest.importorskip("numpy")
    import numpy as np
    from explorer.payload import MAX_NAMED_BLOCK_COLUMNS, _block_column_overlays

    class Config:
        blocks = {"biophys": object()}

    wide = MAX_NAMED_BLOCK_COLUMNS + 1
    names = [f"kmer{i}" for i in range(wide)]
    protids = _biophys_block(tmp_path, names, np.zeros((2, wide)))
    assert _block_column_overlays(str(tmp_path), Config(), protids) == {}


def test_a_manifest_disagreeing_with_its_array_yields_nothing(tmp_path):
    """If the names and the columns disagree the block was written by a
    different version of the provider, and guessing which column is which would
    mislabel every point on the map."""
    pytest.importorskip("numpy")
    from explorer.payload import _block_column_overlays

    class Config:
        blocks = {"biophys": object()}

    protids = _biophys_block(tmp_path, ["only_one_name"], [[0.1, 7.0], [0.2, 5.0]])
    assert _block_column_overlays(str(tmp_path), Config(), protids) == {}


def test_a_descriptor_never_displaces_a_feature_table_column_of_the_same_name():
    """The block columns are added with `setdefault` after the table's, and they
    carry a `block:` prefix so the source travels with the number."""
    import inspect

    from explorer import payload

    source = inspect.getsource(payload.build_payload)
    assert "overlays.setdefault(name, overlay)" in source
    assert 'f"{block_id}:{name}"' in inspect.getsource(payload._block_column_overlays)


def test_each_column_s_share_of_the_distance_is_reported(tmp_path):
    """Euclidean distance on raw columns is a sum of per-column squared
    differences, so a column's share of the variance IS its share of the
    squared distance. This is the quantity, not a proxy for it."""
    pytest.importorskip("numpy")
    from explorer.payload import _column_shares

    class Block:
        normalization = "zscore_within"

    class Space:
        blocks = ("biophys",)

    class Config:
        blocks = {"biophys": Block()}

    # One column with all the variance and one with none.
    _biophys_block(tmp_path, ["flat", "wide"], [[1.0, 0.0], [1.0, 10.0]])
    shares = _column_shares(str(tmp_path), Config(), Space())
    by_name = {row["column"]: row for row in shares}
    assert by_name["flat"]["share"] == pytest.approx(0.0)
    assert by_name["wide"]["share"] == pytest.approx(1.0)
    assert by_name["wide"]["share_if_standardized"] == pytest.approx(0.5)
    assert by_name["wide"]["declared_normalization"] == "zscore_within"


def test_a_fused_space_reports_block_contributions_and_not_column_shares(tmp_path):
    """With several blocks the apportionment between them is the fused
    contribution, which the run already computes. Reporting columns as well
    would be two different answers to one question."""
    pytest.importorskip("numpy")
    from explorer.payload import _column_shares

    class Space:
        blocks = ("biophys", "tmscore")

    class Config:
        blocks: dict = {}

    _biophys_block(tmp_path, ["a", "b"], [[1.0, 0.0], [1.0, 10.0]])
    assert _column_shares(str(tmp_path), Config(), Space()) == []


def test_the_map_says_when_one_column_carries_almost_all_of_it():
    """FOLLOWUPS #32, made visible. The biophysical block declares
    `zscore_within` and nothing reads the field, so its columns enter the
    distance raw: on both shipped cohorts isoelectric point is over 97% of it.
    A reader who does not open the fold-out would otherwise read that map as a
    map of physicochemistry."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "space.column_shares" in html, "the shares never reach the panel"
    assert "This map is mostly" in html
    assert "a fact about the units, not about the" in html
    assert "FOLLOWUPS #32" in html, "the unread field is not named"


def test_a_report_with_no_faithfulness_section_is_not_read_as_a_clean_one():
    """Absence taken for absence-of-problem, which is the #29/#32 shape again.

    Faithfulness is the ONLY diagnostic that looks at the 2-D coordinates, so a
    report without it cannot say whether any layout is readable. Found for real:
    regenerating a cohort's diagnostics without `--embedding` dropped the
    section, and 306 flagged proteins silently became none while every banner
    still read "diagnostics found no reason to distrust this map".
    """
    from explorer.payload import space_verdict

    partial = {
        "stability": [{"informative": True, "stability_mean": 0.9}],
        "partition": {"n_clusters": 4},
    }
    verdict = space_verdict(partial, 100)
    assert verdict["level"] == "caution"
    assert any("no faithfulness section" in reason for reason in verdict["reasons"])
    assert any("undiagnosed rather than sound" in reason for reason in verdict["reasons"])

    # And a report that HAS the section is judged on its numbers as before.
    complete = dict(
        partial,
        faithfulness=[
            {"reducer": "pca_umap", "trustworthiness_mean": 0.95, "continuity_mean": 0.96}
        ],
    )
    assert space_verdict(complete, 100)["level"] == "ok"


def test_diagnose_space_says_so_when_it_measures_no_layout():
    """Every other skipped section calls `_skip`. This one vanished in silence,
    and it is the worst one to lose quietly."""
    import inspect

    import diagnose_space

    source = inspect.getsource(diagnose_space)
    branch = source[source.index("# 4. did the map survive two dimensions") :][:1400]
    assert "else:" in branch, "the no-embedding case is still silent"
    assert "no --embedding was supplied" in branch
    assert "nothing here judges any layout" in branch


def test_the_structural_space_is_resolved_from_blocks_not_from_its_name(tmp_path):
    """A cohort whose structural space is not called "structure" still draws.

    Both panels defaulted to the literal id `"structure"`. That is a naming
    convention, not a fact about a run, and the failure it produced was silent:
    an absent payload key and an unbuildable panel render identically, so a
    cohort using any other id got an empty matrix panel and a censoring panel
    claiming its input did not exist, while the matrix sat on disk.
    """
    from explorer.payload import _structural_space

    matrix = tmp_path / "m.tsv"
    matrix.write_text("protid\ta\tb\na\t1.0\t0.5\nb\t0.5\t1.0\n")

    class Block:
        provider = "tmscore"

        def __init__(self, path):
            self.params = {"matrix_path": path}

    class Space:
        def __init__(self, blocks):
            self.blocks = blocks

    class Config:
        blocks = {"tm": Block(str(matrix))}
        spaces = {"shape": Space(("tm",))}

    spaces = [type("S", (), {"space_id": "shape"})()]
    assert _structural_space(Config(), spaces) == "shape"


def test_two_tmscore_blocks_resolve_to_the_first_space_in_config_order(tmp_path):
    """The censoring comparison adds a capped twin, so two qualify.

    Asserted by space id rather than by matrix contents: the point is that the
    ORDER is fixed by the config, not that one file differs from another. A rule
    that did not fix an order would choose by dict insertion order, which is a
    property of how the config was typed.
    """
    from explorer.payload import _structural_space

    full = tmp_path / "full.tsv"
    capped = tmp_path / "capped.tsv"
    for f in (full, capped):
        f.write_text("protid\ta\tb\na\t1.0\t0.5\nb\t0.5\t1.0\n")

    class Block:
        provider = "tmscore"

        def __init__(self, path):
            self.params = {"matrix_path": path}

    class Space:
        def __init__(self, blocks):
            self.blocks = blocks

    class Config:
        blocks = {"tm": Block(str(full)), "tm_capped": Block(str(capped))}
        spaces = {"structure": Space(("tm",)), "structure_capped": Space(("tm_capped",))}

    spaces = [
        type("S", (), {"space_id": "structure"})(),
        type("S", (), {"space_id": "structure_capped"})(),
    ]
    assert _structural_space(Config(), spaces) == "structure"


def test_no_tmscore_block_means_no_structural_space_and_no_keys():
    """The refusal has to be reachable, or the panels refuse for the wrong reason.

    A biophys-only cohort has no matrix at all. `_structural_space` returns ""
    and both payload keys stay out of `available`, so each panel prints its own
    `requires` rather than drawing someone else's data.
    """
    from explorer.payload import _censoring, _structural_space, _tm_matrix

    class Block:
        provider = "biophys"
        params: dict = {}

    class Space:
        blocks = ("bio",)

    class Config:
        blocks = {"bio": Block()}
        spaces = {"chem": Space()}

    spaces = [type("S", (), {"space_id": "chem"})()]
    assert _structural_space(Config(), spaces) == ""
    assert not _tm_matrix(Config(), spaces, "")
    assert not _censoring(Config(), spaces, "")


def test_the_axes_sentence_reports_the_components_the_run_actually_used(tmp_path):
    """Thirty is what the config REQUESTS, not what PCA returns.

    `PCA` returns min(n_components, n_samples, n_features), so a four-column
    biophysics block yields four however many are asked for. The typed sentence
    said "a 30-component PCA" under every map and was wrong on physicochemistry
    on BOTH shipped cohorts -- sixteen times on the built page -- while
    `n_components_used` in the manifest carried the true number all along.
    """
    from explorer.descriptions import describe_axes

    text = describe_axes(
        "pca_umap",
        {
            "axis_names": ["UMAP1", "UMAP2"],
            "pca_components_requested": 30,
            "pca_components_used": 4,
        },
    )
    assert "4-component PCA" in text
    assert "30-component" not in text
    # The divergence is stated, not silently resolved: a reader who knows the
    # config says 30 has to be able to see why the picture says 4.
    assert "config asked for 30" in text
    assert "min(n_components, n_samples, n_features)" in text


def test_the_axes_sentence_stays_quiet_when_requested_equals_used():
    """No divergence, no explanation -- the caveat would be noise on 9 of 11."""
    from explorer.descriptions import describe_axes

    text = describe_axes(
        "pca_umap",
        {
            "axis_names": ["UMAP1", "UMAP2"],
            "pca_components_requested": 30,
            "pca_components_used": 30,
        },
    )
    assert "30-component PCA" in text
    assert "config asked for" not in text


def test_a_run_that_recorded_no_steps_says_nothing_about_its_axes(tmp_path):
    """A refusal is recoverable; a confident wrong number is not.

    A page built from a tree older than the `steps` key must not fall back to
    the typed sentence, which is the defect being removed.
    """
    from explorer.descriptions import describe_axes
    from explorer.payload import _reducer_axes

    assert _reducer_axes(str(tmp_path), "pca_umap") == {}
    assert describe_axes("pca_umap", {}) == ""
    assert describe_axes("pca_umap", None) == ""


def test_no_component_count_is_typed_into_the_axes_prose():
    """The guard that makes this stay fixed.

    The invariant is not "the number is 4" -- it is that NO component count may
    be typed into the reducer-independent prose at all, because a typed one
    cannot track a run. Anyone reintroducing "a 30-component PCA" as a literal
    fails here rather than on a page nobody greps.
    """
    import re

    from explorer.descriptions import AXES

    joined = " ".join(AXES["paragraphs"])
    assert not re.search(r"\b\d+-component\b", joined), (
        "a component count is typed into the invariant axes prose; it must be "
        "composed from the run's manifest by describe_axes instead"
    )
    assert "UMAP1" not in joined, "an axis name is typed into the invariant prose"


def test_the_inspector_offers_entry_pages_and_never_a_guessed_model_url():
    """A constructed AF-{acc}-F1 url 404s for isoform-backed entries.

    `fetch_accession.py` records that, so the only honest link is the entry page
    each database routes itself. Both anchors also carry rel="noopener", because
    target="_blank" without it hands the opened tab a window.opener handle.
    """
    from explorer.template import _TEMPLATE

    assert "https://www.uniprot.org/uniprotkb/{acc}/entry" in _TEMPLATE
    assert "https://alphafold.ebi.ac.uk/entry/{acc}" in _TEMPLATE
    assert 'rel="noopener"' in _TEMPLATE
    # Targets CODE, not prose. A bare "AF-" also matches the comment explaining
    # why the model url is NOT built, so the assertion would fail on the very
    # sentence documenting the decision. What must stay absent is a CONSTRUCTED
    # one: a literal opening "AF-", an interpolation into it, or the /files/
    # path those urls live under.
    assert '"AF-' not in _TEMPLATE, "a guessed model-file url literal is back"
    assert "AF-${" not in _TEMPLATE, "a model url is being interpolated"
    assert "alphafold.ebi.ac.uk/files" not in _TEMPLATE, "the model-file path is back"


def test_the_pages_accession_guard_is_the_same_regex_python_uses():
    """The one real cross-check here, and the reason it exists.

    The guard is duplicated -- once in Python, once as a JS literal -- because
    the page cannot import Python. Two copies of a regex in two languages drift
    silently, and the drift would show up as proteins quietly losing their links
    rather than as any error. So the literal is pinned to its source of truth.
    """
    import re

    from domain_utils import UNIPROT_ACCESSION
    from explorer.template import _TEMPLATE

    block = re.search(r"const UNIPROT_ACCESSION = new RegExp\(\s*(.+?)\s*\);", _TEMPLATE, re.S)
    assert block, "the JS accession guard is missing or is no longer built from strings"

    # The JS is split at the same point domain_utils splits its own pattern, so
    # rejoin the halves before comparing.
    halves = re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1))
    assert len(halves) == 2, f"expected two string halves, found {len(halves)}"
    js_pattern = "".join(halves)

    # ONE transformation, stated rather than absorbed: a backslash inside a JS
    # string literal is written doubled, so the Python pattern must be escaped
    # the same way before the two can be compared. A test that skipped this
    # would still pass on any pattern containing no backslash at all, which is
    # exactly the kind of vacuous green this branch has been bitten by.
    expected = UNIPROT_ACCESSION.pattern.replace("\\", "\\\\")
    assert js_pattern == expected, (
        "the page's accession guard has drifted from domain_utils.UNIPROT_ACCESSION:\n"
        f"  page:   {js_pattern}\n  python: {expected}"
    )


def test_the_full_selection_is_offered_even_though_the_code_line_truncates():
    """The count and the textarea must not be readable as disagreeing.

    The inline `<code>` list truncates at 12 so it stays readable. The textarea
    carries the whole selection, and its label repeats the FULL count.

    A copy button was rejected rather than forgotten, and on a MEASURED reason
    rather than the assumed one. file:// is a secure context in Chrome and
    `navigator.clipboard.writeText` exists there; it is gated on document focus
    and rejects with NotAllowedError without it. A button would therefore work
    sometimes and fail silently otherwise. Selecting text always works.
    """
    from explorer.template import _TEMPLATE

    assert "chosen.slice(0, 12)" in _TEMPLATE, "the truncated code line is gone"
    assert 'textarea class="egress" readonly' in _TEMPLATE
    assert "chosen.join(" in _TEMPLATE, "the textarea does not carry the whole selection"
    assert "All ${chosen.length}" in _TEMPLATE, "the label does not repeat the full count"
    # CODE, not prose, and the trailing paren is the whole distinction: the
    # comment above the textarea NAMES navigator.clipboard.writeText to explain
    # why it is not used, so matching the bare identifier fails on the very
    # sentence documenting the decision. A call has parentheses; a mention does
    # not. This assertion has now been wrong twice in that exact way.
    assert (
        "navigator.clipboard.writeText(" not in _TEMPLATE
    ), "a clipboard write is back; it is focus-gated and fails silently unfocused"


def test_no_metricity_figure_reaches_the_payload(built):
    """A decided refusal, pinned so it cannot drift back in.

    FOLLOWUPS #59 measured that roughly forty points of any metricity figure are
    manufactured by the `1 - S` transform: on data Euclidean by construction it
    scores 41.8-43.4% where `sqrt(2(1-S))` scores ~1e-15. A panel reporting it
    would be reporting its own convention with a cohort-shaped wobble on top,
    and a reader could not tell that from the picture. So no such panel is
    built, and no figure travels to the page where someone could quote it.

    Asserted over the WHOLE serialized payload rather than over a key list,
    because the point is that the number reaches no surface at all -- a key-name
    check would pass while the value rode along inside a nested diagnostics blob.
    """
    import json

    blob = json.dumps(built.to_dict())

    # Not vacuous: a payload that serialized to almost nothing would satisfy the
    # loop below for the wrong reason. This fixture is a real run.
    assert len(blob) > 5000, f"payload is only {len(blob)} chars; the guard would be vacuous"

    for term in ("metricity", "negative_mass", "negative_eigen"):
        assert term not in blob, f"{term!r} reached the payload; #59 says it must not"

    # And the detector bites: the same check against a payload carrying the
    # figure must fail. A guard never seen to fail is not a guard.
    planted = json.dumps({**built.to_dict(), "derived": {"metricity": 0.446}})
    assert "metricity" in planted


def _producible_keys():
    """What `build_payload` can actually put into `available`, read from source.

    Parsed rather than retyped, so the check reads the code instead of a copy of
    it. `explorer/payload.py` has no exported vocabulary yet -- PC-009 phase 2
    would add one -- and until it does, this is the honest way to ask.
    """
    import pathlib
    import re

    source = pathlib.Path(__file__).resolve().parents[1] / "explorer" / "payload.py"
    return set(re.findall(r'available\.add\("([^"]+)"\)', source.read_text()))


def test_the_seventh_empty_panel_needs_a_third_dataset_not_the_tree():
    """`tree_space` is not part of the phylogeny group, and the count says so.

    Summary lines on this branch have repeatedly said "ten panels blocked on the
    two data-contract datasets". It is 8 + 1 + 1: six panels want a reconciled
    tree, two want a perturbation scan, `tree_space` wants a CORPUS of
    per-family trees, and `identity_vs_tm` wants neither -- one foldseek pass
    per cohort emitting fident, which is compute rather than a decision.
    """
    from explorer.panels import CATALOGUE

    tree_space = next(s for s in CATALOGUE if s.panel_id == "tree_space")
    assert tree_space.needs == ("tree_corpus",)

    tree_blocked = {s.panel_id for s in CATALOGUE if "reconciled gene/species tree" in s.requires}
    assert "tree_space" not in tree_blocked, (
        "tree_space is being counted with the tree-blocked panels; it needs a "
        "corpus across families, not this cohort's one tree"
    )


def test_exactly_nine_declared_inputs_are_ones_this_pipeline_cannot_produce():
    """The count claim, pinned where it can fail.

    If anything ever starts producing `patristic`, the claim that nine inputs
    are unsatisfiable becomes false. It should fail HERE, next to the list, and
    not on a page where a reader would have to notice a panel that stopped
    refusing.
    """
    from explorer.panels import CATALOGUE

    declared = {n for s in CATALOGUE for n in s.needs}
    unsatisfiable = declared - _producible_keys()
    assert unsatisfiable == {
        "ancestors",
        "clades",
        "gene_tree",
        "identity",
        "metric_embedding",
        "patristic",
        "perturbation",
        "species_tree",
        "tree_corpus",
    }, f"the unsatisfiable set has moved: {sorted(unsatisfiable)}"


def test_one_produced_key_is_declared_by_no_panel_at_all():
    """The asymmetry in the other direction, which nothing else looks at.

    `overlays` is put into `available` and appears in no panel's `needs`. That
    is correct today -- overlays are a colour-by control rather than a panel
    input -- but it is the shape that would hide a panel silently losing its
    declaration, so it is stated rather than left as a coincidence.
    """
    from explorer.panels import CATALOGUE

    declared = {n for s in CATALOGUE for n in s.needs}
    assert _producible_keys() - declared == {"overlays"}


def test_no_panel_names_the_fident_aggregator_as_the_route_to_an_identity_table():
    """`aggregate_foldseek_fraction_seq_identity.py` cannot serve these cohorts.

    Two panels' prose used to name it as the route, and the second one got there
    by way of a correction: PC-018 rewrote `fills_in` to stop pointing at a
    document, and carried the broken route into the replacement. So this pins
    the claim rather than the wording.

    Why it cannot serve them: the script keeps only targets matching
    `-F1-model` and pulls the protid out with `re.findall("AF-(.*)-F1-model")`,
    while a cohort's own all-vs-all names targets like `A0A068F598.pdb` -- so
    the filter drops every row. Its input is the Foldseek WEB API rather than a
    local pass, `key_protids` is empty for both shipped cohorts, and its output
    is a wide per-key-protid feature table rather than the pairwise table a
    scatter needs.
    """
    from explorer.panels import CATALOGUE

    for spec in CATALOGUE:
        for field in ("requires", "fills_in"):
            assert "aggregate_foldseek_fraction_seq_identity" not in getattr(spec, field), (
                f"{spec.panel_id}.{field} names a route that cannot produce an "
                "identity table for either shipped cohort"
            )

    identity = next(s for s in CATALOGUE if s.panel_id == "identity_vs_tm")
    assert "superposition" in identity.requires, (
        "the refusal no longer says the identity is read off a structural "
        "superposition, which is the reason it is not an independent axis"
    )


def test_the_stability_foldout_says_the_number_reads_two_ways(built):
    """The one idea worth keeping from the refused orientation sheet (#70).

    It lives beside the number it describes rather than on a sheet a reader
    would have to find first. Asserted on the RENDERED html, not on the
    constant, because a paragraph that never reaches the page is the same as
    one nobody wrote.

    The hazard count is asserted too: this is a PARAGRAPH, not a hazard. Moving
    it into `hazards` would change the fold-out's summary line from "What this
    shows" to "and N thing(s) it cannot be read for", which would tell a reader
    the opposite of what the sentence says.
    """
    from explorer.descriptions import PANEL_DESCRIPTIONS
    from explorer.template import render

    html = render(built.to_dict(), "", "t")
    assert "where the measurements disagree" in html
    assert "indeterminate is a warning" in html

    assert (
        len(PANEL_DESCRIPTIONS["stability_map"]["hazards"]) == 1
    ), "the two-ways sentence belongs in paragraphs, not hazards"


#: Spelled numbers the inventory's question could plausibly need. The map is
#: deliberately small: a count outside it fails loudly rather than silently
#: skipping the assertion, which is how a guard ends up checking nothing.
INVENTORY_COUNT_WORDS = {
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
}


def test_the_inventorys_hand_typed_count_is_pinned_to_the_guard_it_reads():
    """The rows are read from the guard; the word "Eight" above them is typed.

    Both existing tests over this panel assert the SET of signals matches
    `NOT_FUSABLE_PROVIDERS`, and both keep passing if a ninth signal is added
    to the guard -- while the question above the table goes on saying eight.
    A panel that miscounts its own rows reads as a check and is not one.
    """
    from config_schema import NOT_FUSABLE_PROVIDERS
    from explorer.panels import CATALOGUE

    entry = next(p for p in CATALOGUE if p.panel_id == "signal_inventory")
    count = len(set(NOT_FUSABLE_PROVIDERS.values()))
    word = INVENTORY_COUNT_WORDS.get(count)
    assert word, f"the guard names {count} signals and this test has no word for it"
    assert entry.question.startswith(f"{word} quantities"), (
        f"the guard now names {count} signals, so the question should open "
        f"{word!r} -- it opens {entry.question[:24]!r}"
    )
    assert len(entry.content["rows"]) == count


def test_a_panel_drawn_on_another_sheet_says_where_it_went():
    """`contributions` was catalogued on mechanics and appeared on no sheet.

    `BUILT_ELSEWHERE` stops the mechanics body from drawing it as a card --
    correctly, because it is already drawn as the `contribution:` strip inside
    every fused map. The effect was that the mechanics sheet counted a panel it
    never showed and never said where it had gone.

    Asserted against CODE, not against the comment that explains it: a bare
    `DRAWN_AT` also matches the prose above the constant.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "const DRAWN_AT = {" in html, "the pointer table is not in the page"
    assert "DRAWN_AT[p.panel_id]" in html, "the sheet body does not select pointers"
    assert "DRAWN_AT[panel.panel_id]" in html, "the pointer text is never read"
    assert "contributions:" in html[html.index("const DRAWN_AT = {") :][:600]


def test_the_pointer_is_keyed_on_panel_id_and_never_on_panel_type():
    """`panel_type` is shared; `panel_id` is not.

    `BUILT_ELSEWHERE` is keyed on the type on purpose -- it asks "does an older
    renderer already draw this kind". The pointer asks a different question,
    "where did THIS panel go", and a second panel of type `contributions` would
    inherit a false answer if the pointer were keyed the same way.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "DRAWN_AT[p.panel_type]" not in html
    assert "DRAWN_AT[panel.panel_type]" not in html
    assert (
        "p.drawable && DRAWN_AT[p.panel_id]" in html
    ), "the sheet body no longer selects pointer panels by panel_id"


def test_the_pointer_card_is_not_counted_as_an_empty_panel():
    """Nothing is missing on a pointer card, so it must not read as a refusal.

    The `(N empty)` counter filters on `BUILT_ELSEWHERE` before counting, and
    the pointer deliberately does not change that filter -- so the mechanics
    tab's count is the same number before and after this card exists. It also
    must not borrow the `.awaiting` hatch, which is the page's visual language
    for an input that is absent.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    counter = html[html.index("const blank = sheetPanels") - 700 :][:900]
    assert "!BUILT_ELSEWHERE.has(p.panel_type)" in counter, (
        "the empty counter stopped excluding panels drawn elsewhere, so the "
        "pointer card would now be counted as blank"
    )
    card = html[html.index("elsewhere.forEach((panel)") :][:400]
    assert '"elsewhere"' in card
    assert "awaitingBlock" not in card


def test_the_cutoff_that_decides_a_position_is_unreadable_is_read_and_printed():
    """A count whose threshold is nowhere on the page cannot be checked.

    `payload._thresholds()` ships `distorted` (0.70) precisely so the page can
    state the line it enforces without retyping it, and the template took only
    `coin_flip` -- so "positions not readable: 306 of 367" appeared twice on the
    page with the rule that produced 306 stated nowhere.

    This repo's recorded failure mode is that a value written to a manifest is
    not thereby honored (FOLLOWUPS #29, #32), so the assertion is that the
    template CONSUMES the key, not that it mentions it.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "THRESHOLDS.distorted" in html, "the cutoff never reaches the page"
    assert "distortedCutoffNote()" in html, "nothing calls the note builder"
    assert (
        "trustworthiness or continuity at or below ${THRESHOLDS.distorted}" in html
    ), "the note no longer interpolates the payload value, so it is retyped"


def test_the_cutoff_note_is_dropped_rather_than_printed_half_written():
    """An older page carries no `thresholds` key at all.

    The note must then be absent, not "at or below undefined" -- a page saying
    that is worse than one saying nothing, because it looks like a measurement.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("function distortedCutoffNote") :][:420]
    assert "THRESHOLDS.distorted === undefined" in body
    assert 'return THRESHOLDS.distorted === undefined\n    ? ""' in body.replace("\r", "")


def test_both_places_that_print_the_unreadable_count_carry_the_cutoff():
    """The count appears twice: the geometry table and the inspector.

    One helper feeds both, so the two can never state different lines, and a
    reader who never opens the diagnostics report still sees the cutoff beside
    the count in the panel they are actually using.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    calls = html.count("distortedCutoffNote()") - html.count("function distortedCutoffNote()")
    assert calls == 2, f"expected exactly the two count sites to call the note, found {calls}"
    inspector = html[html.index("positions not readable<span") :][:220]
    assert "distortedCutoffNote()" in inspector
    geometry = html[html.index('label: "positions not readable"') :][:200]
    assert "note: distortedCutoffNote()" in geometry


# ==========================================================================
# The sentences the run's diagnostics wrote, which reached the browser and were
# rendered nowhere. Measured on the two-cohort page at commit 140: 32 strings
# over 11 space/cohort combinations, 24 distinct -- faithfulness 15, stability
# 8, resolution sweep 7, redundancy 2 -- and `grep -c warnings` over
# template.py returned 0.
# ==========================================================================


def _space_with_warnings(**diagnostics):
    return {
        "space_id": "s",
        "verdict": {"level": "ok", "headline": "ok", "reasons": []},
        "protids": [],
        "readable": [],
        "clusters": {},
        "diagnostics": diagnostics,
    }


def test_a_diagnostic_sentence_reaches_the_rendered_page():
    """Nothing asserted this before, and nothing rendered them.

    The strings travel inside whole diagnostic sections that `SUMMARY_SECTIONS`
    copies, so no payload test noticed they arrived, and no template test
    noticed they were dropped.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function diagnosticsBlock(space)" in html
    # Read under BOTH names a diagnostic module uses -- see
    # test_the_censoring_sections_prose_is_read_under_its_own_key.
    assert '"warnings"' in html, "nothing reads a section's warnings"
    assert "entry[prose]" in html, "nothing reads a section's prose at all"
    shell = html[html.index("function panelShell(space)") : html.index("PANELS.scatter = {")]
    assert "diagnosticsBlock(space)" in shell, "the space panel never calls it"


def test_the_disclosure_is_titled_by_what_it_contains_not_called_warnings():
    """4 of 11 combinations carry an all-clear.

    `embedding.warnings()` appends "the layout is faithful at k=15:
    trustworthiness 0.94, continuity 0.96" when nothing fired. A fold-out
    labelled "warnings" containing that sentence contradicts itself, and it
    would do so on more than a third of the page.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function diagnosticsBlock(space)") :][:3000]
    assert "What the diagnostics wrote about this space" in block
    assert "an all-clear" in block, "the all-clear count is not disclosed"
    assert '"cleared"' in block, "an all-clear is not distinguished from a warning"
    summary_line = block[block.index("summary.textContent") :][:260]
    assert (
        "warning" not in summary_line.lower()
    ), "the disclosure calls its contents warnings, which is false on the all-clears"


def test_the_block_reads_the_space_it_is_given_and_never_walks_the_payload():
    """Cohort 0 is duplicated at the payload top level by design.

    A renderer that walked the payload rather than the space handed to it would
    draw the first cohort's sixteen sentences twice. `panelShell` is called per
    space of the ACTIVE cohort, so taking `space.diagnostics` is what keeps the
    count right.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function diagnosticsBlock(space)") :][:3000]
    assert "space.diagnostics" in block
    for walked in ("PAYLOAD.cohorts", "PAYLOAD.spaces", "cohorts.forEach"):
        assert walked not in block, f"the block reaches for {walked}"


def test_a_section_with_no_prose_renders_no_disclosure():
    """An empty triangle promises content and then has none.

    This is the negative half the memo asked for: a space whose diagnostics
    carry sections but no `warnings` at all must render nothing, not an empty
    fold-out.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function diagnosticsBlock(space)") :][:3000]
    assert "if (!total) return null;" in block
    assert "if (!lines.length) return;" in block


def test_the_section_order_is_fixed_and_not_json_insertion_order():
    """`Object.keys` over a payload section would order by however it was written."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "const DIAGNOSTIC_SECTIONS = [" in html
    listed = html[html.index("const DIAGNOSTIC_SECTIONS = [") :][:520]
    for key in ("faithfulness", "stability", "redundancy", "resolution_sweep"):
        assert f'"{key}"' in listed, f"{key} writes prose and is not in the order"


def test_the_banner_prints_the_stability_mean_at_the_diagnostics_own_precision():
    """One measurement, one precision, now that both appear on one screen.

    The banner rounded to two decimals and `diagnostics.stability` writes three,
    so the same number read 0.12 in the verdict and 0.119 in the sentence three
    lines below it. Two roundings of one measurement look like two measurements
    that disagree; the fold-out is what puts them together.

    Asserted structurally rather than against a literal: the check is that the
    two agree, not that either says any particular thing.
    """
    from explorer.payload import space_verdict

    mean = 0.1194
    verdict = space_verdict(
        {"stability": [{"stability_mean": mean, "informative": True}]}, n_proteins=367
    )
    reason = next(r for r in verdict["reasons"] if "coin-flip" in r)
    assert f"{mean:.3f}" in reason, reason
    assert (
        f"{mean:.2f}," not in reason
    ), "the banner is back to two decimals while the diagnostic writes three"


def test_the_cluster_count_is_shown_as_one_point_on_a_sweep():
    """A count printed alone reads as a property of the cohort.

    chymo_A1's fused_late goes 4, 6, 10 and 13 clusters across resolutions 0.25
    to 2.0, and the geometry table printed "10 at resolution 1" with none of
    that beside it. The sweep has shipped in the payload since it was written
    and `grep resolution_sweep` over template.py returned 0 hits.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function sweepNote(space)" in html
    assert "sweep.steps" in html, "nothing reads the swept resolutions"
    assert "sweep.adjacent_ari" in html, "nothing reads the agreement between them"
    cell = html[html.index('label: "clusters"') :][:420]
    assert "sweepNote(s)" in cell, "the clusters cell does not draw the sweep"


def test_the_resolution_the_page_actually_drew_is_marked_in_the_sweep():
    """Four counts with no indication of which one is on screen is a puzzle.

    The mark is matched on the partition's own resolution rather than on a
    position in the list, so a sweep that adds or reorders a step cannot
    silently mark the wrong row.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function sweepNote(space)") :][:1400]
    assert "step.resolution === drawn" in block
    assert "partition || {}).resolution" in block


def test_the_sweep_states_numbers_and_leaves_the_verdict_to_the_diagnostics():
    """The 0.80-ARI judgement is already written, once, by the diagnostics.

    It reaches the reader through the per-space fold-out added in the previous
    commit. Restating it in the table would put one judgement in two places with
    nothing holding them in step -- and the threshold would have to be retyped,
    because the payload does not carry it.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function sweepNote(space)") :][:1400]
    assert "0.8" not in block, "an ARI threshold was retyped into the sweep"
    assert "plateau" not in block, "the sweep restates a judgement it should not"


def test_a_space_with_no_sweep_adds_nothing_to_its_cluster_count():
    """actin_B's `structure` space carries no resolution_sweep at all."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    block = html[html.index("function sweepNote(space)") :][:1400]
    assert 'if (!steps.length) return "";' in block


# ==========================================================================
# The per-space censoring section. Distinct from the cohort-level `censoring`
# key, which `_censoring` builds from ONE matrix -- the resolved structural
# space's. This one is written per space by `diagnose_space` about the matrix
# that space was actually built from, which is how actin_B's `structure_capped`
# can report 82.5% censoring on a cohort whose top-level summary says 0.0.
# ==========================================================================


def test_each_spaces_own_censoring_section_reaches_the_browser():
    """Before this it did not travel at all, and the report said so.

    `SUMMARY_SECTIONS` named four sections and this was not one of them, so the
    coverage table printed "per-space retention at each k: not in this payload".
    """
    from explorer.payload import SUMMARY_SECTIONS

    assert "censoring" in SUMMARY_SECTIONS


def test_the_cluster_pair_table_is_dropped_and_its_reduction_is_kept():
    """`cross_cluster_table` is one row per ORDERED cluster pair.

    It grows as the square of the cluster count -- 49 rows at k=7 on chymo_A1's
    structure space, 64 at k=8 and 36 at k=6 on actin_B -- and no panel draws
    it. `cross_cluster_edge_retention` is its reduction and is six numbers.

    The drop is keyed by NAME, not by an allow-list: a new key added to the
    censoring diagnostic should arrive and be noticed rather than be dropped
    silently by a list nobody updated.
    """
    from explorer.payload import CENSORING_DROPPED_KEYS

    assert CENSORING_DROPPED_KEYS == ("cross_cluster_table",)
    assert "cross_cluster_edge_retention" not in CENSORING_DROPPED_KEYS


def test_the_coverage_row_reports_retention_instead_of_refusing_to():
    """The refusal said the sections were not plumbed into the payload.

    That was true when it was written and is false now, which is the shape this
    branch keeps catching: a refusal that outlives its reason reads as a
    limitation of the method rather than of one commit.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "not plumbed into the payload yet" not in html, "the stale refusal survives"
    assert "PC-003 phase 2" not in html, "the page still cites the ticket as its excuse"
    assert "retentionSummaryCell()" in html


def test_retention_is_never_shown_without_the_cluster_count_it_was_measured_at():
    """A retention figure at k=6 and one at k=10 are not comparable.

    actin_B's capped twin reports at k=6 and its fused_late at k=8 in the same
    table, so the count is not decoration here -- it is what stops the column
    from being read as a ranking.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    cell = html[html.index("function retentionCell(space)") :][:2200]
    assert "retention.n_clusters" in cell
    summary = html[html.index("function retentionSummaryCell()") :][:900]
    assert "retention.n_clusters" in summary


def test_the_ratio_refuses_rather_than_printing_nan():
    """`between_over_within` divides by the within figure.

    A space with no measured within-cluster pair yields Infinity or NaN, and
    "NaN" printed beside two real fractions reads as a measurement that failed
    rather than one that could not be formed.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    cell = html[html.index("function retentionCell(space)") :][:2200]
    assert "Number.isFinite(within)" in cell
    assert "no ratio: nothing was measured within a cluster" in cell


def test_the_per_space_censoring_is_named_apart_from_the_cohorts():
    """`active.censoring` is the cohort's; `spaceCensoring` is one space's.

    On actin_B they disagree completely -- 0.0% against 82.5% -- because the
    cohort summary is built from the resolved structural space's matrix and the
    capped twin is built from another. A call site that took one for the other
    would print a true number about the wrong thing.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function spaceCensoring(space)" in html
    block = html[html.index("function spaceCensoring(space)") :][:200]
    assert "active.censoring" not in block


def test_the_censoring_sections_prose_is_read_under_its_own_key():
    """Four modules write `warnings`; censoring writes `interpretation`.

    Reading only the first name would silently drop every sentence censoring
    writes -- including the only ones on either shipped cohort that describe a
    matrix that really was censored.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert 'const PROSE_KEYS = ["warnings", "interpretation"];' in html
    block = html[html.index("function diagnosticsBlock(space)") :][:2400]
    assert "PROSE_KEYS.forEach" in block
    assert "entry[prose]" in block


def test_the_censoring_all_clear_is_recognised_as_one():
    """ "No censoring problems detected in this matrix." is not a warning.

    It fires on every exhaustive matrix, which is every space carrying this
    section on both cohorts except the capped twin. Left unrecognised it would
    be filed and coloured as a hazard.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "const ALL_CLEAR_PHRASES = [" in html
    listed = html[html.index("const ALL_CLEAR_PHRASES = [") :][:400]
    assert "No censoring problems detected" in listed
    assert "the layout is faithful at k=" in listed


# ==========================================================================
# The axes caption and the Layout dropdown. Commit 128 gave the sentence its own
# node and tagged it with the reducer that produced it; this is the half that
# makes the node move.
# ==========================================================================


def _two_reducer_space(space_id, reducers):
    return {
        "space_id": space_id,
        "verdict": {"level": "ok", "headline": "ok", "reasons": []},
        "protids": ["a", "b"],
        "readable": [True, True],
        "clusters": {},
        "diagnostics": {},
        "embeddings": {reducer: [[0.0, 0.0], [1.0, 1.0]] for reducer in reducers},
        "description": {
            "paragraphs": [f"Axes: {reducer} over N components." for reducer in reducers],
            "axes_by_reducer": {
                reducer: f"Axes: {reducer} over N components." for reducer in reducers
            },
        },
    }


def test_every_reducers_axes_sentence_reaches_the_page_not_only_the_opening_one():
    """The JS can only switch to text the page already carries.

    The node is built with one sentence. If the payload shipped only that one,
    the dropdown would have nothing to switch to and the caption would be stuck
    describing the layout the page happened to open on.
    """
    import json

    from explorer.template import render

    document = {"spaces": [_two_reducer_space("s", ["pca_umap", "pca_tsne"])]}
    html = render(document, plotly_js="", title="t")
    start = html.index("const PAYLOAD = ") + len("const PAYLOAD = ")
    payload, _ = json.JSONDecoder().raw_decode(html, start)
    by_reducer = payload["spaces"][0]["description"]["axes_by_reducer"]
    assert set(by_reducer) == {"pca_umap", "pca_tsne"}
    for reducer, text in by_reducer.items():
        assert reducer in text


def test_the_caption_reports_the_layout_drawn_and_not_the_one_selected():
    """`traceFor` falls back to the first embedding when the selection is absent.

    So a space without `pca_tsne` goes on showing its `pca_umap` picture while
    the dropdown reads pca_tsne. A caption that named the selection would
    describe a reduction that is not on screen — and unlike the typed caption
    this replaced, it would look verified.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function drawnReducer(space)" in html
    block = html[html.index("function drawnReducer(space)") :][:420]
    assert "embeddings[state.reducer] ? state.reducer" in block
    assert "Object.keys(embeddings)[0]" in block
    refresh = html[html.index("function refreshAxes(space)") :][:900]
    assert "drawnReducer(space)" in refresh, "the caption is not read off the drawn layout"


def test_the_caption_is_rewritten_on_every_draw_not_only_at_build():
    """The grid is built once and re-rendered on every interaction."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("function draw()") : html.index("// --- sheets")]
    assert "refreshAxes(space)" in body, "draw() does not refresh the caption"
    assert 'grid.dataset.built = "1"' in body
    assert (
        "refreshAxes" not in body[: body.index('grid.dataset.built = "1"')]
    ), "the refresh is inside the build-once branch, so it would run only once"


def test_the_panel_is_found_by_name_and_never_by_position():
    """Matching on index breaks the moment a space is filtered or reordered."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "panel.dataset.space = space.space_id;" in html
    refresh = html[html.index("function refreshAxes(space)") :][:900]
    assert "p.dataset.space === space.space_id" in refresh


def test_a_layout_with_no_recorded_axes_sentence_refuses_rather_than_going_stale():
    """Leaving the previous layout's text is a caption about the wrong picture."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    refresh = html[html.index("function refreshAxes(space)") :][:900]
    assert "no axes sentence was recorded for this space" in refresh
    assert "reportMissing(" in refresh


def test_the_stability_note_states_where_the_reference_neighbourhood_comes_from():
    """It said the subsample is compared with "its neighbourhood in the full one".

    `diagnostics/stability.py` says the opposite, in bold, and says why it
    matters: **the reference is recomputed inside each subsample**, not taken
    from the full cohort. That is what makes resampling a genuine null -- both
    sides lose the same proteins and promote the same replacement -- and it is
    the reason a replicate with no noise scores exactly 1.0. A reader who
    believed the note would expect a no-noise replicate to score BELOW 1.0 and
    would read every number on the sheet against the wrong baseline.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "in the full one" not in html, "the note still describes a full-cohort reference"
    note = html[html.index("SHEET_PANELS.stability") :][:2400]
    assert "in that same subsample" in note
    assert "exactly 1.0" in note, "the consequence that makes the correction checkable is gone"


def test_the_stability_note_names_faithfulness_rather_than_pointing_off_screen():
    """ "the panel banners above" was wrong twice over.

    They are not above: `renderSheet` hides `#grid` on every sheet but `maps`,
    so on this sheet the banners are not rendered at all. And their first two
    reasons come from this very statistic, so a reader sent there to check the
    drawing would be sent back to the number they were already looking at. The
    measurements that actually judge the drawing are trustworthiness and
    continuity, and the note now names them and where to find them.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    note = html[html.index("SHEET_PANELS.stability") :][:2400]
    assert "panel banners above" not in note
    assert "trustworthiness and continuity" in note
    assert "diagnostics report" in note, "the reader is not told where to look"


def test_the_grid_really_is_hidden_on_every_sheet_but_maps():
    """The claim the note's correction rests on, pinned.

    If the grid were ever shown on another sheet, "the panel banners above"
    would stop being wrong and this correction would need revisiting.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert 'const isMaps = sheetId === "maps";' in html
    assert 'el("grid").style.display = isMaps ? "" : "none";' in html


# ==========================================================================
# The chance level. A stability number is not interpretable until you know what
# ZERO information would have scored, and chance depends on k and on how many
# candidates a replicate offers -- so the same 0.119 means different things on
# different cohorts. This repo has already retracted a figure for having no
# stated denominator.
# ==========================================================================


def _stability(**overrides):
    entry = {
        "k": 15,
        "stability_mean": 0.119,
        "subsample_size": 276,
        "subsample_fraction": 0.75,
        "informative": True,
    }
    entry.update(overrides)
    return {"stability": [entry]}


def test_the_verdict_finally_reads_the_n_proteins_it_has_always_been_given():
    """`space_verdict(diagnostics, n_proteins)` never read the second argument.

    It has been in the signature since the function was written and nothing
    consumed it -- the same shape as FOLLOWUPS #29 and #32, an argument threaded
    through and never honored. Asserted on the REASON text and not the level,
    because the level is deliberately unchanged by this phase.
    """
    from explorer.payload import space_verdict

    small = space_verdict(_stability(subsample_size=None), n_proteins=100)
    large = space_verdict(_stability(subsample_size=None), n_proteins=2465)
    small_reason = next(r for r in small["reasons"] if "coin-flip" in r)
    large_reason = next(r for r in large["reasons"] if "coin-flip" in r)
    assert small_reason != large_reason, "n_proteins still changes nothing"
    assert "n=100" in small_reason and "n=2465" in large_reason
    assert small["level"] == large["level"], "this phase must not move a band"


def test_the_chance_in_the_banner_is_the_librarys_chance():
    """The page and the library cannot be allowed to drift.

    `chance_jaccard` now lives in `diagnostics.stability` and the test fixture
    re-exports it, so there is exactly one definition. This asserts the banner
    prints that one.
    """
    from diagnostics.stability import chance_jaccard
    from explorer.payload import space_verdict

    verdict = space_verdict(_stability(), n_proteins=367)
    reason = next(r for r in verdict["reasons"] if "coin-flip" in r)
    expected = chance_jaccard(275, 15)
    assert f"chance is {expected:.3f}" in reason, reason
    assert f"{0.119 / expected:.0f}x chance" in reason


def test_the_fixtures_chance_is_the_librarys_and_not_a_second_copy():
    """Two definitions of chance would eventually disagree."""
    from diagnostics.stability import chance_jaccard as library
    from stability_cohort import chance_jaccard as fixture

    assert fixture is library


def test_the_pool_is_the_subsamples_and_not_the_cohorts():
    """A replicate chooses among the SUBSAMPLE's candidates, not the cohort's.

    Using the cohort would understate chance on every run, because
    `subsample_fraction` is 0.75 by default -- and understating chance makes
    every stability number look better than it is.
    """
    from explorer.payload import space_verdict

    verdict = space_verdict(_stability(subsample_size=276), n_proteins=367)
    reason = next(r for r in verdict["reasons"] if "coin-flip" in r)
    assert "k=15 of 275 candidates" in reason, reason
    assert "of 366 candidates" not in reason, "the pool is the whole cohort's"


def test_a_verdict_with_no_usable_chance_loses_the_clause_rather_than_printing_nan():
    """A verdict carrying "so this is NaNx chance" is worse than one carrying none."""
    from explorer.payload import space_verdict

    verdict = space_verdict(
        _stability(k=None, subsample_size=None, subsample_fraction=None), n_proteins=0
    )
    for reason in verdict["reasons"]:
        assert "chance" not in reason
        assert "nan" not in reason.lower()


# ==========================================================================
# What the page weighs. The size argument for this page has been made from
# estimates; this is the measurement, and the quadratic term is pinned by
# arithmetic rather than by a snapshot so it cannot drift quietly.
# ==========================================================================


def test_the_accounting_is_complete_and_says_what_it_cannot_attribute():
    """Per-key sizes do NOT sum to the document's length and must not pretend to.

    The separators and quotes between keys belong to no key. Reporting only the
    per-key sizes would leave a silent remainder; `unattributed` states it, so a
    reader can check the accounting without it having to balance to zero.
    """
    import json

    from explorer.payload import payload_bytes

    document = {"a": [1, 2, 3], "b": {"c": "d"}, "e": "f"}
    accounting = payload_bytes(document)
    assert accounting["total"] == len(json.dumps(document, separators=(",", ":")))
    assert accounting["unattributed"] == accounting["total"] - sum(accounting["per_key"].values())
    assert set(accounting["per_key"]) == set(document)


def test_the_quadratic_term_is_pinned_by_arithmetic_and_not_by_a_snapshot():
    """`tm_matrix` is the only part that grows faster than the cohort.

    Its cells are one byte each, base64-encoded, so n**2 bytes become
    4*ceil(n**2/3) characters. Asserting a recorded number instead would pass
    forever after someone changed the encoding; asserting the arithmetic fails
    the moment the relationship changes.
    """
    import base64
    import math

    from explorer.payload import payload_bytes

    n = 40
    cells = bytes(range(256))[: n * n % 256] * (n * n // 256) + bytes(n * n % 256)
    encoded = base64.b64encode(bytes(n * n)).decode()
    assert len(encoded) == 4 * math.ceil(n * n / 3)
    document = {"cohorts": [{"cohort_name": "c", "tm_matrix": {"cells": encoded, "n": n}}]}
    reported = payload_bytes(document)["cohorts"][0]["tm_matrix_bytes"]
    # The envelope is the JSON around the string: quotes, the key names, braces.
    assert len(encoded) < reported < len(encoded) + 64, reported
    assert cells is not None  # the byte pattern is incidental; the length is the claim


def test_the_budget_is_named_and_is_not_a_gate():
    """A build that FAILED on size would refuse the artifact at the moment
    someone most needs to see how big it got. The numbers are printed and the
    judgement is left to a person, so nothing here raises."""
    from explorer import payload as payload_module

    assert payload_module.HARD_BUDGET_BYTES == 20 * 1024**2
    assert payload_module.PREFERRED_BUDGET_BYTES == 10 * 1024**2
    source = open(payload_module.__file__).read()
    assert "raise" not in source.split("HARD_BUDGET_BYTES")[1][:400]


def test_the_size_report_goes_to_stderr_and_never_into_the_page():
    """The page carries no generation timestamp on purpose, so that two runs of
    the same inputs produce the same bytes. A size line inside it would be a
    second thing that varies with the machine."""
    import inspect

    import build_explorer

    source = inspect.getsource(build_explorer._report_size)
    assert source.count("file=sys.stderr") >= 4
    assert "handle.write" not in source


def test_a_multi_cohort_page_ships_the_first_cohort_once():
    """It used to carry cohort 0 twice: inside `cohorts` and at the top level.

    Measured on the two-cohort page before this change, the top-level copy was
    670,709 bytes -- 33.8% of the payload and 11.6% of the file -- against
    cohort 0's own 670,948. It scales as n**2, because the copy carries its own
    `tm_matrix`.

    The renderer never read it. `COHORTS = PAYLOAD.cohorts || [...]` takes the
    array whenever it exists, and the only top-level key the page reads is
    `thresholds`.
    """
    from explorer.payload import payload_bytes

    document = {
        "analysis_name": "a",
        "thresholds": {"coin_flip": 0.3},
        "spaces": [{"space_id": "s"}],
        "tm_matrix": {"cells": "x" * 500},
        "cohorts": [
            {"cohort_name": "a", "spaces": [{"space_id": "s"}], "tm_matrix": {"cells": "x" * 500}},
            {"cohort_name": "b", "spaces": [{"space_id": "t"}], "tm_matrix": {"cells": "y" * 500}},
        ],
    }
    reduced = {k: v for k, v in document.items() if k in ("analysis_name", "thresholds")}
    reduced["cohorts"] = document["cohorts"]
    assert payload_bytes(reduced)["total"] < payload_bytes(document)["total"]
    assert set(reduced) == {"analysis_name", "thresholds", "cohorts"}


def test_the_reducer_keeps_exactly_the_keys_the_page_reads_off_the_top_level():
    """Pinned against the template, not against a list someone remembered.

    If the page ever reads a third top-level key, this fails rather than the
    page silently losing it -- which would show up as a blank map long after the
    commit that caused it.
    """
    import re

    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    read = set(re.findall(r"PAYLOAD\.([a-z_]+)", html))
    # `cohorts` and `cohort_name` belong to the fallback that builds the
    # one-cohort list; `analysis_name` is the page identity.
    assert read <= {"thresholds", "cohorts", "cohort_name", "analysis_name"}, read
    assert "thresholds" in read


def test_the_single_cohort_page_still_carries_its_keys_at_the_top_level():
    """The fallback REBUILDS a one-cohort list from them.

    A page built without `--also-cohort` has no `cohorts` key, so
    `Object.assign({}, PAYLOAD, ...)` is what makes it a cohort at all. Reducing
    that document would produce a page with no spaces and no error.
    """
    import inspect

    import build_explorer
    from explorer.template import render

    source = inspect.getsource(build_explorer.main)
    reducer = source[source.index("if len(cohorts) > 1:") :]
    assert "document = {" in reducer, "the reduction is not inside the multi-cohort branch"
    html = render({"spaces": []}, plotly_js="", title="t")
    assert "PAYLOAD.cohorts || [Object.assign({}, PAYLOAD, {" in html


# ==========================================================================
# The heatmap's n**2 bound. Matt's rule, 2026-08-21: "make it visible, I want
# all the points" -- so the MATRIX is bounded and the MAPS never are.
# ==========================================================================


def test_the_heatmap_bound_is_derived_from_the_page_budget_not_typed():
    """A magic number here would be a second budget nobody could trace."""
    from explorer.payload import (
        PREFERRED_BUDGET_BYTES,
        TM_MATRIX_MAX_BYTES,
        TM_MATRIX_MAX_PROTEINS,
    )

    assert TM_MATRIX_MAX_BYTES == PREFERRED_BUDGET_BYTES // 4
    # n**2 bytes -> 4*ceil(n**2/3) base64 chars, so the bound is sqrt(3/4 * B).
    n = TM_MATRIX_MAX_PROTEINS
    assert 4 * ((n * n + 2) // 3) <= TM_MATRIX_MAX_BYTES
    assert 4 * (((n + 1) ** 2 + 2) // 3) > TM_MATRIX_MAX_BYTES, "the bound is not tight"


def test_the_two_shipped_cohorts_are_comfortably_inside_the_bound():
    """367 and 308 must keep their heatmaps; this is a regression guard on the
    constant, not a claim about those cohorts."""
    from explorer.payload import TM_MATRIX_MAX_PROTEINS

    assert TM_MATRIX_MAX_PROTEINS > 400
    # ...and a full production cohort must be outside it, or the bound does
    # nothing for the case it was added for.
    assert TM_MATRIX_MAX_PROTEINS < 2465


def test_a_cohort_over_the_bound_refuses_the_heatmap_and_says_why():
    """The refusal has to carry BOTH numbers.

    "too big" is not actionable; "2530 against a cap of 1400" is. It must also
    say the maps are unaffected, because a reader who sees one panel withheld
    has no way to know the others are complete.
    """
    import inspect

    from explorer import payload as payload_module

    source = inspect.getsource(payload_module._tm_matrix)
    assert "TM_MATRIX_MAX_PROTEINS" in source
    assert '"refused"' in source
    assert "Every protein is still" in source


def test_the_bound_refuses_and_never_subsamples():
    """A subsampled heatmap is a picture of a cohort nobody chose, and a reader
    cannot tell it from the real one."""
    import inspect

    from explorer import payload as payload_module

    source = inspect.getsource(payload_module._tm_matrix)
    guard = source[source.index("if len(order) > TM_MATRIX_MAX_PROTEINS") :][:900]
    for forbidden in ("random", "sample", "choice", "::", "[:TM_MATRIX"):
        assert forbidden not in guard, f"the bound appears to subsample ({forbidden})"


def test_a_withheld_heatmap_is_a_third_state_and_not_awaiting_a_matrix():
    """ "Awaiting" would send a reader to produce a file that already exists.

    The matrix is on disk and is fine; it is the cohort that is too large to
    ship it. HTML-PLAN §6's rule is that these empty states stay distinct,
    because collapsing them hides which panels are one step from working.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("SHEET_PANELS.heatmap") :][:2200]
    assert "matrix.refused" in body, "the renderer has no withheld branch"
    assert "the heatmap is withheld for this cohort" in body
    # ...and it must come BEFORE the decode, or the decode's own "not n x n"
    # refusal fires first and says something false.
    assert body.index("matrix.refused") < body.index(
        "decodeMatrix"
    ), "the withheld branch is after the decode, so the wrong refusal renders"


# ==========================================================================
# Why the colour-by list is short. Matt, 2026-08-21, on the full-cohort page:
# "why do we only have the 4 things in the 2.5k groups?" The answer was a
# missing file, and the page said nothing.
# ==========================================================================


def test_the_overlay_builder_reports_what_it_could_not_use(tmp_path):
    """Two different silences, one report.

    A missing table and an unusable column both shorten the dropdown, and a
    reader cannot tell them apart -- or even notice -- from the dropdown alone.
    """
    import pandas as pd
    from explorer.payload import _overlays_from_features

    seed = {"path": str(tmp_path / "nope.tsv"), "found": False, "source": "none"}
    overlays, report = _overlays_from_features(None, ["a", "b"], seed)
    assert overlays == {}
    assert report["found"] is False
    assert report["path"].endswith("nope.tsv"), "the path that was missed is not reported"

    table = tmp_path / "t.tsv"
    table.write_text(
        "protid\tgood\tsame\tmany\n" + "\n".join(f"p{i}\t{i % 3}\tX\tv{i}" for i in range(40)),
        encoding="utf-8",
    )
    protids = [f"p{i}" for i in range(40)]
    frame = pd.read_csv(table, sep="\t")
    overlays, report = _overlays_from_features(
        frame, protids, {"path": str(table), "found": True, "source": "aggregate_features"}
    )
    assert report["found"] is True
    assert report["n_columns"] == 3
    dropped = {d["column"]: d["why"] for d in report["dropped"]}
    assert "same" in dropped and "same value" in dropped["same"]
    assert "many" in dropped and "categories" in dropped["many"]
    assert report["n_kept"] == len(overlays)


def test_the_levels_ceiling_is_named_once_and_quoted_from_there():
    """A second copy of 24 would eventually disagree with the filter."""
    import inspect

    from explorer import payload as payload_module

    assert payload_module.MAX_OVERLAY_LEVELS == 24
    source = inspect.getsource(payload_module._overlays_from_features)
    assert source.count("24") == 0, "the ceiling is typed as a literal somewhere"


def test_the_page_says_why_the_colour_by_list_is_short():
    """The dropdown cannot explain its own length."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert "function overlaySourceCell()" in html
    assert '["colour-by vocabulary", overlaySourceCell(), "aggregate_features"]' in html
    body = html[html.index("function overlaySourceCell()") :][:1600]
    assert "no aggregated features table" in body
    # The dropped list is rendered by `droppedNote`, which every branch shares.
    note = html[html.index("function droppedNote(dropped)") :][:900]
    assert (
        "not usable, nearest first:" in note
    ), "the dropped list is not ordered by how close each column came to being usable"


def test_the_overlay_source_is_read_from_the_active_cohort():
    """On a multi-cohort page some cohorts have the table and some do not."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("function overlaySourceCell()") :][:900]
    assert "active.overlay_source" in body
    assert "PAYLOAD.overlay_source" not in html


# ==========================================================================
# PC-035: a run with no aggregated features table is still colourable. Every
# full production run in this project's archive lacks one -- none of them wrote
# a final_results/ directory at all -- so this is the common case, not the
# exotic one.
# ==========================================================================


def test_the_colour_frame_prefers_the_aggregated_table(tmp_path):
    """When aggregate_features did run, nothing changes."""
    from explorer.payload import _colour_frame

    final = tmp_path / "final_results"
    final.mkdir()
    (final / "run_aggregated_features.tsv").write_text(
        "protid\tLeidenCluster\nA\tLC0\n", encoding="utf-8"
    )
    frame, seed = _colour_frame(str(tmp_path), "run")
    assert seed["found"] is True
    assert seed["source"] == "aggregate_features"
    assert list(frame["protid"]) == ["A"]


def test_a_run_with_no_aggregated_table_is_assembled_from_what_it_does_have(tmp_path):
    """uniprot_features LEFT JOIN leiden_features -- the subset colouring needs.

    Both files are already read by this module for other purposes, so the
    fallback adds no new dependency. Without it a 2530-protein cohort offered
    four colours instead of thirteen and could not be coloured by cluster at
    all, which is the first thing anyone asks a map to do.
    """
    from explorer.payload import _colour_frame

    (tmp_path / "protein_features").mkdir()
    (tmp_path / "protein_features" / "uniprot_features.tsv").write_text(
        "protid\tOrganism\nA\tmouse\nB\trat\n", encoding="utf-8"
    )
    (tmp_path / "foldseek_clustering_results").mkdir()
    (tmp_path / "foldseek_clustering_results" / "leiden_features.tsv").write_text(
        "protid\tLeidenCluster\nA\tLC0\nB\tLC1\n", encoding="utf-8"
    )
    frame, seed = _colour_frame(str(tmp_path), "run")
    assert seed["found"] is False, "it must not claim the aggregated table exists"
    assert seed["source"] == "assembled"
    assert "LeidenCluster" in frame.columns, "the cluster column is what this is for"
    assert seed["assembled_from"] == [
        "protein_features/uniprot_features.tsv",
        "foldseek_clustering_results/leiden_features.tsv",
    ]


def test_the_assembly_survives_a_run_with_no_leiden_table(tmp_path):
    """The join is optional; the base table alone still colours."""
    from explorer.payload import _colour_frame

    (tmp_path / "protein_features").mkdir()
    (tmp_path / "protein_features" / "uniprot_features.tsv").write_text(
        "protid\tOrganism\nA\tmouse\n", encoding="utf-8"
    )
    frame, seed = _colour_frame(str(tmp_path), "run")
    assert seed["source"] == "assembled"
    assert seed["assembled_from"] == ["protein_features/uniprot_features.tsv"]
    assert "Organism" in frame.columns


def test_a_run_with_nothing_to_colour_by_says_so_rather_than_raising(tmp_path):
    """An empty output tree is a refusal, not a traceback."""
    from explorer.payload import _colour_frame

    frame, seed = _colour_frame(str(tmp_path), "run")
    assert frame is None
    assert seed["source"] == "none"
    assert seed["found"] is False


def test_an_assembled_vocabulary_says_so_and_names_what_it_cannot_recover():
    """ "No aggregated features table, so the columns are absent" is FALSE once
    the vocabulary is assembled from the base tables -- the columns are there.

    What is genuinely gone is anything `assess_pdbs` produces, and a reader who
    is not told that would conclude pdb_confidence was merely unusable.
    """
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    body = html[html.index("function overlaySourceCell()") :][:2600]
    assert 'source.source === "assembled"' in body
    assert "assembled from" in body
    assert "assess_pdbs" in body
    assert "cannot be recovered this way" in body
    assert 'source.source === "none"' in body, "an empty tree needs its own branch"


def test_the_dropped_note_is_built_once_for_every_branch():
    """Two copies would eventually disagree about the ordering or the cut-off."""
    from explorer.template import render

    html = render({"spaces": []}, plotly_js="", title="t")
    assert html.count("function droppedNote(dropped)") == 1
    assert html.count("nearest first:") == 1, "the note is spelled out more than once"
