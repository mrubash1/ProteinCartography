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

import pytest
from explorer.payload import space_verdict

pytest.importorskip("pandas")


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
