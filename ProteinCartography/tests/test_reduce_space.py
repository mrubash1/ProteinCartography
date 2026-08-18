"""The space entry point, now that it fuses.

`reduce_space.py` had no direct tests before this: it was one strategy over one
block, and the demo exercised it. Fusion gives it three things worth testing on
their own -- reading several blocks, putting them on a shared index, and
refusing the cases where the answer would be wrong rather than absent.

Two levels, deliberately:

* `features_for` and its helpers, which need nothing installed. This is where
  the alignment and the refusals live, and they are the parts that produce a
  *wrong* map rather than no map.
* `main()`, behind `importorskip("sklearn")`, because reducing needs a reducer.
  Group 6 twice shipped code whose unit tests passed and whose entry point had
  never run (`compute_block` did the same before it), so the entry point
  gets driven end to end here as well as through the demo.

The sklearn-gated tests do not run in `cartography_tidy`. Run them somewhere
they do not skip -- see PLAN §0.4.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest
import reduce_space
from config_schema import ConfigError, SpaceConfig
from fusion_cohort import (
    DEGENERATE_BLOCK,
    NARROW_BLOCK,
    NOISE_BLOCK,
    WIDE_BLOCK,
    fusion_cohort,
    write_fusion_cohort,
)
from spaces.store import BlockStore


@pytest.fixture(scope="module")
def cohort():
    # 60 proteins rather than the default 240: this file exercises wiring, and
    # the geometry claims are made in test_fusion.py where the full cohort is.
    return fusion_cohort(n=60)


@pytest.fixture
def store_root(tmp_path, cohort):
    write_fusion_cohort(tmp_path, cohort)
    return tmp_path


def space(space_id, blocks, strategy, **kwargs):
    return SpaceConfig(id=space_id, blocks=tuple(blocks), strategy=strategy, **kwargs)


# ===========================================================================
# reading blocks
# ===========================================================================


def test_a_single_block_space_is_unchanged_by_going_through_fusion(store_root, cohort):
    """The path every existing space takes. It must reduce to the identity.

    Adding fusion routed `strategy: none` through `fuse_none`, which is only
    safe if the values arrive at the reducer bit-identical to before -- the
    store's float32, not a promoted copy.
    """
    fused, index, blocks = reduce_space.features_for(
        space("s", [WIDE_BLOCK], "none"), BlockStore(str(store_root))
    )
    stored = BlockStore(str(store_root)).read_block(WIDE_BLOCK)
    assert index.as_list == cohort.protids
    assert fused.values.dtype == np.float32
    np.testing.assert_array_equal(fused.values, stored.features)
    assert [b.spec.id for b in blocks] == [WIDE_BLOCK]


def test_a_missing_block_is_named(store_root):
    from spaces.store import StoreError

    with pytest.raises(StoreError, match="absent_block"):
        reduce_space.features_for(space("s", ["absent_block"], "none"), BlockStore(str(store_root)))


def test_a_skipped_block_reports_the_providers_reason(store_root):
    directory = Path(BlockStore(str(store_root)).block_dir(NOISE_BLOCK))
    (directory / "SKIPPED.json").write_text(json.dumps({"reason": "needs torch"}))
    with pytest.raises(SystemExit, match="which was skipped: needs torch"):
        reduce_space.features_for(
            space("s", [WIDE_BLOCK, NOISE_BLOCK], "late"), BlockStore(str(store_root))
        )


# ===========================================================================
# the shared index
# ===========================================================================


def test_blocks_over_the_same_proteins_keep_every_one(store_root, cohort):
    blocks = reduce_space.read_blocks(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), BlockStore(str(store_root))
    )
    index, dropped = reduce_space.shared_index(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), blocks
    )
    assert index.as_list == cohort.protids
    assert dropped == {}


def test_blocks_over_different_proteins_are_intersected_and_the_loss_is_named(
    tmp_path, cohort, capsys
):
    """The precondition fusion cannot check for itself.

    Fusion combines row `i` of each block as one protein. If two blocks cover
    slightly different cohorts -- and they do, because they are built by
    different rules from different files -- then row `i` is two different
    proteins and every fused distance is wrong, silently. So the index is the
    intersection, and what it cost is on stderr rather than nowhere.
    """
    store = BlockStore(str(tmp_path))
    full = cohort.block_result(WIDE_BLOCK)
    store.write_block(full)

    from spaces.base import BlockResult

    short = BlockResult(
        spec=cohort.block_spec(NARROW_BLOCK),
        protids=cohort.protids[:50],
        features=cohort.blocks[NARROW_BLOCK][:50],
    )
    store.write_block(short)

    fused, index, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), store
    )
    assert len(index) == 50
    assert index.as_list == cohort.protids[:50]
    assert fused.values.shape == (50, 50)
    assert "measured 10 protein(s) that another block" in capsys.readouterr().err


def test_the_intersection_keeps_the_first_blocks_order(tmp_path, cohort):
    """Order has to come from somewhere nameable, and sorting is not it."""
    store = BlockStore(str(tmp_path))
    reversed_ids = list(reversed(cohort.protids))
    from spaces.base import BlockResult

    store.write_block(
        BlockResult(
            spec=cohort.block_spec(WIDE_BLOCK),
            protids=reversed_ids,
            features=cohort.blocks[WIDE_BLOCK][::-1],
        )
    )
    store.write_block(cohort.block_result(NARROW_BLOCK))
    _, index, _ = reduce_space.features_for(space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), store)
    assert index.as_list == reversed_ids


def test_blocks_sharing_no_proteins_is_an_error(tmp_path, cohort):
    store = BlockStore(str(tmp_path))
    from spaces.base import BlockResult

    store.write_block(cohort.block_result(WIDE_BLOCK))
    store.write_block(
        BlockResult(
            spec=cohort.block_spec(NARROW_BLOCK),
            protids=[f"OTHER{i}" for i in range(len(cohort.protids))],
            features=cohort.blocks[NARROW_BLOCK],
        )
    )
    with pytest.raises(SystemExit, match="share no proteins"):
        reduce_space.features_for(space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), store)


def test_a_reordered_block_is_realigned_not_read_positionally(tmp_path, cohort):
    """ADR 0007, met inside fusion.

    The second block holds the same proteins in a shuffled order. Reading it
    positionally would fuse each protein with a different one, and the result
    would still be a well-formed square matrix of plausible numbers.
    """
    store = BlockStore(str(tmp_path))
    store.write_block(cohort.block_result(WIDE_BLOCK))

    from spaces.base import BlockResult

    order = np.random.RandomState(4).permutation(len(cohort.protids))
    store.write_block(
        BlockResult(
            spec=cohort.block_spec(NARROW_BLOCK),
            protids=[cohort.protids[i] for i in order],
            features=cohort.blocks[NARROW_BLOCK][order],
        )
    )
    shuffled, _, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), store
    )

    aligned_store = BlockStore(str(tmp_path / "aligned"))
    write_fusion_cohort(tmp_path / "aligned", cohort, [WIDE_BLOCK, NARROW_BLOCK])
    aligned, _, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), aligned_store
    )
    np.testing.assert_allclose(shuffled.values, aligned.values, rtol=1e-6, atol=1e-9)


# ===========================================================================
# fusing
# ===========================================================================


@pytest.mark.parametrize(
    ("strategy", "params", "representation"),
    [
        ("early", {}, "features"),
        ("late", {}, "distance_profile"),
        ("graph", {"k": 5}, "affinity_profile"),
    ],
)
def test_each_strategy_reaches_the_entry_point(
    store_root, cohort, strategy, params, representation
):
    fused, index, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], strategy, params=params),
        BlockStore(str(store_root)),
    )
    assert fused.strategy == strategy
    assert fused.representation == representation
    assert fused.values.shape[0] == len(index) == len(cohort.protids)
    assert sum(fused.shares.values()) == pytest.approx(1.0, rel=1e-12)


def test_configured_weights_reach_the_fusion(store_root):
    fused, _, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late", weights={WIDE_BLOCK: 3.0}),
        BlockStore(str(store_root)),
    )
    assert fused.shares[WIDE_BLOCK] == pytest.approx(0.75, rel=1e-9)
    assert fused.shares[NARROW_BLOCK] == pytest.approx(0.25, rel=1e-9)


def test_configured_graph_parameters_reach_the_algorithm(store_root):
    fused, _, _ = reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "graph", params={"k": 4, "iterations": 3}),
        BlockStore(str(store_root)),
    )
    assert fused.params_used["k"] == 4
    assert fused.params_used["iterations"] == 3


def test_a_parameter_the_strategy_will_not_read_is_refused_at_config_time():
    """Caught while parsing, not carried into the manifest looking configured."""
    with pytest.raises(ConfigError, match=r"strategy 'late' does not read \['k'\]"):
        SpaceConfig(id="s", blocks=("a", "b"), strategy="late", params={"k": 5})


def test_a_misspelled_graph_parameter_is_refused_at_config_time():
    with pytest.raises(ConfigError, match="does not read"):
        SpaceConfig(id="s", blocks=("a", "b"), strategy="graph", params={"iteratons": 3})


def test_the_graph_parameters_a_config_may_set_are_exactly_the_ones_it_reads():
    for name in ("k", "mu", "iterations"):
        SpaceConfig(id="s", blocks=("a", "b"), strategy="graph", params={name: 2})


def test_a_degenerate_block_fails_with_the_reason_rather_than_a_nan_map(store_root):
    with pytest.raises(SystemExit, match="every pairwise distance is zero"):
        reduce_space.features_for(
            space("s", [WIDE_BLOCK, DEGENERATE_BLOCK], "late"), BlockStore(str(store_root))
        )


def test_the_shares_are_printed_for_a_fused_space(store_root, capsys):
    """ADR 0002: no fused map exists without its weight vector beside it."""
    reduce_space.features_for(
        space("s", [WIDE_BLOCK, NARROW_BLOCK], "late"), BlockStore(str(store_root))
    )
    err = capsys.readouterr().err
    assert "fusion strategy 'late'" in err
    assert WIDE_BLOCK in err and "realized" in err


def test_nothing_is_printed_for_a_single_block_space(store_root, capsys):
    reduce_space.features_for(space("s", [WIDE_BLOCK], "none"), BlockStore(str(store_root)))
    assert capsys.readouterr().err == ""


# ===========================================================================
# the entry point, end to end
# ===========================================================================


def _config_file(tmp_path, spaces: dict) -> Path:
    """JSON, because that is what the pipeline hands these scripts.

    `envs/analysis.yml` has no PyYAML, so the Snakefile writes the resolved
    config as JSON and `config_io` reads it with the standard library. A test
    that wrote YAML would be testing a path the pipeline never takes -- and
    would skip in exactly the environment where the reducers exist.
    """
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "analysis_name": "fusion-test",
                "input_dir": str(tmp_path),
                "output_dir": str(tmp_path),
                "blocks": {
                    WIDE_BLOCK: {"provider": "tmscore"},
                    NARROW_BLOCK: {"provider": "biophys"},
                },
                "spaces": spaces,
            }
        )
    )
    return path


def _run(monkeypatch, config_path, output_dir, space_id, reducer="pca"):
    monkeypatch.setattr(
        "sys.argv",
        [
            "reduce_space.py",
            "--configfile",
            str(config_path),
            "--space-id",
            space_id,
            "--reducer",
            reducer,
            "--output-dir",
            str(output_dir),
        ],
    )
    return reduce_space.main()


def test_main_writes_an_embedding_and_a_fusion_record(store_root, monkeypatch, cohort):
    pytest.importorskip("sklearn")
    import pandas as pd

    config_path = _config_file(
        store_root,
        {"fused": {"blocks": [WIDE_BLOCK, NARROW_BLOCK], "strategy": "late"}},
    )
    assert _run(monkeypatch, config_path, store_root, "fused") == 0

    space_dir = store_root / "spaces" / "fused"
    frame = pd.read_csv(space_dir / "embedding_pca.tsv", sep="\t", index_col=0)
    assert list(frame.index) == cohort.protids

    manifest = json.loads((space_dir / "manifest_pca.json").read_text())
    record = manifest["extra"]["fusion"]
    assert record["strategy"] == "late"
    assert record["representation"] == "distance_profile"
    shares = {c["block_id"]: c["share"] for c in record["contributions"]}
    assert shares == pytest.approx({WIDE_BLOCK: 0.5, NARROW_BLOCK: 0.5}, rel=1e-9)
    # Every block is an input, not just the first. Before fusion the manifest
    # named one block, which for a fused space would make a changed second block
    # look like a cache hit.
    assert set(manifest["inputs"]) == {f"block:{WIDE_BLOCK}", f"block:{NARROW_BLOCK}"}
    assert manifest["params"]["weights"] == {WIDE_BLOCK: 1.0, NARROW_BLOCK: 1.0}


def test_main_runs_graph_fusion_end_to_end(store_root, monkeypatch):
    pytest.importorskip("sklearn")
    config_path = _config_file(
        store_root,
        {
            "snf": {
                "blocks": [WIDE_BLOCK, NARROW_BLOCK],
                "strategy": "graph",
                "params": {"k": 5, "iterations": 5},
            }
        },
    )
    assert _run(monkeypatch, config_path, store_root, "snf") == 0
    manifest = json.loads((store_root / "spaces" / "snf" / "manifest_pca.json").read_text())
    record = manifest["extra"]["fusion"]
    assert record["params_used"]["k"] == 5
    assert record["params_used"]["algorithm"].startswith("SNF")
    assert all("per_protein_share" in c for c in record["contributions"])


def test_main_still_reduces_a_single_block_space(store_root, monkeypatch, cohort):
    pytest.importorskip("sklearn")
    config_path = _config_file(store_root, {"solo": {"blocks": [WIDE_BLOCK]}})
    assert _run(monkeypatch, config_path, store_root, "solo") == 0
    manifest = json.loads((store_root / "spaces" / "solo" / "manifest_pca.json").read_text())
    assert manifest["extra"]["fusion"]["strategy"] == "none"
    assert manifest["extra"]["fusion"]["contributions"][0]["share"] == 1.0
