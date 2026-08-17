"""Tests for the `coregister` entry point.

Driven through `main()` with an argv, for the reason `test_compute_block.py`
is: the entry point is where the config meets the provider, and a defect that
lives in the caller is invisible to every test that calls the callee. That
lesson cost a wrong normalization in a shipped manifest once already.

The tests build real blocks with `compute_block` rather than writing arrays
into the store by hand. Hand-written fixtures would let the two entry points
disagree about the on-disk layout without anything noticing, which is the
failure this whole layer exists to prevent.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from coregister import main, pair_filename, parse_embeddings, read_embedding
from coregistration import CoregistrationError

SEQUENCES = {
    "P1": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQ",
    "P2": "MKKKKKAAAAAEEEEEGGGGGWWWWWYYYYYFFFFFLLLLL",
    "P3": "MTTTTTSSSSSNNNNNQQQQQHHHHHRRRRRIIIIIVVVVV",
    "P4": "MCCCCCPPPPPGGGGGAAAAADDDDDEEEEEKKKKKLLLLL",
    "P5": "MWWWWWFFFFFYYYYYIIIIILLLLLVVVVVAAAAAGGGGG",
}
DOMAINS = {"P1": "PF00022;", "P2": "PF00022;PF00125;", "P3": "PF00125;", "P4": "", "P5": "PF00022;"}


@pytest.fixture
def run_dir(tmp_path):
    """An output directory with the features table both providers read."""
    features = tmp_path / "output" / "protein_features"
    features.mkdir(parents=True)
    rows = ["protid\tSequence\tPfam\tInterPro"]
    for protid, sequence in SEQUENCES.items():
        rows.append(f"{protid}\t{sequence}\t{DOMAINS[protid]}\t")
    (features / "uniprot_features.tsv").write_text("\n".join(rows) + "\n")
    return tmp_path


def write_config(tmp_path, coregistration, spaces=None):
    """A two-space config over two blocks that need only the features table."""
    config = {
        "blocks": {
            "biophys": {"provider": "biophys"},
            "domains": {"provider": "domains"},
        },
        "spaces": spaces
        or {
            "physicochemistry": {"blocks": ["biophys"], "strategy": "none"},
            "families": {"blocks": ["domains"], "strategy": "none"},
        },
        "coregistration": coregistration,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def build_blocks(monkeypatch, config_path, output_dir, block_ids=("biophys", "domains")):
    import compute_block

    for block_id in block_ids:
        argv = [
            "compute_block.py",
            "--configfile",
            str(config_path),
            "--block-id",
            block_id,
            "--output-dir",
            str(output_dir),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        assert compute_block.main() == 0


def run(monkeypatch, config_path, output_dir, *extra):
    argv = [
        "coregister.py",
        "--configfile",
        str(config_path),
        "--output-dir",
        str(output_dir),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return main()


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_a_two_space_comparison_writes_all_three_kinds_of_file(monkeypatch, run_dir):
    output = run_dir / "output"
    config = write_config(
        run_dir, {"compare": ["physicochemistry", "families"], "reference_space": None, "k": 2}
    )
    build_blocks(monkeypatch, config, output)
    assert run(monkeypatch, config, output) == 0

    directory = output / "coregistration"
    index = json.loads((directory / "index.json").read_text())
    assert index["n_shared"] == 5
    assert index["is_exact"] is True

    summary = pd.read_csv(directory / "summary.tsv", sep="\t")
    assert len(summary) == 1  # one pair
    assert summary.loc[0, "n_proteins"] == 5
    assert 0.0 <= summary.loc[0, "jaccard_mean"] <= 1.0

    per_protein = pd.read_csv(
        directory / pair_filename("physicochemistry", "families"), sep="\t", index_col=0
    )
    assert list(per_protein.index) == list(SEQUENCES)
    assert list(per_protein.columns) == ["neighborhood_jaccard", "rank_correlation"]


def test_every_pair_gets_its_own_file(monkeypatch, run_dir):
    """Three spaces are three pairs, not one comparison against a reference."""
    output = run_dir / "output"
    config = write_config(
        run_dir,
        {"compare": ["physicochemistry", "families", "shape"], "k": 2},
        spaces={
            "physicochemistry": {"blocks": ["biophys"], "strategy": "none"},
            "families": {"blocks": ["domains"], "strategy": "none"},
            "shape": {"blocks": ["biophys"], "strategy": "none"},
        },
    )
    build_blocks(monkeypatch, config, output)
    assert run(monkeypatch, config, output) == 0

    directory = output / "coregistration"
    assert len(pd.read_csv(directory / "summary.tsv", sep="\t")) == 3
    for a, b in [
        ("physicochemistry", "families"),
        ("physicochemistry", "shape"),
        ("families", "shape"),
    ]:
        assert (directory / pair_filename(a, b)).exists()


def test_the_manifest_records_what_the_numbers_rest_on(monkeypatch, run_dir):
    output = run_dir / "output"
    config = write_config(run_dir, {"compare": ["physicochemistry", "families"], "k": 2})
    build_blocks(monkeypatch, config, output)
    run(monkeypatch, config, output)

    manifest = json.loads((output / "coregistration" / "manifest.json").read_text())
    assert manifest["params"]["k"] == 2
    assert manifest["extra"]["geometry_caveats"]
    assert manifest["extra"]["index"]["n_shared"] == 5
    # The blocks the comparison read must be in `inputs`, so a changed block
    # changes the cache key rather than leaving a stale comparison looking fresh.
    assert set(manifest["inputs"]) == {"block:biophys", "block:domains"}


def test_the_shared_index_is_reported_on_stderr(monkeypatch, run_dir, capsys):
    """A reader who never opens index.json still has to be told."""
    output = run_dir / "output"
    config = write_config(run_dir, {"compare": ["physicochemistry", "families"], "k": 2})
    build_blocks(monkeypatch, config, output)
    run(monkeypatch, config, output)
    assert "co-registered over 5 proteins" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the reference space
# ---------------------------------------------------------------------------


def test_the_reference_must_be_one_of_the_compared_spaces(monkeypatch, run_dir):
    """It supplies the shared index's order, so it has to be in the comparison."""
    output = run_dir / "output"
    config = write_config(
        run_dir,
        {
            "compare": ["physicochemistry", "families"],
            "reference_space": "shape",
            "k": 2,
        },
        spaces={
            "physicochemistry": {"blocks": ["biophys"], "strategy": "none"},
            "families": {"blocks": ["domains"], "strategy": "none"},
            "shape": {"blocks": ["biophys"], "strategy": "none"},
        },
    )
    build_blocks(monkeypatch, config, output)
    with pytest.raises(SystemExit, match="not among the spaces being compared"):
        run(monkeypatch, config, output)


def test_fewer_than_two_spaces_is_an_error_that_says_how_to_opt_out(monkeypatch, run_dir):
    output = run_dir / "output"
    config = write_config(run_dir, {"compare": ["physicochemistry"], "k": 2})
    build_blocks(monkeypatch, config, output)
    with pytest.raises(SystemExit, match="at least two"):
        run(monkeypatch, config, output)


# ---------------------------------------------------------------------------
# embeddings, and reading them by label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["structure", "=/path", "  =/path"])
def test_a_malformed_embedding_argument_is_rejected(bad):
    with pytest.raises(SystemExit, match="SPACE_ID=PATH"):
        parse_embeddings([bad])


def test_an_embedding_path_containing_an_equals_sign_is_kept_whole():
    assert parse_embeddings(["s=/tmp/a=b/e.tsv"]) == {"s": "/tmp/a=b/e.tsv"}


def test_an_embedding_is_read_by_protid_and_not_by_position(tmp_path):
    """The embedding covers the space's own protids, which is a superset of the
    shared index whenever any space dropped something. Taking the first N rows
    would silently pair the wrong coordinates with the wrong proteins."""
    path = tmp_path / "embedding.tsv"
    pd.DataFrame(
        {"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0]},
        index=pd.Index(["P3", "P1", "P2"], name="protid"),
    ).to_csv(path, sep="\t")

    values = read_embedding(str(path), ["P1", "P2"])
    assert values.tolist() == [[20.0, 2.0], [30.0, 3.0]]


def test_an_embedding_missing_a_shared_protein_is_an_error(tmp_path):
    path = tmp_path / "embedding.tsv"
    pd.DataFrame({"x": [1.0]}, index=pd.Index(["P1"], name="protid")).to_csv(path, sep="\t")
    with pytest.raises(CoregistrationError, match="disagree about the cohort"):
        read_embedding(str(path), ["P1", "P2"])


def test_supplying_embeddings_brings_the_procrustes_disparity_in(monkeypatch, run_dir):
    output = run_dir / "output"
    config = write_config(run_dir, {"compare": ["physicochemistry", "families"], "k": 2})
    build_blocks(monkeypatch, config, output)

    embeddings = {}
    for i, space_id in enumerate(["physicochemistry", "families"]):
        directory = output / "spaces" / space_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "embedding_pca_umap.tsv"
        coordinates = np.random.RandomState(i).normal(size=(len(SEQUENCES), 2))
        pd.DataFrame(
            coordinates, index=pd.Index(list(SEQUENCES), name="protid"), columns=["x", "y"]
        ).to_csv(path, sep="\t")
        embeddings[space_id] = path

    assert (
        run(
            monkeypatch,
            config,
            output,
            "--embedding",
            f"physicochemistry={embeddings['physicochemistry']}",
            "--embedding",
            f"families={embeddings['families']}",
        )
        == 0
    )
    summary = pd.read_csv(output / "coregistration" / "summary.tsv", sep="\t")
    assert 0.0 <= summary.loc[0, "procrustes_disparity"] <= 1.0


def test_the_disparity_is_empty_rather_than_invented_without_embeddings(monkeypatch, run_dir):
    output = run_dir / "output"
    config = write_config(run_dir, {"compare": ["physicochemistry", "families"], "k": 2})
    build_blocks(monkeypatch, config, output)
    run(monkeypatch, config, output)
    summary = pd.read_csv(output / "coregistration" / "summary.tsv", sep="\t")
    assert pd.isna(summary.loc[0, "procrustes_disparity"])


# ---------------------------------------------------------------------------
# filename collisions
# ---------------------------------------------------------------------------


def test_a_space_id_that_would_collide_with_the_pair_separator_is_refused(monkeypatch, run_dir):
    """`a` vs `b__vs__c` and `a__vs__b` vs `c` both want the same filename, and
    one would overwrite the other while the summary still listed both rows."""
    output = run_dir / "output"
    config = write_config(
        run_dir,
        {"compare": ["physicochemistry", "a__vs__b"], "k": 2},
        spaces={
            "physicochemistry": {"blocks": ["biophys"], "strategy": "none"},
            "a__vs__b": {"blocks": ["domains"], "strategy": "none"},
        },
    )
    build_blocks(monkeypatch, config, output)
    with pytest.raises(SystemExit, match="__vs__"):
        run(monkeypatch, config, output)
