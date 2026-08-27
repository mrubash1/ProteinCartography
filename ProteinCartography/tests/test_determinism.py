#!/usr/bin/env python
"""Phase 5 item 9: two runs at the same versions produce the same map.

A standing guard rather than a diagnostic that appears in a report, because
what it protects against is not a property of anyone's data. Nondeterminism
here comes from the reducers, and it is silent: two runs of the same pipeline
on the same input produce two maps that both look right, and nothing in either
one says the other exists.

**The fixture size is load-bearing, and this file is where that is proved.**
scikit-learn's PCA picks its SVD solver from the input shape when
``svd_solver="auto"``, and the randomized solver it picks above 500 samples is
not reproducible without a seed. At N=400 `auto` resolves to `full` and the
same code is deterministic, so a fixture at or below 500 proteins passes this
guard no matter how the reducer is configured -- the check would exist, run,
and be incapable of failing.

:func:`test_the_guard_can_actually_fail_at_this_fixture_size` is the negative
control for exactly that. It configures the reducer the way this repository
deliberately does not, and requires the two runs to *differ*. If that test ever
starts passing trivially, the rest of this file has stopped meaning anything.

The pipeline's own configuration is ``svd_solver="full"`` with a fixed
``random_state``, which is reproducible at every N. That is what the other
tests here assert, at a size where it is a real assertion.
"""

from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import parity
import pytest

#: Above the 500-sample boundary where scikit-learn's `auto` solver switches to
#: the randomized path. See the module docstring; this is not a round number
#: chosen for comfort.
FIXTURE_N = 750

REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("determinism")
    return parity.synthetic_matrix(root / "all_by_all_tmscore_pivoted.tsv", n=FIXTURE_N)


@pytest.fixture(scope="module")
def features(matrix):
    """The matrix as a feature block, read label-safely."""
    from matrix_io import load_labeled_matrix

    loaded = load_labeled_matrix(str(matrix))
    return np.asarray(loaded.values, dtype=np.float64), list(loaded.protids)


# --- the reducer core -------------------------------------------------------


@pytest.mark.parametrize("reducer", ["pca", "umap", "tsne"])
def test_the_shared_reducer_core_repeats_itself_exactly(features, reducer):
    """Bit for bit, not to a tolerance. Every space in this work goes through
    these three functions, so a drift here moves every map at once."""
    pytest.importorskip("sklearn")
    if reducer == "umap":
        pytest.importorskip("umap")
    from spaces.reducers.core import reduce_pca, reduce_tsne, reduce_umap

    run = {"pca": reduce_pca, "umap": reduce_umap, "tsne": reduce_tsne}[reducer]
    values, protids = features
    first = run(values, protids, random_state=123456)
    second = run(values, protids, random_state=123456)
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    assert first.params_used == second.params_used


def test_the_guard_can_actually_fail_at_this_fixture_size(features):
    """The negative control, and the reason FIXTURE_N is 750 rather than 200.

    Configured the way the pipeline deliberately is not -- `auto` solver, no
    seed -- two runs of PCA on this fixture must disagree. If they agree, the
    fixture has fallen below the boundary where scikit-learn switches to the
    randomized solver and every other test in this file has quietly become
    unfalsifiable.
    """
    pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA

    values, _ = features
    first = PCA(n_components=30, svd_solver="auto", random_state=None).fit_transform(values)
    second = PCA(n_components=30, svd_solver="auto", random_state=None).fit_transform(values)
    assert not np.array_equal(first, second), (
        f"PCA is reproducible without a seed at N={values.shape[0]}, so this fixture "
        "cannot exercise the nondeterminism the rest of this file guards against. "
        "It is below the 500-sample boundary, or scikit-learn changed its solver "
        "selection. Either way the guard is now decoration."
    )


def test_the_pipeline_s_own_pca_is_reproducible_without_a_seed_being_lucky(features):
    """The positive half of the same comparison: `full` plus a fixed seed is
    reproducible where `auto` unseeded is not, on the same data in the same
    process."""
    pytest.importorskip("sklearn")
    from spaces.reducers.core import reduce_pca

    values, protids = features
    runs = [reduce_pca(values, protids, n_components=30).coordinates for _ in range(3)]
    for later in runs[1:]:
        np.testing.assert_array_equal(runs[0], later)


# --- the entry points, end to end -------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["pca_umap", "pca_tsne"])
def test_the_legacy_reducer_writes_identical_bytes_twice(matrix, tmp_path, mode):
    """Two subprocesses, not two calls. A module-level cache or a hash seed
    that happened to be stable within one interpreter would hide the failure
    this is looking for, and the pipeline runs each rule in its own process.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("umap")
    first = parity.run_reducer(REPO_ROOT, matrix, tmp_path / "first", mode=mode)
    second = parity.run_reducer(REPO_ROOT, matrix, tmp_path / "second", mode=mode)

    produced = sorted(p.name for p in first.iterdir() if p.suffix == ".tsv")
    assert produced, "the reducer wrote no tables"
    for name in produced:
        assert _digest(first / name) == _digest(second / name), f"{name} differs between runs"


@pytest.mark.slow
def test_a_space_reduces_to_identical_coordinates_twice(features, tmp_path):
    """The new path, through `reduce_space`'s own entry point.

    Group 8a's fourth sighting of "a passing test suite is not evidence the
    entry point works" was a signature change nineteen unit tests could not
    see. This one runs the entry point.
    """
    pytest.importorskip("sklearn")
    import json
    import subprocess
    import sys

    from spaces.base import BlockResult, BlockSpec
    from spaces.store import BlockStore

    values, protids = features
    output_dir = tmp_path / "output"
    store = BlockStore(str(output_dir))
    store.write_block(
        BlockResult(
            spec=BlockSpec(
                id="tmscore",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="determinism_fixture",
            ),
            protids=protids,
            features=values,
        )
    )

    # JSON, not YAML: `envs/analysis.yml` carries no PyYAML, and JSON is what
    # the Snakefile hands these scripts anyway via `multispace_config`.
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "blocks": {"tmscore": {"provider": "tmscore"}},
                "spaces": {
                    "structure": {
                        "blocks": ["tmscore"],
                        "strategy": "none",
                        "reducers": ["pca"],
                    }
                },
            }
        )
    )

    digests = []
    for attempt in ("first", "second"):
        target = tmp_path / attempt
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "ProteinCartography" / "reduce_space.py"),
                "--configfile",
                str(config_path),
                "--space-id",
                "structure",
                "--reducer",
                "pca",
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        written = output_dir / "spaces" / "structure" / "embedding_pca.tsv"
        target.mkdir()
        (target / "embedding_pca.tsv").write_bytes(written.read_bytes())
        digests.append(_digest(target / "embedding_pca.tsv"))

    assert digests[0] == digests[1], "reduce_space produced a different embedding on re-run"
