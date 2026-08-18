"""Tests for the TM-score block.

The interesting cases are about the two representations behaving differently on
a permuted matrix, because that difference is the entire reason `direct` is
gated: `profile` is invariant to a consistent column permutation and `direct` is
not.
"""

import numpy as np
import pytest
from blocks.tmscore import PipelineContext, TMScoreProvider, validate_params
from matrix_io import CENSORED_FILL_TOKEN

FILL = CENSORED_FILL_TOKEN


def fmt(v):
    return f"{v:.3E}"


def write_matrix(root, row_labels, col_labels, cells):
    d = root / "foldseek_clustering_results"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(["protid"] + list(col_labels))]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    (d / "all_by_all_tmscore_pivoted.tsv").write_text("\n".join(lines) + "\n")
    return root


def symmetric_cells(labels, off=0.7):
    return [
        [fmt(1.0) if i == j else fmt(off) for j in range(len(labels))] for i in range(len(labels))
    ]


@pytest.fixture
def labels():
    return ["P1", "P2", "P3", "P4"]


@pytest.fixture
def ctx(tmp_path, labels):
    write_matrix(tmp_path, labels, labels, symmetric_cells(labels))
    return PipelineContext(output_dir=str(tmp_path))


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------


def test_default_representation_is_profile():
    assert validate_params({})["representation"] == "profile"


def test_unknown_representation_rejected():
    with pytest.raises(ValueError, match="representation"):
        validate_params({"representation": "sideways"})


def test_direct_requires_verified_alignment():
    with pytest.raises(ValueError, match="alignment_verified"):
        validate_params({"representation": "direct"})


def test_direct_requires_a_valid_symmetrization():
    with pytest.raises(ValueError, match="symmetrization"):
        validate_params(
            {
                "representation": "direct",
                "alignment_verified": True,
                "symmetrization": "whatever",
            }
        )


def test_direct_defaults_to_mean_symmetrization():
    params = validate_params({"representation": "direct", "alignment_verified": True})
    assert params["symmetrization"] == "mean"


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def test_profile_block_shape_and_spec(ctx, labels):
    result = TMScoreProvider().compute(ctx, {})
    assert result.protids == labels
    assert result.features.shape == (4, 4)
    assert result.features.dtype == np.float32
    assert result.spec.kind == "features"
    assert result.spec.metric == "euclidean"
    assert result.spec.fusable


def test_profile_carries_the_censoring_channel(tmp_path, labels):
    cells = symmetric_cells(labels)
    cells[0][2] = FILL
    cells[2][0] = FILL
    write_matrix(tmp_path, labels, labels, cells)
    result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), {})

    assert "censored" in result.channels
    assert result.channels["censored"].sum() == 2
    assert result.censoring_rate == pytest.approx(2 / 16)


def test_profile_repairs_a_permuted_matrix_and_warns(tmp_path, labels):
    """`profile` tolerates a permutation, but should not accept one silently."""
    import warnings

    permuted = [labels[2], labels[0], labels[3], labels[1]]
    cells = [[fmt(1.0) if r == c else fmt(0.7) for c in permuted] for r in labels]
    write_matrix(tmp_path, labels, permuted, cells)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), {})

    assert any("PR #106" in str(w.message) for w in caught)
    # After repair the label diagonal is 1.0, as it should be.
    np.testing.assert_allclose(np.diag(result.features), 1.0, rtol=0, atol=1e-6)


def test_profile_distances_are_invariant_to_a_column_permutation(tmp_path, labels):
    """The property that makes `profile` safe, asserted rather than assumed.

    Euclidean distance between rows does not change when the columns are
    consistently reordered. This is why the shipped UMAP is not garbage on
    pre-#106 output, and why `profile` is the default.
    """
    import warnings

    rng = np.random.RandomState(0)
    base = rng.uniform(0.2, 0.9, size=(4, 4))
    np.fill_diagonal(base, 1.0)

    write_matrix(tmp_path / "a", labels, labels, [[fmt(v) for v in row] for row in base])
    aligned = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path / "a")), {})

    order = [2, 0, 3, 1]
    permuted_labels = [labels[i] for i in order]
    permuted_cells = [[fmt(base[r][i]) for i in order] for r in range(4)]
    write_matrix(tmp_path / "b", labels, permuted_labels, permuted_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repaired = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path / "b")), {})

    def pairwise(x):
        diff = x[:, None, :] - x[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    np.testing.assert_allclose(
        pairwise(aligned.features), pairwise(repaired.features), rtol=1e-5, atol=1e-5
    )


def test_missing_matrix_names_the_rule_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="foldseek_clustering"):
        TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), {})


def test_manifest_records_the_input_digest_and_censoring(ctx):
    result = TMScoreProvider().compute(ctx, {})
    manifest = result.manifest
    assert manifest["provider"] == "tmscore"
    assert manifest["inputs"]["similarity_matrix"].startswith("sha256:")
    assert "censoring" in manifest["extra"]
    assert manifest["n_proteins"] == 4


def test_a_changed_matrix_changes_the_cache_key(tmp_path, labels):
    write_matrix(tmp_path / "a", labels, labels, symmetric_cells(labels, off=0.7))
    write_matrix(tmp_path / "b", labels, labels, symmetric_cells(labels, off=0.6))
    a = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path / "a")), {})
    b = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path / "b")), {})
    assert a.manifest["cache_key"] != b.manifest["cache_key"]


# --------------------------------------------------------------------------
# direct -- needs scipy, which the bare test environment does not have
# --------------------------------------------------------------------------


@pytest.fixture
def direct_params():
    return {"representation": "direct", "alignment_verified": True}


def test_direct_block_is_pairwise_and_records_its_symmetrization(ctx, direct_params):
    pytest.importorskip("scipy")
    result = TMScoreProvider().compute(ctx, direct_params)
    assert result.spec.kind == "pairwise"
    assert result.spec.metric == "precomputed"
    assert result.spec.symmetrization == "mean"
    assert result.spec.distance_metric == "1 - TM-score"
    # Condensed upper triangle for 4 proteins.
    assert result.distances.shape == (6,)


def test_direct_distance_is_one_minus_tmscore(tmp_path, labels, direct_params):
    pytest.importorskip("scipy")
    write_matrix(tmp_path, labels, labels, symmetric_cells(labels, off=0.7))
    result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), direct_params)
    np.testing.assert_allclose(result.distances, 0.3, rtol=1e-5, atol=1e-5)


def test_direct_symmetrization_rules_differ(tmp_path, labels):
    """The three rules must actually produce different numbers.

    If they did not, recording the choice would be theatre.
    """
    pytest.importorskip("scipy")
    cells = symmetric_cells(labels)
    cells[0][1] = fmt(0.9)
    cells[1][0] = fmt(0.5)
    write_matrix(tmp_path, labels, labels, cells)
    ctx = PipelineContext(output_dir=str(tmp_path))

    results = {}
    for rule in ("min", "max", "mean"):
        block = TMScoreProvider().compute(
            ctx,
            {"representation": "direct", "alignment_verified": True, "symmetrization": rule},
        )
        results[rule] = float(block.distances[0])

    # 1 - max(0.9, 0.5) = 0.1 ; 1 - min = 0.5 ; 1 - mean = 0.3
    assert results["max"] == pytest.approx(0.1, abs=1e-5)
    assert results["min"] == pytest.approx(0.5, abs=1e-5)
    assert results["mean"] == pytest.approx(0.3, abs=1e-5)


def test_direct_censors_a_pair_if_either_direction_is_missing(tmp_path, labels, direct_params):
    pytest.importorskip("scipy")
    cells = symmetric_cells(labels)
    cells[1][0] = FILL  # P2 -> P1 missing; P1 -> P2 present
    write_matrix(tmp_path, labels, labels, cells)
    result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), direct_params)
    # The {P1, P2} pair is the first entry of the condensed triangle.
    assert bool(result.channels["censored"][0])


def test_direct_refuses_a_permuted_matrix(tmp_path, labels, direct_params):
    """No repair for `direct`: its config gate promised the matrix was verified."""
    pytest.importorskip("scipy")
    from matrix_io import MatrixAlignmentError

    permuted = [labels[2], labels[0], labels[3], labels[1]]
    write_matrix(tmp_path, labels, permuted, symmetric_cells(labels))
    with pytest.raises(MatrixAlignmentError):
        TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), direct_params)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_provider_registers_as_a_builtin():
    from blocks.tmscore import register
    from spaces.registry import BLOCK_GROUP, get_provider, list_providers

    register()
    try:
        assert "tmscore" in list_providers(BLOCK_GROUP)
        assert isinstance(get_provider(BLOCK_GROUP, "tmscore"), TMScoreProvider)
    finally:
        from spaces import registry

        registry._BUILTINS[BLOCK_GROUP].pop("tmscore", None)


def test_is_available_does_not_depend_on_the_matrix_existing():
    """A survey before the pipeline has run must not report the provider broken."""
    available, reason = TMScoreProvider().is_available()
    assert available
    assert reason == ""
