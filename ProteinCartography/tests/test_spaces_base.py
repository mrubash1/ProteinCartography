"""Tests for the Block / Space core types.

Most of these are about the specification refusing to describe something
incoherent, because an incoherent spec that validates becomes a wrong map.
"""

import numpy as np
import pytest
from spaces.base import (
    BlockResult,
    BlockSpec,
    BlockSpecError,
    NotFusableError,
    SpaceSpec,
)


def features_spec(**overrides):
    kwargs = {
        "id": "biophys",
        "kind": "features",
        "fusable": True,
        "metric": "euclidean",
        "normalization": "zscore_within",
        "provider": "biophys",
    }
    kwargs.update(overrides)
    return BlockSpec(**kwargs)


def pairwise_spec(**overrides):
    kwargs = {
        "id": "tmscore",
        "kind": "pairwise",
        "fusable": True,
        "metric": "precomputed",
        "normalization": "unit_mean_distance",
        "provider": "tmscore",
        "symmetrization": "mean",
    }
    kwargs.update(overrides)
    return BlockSpec(**kwargs)


# --------------------------------------------------------------------------
# BlockSpec
# --------------------------------------------------------------------------


def test_valid_specs_construct():
    assert features_spec().kind == "features"
    assert pairwise_spec().metric == "precomputed"


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"kind": "wat"}, "kind"),
        ({"metric": "manhattan"}, "metric"),
        ({"normalization": "minmax"}, "normalization"),
        ({"id": ""}, "non-empty string"),
    ],
)
def test_invalid_field_values_are_rejected(overrides, fragment):
    with pytest.raises(BlockSpecError, match=fragment):
        features_spec(**overrides)


def test_pairwise_block_must_use_precomputed_metric():
    with pytest.raises(BlockSpecError, match="must be 'precomputed'"):
        pairwise_spec(metric="euclidean")


def test_features_block_may_not_claim_precomputed():
    with pytest.raises(BlockSpecError, match="kind must be 'pairwise'"):
        features_spec(metric="precomputed")


def test_non_fusable_block_must_carry_a_reason():
    """ADR 0003: the reason is what stops the flag being removed as an obstacle."""
    with pytest.raises(BlockSpecError) as excinfo:
        features_spec(id="taxonomy", fusable=False)
    message = str(excinfo.value)
    assert "not_fusable_reason" in message
    assert "ADR 0003" in message


def test_require_fusable_passes_for_a_fusable_block():
    assert features_spec().require_fusable("multiview") is None


def test_require_fusable_raises_with_the_reason():
    spec = features_spec(
        id="taxonomy",
        fusable=False,
        not_fusable_reason="fusing taxonomy makes cluster claims circular",
    )
    with pytest.raises(NotFusableError) as excinfo:
        spec.require_fusable("multiview")
    message = str(excinfo.value)
    assert "cannot be fused into space 'multiview'" in message
    assert "circular" in message
    # It must also say what the user *can* do.
    assert "overlay" in message


# --------------------------------------------------------------------------
# BlockResult
# --------------------------------------------------------------------------


def test_features_result_round_trips():
    result = BlockResult(spec=features_spec(), protids=["A", "B"], features=np.zeros((2, 5)))
    assert result.n_proteins == 2
    assert result.n_features == 5
    # None, not 0.0: this block declared no censoring channel, so it has made no
    # claim about censoring, and the manifest must not record a non-claim as a
    # measurement.
    assert result.censoring_rate is None


def test_pairwise_result_takes_condensed_distances():
    # 3 proteins -> 3 pairs in the condensed upper triangle.
    result = BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3))
    assert result.n_proteins == 3
    assert result.n_features is None


def test_result_needs_either_features_or_distances():
    with pytest.raises(BlockSpecError, match="either features or distances"):
        BlockResult(spec=features_spec(), protids=["A"])


def test_result_rejects_both():
    with pytest.raises(BlockSpecError, match="not both"):
        BlockResult(
            spec=features_spec(),
            protids=["A", "B"],
            features=np.zeros((2, 2)),
            distances=np.zeros(1),
        )


def test_feature_row_count_must_match_protids():
    with pytest.raises(BlockSpecError, match="3 protids but features has 2 rows"):
        BlockResult(spec=features_spec(), protids=["A", "B", "C"], features=np.zeros((2, 4)))


def test_square_distance_matrix_is_rejected_with_advice():
    """A square matrix where a condensed one belongs is a very easy mistake."""
    with pytest.raises(BlockSpecError) as excinfo:
        BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros((3, 3)))
    message = str(excinfo.value)
    assert "condensed upper triangle" in message
    assert "squareform" in message


def test_mask_shape_must_match():
    with pytest.raises(BlockSpecError, match="does not match the data shape"):
        BlockResult(
            spec=features_spec(),
            protids=["A", "B"],
            features=np.zeros((2, 3)),
            mask=np.zeros((2, 2), dtype=bool),
        )


def test_mask_must_be_boolean():
    """A float mask invites arithmetic that treats 'missing' as a number."""
    with pytest.raises(BlockSpecError, match="must be boolean"):
        BlockResult(
            spec=features_spec(),
            protids=["A", "B"],
            features=np.zeros((2, 3)),
            mask=np.zeros((2, 3), dtype=float),
        )


def test_censoring_rate_from_mask():
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, :2] = True
    result = BlockResult(
        spec=features_spec(), protids=["A", "B"], features=np.zeros((2, 4)), mask=mask
    )
    assert result.censoring_rate == pytest.approx(0.25)


# --------------------------------------------------------------------------
# SpaceSpec
# --------------------------------------------------------------------------


def test_single_block_space():
    space = SpaceSpec(id="structure", blocks=("tmscore",))
    assert not space.is_multiblock
    assert space.weight_for("tmscore") == 1.0


def test_strategy_none_rejects_multiple_blocks():
    with pytest.raises(BlockSpecError) as excinfo:
        SpaceSpec(id="s", blocks=("a", "b"), strategy="none")
    assert "co-registered spaces" in str(excinfo.value)


def test_multiblock_space_with_late_fusion():
    space = SpaceSpec(id="s", blocks=("a", "b"), strategy="late", weights={"a": 1.0, "b": 0.85})
    assert space.is_multiblock
    assert space.weight_for("b") == 0.85


def test_duplicate_blocks_rejected():
    with pytest.raises(BlockSpecError, match="more than once"):
        SpaceSpec(id="s", blocks=("a", "a"), strategy="late")


def test_weight_for_unknown_block_rejected():
    with pytest.raises(BlockSpecError, match="not in this space"):
        SpaceSpec(id="s", blocks=("a",), weights={"b": 1.0})


def test_negative_weight_rejected():
    with pytest.raises(BlockSpecError, match="no meaning"):
        SpaceSpec(id="s", blocks=("a", "b"), strategy="late", weights={"a": -1.0})


def test_boolean_weight_rejected():
    """True is an int in Python; accepting it as a weight is never intended."""
    with pytest.raises(BlockSpecError, match="expected a number"):
        SpaceSpec(id="s", blocks=("a", "b"), strategy="late", weights={"a": True})


def test_unknown_strategy_rejected():
    with pytest.raises(BlockSpecError, match="strategy"):
        SpaceSpec(id="s", blocks=("a",), strategy="magic")


def test_empty_block_list_rejected():
    with pytest.raises(BlockSpecError, match="at least one block"):
        SpaceSpec(id="s", blocks=())
