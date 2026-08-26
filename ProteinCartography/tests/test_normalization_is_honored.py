"""`spec.normalization` must change the geometry, not just the manifest.

FOLLOWUPS #32: the field was recorded on every block and applied by nothing.
`reduce_space` fed `block.features` straight into fusion, so `zscore_within` and
`unit_mean_distance` were declared in every manifest on disk and honored
nowhere.

The consequence was visible on the shipped page and was read as a property of
the proteins: `physicochemistry` on the production actin cohort reported
"made of: isoelectric_point 97.1%", because pI runs 4 to 12 while gravy runs
about -2 to 2 and charge per residue sits near zero, and the columns entered the
distance raw. The fixture here is that shape, deliberately.
"""

import numpy as np
import pytest
from spaces.base import NORMALIZATIONS
from spaces.normalize import mean_pairwise_distance, normalize_block


@pytest.fixture
def incomparable_scales():
    """Three columns with the spread of pI, gravy and charge per residue."""
    rng = np.random.default_rng(20260825)
    n = 60
    return np.column_stack(
        [
            rng.uniform(4.0, 12.0, n),  # isoelectric point, the real range
            rng.uniform(-1.0, 0.5, n),  # GRAVY, the real range for proteins
            rng.uniform(-0.05, 0.05, n),  # charge per residue
        ]
    )


def _column_share(values):
    """Each column's share of the total variance the distance sees."""
    var = values.var(axis=0)
    return var / var.sum()


def test_the_raw_matrix_is_dominated_by_its_widest_column(incomparable_scales):
    """The defect, stated as a measurement rather than as a complaint.

    This is what the page was reporting. It is not a bug in this test; it is the
    input the fix has to change.
    """
    share = _column_share(incomparable_scales)
    assert share[0] > 0.95, f"expected pI to dominate the raw matrix, got {share}"
    # The production actin cohort reported 97.1% for isoelectric_point. This
    # fixture lands near it because the ranges are the real ones: a first draft
    # gave GRAVY a +/-2.0 spread it does not have and produced 80%, which
    # understated the defect.


def test_zscore_within_equalises_the_columns(incomparable_scales):
    share = _column_share(normalize_block(incomparable_scales, "zscore_within"))
    assert np.allclose(
        share, 1 / 3, atol=0.02
    ), f"after zscore_within no column should dominate, got {share}"


def test_zscore_within_is_per_column_not_per_block(incomparable_scales):
    """The distinction that kept FOLLOWUPS #32 alive.

    Fusion already normalizes a block to unit mean distance, and that is a
    single scalar, so it cannot change the relative scale of the columns. If it
    could, this defect would never have reached the page.
    """
    scaled = normalize_block(incomparable_scales, "unit_mean_distance")
    assert np.allclose(_column_share(scaled), _column_share(incomparable_scales)), (
        "a block-level scalar must not change column shares -- that is why "
        "unit_mean_distance cannot fix the pI problem and zscore_within can"
    )


def test_a_constant_column_becomes_zero_rather_than_nan(incomparable_scales):
    """A descriptor identical across the cohort has an undefined z-score.

    Zero is the honest answer: it says the column separates nobody. NaN would
    propagate into every distance and take the whole map with it.
    """
    values = np.column_stack([incomparable_scales, np.full(len(incomparable_scales), 7.0)])
    out = normalize_block(values, "zscore_within")
    assert np.isfinite(out).all()
    assert (out[:, -1] == 0).all()


def test_unit_mean_distance_arrives_at_unit_mean_distance(incomparable_scales):
    out = normalize_block(incomparable_scales, "unit_mean_distance")
    assert mean_pairwise_distance(out) == pytest.approx(1.0, rel=1e-9)


def test_mean_pairwise_distance_agrees_with_the_naive_computation(incomparable_scales):
    """The chunked implementation exists because scipy is not importable here.

    Chunking is where an off-by-one hides, so it is checked against the direct
    computation, and at a chunk size small enough to force several chunks.
    """
    values = incomparable_scales
    n = len(values)
    naive = [
        float(np.linalg.norm(values[i] - values[j])) for i in range(n) for j in range(i + 1, n)
    ]
    assert mean_pairwise_distance(values, chunk=7) == pytest.approx(sum(naive) / len(naive))


def test_none_is_the_identity_and_every_vocabulary_member_is_implemented():
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert np.array_equal(normalize_block(values, "none"), values)
    for rule in NORMALIZATIONS:
        out = normalize_block(values, rule)
        assert out.shape == values.shape, f"{rule} changed the shape"
        assert np.isfinite(out).all(), f"{rule} produced a non-finite value"


def test_an_unknown_rule_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize_block(np.zeros((3, 2)), "whatever-the-config-said")
