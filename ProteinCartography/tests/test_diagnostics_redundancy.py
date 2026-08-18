#!/usr/bin/env python
"""Block redundancy: against the definition, the fusion fixture, and scipy.

``tests/fusion_cohort.py`` turns out to be exactly the fixture this needs and
it was built for something else, which is worth saying plainly rather than
quietly reusing it. It plants two *crossed* partitions -- `wide` shows `fold`
and is blind to `chemistry`, `narrow` does the reverse, and the crossing is
exact rather than in expectation -- so two of its blocks are independent by
construction and must correlate at zero. It also carries `narrow_rescaled`,
which is `narrow` times 1000, so two of its blocks are the same information in
different units and must correlate at exactly one.

Those are the two ends of the scale this diagnostic measures, both planted, and
neither is a property this file had to arrange.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics.redundancy import (
    REDUNDANT_THRESHOLD,
    BlockPair,
    RedundancyError,
    correlate_distances,
    redundancy,
)
from fusion_cohort import (
    CONSTANT_COLUMN_BLOCK,
    DEGENERATE_BLOCK,
    NARROW_BLOCK,
    NOISE_BLOCK,
    RESCALE_FACTOR,
    RESCALED_BLOCK,
    WIDE_BLOCK,
    fusion_cohort,
)


@pytest.fixture(scope="module")
def cohort():
    return fusion_cohort()


# --- the two planted ends of the scale --------------------------------------


def test_a_block_against_itself_correlates_at_exactly_one(cohort):
    for block_id in (WIDE_BLOCK, NARROW_BLOCK, NOISE_BLOCK):
        pearson, spearman = correlate_distances(cohort.values(block_id), cohort.values(block_id))
        assert pearson == 1.0
        assert spearman == 1.0


def test_a_rescaled_block_is_perfectly_redundant_with_its_original(cohort):
    """`narrow_rescaled` is `narrow` times 1000. Same information, different
    units, and the diagnostic has to say so -- this is the case a reader most
    needs caught, because two blocks in different units look independent in
    every other summary the pipeline prints."""
    pearson, spearman = correlate_distances(
        cohort.values(NARROW_BLOCK), cohort.values(RESCALED_BLOCK)
    )
    assert spearman == 1.0, "Spearman is a pure rank statistic and must be exact"
    assert abs(pearson - 1.0) < 1e-12, f"Pearson off by {abs(pearson - 1.0):.2e}"


def test_blocks_carrying_crossed_partitions_are_not_redundant(cohort):
    """`wide` and `narrow` show different, exactly independent partitions of
    the same proteins, so there is nothing for them to agree about."""
    pearson, spearman = correlate_distances(cohort.values(WIDE_BLOCK), cohort.values(NARROW_BLOCK))
    assert abs(pearson) < 0.05
    assert abs(spearman) < 0.05


def test_noise_is_redundant_with_nothing(cohort):
    for block_id in (WIDE_BLOCK, NARROW_BLOCK, CONSTANT_COLUMN_BLOCK):
        _, spearman = correlate_distances(cohort.values(NOISE_BLOCK), cohort.values(block_id))
        assert abs(spearman) < 0.05


# --- the scale invariance that lets this run before fusion ------------------


@pytest.mark.parametrize("factor", [1e-6, 0.5, 3.0, RESCALE_FACTOR, 1e9])
def test_rescaling_either_block_changes_nothing(cohort, factor):
    """The property that makes this diagnostic reportable before a fusion
    strategy has been chosen: it cannot depend on ADR 0002's unit-mean-distance
    step having run, because a positive rescale moves neither correlation."""
    baseline = correlate_distances(cohort.values(WIDE_BLOCK), cohort.values(NARROW_BLOCK))
    scaled = correlate_distances(
        np.asarray(cohort.values(WIDE_BLOCK)) * factor, cohort.values(NARROW_BLOCK)
    )
    np.testing.assert_allclose(scaled, baseline, rtol=1e-12)


@pytest.mark.parametrize("block_id", [WIDE_BLOCK, NARROW_BLOCK])
def test_translating_a_block_changes_nothing(cohort, block_id):
    """Distances are translation invariant, so the correlation between them is
    too. Worth pinning because a provider that centers its output would
    otherwise look like it changed the answer.

    The shift is a multiple of the block's own scale on purpose. Shifting a
    block by an amount large relative to its spread loses precision in the
    distance computation itself -- ``(a + s) - (b + s)`` cancels -- and that is
    a property of computing distances, not of this diagnostic. Measured: a flat
    17.5 on `narrow`, whose values are of order 0.01, moves the correlation by
    2.6e-10 relative, four digits of cancellation for a shift/scale ratio of
    about 6000. Scaling the shift keeps the test on the invariance.
    """
    scale = cohort.scales[block_id]
    other = NARROW_BLOCK if block_id == WIDE_BLOCK else WIDE_BLOCK
    shifted = np.asarray(cohort.values(block_id)) + 1.75 * scale
    np.testing.assert_allclose(
        correlate_distances(shifted, cohort.values(other)),
        correlate_distances(cohort.values(block_id), cohort.values(other)),
        rtol=1e-12,
    )


def test_the_comparison_is_symmetric(cohort):
    forward = correlate_distances(cohort.values(WIDE_BLOCK), cohort.values(NOISE_BLOCK))
    backward = correlate_distances(cohort.values(NOISE_BLOCK), cohort.values(WIDE_BLOCK))
    assert forward == backward


# --- a block with no geometry -----------------------------------------------


def test_a_degenerate_block_yields_nan_rather_than_zero(cohort):
    """Every row of `degenerate` is identical, so every pairwise distance is
    zero and there is no ordering to correlate. NaN says "unanswerable"; zero
    would say "they disagree", which is a different and false claim."""
    pearson, spearman = correlate_distances(
        cohort.values(DEGENERATE_BLOCK), cohort.values(WIDE_BLOCK)
    )
    assert np.isnan(pearson)
    assert np.isnan(spearman)


def test_a_degenerate_block_is_not_reported_as_redundant(cohort):
    report = redundancy(
        {DEGENERATE_BLOCK: cohort.values(DEGENERATE_BLOCK), WIDE_BLOCK: cohort.values(WIDE_BLOCK)}
    )
    assert not report.pairs[0].redundant
    assert any("no geometry" in note for note in report.warnings())


# --- the guards -------------------------------------------------------------


def test_blocks_over_different_proteins_are_refused(cohort):
    with pytest.raises(RedundancyError, match="aligned to a shared index"):
        correlate_distances(cohort.values(WIDE_BLOCK), cohort.values(NARROW_BLOCK)[:100])


def test_too_few_proteins_is_refused():
    with pytest.raises(RedundancyError, match="too few to mean anything"):
        correlate_distances(np.zeros((2, 3)), np.ones((2, 3)))


def test_a_single_block_is_refused(cohort):
    with pytest.raises(RedundancyError, match="makes no pair"):
        redundancy({WIDE_BLOCK: cohort.values(WIDE_BLOCK)})


def test_mismatched_block_sizes_are_refused(cohort):
    with pytest.raises(RedundancyError, match="different numbers of proteins"):
        redundancy(
            {
                WIDE_BLOCK: cohort.values(WIDE_BLOCK),
                NARROW_BLOCK: np.asarray(cohort.values(NARROW_BLOCK))[:100],
            }
        )


# --- the report -------------------------------------------------------------


@pytest.fixture(scope="module")
def report(cohort):
    return redundancy(
        {
            WIDE_BLOCK: cohort.values(WIDE_BLOCK),
            NARROW_BLOCK: cohort.values(NARROW_BLOCK),
            RESCALED_BLOCK: cohort.values(RESCALED_BLOCK),
        }
    )


def test_every_unordered_pair_appears_once(report):
    assert len(report.pairs) == 3
    seen = {frozenset((p.block_a, p.block_b)) for p in report.pairs}
    assert len(seen) == 3
    assert report.block_ids == (WIDE_BLOCK, NARROW_BLOCK, RESCALED_BLOCK)


def test_the_pair_count_is_the_number_of_protein_pairs(report, cohort):
    n = cohort.n_proteins
    for pair in report.pairs:
        assert pair.n_pairs == n * (n - 1) // 2


def test_the_redundant_pair_is_the_one_that_is_redundant(report):
    assert report.most_redundant.spearman == 1.0
    assert {report.most_redundant.block_a, report.most_redundant.block_b} == {
        NARROW_BLOCK,
        RESCALED_BLOCK,
    }
    assert [p.redundant for p in report.pairs].count(True) == 1


def test_the_warning_says_the_shares_will_be_misleading(report):
    """The whole point of the diagnostic. A contribution share cannot reveal
    redundancy, so the warning has to."""
    joined = " ".join(report.warnings())
    assert "counts one view twice" in joined
    assert "describes the arithmetic rather than the evidence" in joined


def test_a_report_with_no_redundant_pair_says_so(cohort):
    """A diagnostic that is silent when everything is fine leaves a reader
    unable to tell 'checked' from 'not checked'."""
    report = redundancy(
        {WIDE_BLOCK: cohort.values(WIDE_BLOCK), NARROW_BLOCK: cohort.values(NARROW_BLOCK)}
    )
    notes = report.warnings()
    assert len(notes) == 1
    assert "each is contributing something the others do not" in notes[0]


def test_the_threshold_is_what_decides_redundancy():
    just_below = BlockPair("a", "b", 0.5, REDUNDANT_THRESHOLD - 1e-9, 10)
    just_above = BlockPair("a", "b", 0.5, REDUNDANT_THRESHOLD, 10)
    assert not just_below.redundant
    assert just_above.redundant


def test_a_nan_pair_is_never_redundant():
    assert not BlockPair("a", "b", float("nan"), float("nan"), 10).redundant


def test_the_report_serializes_to_plain_json_types(report):
    import json

    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_the_report_frame_has_one_row_per_pair(report):
    frame = report.to_frame()
    assert len(frame) == 3
    assert list(frame.columns) == [
        "block_a",
        "block_b",
        "pearson",
        "spearman",
        "n_pairs",
        "redundant",
    ]


def test_a_pair_can_be_looked_up_in_either_order(report):
    assert report.pair(NARROW_BLOCK, RESCALED_BLOCK) is report.pair(RESCALED_BLOCK, NARROW_BLOCK)
    with pytest.raises(KeyError):
        report.pair(WIDE_BLOCK, "nope")


# --- against scipy ----------------------------------------------------------


@pytest.mark.parametrize(
    "block_a,block_b",
    [
        (WIDE_BLOCK, NARROW_BLOCK),
        (WIDE_BLOCK, NOISE_BLOCK),
        (NARROW_BLOCK, RESCALED_BLOCK),
        (NOISE_BLOCK, CONSTANT_COLUMN_BLOCK),
    ],
)
def test_both_correlations_agree_with_scipy(cohort, block_a, block_b):
    """Relative tolerance, not absolute.

    A correlation of -0.0039 is a perfectly ordinary value here -- two
    independent blocks produce them -- and `abs(a - b) < 1e-12` would pass on
    an implementation that returned zero for it. 1e-12 relative is a real
    check on a number that small.
    """
    stats = pytest.importorskip("scipy.stats")
    from coregistration import pairwise_distances

    def upper(values):
        matrix = pairwise_distances(np.asarray(values, dtype=np.float64))
        return matrix[np.triu_indices(matrix.shape[0], k=1)]

    a, b = upper(cohort.values(block_a)), upper(cohort.values(block_b))
    pearson, spearman = correlate_distances(cohort.values(block_a), cohort.values(block_b))
    assert abs(pearson - stats.pearsonr(a, b)[0]) <= 1e-12 * abs(stats.pearsonr(a, b)[0])
    assert abs(spearman - stats.spearmanr(a, b)[0]) <= 1e-12 * abs(stats.spearmanr(a, b)[0])
