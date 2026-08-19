"""Tests for the metricity diagnostic.

Two kinds, and the reference points are the load-bearing half.

A negative-mass fraction is uninterpretable on its own -- 44.6% is only alarming
if you know what a well-behaved matrix scores. So the tests pin both ends: a
genuinely Euclidean distance matrix must come back at essentially zero, and a
matrix with the same marginals but no consistent geometry must not. Without
those two, an implementation that returned a plausible-looking constant would
pass every shape assertion here -- which is exactly how a fused map once
scored a separation of 1.005 while carrying no structure at all.

The other half is the convention. This statistic has appeared in the project's
notes as four different percentages for one name, all of them correctly
computed under four unstated conventions. The tests below require that the two
convention axes measurably change the answer and that both are echoed into the
result, so a bare number cannot be obtained from this module at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics.metricity import DEFAULT_DENOMINATOR, metricity_report


def euclidean_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


@pytest.fixture
def euclidean():
    """40 points in 5-D. By construction these distances are exactly Euclidean."""
    rng = np.random.RandomState(0)
    return euclidean_distances(rng.normal(size=(40, 5)))


@pytest.fixture
def non_euclidean():
    """A symmetric, zero-diagonal, positive matrix that is not a Euclidean
    distance matrix: uniform noise satisfies every surface property and no
    triangle inequality worth the name."""
    rng = np.random.RandomState(1)
    values = rng.uniform(0.1, 1.0, size=(40, 40))
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    return values


# --------------------------------------------------------------------------
# reference points: what the number means at each end
# --------------------------------------------------------------------------


def test_a_euclidean_matrix_has_essentially_no_negative_mass(euclidean):
    report = metricity_report(euclidean)
    assert report["negative_mass_fraction"] < 1e-9, (
        "a matrix of true Euclidean distances double-centres to a positive "
        "semi-definite Gram matrix, so anything above rounding error means the "
        "computation is wrong, not that the data is interesting"
    )


def test_a_euclidean_matrix_spends_its_mass_in_its_true_dimension(euclidean):
    """The spectrum is the other half of the answer and it is checkable here:
    40 points drawn in 5-D have exactly 5 non-negligible positive eigenvalues."""
    top = metricity_report(euclidean, top_k=10)["top_positive_eigenvalues"]
    assert sum(v > 1e-6 * top[0] for v in top) == 5


def test_a_non_euclidean_matrix_is_far_from_zero(non_euclidean):
    report = metricity_report(non_euclidean)
    assert report["negative_mass_fraction"] > 0.05, (
        "the control must separate from the Euclidean reference, or the "
        "diagnostic cannot distinguish the two cases it exists to distinguish"
    )


def test_the_two_reference_points_are_orders_of_magnitude_apart(euclidean, non_euclidean):
    lo = metricity_report(euclidean)["negative_mass_fraction"]
    hi = metricity_report(non_euclidean)["negative_mass_fraction"]
    assert hi > 1000 * max(lo, 1e-12)


def test_shuffling_a_euclidean_matrix_destroys_its_metricity(euclidean):
    """A permutation control. Shuffling the off-diagonal entries keeps every
    marginal -- same values, same symmetry, same zero diagonal -- and discards
    the geometry. A diagnostic that scored the two the same would be reading
    the distribution rather than the structure."""
    rng = np.random.RandomState(2)
    n = euclidean.shape[0]
    iu = np.triu_indices(n, 1)
    values = euclidean[iu].copy()
    rng.shuffle(values)
    shuffled = np.zeros_like(euclidean)
    shuffled[iu] = values
    shuffled = shuffled + shuffled.T

    assert metricity_report(euclidean)["negative_mass_fraction"] < 1e-9
    assert metricity_report(shuffled)["negative_mass_fraction"] > 0.05


# --------------------------------------------------------------------------
# the convention travels with the number
# --------------------------------------------------------------------------


def test_the_two_convention_axes_change_the_answer(non_euclidean):
    """Four cells, four answers. This is the whole reason both axes are
    arguments rather than choices buried in the implementation."""
    answers = {
        (sq, den): metricity_report(non_euclidean, square_first=sq, denominator=den)[
            "negative_mass_fraction"
        ]
        for sq in (True, False)
        for den in ("positive", "total")
    }
    assert len(set(answers.values())) == 4, (
        f"the conventions must be distinguishable, got {answers}. If two cells "
        "agreed, a figure quoted under one could be silently read as the other."
    )
    # neg/total is bounded by neg/pos for the same centring, always.
    for sq in (True, False):
        assert answers[(sq, "total")] < answers[(sq, "positive")]


def test_the_report_states_its_own_convention(non_euclidean):
    report = metricity_report(non_euclidean, square_first=True, denominator="positive")
    assert report["convention"]["double_centered"] == "squared_distances"
    assert report["convention"]["denominator"] == "positive"
    assert "negative eigenvalues" in report["convention"]["formula"]

    raw = metricity_report(non_euclidean, square_first=False, denominator="total")
    assert raw["convention"]["double_centered"] == "raw_distances"
    assert raw["convention"]["denominator"] == "total"


def test_the_default_is_the_classical_scaling_convention(non_euclidean):
    default = metricity_report(non_euclidean)
    explicit = metricity_report(non_euclidean, square_first=True, denominator="positive")
    assert default == explicit
    assert DEFAULT_DENOMINATOR == "positive"


def test_an_unknown_denominator_is_refused(non_euclidean):
    with pytest.raises(ValueError, match="denominator"):
        metricity_report(non_euclidean, denominator="all")


def test_it_reports_and_does_not_gate(non_euclidean):
    """A verdict field that quietly acquired a value would be a threshold
    derived from one cohort, which is the mistake this diagnostic exists to
    avoid repeating.

    The note has to say *why* it refuses, and the reason has now been wrong
    twice. "No second family yet" was wrong because the archive had one all
    along. Its replacement -- "it moves more between two runs of one query than
    between families" -- was wrong because repeated runs of one query are
    bit-identical, so that variance is exactly zero; what moved was which subset
    an analyst selected. The refusal now rests on the transform instead, which
    is a property of the code and cannot be invalidated by another cohort.
    """
    report = metricity_report(non_euclidean)
    assert report["verdict"] is None
    assert "do not compare" in report["verdict_note"].lower()
    assert "#59" in report["verdict_note"]
    assert "#49" in report["verdict_note"]


def test_the_cohort_that_produced_the_number_travels_with_it(non_euclidean):
    report = metricity_report(non_euclidean, censoring_rate=0.605, n_proteins=2530)
    assert report["cohort"] == {"n_proteins": 2530, "censoring_rate": 0.605}


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_condensed_vector_is_refused():
    with pytest.raises(ValueError, match="square"):
        metricity_report(np.array([0.1, 0.2, 0.3]))


def test_an_asymmetric_matrix_is_refused():
    """TM-score is normalized per query, so the raw matrix is asymmetric and
    silently averaging it here would hide which rule was applied."""
    values = np.array([[0.0, 0.2, 0.4], [0.9, 0.0, 0.5], [0.4, 0.5, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        metricity_report(values)


def test_a_cohort_too_small_for_a_spectrum_reports_rather_than_raises():
    """Two proteins have no spectrum, and that is a fact to record, not a crash.

    This is called unconditionally from the tmscore block and nothing gates on
    the value, so raising failed a whole block build over a provenance number
    no consumer reads. The refusal survives as a stated `verdict_note`.
    """
    report = metricity_report(np.zeros((2, 2)))
    assert report["negative_mass_fraction"] is None
    assert report["n"] == 2
    assert "at least 3" in report["verdict_note"]
    assert report["top_positive_eigenvalues"] == []


# --------------------------------------------------------------------------
# the wiring: the writer puts it where a reader can find it
# --------------------------------------------------------------------------


def test_the_direct_block_records_metricity_in_its_manifest(tmp_path):
    """Drive the real provider and read the real manifest.

    No literal from `metricity_report` appears on the reading side beyond the
    key itself, and the block is built by `TMScoreProvider.compute` rather than
    by a hand-written fixture -- the lesson of commit 68 is that a fixture built
    with the reader's spelling proves only that the reader agrees with the test.
    """
    pytest.importorskip("scipy")
    from blocks.tmscore import PipelineContext, TMScoreProvider

    labels = ["P1", "P2", "P3", "P4", "P5"]
    rng = np.random.RandomState(3)
    tm = rng.uniform(0.2, 0.9, size=(5, 5))
    tm = (tm + tm.T) / 2.0
    np.fill_diagonal(tm, 1.0)

    d = tmp_path / "foldseek_clustering_results"
    d.mkdir(parents=True)
    lines = ["\t".join(["protid"] + labels)]
    for i, label in enumerate(labels):
        lines.append("\t".join([label] + [f"{v:.3E}" for v in tm[i]]))
    (d / "all_by_all_tmscore_pivoted.tsv").write_text("\n".join(lines) + "\n")

    result = TMScoreProvider().compute(
        PipelineContext(output_dir=str(tmp_path)),
        {"representation": "direct", "alignment_verified": True},
    )
    report = result.manifest["derived"]["metricity"]
    assert 0.0 <= report["negative_mass_fraction"] <= 1.0
    assert report["n"] == 5
    assert report["convention"]["double_centered"] == "squared_distances"
    # The cohort context has to come from the block, not from a default.
    assert report["cohort"]["n_proteins"] == 5
    assert report["cohort"]["censoring_rate"] == 0.0
    # And it must NOT be in `extra`, which is folded into the cache key. A block
    # whose key depended on a figure derived from its own output could never be
    # matched by a caller asking "do I need to build this?".
    assert "metricity" not in result.manifest["extra"], (
        "metricity landed in `extra`, so it is now part of `cache_key` and the "
        "block can never cache-hit across processes"
    )


def test_metricity_survives_the_trip_to_disk(tmp_path):
    """The store, not the return value, is what a later reader sees.

    `write_block` prefers the provider's manifest but rebuilds from the spec
    when it cannot find one, and it copies `derived` before adding its own keys.
    Either path could drop this silently, and the failure would look like a
    manifest that simply has no metricity section -- which is indistinguishable
    from a block that never computed one. So read it back off disk.
    """
    pytest.importorskip("scipy")
    from blocks.tmscore import PipelineContext, TMScoreProvider
    from spaces.store import BlockStore

    labels = ["P1", "P2", "P3", "P4", "P5"]
    rng = np.random.RandomState(4)
    tm = rng.uniform(0.2, 0.9, size=(5, 5))
    tm = (tm + tm.T) / 2.0
    np.fill_diagonal(tm, 1.0)

    d = tmp_path / "foldseek_clustering_results"
    d.mkdir(parents=True)
    lines = ["\t".join(["protid"] + labels)]
    for i, label in enumerate(labels):
        lines.append("\t".join([label] + [f"{v:.3E}" for v in tm[i]]))
    (d / "all_by_all_tmscore_pivoted.tsv").write_text("\n".join(lines) + "\n")

    result = TMScoreProvider().compute(
        PipelineContext(output_dir=str(tmp_path)),
        {"representation": "direct", "alignment_verified": True},
    )
    store = BlockStore(str(tmp_path / "blocks"))
    store.write_block(result)

    on_disk = store.read_block("tmscore").manifest["derived"]["metricity"]
    assert on_disk == result.manifest["derived"]["metricity"], (
        "the metricity report did not survive the store round trip, so every "
        "block on disk would carry no metricity section and nothing would raise"
    )
    # And the store's own derived keys are still there -- this must add, not replace.
    assert "values_digest" in store.read_block("tmscore").manifest["derived"]


def test_metricity_does_not_enter_the_cache_key(tmp_path):
    """`derived` is excluded from `cache_key` by design, and it has to stay that
    way: a key that folded in a figure computed from the output could only be
    reproduced by a caller who had already built the block.
    """
    pytest.importorskip("scipy")
    from spaces.manifest import Manifest

    manifest = Manifest.build("block", "tmscore", provider="tmscore", protids=["A", "B"])
    before = manifest.cache_key
    manifest.derived["metricity"] = {"negative_mass_fraction": 0.446}
    assert manifest.cache_key == before


def test_the_profile_block_does_not_claim_a_metricity(tmp_path):
    """`profile` produces features, not distances. Reporting a metricity for it
    would be reporting a property of a matrix it never forms."""
    from blocks.tmscore import PipelineContext, TMScoreProvider

    labels = ["P1", "P2", "P3", "P4"]
    d = tmp_path / "foldseek_clustering_results"
    d.mkdir(parents=True)
    lines = ["\t".join(["protid"] + labels)]
    for i, label in enumerate(labels):
        row = [f"{1.0 if i == j else 0.7:.3E}" for j in range(4)]
        lines.append("\t".join([label] + row))
    (d / "all_by_all_tmscore_pivoted.tsv").write_text("\n".join(lines) + "\n")

    result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), {})
    assert "metricity" not in result.manifest.get("derived", {})


def test_one_minus_similarity_manufactures_negative_mass_on_euclidean_data():
    """The pipeline's own transform is not the Euclidean one, and this pins it.

    `blocks/tmscore.py` forms distances as `1 - TM`. Schoenberg's criterion says
    that for a positive semi-definite `S` with a unit diagonal it is
    `sqrt(2(1 - S))` that is Euclidean, not `1 - S`. So the fraction reported on
    any `1 - TM` matrix carries a large offset that is a property of the
    transform rather than of the data.

    This is a characterization test: it asserts the artifact is present and
    large, so that anyone who changes the transform sees this fail and reads
    FOLLOWUPS #59 before deciding whether the change is wanted. The existing
    Euclidean test above passes an already-metric distance matrix, which is a
    shape the pipeline never constructs, so it cannot catch this.
    """
    rng = np.random.RandomState(0)
    points = rng.normal(size=(120, 8))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    similarity = points @ points.T
    np.fill_diagonal(similarity, 1.0)
    # Cosine similarity of real vectors is a Gram matrix, so it is PSD by
    # construction. Assert it rather than trusting it.
    assert np.linalg.eigvalsh((similarity + similarity.T) / 2).min() > -1e-10

    affine = 1.0 - similarity
    np.fill_diagonal(affine, 0.0)
    affine = (affine + affine.T) / 2

    schoenberg = np.sqrt(np.maximum(2.0 * (1.0 - similarity), 0.0))
    np.fill_diagonal(schoenberg, 0.0)
    schoenberg = (schoenberg + schoenberg.T) / 2

    manufactured = metricity_report(affine)["negative_mass_fraction"]
    genuine = metricity_report(schoenberg)["negative_mass_fraction"]

    assert genuine < 1e-9, (
        "sqrt(2(1-S)) of a PSD similarity is Euclidean, so this must be rounding "
        f"error; got {genuine}. If this fails the computation is wrong."
    )
    assert manufactured > 0.30, (
        "the point of this test is that `1 - S` does NOT return zero on Euclidean "
        f"data; got {manufactured}, which would mean the artifact has gone away "
        "and the docstring's warning is now wrong"
    )
    assert manufactured > 1e6 * genuine, (
        f"the transform contributes {manufactured} where the data contributes "
        f"{genuine}: the reported fraction is mostly the convention, which is why "
        "docs/FOLLOWUPS.md #59 says not to read it as a property of the data"
    )
