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


def test_a_block_declaring_a_metric_the_reducer_cannot_honor_is_refused(tmp_path):
    """FOLLOWUPS #29, the sibling of #32.

    `spec.metric` was validated against the vocabulary, written into every
    manifest and consulted by nothing, so a block declaring `cosine` was reduced
    with a euclidean PCA and carried a manifest that said otherwise. Refusing is
    the honest half of the pair: #32 was fixed by honoring the declaration,
    #29 by admitting it cannot be.
    """
    import reduce_space
    from spaces.base import BlockResult, BlockSpec

    class _Store:
        def block_dir(self, block_id):
            return str(tmp_path)

        def read_block(self, block_id):
            spec = BlockSpec(
                id=block_id,
                kind="features",
                fusable=True,
                metric="cosine",
                normalization="none",
                provider="biophys",
            )
            return BlockResult(
                protids=["a", "b"],
                features=np.zeros((2, 2), dtype=np.float32),
                spec=spec,
            )

    class _Space:
        id = "s"
        blocks = ("chem",)

    with pytest.raises(SystemExit, match="declares metric 'cosine'"):
        reduce_space.read_blocks(_Space(), _Store())


def test_mean_pairwise_distance_agrees_with_a_naive_reference():
    """The Gram identity `|a-b|^2 = |a|^2 + |b|^2 - 2a.b` is an exact rewrite,
    so this must hold to machine precision rather than approximately.

    It is worth pinning because the identity has a known failure mode -- it
    subtracts two large numbers to get a small one, so it loses relative
    accuracy when points are close together and far from the origin. The last
    case below is that shape deliberately.
    """
    import itertools

    def naive(a):
        a = np.asarray(a, dtype=np.float64)
        pairs = list(itertools.combinations(range(len(a)), 2))
        return sum(float(np.sqrt(((a[i] - a[j]) ** 2).sum())) for i, j in pairs) / len(pairs)

    rng = np.random.default_rng(20260826)
    for shape in [(5, 3), (11, 4), (60, 7), (37, 37)]:
        values = rng.random(shape).astype(np.float32)
        assert mean_pairwise_distance(values) == pytest.approx(naive(values), rel=1e-9)

    # Across a chunk boundary, which is where an off-by-one in the triangle
    # mask would hide: 700 rows at chunk=128 is six chunks, the last partial.
    values = rng.random((700, 9)).astype(np.float32)
    assert mean_pairwise_distance(values, chunk=128) == pytest.approx(naive(values), rel=1e-9)

    # Tight cluster far from the origin -- the identity's worst case, and the
    # reason the implementation centres before it does anything else. Without
    # centring this case fails at 1e-5; with it, 1e-12 holds.
    values = (rng.random((40, 6)) * 1e-3 + 500.0).astype(np.float64)
    assert mean_pairwise_distance(values) == pytest.approx(naive(values), rel=1e-12)


def test_mean_pairwise_distance_does_not_materialise_a_cubic_intermediate(tmp_path):
    """The production-scale bug that a 367-protein cohort cannot show.

    `tmscore` with `representation: profile` -- the pipeline's own
    representation, where a protein IS its row of the similarity matrix -- gives
    a block whose column count EQUALS its row count. The obvious way to write
    this function, `block[:, None, :] - values[None, :, :]`, then allocates
    `chunk * N**2`. Measured on the old implementation: 66 MB peak at N=200,
    325 MB at N=400, 938 MB at N=600, and **29.9 GB at the production cohort
    size of 2,703** -- which is not slow, it is dead.

    Nothing exercised it until PC-011, because `unit_mean_distance` was declared
    by that block and applied by nothing. Honoring the field is what put a
    cubic allocation on the production path, and PC-041 phase 1 at N=367 ran
    PC-041 phase 1 at N=367 ran green through it at 395 MB. A fixture at or
    below 500 proteins cannot exercise this bug, which is the same shape as the
    PCA determinism one: `svd_solver="auto"` is reproducible at N=400 because
    it picks `full` there, and only stops being so above 500.

    Measured in a SUBPROCESS because `ru_maxrss` is a high-water mark: read in
    this process it would report the whole suite's peak and prove nothing.
    """
    import subprocess
    import sys

    pytest.importorskip("numpy")
    if sys.platform.startswith("win"):
        pytest.skip("ru_maxrss is not available on Windows")

    n = 1200
    # What the old implementation would have needed, stated so a reader can see
    # the bound below is not arbitrary.
    cubic_bytes = min(512, n) * n * n * 8
    assert cubic_bytes > 5e9, "the fixture stopped being big enough to prove anything"

    script = tmp_path / "peak.py"
    script.write_text(
        "import resource, sys, numpy as np\n"
        f"sys.path.insert(0, {str(_source_root())!r})\n"
        "from spaces.normalize import mean_pairwise_distance\n"
        f"a = np.random.default_rng(0).random(({n}, {n})).astype(np.float32)\n"
        "mean_pairwise_distance(a)\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        # macOS reports bytes, Linux kilobytes.
        "print(peak if sys.platform == 'darwin' else peak * 1024)\n"
    )
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert finished.returncode == 0, finished.stderr
    peak_bytes = int(finished.stdout.strip())

    # The array itself is 1200**2 float64 = 11.5 MB, plus a (chunk, N) working
    # set of 4.9 MB and the interpreter. A gigabyte is far above that and far
    # below the 5.9 GB the cubic form needs at this N.
    assert peak_bytes < 1_000_000_000, (
        f"peak RSS {peak_bytes / 1e6:.0f} MB at N={n}: this is allocating per-pair "
        f"differences again, which needs {cubic_bytes / 1e9:.1f} GB at this size and "
        "29.9 GB on the production cohort"
    )


def _source_root() -> str:
    """The `ProteinCartography/` directory, for the subprocess's `sys.path`."""
    import pathlib

    return str(pathlib.Path(__file__).resolve().parent.parent)
