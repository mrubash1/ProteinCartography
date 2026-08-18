"""Tests for the censoring diagnostics.

Two of these encode lessons that cost real time to learn:

* ``test_asymmetry_denominators_are_all_reported`` -- the same asymmetry can be
  quoted as 36.6% or 53.6% depending on the denominator, and a max-difference of
  0.99 or 0.67 depending on whether fills are counted as data. Reporting one
  number without its denominator is how those get confused.
* ``test_cross_cluster_retention_detects_the_crisp_cluster_trap`` -- heavy
  censoring makes a map *look* better while making its arrangement meaningless.
"""

import numpy as np
import pytest
from diagnostics.censoring import (
    asymmetry_report,
    censoring_report,
    cross_cluster_edge_retention,
    per_protein_censoring,
)
from matrix_io import CENSORED_FILL_TOKEN, load_labeled_matrix

FILL = CENSORED_FILL_TOKEN


def fmt(v):
    return f"{v:.3E}"


def write(path, row_labels, col_labels, cells):
    lines = ["\t".join(["protid"] + list(col_labels))]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    path.write_text("\n".join(lines) + "\n")
    return path


def load(path, **kwargs):
    return load_labeled_matrix(path, **kwargs)


@pytest.fixture
def labels():
    return [f"P{i}" for i in range(6)]


def dense_matrix(tmp_path, labels, value=0.8):
    cells = [
        [fmt(1.0) if i == j else fmt(value) for j in range(len(labels))] for i in range(len(labels))
    ]
    return load(write(tmp_path / "dense.tsv", labels, labels, cells))


# --------------------------------------------------------------------------
# per-protein
# --------------------------------------------------------------------------


def test_per_protein_censoring_counts(tmp_path, labels):
    n = len(labels)
    cells = []
    for i in range(n):
        # Protein 0 has half its row censored; everyone else is fully measured.
        row = []
        for j in range(n):
            censored = i == 0 and j >= n // 2
            row.append(FILL if censored else (fmt(1.0) if i == j else fmt(0.8)))
        cells.append(row)
    matrix = load(write(tmp_path / "m.tsv", labels, labels, cells))

    table = per_protein_censoring(matrix)
    row0 = table[table["protid"] == "P0"].iloc[0]
    assert row0["censoring_rate"] == pytest.approx(0.5)
    assert row0["n_censored"] == 3
    assert row0["n_measured"] == 3
    assert table[table["protid"] == "P1"].iloc[0]["censoring_rate"] == 0.0


def test_per_protein_on_a_dense_matrix_is_all_zero(tmp_path, labels):
    table = per_protein_censoring(dense_matrix(tmp_path, labels))
    assert (table["censoring_rate"] == 0).all()


# --------------------------------------------------------------------------
# asymmetry -- the denominators lesson
# --------------------------------------------------------------------------


def test_symmetric_matrix_reports_no_asymmetry(tmp_path, labels):
    report = asymmetry_report(dense_matrix(tmp_path, labels))
    assert report["pairs_one_way"] == 0
    assert report["one_way_fraction_of_pairs"] == 0.0
    assert report["max_abs_difference_measured"] == 0.0
    assert report["fraction_exactly_equal_measured"] == 1.0


def test_asymmetry_denominators_are_all_reported(tmp_path):
    """Three defensible numbers, each with its denominator named.

    Constructed so they differ: with one one-way pair out of two reported pairs,
    the per-pair rate is 50% while the per-measured-cell rate is 1/3.
    """
    lbl = ["A", "B", "C"]
    #        A          B          C
    cells = [
        [fmt(1.0), fmt(0.9), FILL],  # A->B measured, A->C censored
        [fmt(0.9), fmt(1.0), FILL],  # B->A measured
        [fmt(0.5), FILL, fmt(1.0)],  # C->A measured  => A/C is one-way
    ]
    matrix = load(write(tmp_path / "a.tsv", lbl, lbl, cells))
    report = asymmetry_report(matrix)

    assert report["pairs_reported"] == 2  # {A,B} and {A,C}
    assert report["pairs_both_directions"] == 1  # {A,B}
    assert report["pairs_one_way"] == 1  # {A,C}
    assert report["one_way_fraction_of_pairs"] == pytest.approx(0.5)
    # 6 measured cells (3 diagonal + A->B, B->A, C->A); 1/6 by that denominator.
    assert report["one_way_fraction_of_measured_cells"] == pytest.approx(1 / 6)
    # The two denominators genuinely disagree -- that is the point.
    assert report["one_way_fraction_of_pairs"] != report["one_way_fraction_of_measured_cells"]


def test_max_difference_excludes_fills(tmp_path):
    """Comparing a measured value against a fill measures the fill, not asymmetry."""
    lbl = ["A", "B", "C"]
    cells = [
        [fmt(1.0), fmt(0.90), fmt(0.99)],
        [fmt(0.80), fmt(1.0), FILL],  # A/B differ by 0.10 -- both measured
        [FILL, FILL, fmt(1.0)],  # A/C is one-way with a 0.99 value
    ]
    matrix = load(write(tmp_path / "d.tsv", lbl, lbl, cells))
    report = asymmetry_report(matrix)

    # If fills counted, the max difference would be 0.99. Only both-measured
    # pairs are compared, so it is 0.10.
    assert report["max_abs_difference_measured"] == pytest.approx(0.10, abs=1e-6)


def test_asymmetry_uses_labels_not_positions(tmp_path):
    """A permuted matrix must give the same answer as an aligned one."""
    lbl = ["A", "B", "C"]
    cells = [
        [fmt(1.0), fmt(0.9), fmt(0.5)],
        [fmt(0.9), fmt(1.0), fmt(0.4)],
        [fmt(0.5), fmt(0.4), fmt(1.0)],
    ]
    aligned = load(write(tmp_path / "aligned.tsv", lbl, lbl, cells))

    perm = ["C", "A", "B"]
    pcells = [[cells[i][lbl.index(c)] for c in perm] for i in range(3)]
    permuted = load(write(tmp_path / "perm.tsv", lbl, perm, pcells), require_alignment=False)

    assert asymmetry_report(aligned) == asymmetry_report(permuted)


def test_single_protein_matrix_does_not_divide_by_zero(tmp_path):
    matrix = load(write(tmp_path / "one.tsv", ["A"], ["A"], [[fmt(1.0)]]))
    report = asymmetry_report(matrix)
    assert report["n_proteins"] == 1
    assert report["pairs_reported"] == 0


# --------------------------------------------------------------------------
# cross-cluster edge retention -- the crisp-cluster trap
# --------------------------------------------------------------------------


def test_cross_cluster_retention_detects_the_crisp_cluster_trap(tmp_path):
    """Within-cluster edges survive the cut; between-cluster ones do not.

    This is the failure mode that makes a heavily censored map look *more*
    convincing: the clusters stay tight while their positions relative to each
    other are set by whatever few cross edges happened to survive.
    """
    a = [f"A{i}" for i in range(4)]
    b = [f"B{i}" for i in range(4)]
    lbl = a + b
    clusters = {p: ("A" if p.startswith("A") else "B") for p in lbl}

    cells = []
    for i, pi in enumerate(lbl):
        row = []
        for j, pj in enumerate(lbl):
            if i == j:
                row.append(fmt(1.0))
            elif clusters[pi] == clusters[pj]:
                row.append(fmt(0.9))  # within: measured
            else:
                row.append(FILL)  # between: censored away
            _ = j
        cells.append(row)
    matrix = load(write(tmp_path / "clust.tsv", lbl, lbl, cells))

    table, summary = cross_cluster_edge_retention(matrix, clusters)

    assert summary["within_retention"] == pytest.approx(1.0)
    assert summary["between_retention"] == pytest.approx(0.0)
    assert summary["between_over_within"] == pytest.approx(0.0)
    assert summary["n_clusters"] == 2

    within_rows = table[table["within"]]
    assert (within_rows["retention"] == 1.0).all()

    # And the report must say so in words, not only in numbers.
    notes = censoring_report(matrix, clusters)["interpretation"]
    assert any("arrangement of" in note for note in notes)


def test_uniform_retention_gives_ratio_one(tmp_path):
    lbl = [f"P{i}" for i in range(4)]
    clusters = {"P0": "A", "P1": "A", "P2": "B", "P3": "B"}
    matrix = dense_matrix(tmp_path, lbl)
    _table, summary = cross_cluster_edge_retention(matrix, clusters)
    assert summary["within_retention"] == pytest.approx(1.0)
    assert summary["between_retention"] == pytest.approx(1.0)
    assert summary["between_over_within"] == pytest.approx(1.0)


def test_self_pairs_are_excluded_from_retention(tmp_path):
    """A protein against itself is not a relationship and must not inflate within."""
    lbl = ["P0", "P1"]
    clusters = {"P0": "A", "P1": "A"}
    cells = [[fmt(1.0), FILL], [FILL, fmt(1.0)]]
    matrix = load(write(tmp_path / "s.tsv", lbl, lbl, cells))
    _table, summary = cross_cluster_edge_retention(matrix, clusters)
    # Both off-diagonal cells are censored, so retention is 0 despite the
    # diagonal being fully "measured".
    assert summary["within_possible"] == 2
    assert summary["within_retention"] == pytest.approx(0.0)


def test_missing_cluster_assignment_raises(tmp_path, labels):
    matrix = dense_matrix(tmp_path, labels)
    with pytest.raises(ValueError, match="no cluster assignment"):
        cross_cluster_edge_retention(matrix, {"P0": "A"})


# --------------------------------------------------------------------------
# the full report
# --------------------------------------------------------------------------


def test_report_on_a_clean_matrix_says_so(tmp_path, labels):
    report = censoring_report(dense_matrix(tmp_path, labels))
    assert report["matrix"]["n_censored"] == 0
    assert report["interpretation"] == ["No censoring problems detected in this matrix."]


def test_report_warns_about_heavy_censoring_and_a_cap(tmp_path):
    n, cap = 20, 5
    lbl = [f"P{i:03d}" for i in range(n)]
    rng = np.random.RandomState(0)
    cells = []
    for _ in range(n):
        keep = set(rng.choice(n, size=cap, replace=False).tolist())
        cells.append([fmt(0.8) if j in keep else FILL for j in range(n)])
    matrix = load(write(tmp_path / "cap.tsv", lbl, lbl, cells))

    report = censoring_report(matrix)
    text = " ".join(report["interpretation"])
    assert "per-query cap of 5" in text
    assert "missing rather than measured" in text
    assert report["matrix"]["cap_detected"]


def test_report_flags_a_measured_zero_as_invalidating_the_shortcut(tmp_path):
    """If a real 0.000E+00 exists, `values == 0` is no longer a censoring test."""
    lbl = ["A", "B"]
    cells = [[fmt(1.0), "0.000E+00"], [FILL, fmt(1.0)]]
    matrix = load(write(tmp_path / "mz.tsv", lbl, lbl, cells))
    text = " ".join(censoring_report(matrix)["interpretation"])
    assert "NOT a valid censoring test" in text


def test_report_includes_per_protein_summary(tmp_path, labels):
    report = censoring_report(dense_matrix(tmp_path, labels))
    assert set(report["per_protein"]) == {"min", "median", "max", "mean"}
