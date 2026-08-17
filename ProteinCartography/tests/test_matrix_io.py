"""Tests for the labeled-matrix loader.

The permuted-column fixtures here are the positive control on ADR 0007. An
alignment assertion nobody has watched fire is an assertion nobody knows works,
so several of these tests exist to prove the loader rejects a matrix that the
naive read accepts happily.
"""

import dataclasses
import gzip
import warnings

import numpy as np
import pytest
from matrix_io import (
    CENSORED_FILL_TOKEN,
    LabeledMatrix,
    MatrixAlignmentError,
    assert_aligned,
    detect_permutation,
    load_labeled_matrix,
    load_matrix_frame,
    summarize_censoring,
)

# Foldseek writes measured scores in %.3E form. The fill is a plain "0.0".
# Keeping both forms in the fixtures is the point -- it is the difference the
# loader exists to preserve.
MEASURED_ZERO = "0.000E+00"


FILL = CENSORED_FILL_TOKEN


def fmt(value):
    return f"{value:.3E}"


def write(path, row_labels, col_labels, cells):
    """Write a matrix from already-formatted tokens. Used by the regression block."""
    lines = ["\t".join(["protid"] + list(col_labels))]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_matrix(path, row_labels, col_labels, cells, *, column_prefix="", gzipped=False):
    """Write a matrix file. `cells` is a list of rows of already-formatted tokens."""
    lines = ["\t".join(["protid"] + [column_prefix + c for c in col_labels])]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    text = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt", newline="") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def make_cells(row_labels, col_labels, *, censor=(), measured_zero=()):
    """A plausible similarity matrix: 1.0 on the label diagonal, ~0.7 elsewhere.

    `censor` and `measured_zero` are sets of (row_label, col_label) pairs.
    """
    cells = []
    for r in row_labels:
        row = []
        for c in col_labels:
            if (r, c) in censor:
                row.append(CENSORED_FILL_TOKEN)
            elif (r, c) in measured_zero:
                row.append(MEASURED_ZERO)
            elif r == c:
                row.append(fmt(1.0))
            else:
                row.append(fmt(0.7))
        cells.append(row)
    return cells


@pytest.fixture
def labels():
    return ["P00001", "P00002", "P00003", "P00004"]


@pytest.fixture
def aligned_file(tmp_path, labels):
    return write_matrix(tmp_path / "aligned.tsv", labels, labels, make_cells(labels, labels))


@pytest.fixture
def permuted_file(tmp_path, labels):
    """The PR #106 defect: same labels, columns in a different order."""
    permuted = [labels[2], labels[0], labels[3], labels[1]]
    return write_matrix(tmp_path / "permuted.tsv", labels, permuted, make_cells(labels, permuted))


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_aligned_matrix_loads(aligned_file, labels):
    matrix = load_labeled_matrix(aligned_file)
    assert matrix.protids == labels
    assert matrix.columns == labels
    assert matrix.is_aligned
    assert matrix.is_square
    assert matrix.shape == (4, 4)
    assert matrix.values.dtype == np.float32


def test_permuted_columns_raise(permuted_file):
    with pytest.raises(MatrixAlignmentError) as excinfo:
        load_labeled_matrix(permuted_file)
    message = str(excinfo.value)
    # The message has to be actionable, not just "assertion failed".
    assert "Column order does not match row order" in message
    assert "position 0" in message
    assert "PR #106" in message
    assert "repair=True" in message


def test_permuted_columns_quantify_the_damage(permuted_file):
    """The error distinguishes 'slightly out of order' from 'essentially random'."""
    with pytest.raises(MatrixAlignmentError) as excinfo:
        load_labeled_matrix(permuted_file)
    # In this fixture no column happens to land in its row position.
    assert "Only 0 of 4 columns (0.00%)" in str(excinfo.value)


def test_repair_reorders_and_warns(permuted_file, labels):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = load_labeled_matrix(permuted_file, repair=True)

    assert matrix.is_aligned
    assert matrix.columns == labels
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "PR #106" in str(caught[0].message)
    # The repair must be exact: the label diagonal is 1.0 everywhere.
    np.testing.assert_allclose(np.diag(matrix.values), 1.0, rtol=0, atol=1e-6)


def test_repair_is_exact_not_approximate(tmp_path, labels):
    """Every cell must land where its labels say, not merely the diagonal."""
    permuted = [labels[3], labels[1], labels[0], labels[2]]
    cells = []
    for i, _row_label in enumerate(labels):
        cells.append([fmt(0.1 * (i + 1) + 0.01 * (j + 1)) for j, _ in enumerate(permuted)])
    path = write_matrix(tmp_path / "p.tsv", labels, permuted, cells)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repaired = load_labeled_matrix(path, repair=True)

    raw = load_labeled_matrix(path, require_alignment=False)
    for i, row_label in enumerate(labels):
        for j, c in enumerate(permuted):
            expected = raw.values[i][j]
            got = repaired.values[i][repaired.columns.index(c)]
            assert got == pytest.approx(expected), f"cell ({row_label}, {c}) moved incorrectly"


def test_disjoint_labels_cannot_be_repaired(tmp_path, labels):
    cols = ["Q00001", "Q00002", "Q00003", "Q00004"]
    path = write_matrix(tmp_path / "bad.tsv", labels, cols, make_cells(labels, cols))
    with pytest.raises(MatrixAlignmentError) as excinfo:
        load_labeled_matrix(path, repair=True)
    message = str(excinfo.value)
    assert "cannot be repaired by reordering" in message
    assert "malformed" in message


def test_require_alignment_false_allows_rectangular(tmp_path, labels):
    """key_protid_tmscore_features.tsv is n_proteins x n_key_protids."""
    key = [labels[0], labels[2]]
    path = write_matrix(tmp_path / "rect.tsv", labels, key, make_cells(labels, key))
    matrix = load_labeled_matrix(path, require_alignment=False)
    assert matrix.shape == (4, 2)
    assert not matrix.is_square


def test_assert_aligned_accepts_matching_order(labels):
    assert assert_aligned(labels, list(labels)) is None


# --------------------------------------------------------------------------
# censoring -- ADR 0009
# --------------------------------------------------------------------------


def test_censoring_mask_tracks_the_fill_token(tmp_path, labels):
    censor = {("P00001", "P00003"), ("P00002", "P00004")}
    path = write_matrix(
        tmp_path / "c.tsv", labels, labels, make_cells(labels, labels, censor=censor)
    )
    matrix = load_labeled_matrix(path)

    assert matrix.censored.sum() == 2
    assert matrix.censored[0][2]
    assert matrix.censored[1][3]
    assert not matrix.censored[0][1]
    assert matrix.censoring_rate == pytest.approx(2 / 16)


def test_measured_zero_is_not_censored(tmp_path, labels):
    """The distinction the loader exists to preserve.

    A measured 0.000E+00 and a fill 0.0 are both float zero. Only the raw token
    tells them apart, and only before conversion.
    """
    path = write_matrix(
        tmp_path / "mz.tsv",
        labels,
        labels,
        make_cells(
            labels,
            labels,
            censor={("P00001", "P00002")},
            measured_zero={("P00003", "P00004")},
        ),
    )
    matrix = load_labeled_matrix(path)

    assert matrix.values[0][1] == 0.0
    assert matrix.values[2][3] == 0.0
    # Same value, different meaning.
    assert matrix.censored[0][1] is np.True_ or matrix.censored[0][1]
    assert not matrix.censored[2][3]
    assert matrix.measured_zero_count() == 1


def test_no_measured_zeros_in_a_normal_matrix(aligned_file):
    assert load_labeled_matrix(aligned_file).measured_zero_count() == 0


def test_censoring_rate_per_protid(tmp_path, labels):
    censor = {("P00001", c) for c in labels[:2]}
    path = write_matrix(
        tmp_path / "c.tsv", labels, labels, make_cells(labels, labels, censor=censor)
    )
    rates = load_labeled_matrix(path).censoring_rate_per_protid()
    assert rates["P00001"] == pytest.approx(0.5)
    assert rates["P00002"] == pytest.approx(0.0)
    assert rates.index.name == "protid"


def test_summarize_censoring_detects_a_per_query_cap(tmp_path):
    """Rows pile up at a cap; columns do not. That asymmetry is the tell."""
    n, cap = 12, 4
    lbls = [f"P{i:05d}" for i in range(n)]
    rng = np.random.RandomState(0)
    cells = []
    for _ in range(n):
        # Each row reports exactly `cap` partners, chosen independently, so the
        # column counts vary while every row sits at the cap.
        keep = set(rng.choice(n, size=cap, replace=False).tolist())
        cells.append([fmt(0.8) if j in keep else CENSORED_FILL_TOKEN for j in range(n)])
    write_matrix(tmp_path / "cap.tsv", lbls, lbls, cells)

    summary = summarize_censoring(load_labeled_matrix(tmp_path / "cap.tsv"))
    assert summary["cap_detected"]
    assert summary["inferred_cap"] == cap
    assert summary["measured_per_row"]["min"] == cap
    assert summary["measured_per_row"]["max"] == cap
    assert summary["censoring_predicted_by_cap"] == pytest.approx(1 - cap / n)
    assert summary["censoring_rate"] == pytest.approx(1 - cap / n)


def test_summarize_censoring_reports_no_cap_when_dense(aligned_file):
    summary = summarize_censoring(load_labeled_matrix(aligned_file))
    assert summary["n_censored"] == 0
    assert not summary["cap_detected"]


# --------------------------------------------------------------------------
# the label-vs-position demonstration
# --------------------------------------------------------------------------


def test_label_aligned_diagonal_survives_a_permutation(permuted_file):
    """The heart of ADR 0007, in four lines.

    On a permuted matrix the positional diagonal is mostly wrong and the
    label-aligned diagonal is exactly right. The data is sound; only the naive
    read is broken.
    """
    matrix = load_labeled_matrix(permuted_file, require_alignment=False)

    positional = np.diag(matrix.values)
    assert not np.allclose(positional, 1.0)

    labeled = matrix.aligned_diagonal()
    np.testing.assert_allclose(labeled, 1.0, rtol=0, atol=1e-6)


def test_detect_permutation_reports_without_raising(permuted_file, aligned_file):
    assert detect_permutation(load_labeled_matrix(aligned_file)) is None

    report = detect_permutation(load_labeled_matrix(permuted_file, require_alignment=False))
    assert report["same_label_set"]
    assert report["n"] == 4
    assert report["columns_in_row_position"] == 0
    assert report["fraction_in_place"] == 0.0


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def test_column_prefix_is_stripped(tmp_path, labels):
    key = [labels[0], labels[1]]
    write_matrix(
        tmp_path / "pfx.tsv",
        labels,
        key,
        make_cells(labels, key),
        column_prefix="TMscore_v_",
    )
    matrix = load_labeled_matrix(
        tmp_path / "pfx.tsv", require_alignment=False, column_prefix="TMscore_v_"
    )
    assert matrix.columns == key
    assert matrix.column_prefix == "TMscore_v_"


def test_gzip_is_transparent(tmp_path, labels):
    write_matrix(tmp_path / "g.tsv.gz", labels, labels, make_cells(labels, labels), gzipped=True)
    matrix = load_labeled_matrix(tmp_path / "g.tsv.gz")
    assert matrix.protids == labels
    assert matrix.is_aligned


def test_to_frame_matches_the_legacy_read(aligned_file, labels):
    import pandas as pd

    frame = load_labeled_matrix(aligned_file).to_frame()
    legacy = pd.read_csv(aligned_file, sep="\t", index_col="protid")

    assert list(frame.index) == list(legacy.index)
    assert list(frame.columns) == list(legacy.columns)
    np.testing.assert_allclose(frame.to_numpy(), legacy.to_numpy(), rtol=1e-6)


def test_load_matrix_frame_wrapper(aligned_file, labels):
    frame = load_matrix_frame(aligned_file)
    assert list(frame.index) == labels
    assert frame.index.name == "protid"


def test_trailing_blank_line_is_tolerated(tmp_path, labels):
    path = tmp_path / "t.tsv"
    write_matrix(path, labels, labels, make_cells(labels, labels))
    path.write_text(path.read_text() + "\n")
    assert load_labeled_matrix(path).shape == (4, 4)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labeled_matrix(tmp_path / "nope.tsv")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("")
    with pytest.raises(MatrixAlignmentError, match="empty"):
        load_labeled_matrix(path)


def test_header_only_raises(tmp_path, labels):
    path = tmp_path / "h.tsv"
    path.write_text("\t".join(["protid"] + labels) + "\n")
    with pytest.raises(MatrixAlignmentError, match="no data rows"):
        load_labeled_matrix(path)


def test_wrong_index_column_raises(tmp_path, labels):
    path = tmp_path / "w.tsv"
    path.write_text("\t".join(["accession"] + labels) + "\n")
    with pytest.raises(MatrixAlignmentError, match="protid"):
        load_labeled_matrix(path)


def test_ragged_row_raises_with_line_number(tmp_path, labels):
    path = tmp_path / "r.tsv"
    lines = ["\t".join(["protid"] + labels), "P00001\t1.0\t2.0"]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MatrixAlignmentError, match="line 2"):
        load_labeled_matrix(path)


def test_unparseable_value_raises_with_line_number(tmp_path, labels):
    path = tmp_path / "u.tsv"
    lines = ["\t".join(["protid"] + labels), "P00001\t1.0\tnot_a_number\t0.5\t0.5"]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MatrixAlignmentError, match="line 2"):
        load_labeled_matrix(path)


def test_labeled_matrix_is_frozen(aligned_file):
    matrix = load_labeled_matrix(aligned_file)
    with pytest.raises(dataclasses.FrozenInstanceError):
        matrix.protids = ["x"]


def test_no_bare_array_accessor(aligned_file):
    """Getting an unlabeled array must be a deliberate act, not the default."""
    matrix = load_labeled_matrix(aligned_file)
    assert isinstance(matrix, LabeledMatrix)
    # values exists, but never without protids/columns alongside it.
    assert len(matrix.protids) == matrix.values.shape[0]
    assert len(matrix.columns) == matrix.values.shape[1]


# ==========================================================================
# Regressions from the adversarial review. Each of these was a real defect;
# see docs/REVIEW_LOG.md. Two returned wrong numbers and reported success.
# ==========================================================================

# ==========================================================================
# B1 (blocks) -- repair=True silently produced wrong numbers on duplicate labels
# ==========================================================================


def test_b1_duplicate_column_labels_are_refused_not_silently_collapsed(tmp_path):
    """Was: repair reordered by a {label: pos} dict, keeping only the last
    duplicate. A 2x3 matrix silently became 2x2, one column was discarded, and
    cell (A,B) reported 0.7 when its true value was 0.4 -- with is_aligned True.
    """
    path = write(
        tmp_path / "dup_cols.tsv",
        ["A", "B"],
        ["A", "B", "B"],
        [[fmt(1.0), fmt(0.4), fmt(0.7)], [fmt(0.4), fmt(1.0), fmt(0.3)]],
    )
    with pytest.raises(MatrixAlignmentError) as excinfo:
        load_labeled_matrix(path, repair=True)
    message = str(excinfo.value)
    assert "Duplicate column label" in message
    assert "'B'" in message
    assert "cannot be repaired by reordering" in message


def test_b1_duplicate_row_labels_are_refused(tmp_path):
    path = write(
        tmp_path / "dup_rows.tsv",
        ["A", "B", "B"],
        ["A", "A", "B"],
        [
            [fmt(1.0), fmt(0.5), fmt(0.2)],
            [fmt(0.5), fmt(1.0), fmt(0.9)],
            [fmt(0.2), fmt(0.9), fmt(1.0)],
        ],
    )
    with pytest.raises(MatrixAlignmentError, match="Duplicate row label"):
        load_labeled_matrix(path, repair=True)


def test_b1_duplicates_refused_even_without_repair(tmp_path):
    path = write(
        tmp_path / "d.tsv",
        ["A", "B"],
        ["A", "B", "B"],
        [[fmt(1.0), fmt(0.4), fmt(0.7)], [fmt(0.4), fmt(1.0), fmt(0.3)]],
    )
    with pytest.raises(MatrixAlignmentError, match="Duplicate"):
        load_labeled_matrix(path)


# ==========================================================================
# B4 -- float32 underflow made a zero that was neither censored nor counted
# ==========================================================================


def test_b4_underflow_is_detected_and_warned(tmp_path):
    """Was: 1e-46 is nonzero in double and 0.0 in float32, so it was not the
    fill token and `float(token) == 0.0` was False -- a zero invisible to both
    the censoring mask and the measured-zero guard.
    """
    path = write(
        tmp_path / "u.tsv",
        ["A", "B"],
        ["A", "B"],
        [[fmt(1.0), "1e-46"], [fmt(0.5), fmt(1.0)]],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = load_labeled_matrix(path)

    assert matrix.values[0][1] == 0.0
    assert not matrix.censored[0][1]
    assert any("underflow" in str(w.message) for w in caught)


def test_b4_overflow_is_detected(tmp_path):
    path = write(
        tmp_path / "o.tsv",
        ["A", "B"],
        ["A", "B"],
        [[fmt(1.0), "1e40"], [fmt(0.5), fmt(1.0)]],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_labeled_matrix(path)
    assert any("overflow" in str(w.message) for w in caught)


def test_b4_ordinary_values_do_not_warn(tmp_path):
    path = write(
        tmp_path / "n.tsv",
        ["A", "B"],
        ["A", "B"],
        [[fmt(1.0), fmt(0.7)], [fmt(0.7), fmt(1.0)]],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_labeled_matrix(path)
    assert not [w for w in caught if "underflow" in str(w.message)]


# ==========================================================================
# B5 -- assert_aligned raised TypeError instead of MatrixAlignmentError
# ==========================================================================


def test_b5_length_mismatch_raises_the_documented_exception(tmp_path):
    """Was: 3 data rows under a 2-column header left first_bad=None, then
    `protids[None]` raised TypeError -- so `except MatrixAlignmentError` crashed.
    """
    path = tmp_path / "len.tsv"
    path.write_text(
        "protid\tA\tB\n"
        "A\t1.000E+00\t9.000E-01\n"
        "B\t9.000E-01\t1.000E+00\n"
        "B\t5.000E-01\t5.000E-01\n"
    )
    with pytest.raises(MatrixAlignmentError):
        load_labeled_matrix(path)


def test_note_repair_with_require_alignment_false_is_refused(tmp_path):
    """Was: silently ignored, returning a still-permuted matrix to a caller who
    had explicitly asked for a repair.
    """
    path = write(
        tmp_path / "p.tsv",
        ["A", "B"],
        ["B", "A"],
        [[fmt(0.4), fmt(1.0)], [fmt(1.0), fmt(0.4)]],
    )
    with pytest.raises(ValueError, match="no effect when require_alignment=False"):
        load_labeled_matrix(path, require_alignment=False, repair=True)


def test_note_cap_detection_needs_the_columns_to_disagree(tmp_path):
    """Was: any matrix with a uniform per-row count reported a per-query cap.
    One censored cell per row is uniform on both axes -- not a cap.
    """
    n = 6
    labels = [f"P{i}" for i in range(n)]
    cells = []
    for i in range(n):
        cells.append([FILL if i == j else fmt(0.8) for j in range(n)])
    matrix = load_labeled_matrix(write(tmp_path / "diag.tsv", labels, labels, cells))
    summary = summarize_censoring(matrix)
    assert summary["measured_per_row"]["min"] == summary["measured_per_row"]["max"]
    assert not summary["cap_detected"]


def test_note_a_genuine_per_query_cap_is_still_detected(tmp_path):
    n, cap = 20, 5
    labels = [f"P{i:03d}" for i in range(n)]
    rng = np.random.RandomState(0)
    cells = []
    for _ in range(n):
        keep = set(rng.choice(n, size=cap, replace=False).tolist())
        cells.append([fmt(0.8) if j in keep else FILL for j in range(n)])
    summary = summarize_censoring(
        load_labeled_matrix(write(tmp_path / "cap.tsv", labels, labels, cells))
    )
    assert summary["cap_detected"]
    assert summary["inferred_cap"] == cap
