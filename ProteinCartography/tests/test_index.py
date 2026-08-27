"""Tests for the canonical protein index.

The point of this module is that it refuses to guess. Most of these tests are
about what it declines to do.
"""

import numpy as np
import pandas as pd
import pytest
from index import IndexAlignmentError, ProteinIndex


@pytest.fixture
def idx():
    return ProteinIndex.from_iterable(["A", "B", "C"])


def test_basic_properties(idx):
    assert len(idx) == 3
    assert list(idx) == ["A", "B", "C"]
    assert "B" in idx
    assert "Z" not in idx
    assert idx[0] == "A"
    assert idx.position("C") == 2


def test_empty_index_is_rejected():
    with pytest.raises(IndexAlignmentError, match="cannot be empty"):
        ProteinIndex.from_iterable([])


def test_duplicates_are_rejected():
    with pytest.raises(IndexAlignmentError) as excinfo:
        ProteinIndex.from_iterable(["A", "B", "A"])
    # The reason matters: a duplicate doubles a protein's weight in every geometry.
    assert "doubles its weight" in str(excinfo.value)


def test_unknown_protid_position_raises(idx):
    with pytest.raises(IndexAlignmentError, match="not in this index"):
        idx.position("Z")


def test_align_reorders_to_index_order(idx):
    values = np.array([[30.0], [10.0], [20.0]])
    out = idx.align(["C", "A", "B"], values)
    np.testing.assert_array_equal(out, [[10.0], [20.0], [30.0]])


def test_align_refuses_to_fill_a_missing_protein(idx):
    """The central guarantee. pandas would reindex to NaN and return happily."""
    values = np.array([[1.0], [2.0]])
    with pytest.raises(IndexAlignmentError) as excinfo:
        idx.align(["A", "B"], values, what="block 'plm'")
    message = str(excinfo.value)
    assert "block 'plm' is missing 1 of the index's 3 proteins" in message
    assert "['C']" in message
    assert "Refusing to reindex" in message
    assert "never measured" in message


def test_align_drops_extras_without_complaint(idx):
    """A block computed over a superset is fine; only absences are an error."""
    values = np.array([[10.0], [20.0], [30.0], [99.0]])
    out = idx.align(["A", "B", "C", "EXTRA"], values)
    np.testing.assert_array_equal(out, [[10.0], [20.0], [30.0]])


def test_align_checks_the_shape_matches_the_labels(idx):
    with pytest.raises(IndexAlignmentError, match="axis 0 has length"):
        idx.align(["A", "B", "C"], np.zeros((2, 1)))


def test_align_along_axis_one(idx):
    values = np.array([[3.0, 1.0, 2.0]])
    out = idx.align(["C", "A", "B"], values, axis=1)
    np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0]])


def test_align_frame_preserves_columns(idx):
    frame = pd.DataFrame({"x": [3, 1, 2], "y": [30, 10, 20]}, index=pd.Index(["C", "A", "B"]))
    out = idx.align_frame(frame)
    assert list(out.index) == ["A", "B", "C"]
    assert list(out.columns) == ["x", "y"]
    assert list(out["x"]) == [1, 2, 3]
    assert out.index.name == "protid"


def test_positions_of_raises_on_absentee(idx):
    with pytest.raises(IndexAlignmentError, match="not in the index"):
        idx.positions_of(["A", "Z"])


def test_intersection_takes_order_from_self(idx):
    """Order must be a deterministic function of the inputs, not of set iteration."""
    other = ["C", "A", "NOPE"]
    result = idx.intersection(other)
    assert list(result) == ["A", "C"]


def test_empty_intersection_raises(idx):
    with pytest.raises(IndexAlignmentError, match="Intersection is empty"):
        idx.intersection(["X", "Y"])


def test_missing_from(idx):
    assert idx.missing_from(["A", "C"]) == ["B"]
    assert idx.missing_from(["A", "B", "C"]) == []


def test_equality_is_order_sensitive():
    a = ProteinIndex.from_iterable(["A", "B"])
    b = ProteinIndex.from_iterable(["B", "A"])
    assert a.equals(ProteinIndex.from_iterable(["A", "B"]))
    assert not a.equals(b)


def test_index_is_hashable_so_it_can_live_on_a_manifest(idx):
    assert isinstance(hash(idx), int)


def test_to_pandas(idx):
    pandas_index = idx.to_pandas()
    assert list(pandas_index) == ["A", "B", "C"]
    assert pandas_index.name == "protid"


def test_from_matrix_uses_row_labels(tmp_path):
    from matrix_io import load_labeled_matrix

    labels = ["P1", "P2"]
    path = tmp_path / "m.tsv"
    path.write_text("protid\tP1\tP2\nP1\t1.000E+00\t7.000E-01\nP2\t7.000E-01\t1.000E+00\n")
    idx = ProteinIndex.from_matrix(load_labeled_matrix(path))
    assert list(idx) == labels


def test_repr_is_readable(idx):
    assert "3 proteins" in repr(idx)
