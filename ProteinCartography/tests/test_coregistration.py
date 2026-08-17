"""Tests for the co-registration index.

The thing under test is a guarantee, not a calculation: that two spaces being
compared are over the same proteins, and that anything lost on the way to that
is named. So most of these tests are about what the report *says* rather than
about the intersection itself, which is three lines.

The failure this prevents has no symptom. Two spaces built over sets differing
by a handful of proteins reduce cleanly, plot cleanly, and produce a
per-protein comparison silently conditioned on the overlap. Nothing downstream
can notice, which is why the check has to be here.
"""

import pytest
from coregistration import CoregistrationError, shared_index
from index import ProteinIndex


def test_identical_sets_are_reported_as_exact():
    """The healthy case, and the one the demo produces."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2", "P3"]})
    assert report.index.as_list == ["P1", "P2", "P3"]
    assert report.is_exact
    assert report.n_shared == 3
    assert "every space had the full set" in report.describe()


def test_the_shared_index_follows_the_reference_order():
    """Order must come from somewhere nameable, not from set iteration."""
    report = shared_index({"a": ["P3", "P1", "P2"], "b": ["P1", "P2", "P3"]}, reference="a")
    assert report.index.as_list == ["P3", "P1", "P2"]

    other = shared_index({"a": ["P3", "P1", "P2"], "b": ["P1", "P2", "P3"]}, reference="b")
    assert other.index.as_list == ["P1", "P2", "P3"]


def test_the_reference_defaults_to_the_first_space():
    report = shared_index({"a": ["P2", "P1"], "b": ["P1", "P2"]})
    assert report.reference == "a"
    assert report.index.as_list == ["P2", "P1"]


def test_a_space_that_measured_extra_proteins_has_them_named():
    """The whole point: what falls out of the comparison is listed, not counted."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2"]})
    assert report.index.as_list == ["P1", "P2"]
    assert not report.is_exact
    assert report.contribution("a").dropped == ("P3",)
    assert report.contribution("b").dropped == ()
    assert report.contribution("a").n_own == 3


def test_the_description_names_the_dropped_proteins():
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1"]})
    described = report.describe()
    assert "conditioned on the overlap" in described
    assert "P2" in described and "P3" in described


def test_a_long_dropped_list_is_truncated_with_a_count():
    """A thousand-protein divergence must not produce a thousand-line warning."""
    report = shared_index({"a": [f"P{i}" for i in range(20)], "b": ["P0"]})
    assert "+14 more" in report.describe()
    assert report.contribution("a").n_dropped == 19


def test_three_spaces_intersect_pairwise_down_to_the_common_set():
    report = shared_index(
        {"a": ["P1", "P2", "P3"], "b": ["P2", "P3", "P4"], "c": ["P3", "P2", "P9"]}
    )
    assert report.index.as_list == ["P2", "P3"]
    assert report.contribution("c").dropped == ("P9",)


def test_disjoint_spaces_are_an_error_that_names_both():
    """An empty intersection is not a small overlap; there is nothing to compare."""
    with pytest.raises(CoregistrationError, match="share no proteins"):
        shared_index({"structure": ["P1"], "families": ["P2"]})


def test_a_duplicate_protid_in_one_space_is_an_error():
    """A repeated row doubles that protein's weight, and set arithmetic hides it."""
    with pytest.raises(CoregistrationError, match="Duplicate protids"):
        shared_index({"a": ["P1", "P1", "P2"], "b": ["P1", "P2"]})


def test_no_spaces_at_all_is_an_error():
    with pytest.raises(CoregistrationError, match="at least one space"):
        shared_index({})


def test_an_unknown_reference_is_an_error_that_lists_the_options():
    with pytest.raises(CoregistrationError, match="nonesuch"):
        shared_index({"a": ["P1"], "b": ["P1"]}, reference="nonesuch")


def test_one_space_co_registers_with_itself():
    """Degenerate but legal: a config may compare a single space against nothing."""
    report = shared_index({"only": ["P1", "P2"]})
    assert report.index.as_list == ["P1", "P2"]
    assert report.is_exact


# ---------------------------------------------------------------------------
# what the report is for: aligning to it must then always succeed
# ---------------------------------------------------------------------------


def test_every_space_can_be_aligned_to_the_shared_index():
    """The payoff. `ProteinIndex.align` refuses to invent rows; after
    intersection it never has to, for any of the spaces."""
    import numpy as np

    space_protids = {"a": ["P1", "P2", "P3"], "b": ["P3", "P2", "P9"]}
    report = shared_index(space_protids)
    for space_id, protids in space_protids.items():
        values = np.arange(len(protids) * 2, dtype=np.float64).reshape(len(protids), 2)
        aligned = report.index.align(protids, values, what=space_id)
        assert aligned.shape == (report.n_shared, 2)


def test_alignment_actually_reorders_rather_than_merely_fitting():
    """Same shape is not the same rows. The row for P2 must follow P2."""
    import numpy as np

    report = shared_index({"a": ["P1", "P2"], "b": ["P2", "P1"]})
    b_values = np.array([[20.0], [10.0]])  # P2 then P1
    aligned = report.index.align(["P2", "P1"], b_values, what="b")
    assert report.index.as_list == ["P1", "P2"]
    assert aligned.tolist() == [[10.0], [20.0]]


def test_the_report_round_trips_through_its_dict_form():
    """It is written to disk beside the comparison it qualifies."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2"]})
    data = report.to_dict()
    assert data["n_shared"] == 2
    assert data["is_exact"] is False
    assert data["protids"] == ["P1", "P2"]
    assert data["spaces"][0] == {
        "space_id": "a",
        "n_own": 3,
        "n_dropped": 1,
        "dropped": ["P3"],
    }


def test_the_shared_index_is_a_protein_index():
    """So it carries the same refusal-to-invent-rows behavior everywhere."""
    report = shared_index({"a": ["P1"], "b": ["P1"]})
    assert isinstance(report.index, ProteinIndex)
