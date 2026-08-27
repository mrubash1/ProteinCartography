"""Each of the auditor's checks, observed failing on a payload built to break it.

Every check here holds on the artifacts that exist, so passing tests would say
nothing on their own. What makes them checks is the planted half: a NaN, a
short buffer, a quantisation step past the declared ceiling, an inverted range,
and an oversized cohort. A guard that has never been seen to fail is one of
this repository's named failure modes.

Bare-env by construction -- the auditor is stdlib-only so it can read a built
page with nothing installed, and these tests must not be the reason that stops
being true.
"""

from __future__ import annotations
import base64
import json

import pytest
from explorer.audit_payload import (
    HARD_BUDGET_BYTES,
    audit,
    check_budget,
    check_finite,
    check_heatmap,
    payload_from_html,
)


def _heatmap(n=2, levels=255, low=0.0, high=1.0, raw=None):
    """A tm_matrix block, correct unless an argument says otherwise."""
    if raw is None:
        raw = bytes([0, levels, levels // 2, 0])[: n * n]
    return {
        "n": n,
        "levels": levels,
        "low": low,
        "high": high,
        "values": base64.b64encode(raw).decode("ascii"),
    }


def _cohort(**overrides):
    cohort = {
        "cohort_name": "fixture",
        "overlays": {"gravy": {"kind": "continuous", "values": [0.1, 0.2]}},
        "tm_matrix": _heatmap(),
    }
    cohort.update(overrides)
    return cohort


def test_a_clean_cohort_reports_nothing():
    """The negative control. Without it every assertion below could be vacuous."""
    assert audit({"cohorts": [_cohort()]})["findings"] == []


def test_a_nan_is_found_and_its_path_is_named():
    """`json.dumps` writes NaN bare, and the page is a JS literal rather than
    parsed JSON, so `NaN` is a valid identifier and lands silently."""
    cohort = _cohort(overlays={"gravy": {"values": [0.1, float("nan")]}})
    problems = check_finite(cohort)
    assert len(problems) == 1
    assert "overlays.gravy.values[1]" in problems[0], problems
    assert "nan" in problems[0].lower()


def test_an_infinity_is_found_too():
    assert check_finite(_cohort(overlays={"o": {"values": [float("inf")]}}))


def test_a_short_heatmap_buffer_is_caught():
    """A truncated buffer draws a truncated map and says nothing."""
    problems = check_heatmap(_cohort(tm_matrix=_heatmap(n=2, raw=bytes([0, 1]))))
    assert any("not n*n" in p for p in problems), problems


def test_a_quantisation_step_above_the_declared_ceiling_is_caught():
    """Decodes past `high`, so the colour runs off the end of the scale."""
    problems = check_heatmap(_cohort(tm_matrix=_heatmap(levels=100, raw=bytes([0, 200, 5, 5]))))
    assert any("above its declared" in p for p in problems), problems


def test_an_inverted_range_is_caught():
    """The case the cheap checks would otherwise miss: every decoded value is
    wrong while each one still looks like a plausible number."""
    problems = check_heatmap(_cohort(tm_matrix=_heatmap(low=1.0, high=0.0)))
    assert any("above high" in p for p in problems), problems


def test_a_heatmap_missing_its_contract_says_so_rather_than_crashing():
    problems = check_heatmap({"tm_matrix": {"n": 2, "values": ""}})
    assert problems == ["tm_matrix is present but does not declare all of n, levels, low, high"]


def test_an_oversized_cohort_is_caught():
    fat = _cohort(padding=["x" * 1024] * (HARD_BUDGET_BYTES // 1024))
    assert any("hard budget" in p for p in check_budget(fat, "fat"))


def test_the_payload_is_recovered_from_a_built_page():
    """Matched on template.py's own assignment line, not a script boundary --
    the page carries plotly's bundle in a script tag as well."""
    page = (
        "<html><script>/* plotly bundle */</script>\n"
        "  const PAYLOAD = " + json.dumps({"cohorts": []}) + ";\n"
        "</html>"
    )
    assert payload_from_html(page) == {"cohorts": []}


def test_a_page_without_the_assignment_line_says_which_failure_it_is():
    """If `template.py` renames the assignment this parser goes blind, and that
    is the failure worth naming rather than returning an empty audit."""
    with pytest.raises(ValueError, match="now blind"):
        payload_from_html("<html>no payload here</html>")


def test_a_bare_document_is_audited_as_one_cohort():
    """A payload JSON without a `cohorts` list is still auditable."""
    assert audit(_cohort())["n_cohorts"] == 1
