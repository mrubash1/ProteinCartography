"""The two optional UniProt fields, and the header strings they come back as.

FOLLOWUPS #35 recorded EC number and subcellular localization as having no data
source. That was true of the DEFAULT config and never true of the pipeline:
`uniprot_additional_fields` has always existed and has always defaulted to
`[]`. What was missing was a checked record of the two field ids and, more
importantly, of the exact headers UniProt answers with -- because a header
spelled from memory does not fail. The column arrives under its real name,
nothing matches the guessed one, `enrich_clusters` reports it absent, and the
run reads as a clean refusal forever.

No network here. The header strings are pinned as constants and this file
asserts the shape of what depends on them; the live verification is recorded in
`fetch_uniprot_metadata.OPTIONAL_FIELDS_DICT`'s comment with its date.
"""

from __future__ import annotations

import pytest
from enrichment import UNPARSEABLE_COLUMNS, detect_encoding, parse_terms
from fetch_uniprot_metadata import (
    DEFAULT_FIELDS,
    OPTIONAL_FIELDS,
    OPTIONAL_FIELDS_DICT,
    fields_for,
)


def test_additional_fields_reach_the_request():
    """The unglamorous one. If they did not, the columns would never arrive and
    every consumer would report them absent -- indistinguishable from not
    having asked."""
    assert fields_for(OPTIONAL_FIELDS) == list(DEFAULT_FIELDS) + ["ec", "cc_subcellular_location"]


def test_asking_for_nothing_adds_nothing():
    """`nargs="*"` yields [] for a bare `-a`; neither it nor None may change the
    default request, because that request is on the default output path."""
    assert fields_for(None) == list(DEFAULT_FIELDS)
    assert fields_for([]) == list(DEFAULT_FIELDS)


def test_the_two_field_ids_are_not_already_default():
    """If either were already requested the whole entry would be moot, and the
    pipeline would have been fetching a column nobody read."""
    assert not set(OPTIONAL_FIELDS) & set(DEFAULT_FIELDS)


def test_ec_numbers_are_single_valued_and_are_not_split_on_their_dots():
    """`3.6.4.-` is ONE term. A per-value guess that split on punctuation would
    turn every EC number into three or four meaningless terms, and the trailing
    `-` of a partial assignment into a fifth."""
    column = ["3.6.4.-", "2.7.11.1", "1.1.1.1"]
    assert detect_encoding(column) == "single"
    assert parse_terms("3.6.4.-", "single") == ("3.6.4.-",)


def test_a_protein_with_two_ec_numbers_makes_the_column_delimited():
    """And that is correct, not a bug. UniProt writes multiple EC numbers
    semicolon-separated, so a bifunctional enzyme anywhere in the cohort flips
    the whole column -- which is what `detect_encoding` is for, and why the
    test above must not be read as "EC is always single"."""
    assert detect_encoding(["3.6.4.-", "1.1.1.1; 2.7.11.1"]) == "delimited"
    assert parse_terms("1.1.1.1; 2.7.11.1", "delimited") == ("1.1.1.1", "2.7.11.1")


def test_subcellular_location_is_refused_by_name_rather_than_parsed():
    """Its value is a free-text comment block. `single` would make one term out
    of a paragraph including its citations; `delimited` would split on whatever
    semicolons the ECO braces happen to contain. Both produce terms and neither
    produces a fact."""
    header = "Subcellular location [CC]"
    assert header in UNPARSEABLE_COLUMNS
    assert header in OPTIONAL_FIELDS_DICT, "the refusal must name the header UniProt actually sends"
    reason = UNPARSEABLE_COLUMNS[header]
    assert "ENCODINGS" in reason and "terms" in reason


@pytest.mark.parametrize("header,field_id", sorted(OPTIONAL_FIELDS_DICT.items()))
def test_each_optional_field_maps_a_real_header_to_a_lowercase_field_id(header, field_id):
    """The header is what comes back in the TSV; the field id is what goes out
    in the query. Confusing the two is the failure this dict exists to stop --
    it is silent in both directions."""
    assert field_id == field_id.lower().replace(" ", "_")
    assert header != field_id
    assert header[0].isupper()
