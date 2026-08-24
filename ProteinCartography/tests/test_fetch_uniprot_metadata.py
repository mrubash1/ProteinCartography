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


def _accessions_file(tmp_path):
    """The 24 accessions the recorded artifacts answer for, one per line."""
    from tests.mocks import API_RESPONSE_ARTIFACTS_DIRPATH, UNIPROT_ARTIFACT_DEFAULT

    rows = (API_RESPONSE_ARTIFACTS_DIRPATH / UNIPROT_ARTIFACT_DEFAULT).read_text().splitlines()
    accessions = [row.split("\t")[0] for row in rows[1:]]
    path = tmp_path / "accessions.txt"
    path.write_text("\n".join(accessions) + "\n")
    return path, accessions


def _fetch(tmp_path, fields, name):
    """Drive `query_uniprot` against the recorded UniProtKB responses.

    Patches `requests.Session.request`, which is the seam `mocks` itself uses;
    scoped to the call rather than started globally, because the module-level
    version is meant for a pipeline subprocess and never stops.
    """
    from unittest import mock as _mock

    import fetch_uniprot_metadata as fum
    from tests import mocks

    accessions_path, accessions = _accessions_file(tmp_path)
    output = tmp_path / name
    with _mock.patch("requests.Session.request", side_effect=mocks.mock_response):
        fum.query_uniprot(str(accessions_path), str(output), fields=fields)
    return output, accessions


def test_asking_for_the_optional_fields_puts_both_real_headers_in_the_table(tmp_path):
    """End to end through `query_uniprot`, against the recorded response.

    The mock answers a request that names the optional fields with a DIFFERENT
    recorded table -- the default artifact plus exactly two columns. Without
    that branch this test would pass whether or not the flag ever reached the
    request, which is the whole thing being checked.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    output, accessions = _fetch(tmp_path, fields_for(OPTIONAL_FIELDS), "with_optional.tsv")
    frame = pd.read_csv(output, sep="\t")
    for header in OPTIONAL_FIELDS_DICT:
        assert header in frame.columns, f"{header!r} missing; got {list(frame.columns)}"
    assert len(frame) == len(accessions)
    ec = frame["EC number"].dropna().astype(str)
    assert set(ec[ec.str.strip() != ""]) == {"3.6.4.-"}


def test_the_default_request_gets_neither_column(tmp_path):
    """The controlled negative. Same code path, same cohort, two fewer columns
    -- so the test above is measuring the fields and not the plumbing."""
    pytest.importorskip("pandas")
    import pandas as pd

    output, _ = _fetch(tmp_path, fields_for(None), "default.tsv")
    frame = pd.read_csv(output, sep="\t")
    for header in OPTIONAL_FIELDS_DICT:
        assert header not in frame.columns


def test_the_snakefile_renders_the_config_list_as_bare_arguments():
    """`--additional-fields {UNIPROT_ADDITIONAL_FIELDS}` interpolates a PYTHON
    LIST into a shell command, and the rendering is snakemake's, not Python's.

    Worth pinning because the two differ in exactly the way that would be
    silent: `str([])` is `"[]"`, which argparse would take as a field literally
    named `[]` and UniProt would reject, while snakemake joins on spaces and an
    empty list contributes nothing. The default path depends on the second
    behaviour.
    """
    from snakemake.utils import format as snakemake_format

    def render(value):
        UNIPROT_ADDITIONAL_FIELDS = value  # noqa: F841 -- read by the formatter
        return snakemake_format(
            "python fetch_uniprot_metadata.py --additional-fields {UNIPROT_ADDITIONAL_FIELDS}",
            **locals(),
        )

    assert render([]).endswith("--additional-fields ")
    assert render(OPTIONAL_FIELDS).endswith("--additional-fields ec cc_subcellular_location")
