"""Tests for the domains block.

The reader tests carry most of the weight, for the same reason they do in the
3Di block: the two columns this reads are adjacent in the features table, both
hold semicolon-separated lists of uppercase accessions, and reading the wrong
one produces a block of the wrong thing that looks entirely normal. Several
tests below exist only to make that fail loudly.

The rest is about what a zero row means. A protein with no Pfam annotation is
at the origin, and at the origin it resembles every other unannotated protein
exactly. That resemblance is an artifact of who has been studied, not a
property of the proteins, so the block reports it rather than emitting it as a
cluster.
"""

import numpy as np
import pytest
from blocks.domains import (
    DEFAULT_SOURCE,
    DomainsError,
    DomainsProvider,
    domain_matrix,
    domain_vocabulary,
    parse_accessions,
    read_domains,
    validate_params,
)


def features_file(*rows, columns=("Pfam", "InterPro")):
    """Rows of (protid, pfam_cell, interpro_cell)."""
    header = "\t".join(["protid", "Length", *columns])
    lines = [header]
    for protid, pfam, interpro in rows:
        lines.append("\t".join([protid, "100", pfam, interpro]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parsing the accession lists
# ---------------------------------------------------------------------------


def test_the_trailing_semicolon_does_not_become_an_empty_accession():
    """UniProt writes `PF00022;PF00125;` -- separated *and* terminated."""
    assert parse_accessions("PF00022;PF00125;") == ["PF00022", "PF00125"]


def test_an_empty_cell_is_no_accessions():
    assert parse_accessions("") == []
    assert parse_accessions(None) == []


def test_whitespace_around_accessions_is_stripped():
    assert parse_accessions(" PF00022 ; PF00125 ") == ["PF00022", "PF00125"]


# ---------------------------------------------------------------------------
# reading the features file
# ---------------------------------------------------------------------------


def test_pfam_is_read_by_default():
    text = features_file(("P1", "PF00022;", "IPR004000;"))
    assert read_domains(text) == {"P1": ["PF00022"]}


def test_interpro_can_be_asked_for_instead():
    text = features_file(("P1", "PF00022;", "IPR004000;IPR020902;"))
    assert read_domains(text, "interpro") == {"P1": ["IPR004000", "IPR020902"]}


def test_both_sources_can_be_combined():
    text = features_file(("P1", "PF00022;", "IPR004000;"))
    assert read_domains(text, "both") == {"P1": ["IPR004000", "PF00022"]}


def test_interpro_accessions_in_the_pfam_column_are_rejected():
    """The swapped-column case. Both columns are lists of uppercase accessions."""
    text = features_file(("P1", "IPR004000;", "IPR004000;"))
    with pytest.raises(DomainsError, match="do not look like|does not look like"):
        read_domains(text, "pfam")


def test_pfam_accessions_in_the_interpro_column_are_rejected():
    text = features_file(("P1", "PF00022;", "PF00022;"))
    with pytest.raises(DomainsError, match="not look like"):
        read_domains(text, "interpro")


def test_the_error_names_the_column_and_the_expected_shape():
    text = features_file(("P1", "GO:0005524;", "IPR004000;"))
    with pytest.raises(DomainsError, match=r"'Pfam'.*PF"):
        read_domains(text, "pfam")


def test_a_missing_column_names_what_it_found():
    text = "protid\tLength\nP1\t100\n"
    with pytest.raises(DomainsError, match="no 'Pfam' column"):
        read_domains(text)


def test_an_unknown_source_is_rejected():
    text = features_file(("P1", "PF00022;", "IPR004000;"))
    with pytest.raises(DomainsError, match="not valid"):
        read_domains(text, "supfam")


def test_accessions_are_deduplicated_and_sorted():
    """So the block does not depend on the order UniProt happened to emit."""
    text = features_file(("P1", "PF00125;PF00022;PF00022;", ""))
    assert read_domains(text)["P1"] == ["PF00022", "PF00125"]


def test_file_order_of_proteins_is_preserved():
    text = features_file(("Z9", "PF00022;", ""), ("A1", "PF00125;", ""))
    assert list(read_domains(text)) == ["Z9", "A1"]


def test_a_duplicate_protid_is_rejected_rather_than_overwritten():
    text = features_file(("P1", "PF00022;", ""), ("P1", "PF00125;", ""))
    with pytest.raises(DomainsError, match="more than once"):
        read_domains(text)


def test_an_unannotated_protein_is_kept_with_an_empty_list():
    """Dropping it would silently shrink the cohort."""
    text = features_file(("P1", "PF00022;", ""), ("P2", "", ""))
    assert read_domains(text) == {"P1": ["PF00022"], "P2": []}


# ---------------------------------------------------------------------------
# the presence matrix
# ---------------------------------------------------------------------------


def test_the_matrix_is_binary_presence():
    protids, features, vocabulary, _ = domain_matrix(
        {"P1": ["PF00022"], "P2": ["PF00022", "PF00125"]}
    )
    assert protids == ["P1", "P2"]
    assert vocabulary == ["PF00022", "PF00125"]
    assert features.tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_the_vocabulary_is_sorted_and_observed_only():
    """Pfam has over twenty thousand families; a cohort touches a handful."""
    vocabulary = domain_vocabulary({"P1": ["PF00125", "PF00022"], "P2": ["PF00022"]})
    assert vocabulary == ["PF00022", "PF00125"]


def test_an_unannotated_protein_is_reported_and_left_at_the_origin():
    protids, features, _, without = domain_matrix({"P1": ["PF00022"], "P2": []})
    assert without == ["P2"]
    assert protids == ["P1", "P2"]
    assert features[1].sum() == 0.0
    assert not np.isnan(features).any()


def test_two_unannotated_proteins_are_identical_which_is_why_they_are_reported():
    """The whole reason `proteins_without_domains` exists, stated as a test.

    Nothing here can stop them coinciding -- there is no evidence to separate
    them on. What the block can do is refuse to let that coincidence be read as
    a finding.
    """
    _, features, _, without = domain_matrix({"P1": ["PF00022"], "P2": [], "P3": []})
    assert np.array_equal(features[1], features[2])
    assert without == ["P2", "P3"]


def test_an_explicit_vocabulary_is_honored():
    """So two cohorts can be compared on the same columns when that is wanted."""
    _, features, vocabulary, _ = domain_matrix(
        {"P1": ["PF00022"]}, vocabulary=["PF00022", "PF99999"]
    )
    assert vocabulary == ["PF00022", "PF99999"]
    assert features[0].tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


def test_the_default_source_is_pfam():
    assert validate_params({})["source"] == DEFAULT_SOURCE == "pfam"


def test_an_unknown_source_is_rejected_in_params():
    with pytest.raises(ValueError, match="not valid"):
        validate_params({"source": "supfam"})


def test_jaccard_is_refused_with_the_reason_rather_than_ignored():
    """It is the right distance for sets, and nothing would apply it.

    `reduce_space` feeds features into a euclidean PCA without consulting
    `spec.metric`, so accepting `jaccard` would write a claim into the manifest
    that no code honors.
    """
    with pytest.raises(ValueError, match="not metric-aware"):
        validate_params({"metric": "jaccard"})


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="domains.metric"):
        validate_params({"metric": "cosine"})


# ---------------------------------------------------------------------------
# the provider
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, output_dir, seed=123456):
        self.output_dir = output_dir
        self.seed = seed

    def path(self, *parts):
        import os

        return os.path.join(self.output_dir, *parts)


def write_features(tmp_path, text):
    directory = tmp_path / "protein_features"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "uniprot_features.tsv"
    path.write_text(text)
    return path


def test_the_provider_is_always_available():
    """ADR 0006: no fetch and no new package -- the columns already arrive."""
    available, reason = DomainsProvider().is_available()
    assert available and reason == ""


def test_a_missing_features_file_names_both_modes_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_uniprot_metadata"):
        DomainsProvider().compute(Ctx(str(tmp_path)), {})


def test_the_block_carries_features_and_a_manifest(tmp_path):
    write_features(
        tmp_path,
        features_file(("P1", "PF00022;", "IPR004000;"), ("P2", "PF00022;PF00125;", "IPR004000;")),
    )
    result = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.kind == "features"
    assert result.spec.provider == "domains"
    assert result.spec.metric == "euclidean"
    assert result.protids == ["P1", "P2"]
    assert result.manifest["extra"]["families"] == ["PF00022", "PF00125"]
    assert result.manifest["extra"]["n_families"] == 2


def test_the_annotated_fraction_is_recorded(tmp_path):
    write_features(tmp_path, features_file(("P1", "PF00022;", ""), ("P2", "", ""), ("P3", "", "")))
    result = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    extra = result.manifest["extra"]
    assert extra["annotated_fraction"] == pytest.approx(1 / 3)
    assert extra["proteins_without_domains"] == ["P2", "P3"]


def test_a_cohort_with_no_annotations_at_all_is_an_error_that_says_why(tmp_path):
    """Zero columns is not a degenerate block; it is a run that cannot have one."""
    write_features(tmp_path, features_file(("P1", "", ""), ("P2", "", "")))
    with pytest.raises(DomainsError, match="cannot be built for this run"):
        DomainsProvider().compute(Ctx(str(tmp_path)), {})


def test_the_block_does_not_standardize_its_sparse_columns(tmp_path):
    """A family seen twice must not outweigh one seen everywhere."""
    write_features(tmp_path, features_file(("P1", "PF00022;", "")))
    result = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.normalization == "unit_mean_distance"


def test_the_block_is_fusable(tmp_path):
    """Curated family membership is external evidence, not a restatement of the map."""
    write_features(tmp_path, features_file(("P1", "PF00022;", "")))
    result = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.fusable is True
    assert result.spec.not_fusable_reason is None


def test_an_empty_features_file_is_an_error(tmp_path):
    write_features(tmp_path, "protid\tPfam\tInterPro\n")
    with pytest.raises(DomainsError, match="lists no proteins"):
        DomainsProvider().compute(Ctx(str(tmp_path)), {})


def test_two_computations_of_the_same_input_agree(tmp_path):
    """The block ends up in the output tree, so it must not be a source of churn."""
    write_features(
        tmp_path, features_file(("P1", "PF00022;", "IPR004000;"), ("P2", "PF00125;", "IPR009072;"))
    )
    first = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    second = DomainsProvider().compute(Ctx(str(tmp_path)), {})
    assert np.array_equal(first.features, second.features)
    assert first.manifest["cache_key"] == second.manifest["cache_key"]


def test_the_source_changes_the_cache_key(tmp_path):
    """Two different blocks, not one block computed twice."""
    write_features(tmp_path, features_file(("P1", "PF00022;", "IPR004000;")))
    pfam = DomainsProvider().compute(Ctx(str(tmp_path)), {"source": "pfam"})
    interpro = DomainsProvider().compute(Ctx(str(tmp_path)), {"source": "interpro"})
    assert pfam.manifest["cache_key"] != interpro.manifest["cache_key"]


def test_the_provider_registers_under_its_name():
    from blocks import domains
    from spaces.registry import BLOCK_GROUP, get_provider

    domains.register()
    assert isinstance(get_provider(BLOCK_GROUP, "domains"), DomainsProvider)
