"""Tests for the 3Di block.

The reader tests carry most of the weight. `foldseek structureto3didescriptor`
emits four fields of which two -- the amino-acid sequence and the 3Di string --
are uppercase letter strings of identical length, so reading the wrong one
produces a *sequence* profile labelled as a *structural* one and nothing about
the output looks wrong. Several tests below exist only to make that fail loudly.

The profile tests are about length. Raw k-mer counts scale with protein size, so
an unnormalized profile's dominant axis is "how long is this protein" -- true,
and not what the block is for.
"""

import numpy as np
import pytest
from blocks.threedi import (
    DEFAULT_K,
    DescriptorError,
    ThreeDiProvider,
    kmer_profile,
    kmer_vocabulary,
    read_descriptors,
    strip_structure_suffix,
    validate_params,
)


def row(name, amino_acids, threedi, coords="0.0,0.0"):
    return "\t".join([name, amino_acids, threedi, coords])


def descriptor_file(*rows):
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# reading the descriptor file
# ---------------------------------------------------------------------------


def test_the_protid_comes_from_the_name_field():
    text = descriptor_file(row("P60709.pdb ALPHA", "ACDE", "DVQL"))
    assert read_descriptors(text) == {"P60709": "DVQL"}


def test_the_third_field_is_the_one_read():
    """Field 2 is amino acids and field 3 is 3Di. This pins which is which."""
    text = descriptor_file(row("P1.pdb A", "AAAA", "DVQL"))
    assert read_descriptors(text)["P1"] == "DVQL"


def test_a_length_mismatch_between_the_sequences_is_rejected():
    """Foldseek emits one 3Di letter per residue; a mismatch means misread fields."""
    text = descriptor_file(row("P1.pdb A", "ACDEF", "DVQL"))
    with pytest.raises(DescriptorError, match="one 3Di letter per"):
        read_descriptors(text)


def test_identical_sequence_fields_are_rejected():
    """A duplicated column would yield a sequence profile called a structural one."""
    text = descriptor_file(row("P1.pdb A", "ACDE", "ACDE"))
    with pytest.raises(DescriptorError, match="duplicated column"):
        read_descriptors(text)


def test_the_wrong_number_of_fields_is_rejected():
    with pytest.raises(DescriptorError, match="tab-separated fields"):
        read_descriptors("P1.pdb A\tACDE\tDVQL\n")


def test_a_duplicate_protid_is_rejected_rather_than_overwritten():
    text = descriptor_file(row("P1.pdb A", "ACDE", "DVQL"), row("P1.pdb B", "ACDE", "QLLL"))
    with pytest.raises(DescriptorError, match="more than once"):
        read_descriptors(text)


def test_blank_lines_are_ignored():
    text = row("P1.pdb A", "ACDE", "DVQL") + "\n\n" + row("P2.pdb A", "ACDE", "QLLL") + "\n"
    assert list(read_descriptors(text)) == ["P1", "P2"]


def test_file_order_is_preserved():
    text = descriptor_file(
        row("Z9.pdb A", "AC", "DV"), row("A1.pdb A", "AC", "QL"), row("M5.pdb A", "AC", "LL")
    )
    assert list(read_descriptors(text)) == ["Z9", "A1", "M5"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("P60709.pdb", "P60709"),
        ("P60709.cif", "P60709"),
        ("P60709.pdb.gz", "P60709"),
        ("P60709", "P60709"),
        # The `rstrip(".pdb")` bug (FOLLOWUPS #5) would eat these.
        ("PROTEINB.pdb", "PROTEINB"),
        ("MYPROTD.pdb", "MYPROTD"),
        ("ABCP.pdb", "ABCP"),
    ],
)
def test_the_suffix_is_removed_as_a_suffix_not_a_character_set(name, expected):
    assert strip_structure_suffix(name) == expected


# ---------------------------------------------------------------------------
# the k-mer profile
# ---------------------------------------------------------------------------


def test_kmers_are_counted_with_a_sliding_window():
    protids, features, vocabulary, _ = kmer_profile({"P1": "AAAB"}, k=2, scaling="counts")
    assert protids == ["P1"]
    counts = dict(zip(vocabulary, features[0]))
    assert counts == {"AA": 2.0, "AB": 1.0}


def test_the_vocabulary_is_sorted_and_observed_only():
    """Not the full 20**k grid: at k=3 that is 8000 columns of mostly zeros."""
    vocabulary = kmer_vocabulary({"P1": "DVQL", "P2": "AAAA"}, k=2)
    assert vocabulary == sorted(vocabulary)
    assert vocabulary == ["AA", "DV", "QL", "VQ"]


def test_frequencies_sum_to_one():
    _, features, _, _ = kmer_profile({"S": "ABAB", "L": "AABB"}, k=2, scaling="frequency")
    assert features.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_frequency_removes_the_length_signal_that_counts_keeps():
    """The property, stated as a limit rather than as a threshold.

    Two sequences of the same composition and different lengths never match
    exactly -- the shorter one's end effects weigh more -- but the gap shrinks
    as both grow, while the gap in raw counts grows with the length ratio.
    """
    pairs = [("AB" * 5, "AB" * 25), ("AB" * 50, "AB" * 250)]
    gaps = []
    for short, long in pairs:
        _, freq, _, _ = kmer_profile({"S": short, "L": long}, k=2, scaling="frequency")
        gaps.append(float(np.abs(freq[0] - freq[1]).max()))
    assert gaps[1] < gaps[0] / 5, gaps

    _, counts, _, _ = kmer_profile({"S": "AB" * 50, "L": "AB" * 250}, k=2, scaling="counts")
    assert counts[1].sum() == pytest.approx(5 * counts[0].sum(), rel=0.02)


def test_counts_scaling_keeps_the_length_signal():
    """The opposite of the default, and the reason the default is not this."""
    _, features, _, _ = kmer_profile({"S": "ABAB", "L": "ABABABABAB"}, k=2, scaling="counts")
    assert features[1].sum() > 2 * features[0].sum()


def test_l2_scaling_gives_unit_rows():
    _, features, _, _ = kmer_profile({"P1": "ABAB", "P2": "AAAA"}, k=2, scaling="l2")
    assert np.linalg.norm(features, axis=1) == pytest.approx([1.0, 1.0])


def test_a_protein_shorter_than_k_is_reported_and_left_as_zeros():
    """Not NaN, and not silently dropped from the cohort."""
    protids, features, _, too_short = kmer_profile({"P1": "AB", "P2": "ABCDE"}, k=3)
    assert too_short == ["P1"]
    assert protids == ["P1", "P2"]
    assert features[0].sum() == 0.0
    assert not np.isnan(features).any()


def test_an_explicit_vocabulary_is_honored():
    """So two cohorts can be compared on the same columns when that is wanted."""
    _, features, vocabulary, _ = kmer_profile(
        {"P1": "AAAB"}, k=2, scaling="counts", vocabulary=["AA", "ZZ"]
    )
    assert vocabulary == ["AA", "ZZ"]
    assert list(features[0]) == [2.0, 0.0]


def test_an_unknown_scaling_is_rejected():
    with pytest.raises(DescriptorError, match="not valid"):
        kmer_profile({"P1": "AB"}, k=2, scaling="softmax")


@pytest.mark.parametrize("k", [0, -1, "3", 2.5])
def test_an_invalid_k_is_rejected(k):
    with pytest.raises(DescriptorError, match="positive integer"):
        kmer_profile({"P1": "AB"}, k=k)


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


def test_defaults_are_frequency_and_k3():
    params = validate_params({})
    assert params["k"] == DEFAULT_K == 3
    assert params["scaling"] == "frequency"


def test_a_boolean_k_is_rejected():
    """`True` is an int in Python and would silently become k=1."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_params({"k": True})


def test_an_unknown_scaling_is_rejected_in_params():
    with pytest.raises(ValueError, match="not valid"):
        validate_params({"scaling": "softmax"})


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


def write_descriptors(tmp_path, text):
    directory = tmp_path / "protein_features"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "3di_descriptors.tsv"
    path.write_text(text)
    return path


def test_the_provider_is_always_available():
    """Foldseek is already a dependency; a missing file is a missing input."""
    available, reason = ThreeDiProvider().is_available()
    assert available and reason == ""


def test_a_missing_descriptor_file_names_the_rule_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_3di_descriptors"):
        ThreeDiProvider().compute(Ctx(str(tmp_path)), {})


def test_the_block_carries_features_and_a_manifest(tmp_path):
    write_descriptors(
        tmp_path,
        descriptor_file(row("P1.pdb A", "ACDEFG", "DVQLLL"), row("P2.pdb A", "ACDEFG", "DVQLLA")),
    )
    result = ThreeDiProvider().compute(Ctx(str(tmp_path)), {"k": 2})
    assert result.spec.kind == "features"
    assert result.spec.provider == "threedi"
    assert result.spec.metric == "euclidean"
    assert result.protids == ["P1", "P2"]
    assert result.features.shape[0] == 2
    assert result.manifest["extra"]["k"] == 2
    assert result.manifest["extra"]["n_kmers"] == result.features.shape[1]


def test_the_block_is_fusable():
    """Unlike taxonomy, local structure is a legitimate geometry input (ADR 0003)."""
    provider = ThreeDiProvider()
    assert provider.spec_schema({})["k"] == DEFAULT_K


def test_the_vocabulary_is_recorded_when_it_is_small(tmp_path):
    write_descriptors(tmp_path, descriptor_file(row("P1.pdb A", "ACDE", "DVQL")))
    result = ThreeDiProvider().compute(Ctx(str(tmp_path)), {"k": 2})
    assert result.manifest["extra"]["kmers"] == ["DV", "QL", "VQ"]


def test_an_empty_descriptor_file_is_an_error(tmp_path):
    write_descriptors(tmp_path, "")
    with pytest.raises(DescriptorError, match="no 3Di descriptors"):
        ThreeDiProvider().compute(Ctx(str(tmp_path)), {})


def test_two_computations_of_the_same_input_agree(tmp_path):
    """The block ends up in the output tree, so it must not be a source of churn."""
    write_descriptors(
        tmp_path, descriptor_file(row("P1.pdb A", "ACDEFG", "DVQLLL"), row("P2.pdb A", "AC", "DV"))
    )
    first = ThreeDiProvider().compute(Ctx(str(tmp_path)), {"k": 2})
    second = ThreeDiProvider().compute(Ctx(str(tmp_path)), {"k": 2})
    assert np.array_equal(first.features, second.features)
    assert first.manifest["cache_key"] == second.manifest["cache_key"]


def test_the_provider_registers_under_its_name():
    from blocks import threedi
    from spaces.registry import BLOCK_GROUP, get_provider

    threedi.register()
    assert isinstance(get_provider(BLOCK_GROUP, "threedi"), ThreeDiProvider)
