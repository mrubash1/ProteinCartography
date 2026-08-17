"""Tests for the biophys block.

Two things carry the weight here.

The first is that these descriptors are *constants transcribed from a paper*,
and a transcription error produces numbers that are wrong by a little and look
entirely reasonable. So the numeric tests are properties wherever a property
exists -- charge is zero at the isoelectric point, hydropathy of a pure
sequence is that residue's index, a fraction sums the way a fraction must --
and the values themselves are checked against Biopython when Biopython happens
to be installed. That cross-check is the reason this module uses Biopython's
constants rather than any other published set.

The second is the intensive/extensive distinction. `molecular_weight` is about
110 Da per residue and nothing else, so a space that includes it is partly a
map of protein length. Several tests below exist to keep that fact attached to
the descriptor rather than resident in someone's memory.
"""

import math

import numpy as np
import pytest
from blocks.biophys import (
    AMINO_ACID_WEIGHTS,
    CTERM_PKS,
    DEFAULT_DESCRIPTORS,
    DESCRIPTORS,
    HYDROPATHY,
    NEGATIVE_PKS,
    NTERM_PKS,
    POSITIVE_PKS,
    BiophysError,
    BiophysProvider,
    charge_at_ph,
    descriptor_matrix,
    extensive_descriptors,
    isoelectric_point,
    read_sequences,
    validate_descriptors,
    validate_params,
)

try:  # pragma: no cover - depends on the environment, not on a code path
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    HAS_BIOPYTHON = True
except ImportError:  # pragma: no cover
    ProteinAnalysis = None
    HAS_BIOPYTHON = False

needs_biopython = pytest.mark.skipif(
    not HAS_BIOPYTHON,
    reason="Biopython is not installed here. It is not a dependency of this block -- "
    "these tests check the transcribed constants against it when it is available.",
)

#: A real sequence, so the cross-checks are not all run on homopolymers.
ACTIN_FRAGMENT = (
    "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTF"
)


def features_file(*rows, sequence_column="Sequence"):
    header = "\t".join(["protid", "Entry", sequence_column, "Length"])
    lines = [header]
    for protid, sequence in rows:
        lines.append("\t".join([protid, protid, sequence, str(len(sequence))]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# reading the features file
# ---------------------------------------------------------------------------


def test_the_sequence_is_read_by_column_name():
    text = features_file(("P60709", "ACDE"))
    assert read_sequences(text) == {"P60709": "ACDE"}


def test_file_order_is_preserved():
    text = features_file(("Z9", "AC"), ("A1", "CD"), ("M5", "DE"))
    assert list(read_sequences(text)) == ["Z9", "A1", "M5"]


def test_a_missing_sequence_column_names_what_it_found():
    text = "protid\tEntry\tLength\nP1\tP1\t4\n"
    with pytest.raises(BiophysError, match="no 'Sequence' column"):
        read_sequences(text)


def test_a_missing_protid_column_is_rejected():
    text = "Entry\tSequence\nP1\tACDE\n"
    with pytest.raises(BiophysError, match="no 'protid' column"):
        read_sequences(text)


def test_an_alternative_sequence_column_can_be_named():
    text = features_file(("P1", "ACDE"), sequence_column="seq")
    assert read_sequences(text, "seq") == {"P1": "ACDE"}


def test_a_duplicate_protid_is_rejected_rather_than_overwritten():
    text = features_file(("P1", "ACDE"), ("P1", "EDCA"))
    with pytest.raises(BiophysError, match="more than once"):
        read_sequences(text)


def test_an_empty_protid_is_rejected():
    text = "protid\tSequence\n\tACDE\n"
    with pytest.raises(BiophysError, match="empty protid"):
        read_sequences(text)


def test_an_empty_file_is_rejected():
    with pytest.raises(BiophysError, match="header row"):
        read_sequences("")


def test_sequences_are_upper_cased():
    """UniProt emits uppercase, but a hand-made cluster-mode file may not."""
    text = features_file(("P1", "acde"))
    assert read_sequences(text)["P1"] == "ACDE"


# ---------------------------------------------------------------------------
# charge and the isoelectric point
# ---------------------------------------------------------------------------


def test_charge_decreases_monotonically_with_ph():
    """The property the bisection in `isoelectric_point` relies on."""
    charges = [charge_at_ph(ACTIN_FRAGMENT, ph) for ph in np.arange(0.0, 14.01, 0.5)]
    assert all(b < a for a, b in zip(charges, charges[1:]))


def test_the_charge_at_the_isoelectric_point_is_zero():
    """The definition, checked rather than assumed."""
    for sequence in ("KKKKK", "EEEEE", ACTIN_FRAGMENT, "ACDEFGHIKLMNPQRSTVWY"):
        assert charge_at_ph(sequence, isoelectric_point(sequence)) == pytest.approx(0.0, abs=1e-3)


def test_a_basic_protein_has_a_high_pi_and_an_acidic_one_a_low_pi():
    assert isoelectric_point("KKKKKKKKKK") > 10.0
    assert isoelectric_point("EEEEEEEEEE") < 4.5


def test_the_pi_search_is_not_clipped_to_a_narrow_window():
    """Biopython searches only [4.05, 12] and returns the edge outside it.

    A poly-arginine peptide's real pI is above 12, so a clipped search reports
    the window edge and calls it a measurement.
    """
    assert isoelectric_point("RRRRRRRRRRRRRRRRRRRR") > 12.0


def test_an_empty_sequence_has_no_isoelectric_point():
    assert math.isnan(isoelectric_point(""))


def test_an_empty_sequence_is_uncharged():
    assert charge_at_ph("", 7.0) == 0.0


# ---------------------------------------------------------------------------
# the descriptors
# ---------------------------------------------------------------------------


def test_gravy_of_a_homopolymer_is_that_residues_hydropathy():
    for residue, index in HYDROPATHY.items():
        protids, features, _, _ = descriptor_matrix({"P": residue * 5}, descriptors=["gravy"])
        assert features[0][0] == pytest.approx(index, abs=1e-5), residue


def test_gravy_is_a_mean_and_not_a_sum():
    """A sum would grow with length; the block promises intensive columns."""
    _, short, _, _ = descriptor_matrix({"P": "ACDE"}, descriptors=["gravy"])
    _, long, _, _ = descriptor_matrix({"P": "ACDE" * 10}, descriptors=["gravy"])
    assert short[0][0] == pytest.approx(long[0][0], abs=1e-5)


def test_aromaticity_is_the_fraction_of_f_w_and_y():
    _, features, _, _ = descriptor_matrix({"P": "FWYA"}, descriptors=["aromaticity"])
    assert features[0][0] == pytest.approx(0.75)


def test_molecular_weight_subtracts_one_water_per_peptide_bond():
    _, one, _, _ = descriptor_matrix({"P": "A"}, descriptors=["molecular_weight"])
    _, two, _, _ = descriptor_matrix({"P": "AA"}, descriptors=["molecular_weight"])
    assert one[0][0] == pytest.approx(89.0932, abs=1e-3)
    assert two[0][0] == pytest.approx(2 * 89.0932 - 18.0153, abs=1e-3)


def test_non_standard_residues_are_excluded_from_the_means_and_counted():
    """`X` has no published hydropathy. Treating it as 0.0 would be a real value."""
    _, plain, _, n_plain = descriptor_matrix({"P": "AAAA"}, descriptors=["gravy"])
    _, mixed, _, n_mixed = descriptor_matrix({"P": "AAXXAA"}, descriptors=["gravy"])
    assert plain[0][0] == pytest.approx(mixed[0][0], abs=1e-5)
    assert (n_plain, n_mixed) == (0, 2)


def test_a_sequence_with_no_standard_residue_is_reported_and_left_as_zeros():
    """Not NaN, and not silently dropped from the cohort."""
    protids, features, unusable, _ = descriptor_matrix({"P1": "XXXX", "P2": "ACDE"})
    assert unusable == ["P1"]
    assert protids == ["P1", "P2"]
    assert not np.isnan(features).any()
    assert features[0].sum() == 0.0


def test_the_columns_follow_the_requested_order():
    names = ["aromaticity", "gravy"]
    _, features, _, _ = descriptor_matrix({"P": "FWYA"}, descriptors=names)
    assert features[0][0] == pytest.approx(0.75)
    assert features[0][1] == pytest.approx(np.mean([HYDROPATHY[r] for r in "FWYA"]), abs=1e-5)


def test_charge_per_residue_responds_to_the_ph():
    _, acidic, _, _ = descriptor_matrix({"P": "KKKEEE"}, descriptors=["charge_per_residue"], ph=2.0)
    _, basic, _, _ = descriptor_matrix({"P": "KKKEEE"}, descriptors=["charge_per_residue"], ph=12.0)
    assert acidic[0][0] > basic[0][0]


# ---------------------------------------------------------------------------
# intensive versus extensive
# ---------------------------------------------------------------------------


def test_every_default_descriptor_is_intensive():
    """The promise the module docstring makes, enforced rather than described."""
    assert extensive_descriptors(DEFAULT_DESCRIPTORS) == []


def test_molecular_weight_and_length_are_declared_extensive():
    assert set(extensive_descriptors(list(DESCRIPTORS))) == {"molecular_weight", "length"}


def test_an_intensive_descriptor_does_not_track_length_and_an_extensive_one_does():
    """The distinction, measured on the descriptors themselves."""
    sequences = {f"P{n}": "ACDEFGHIKL" * n for n in range(1, 11)}
    lengths = np.array([len(s) for s in sequences.values()], dtype=float)

    _, intensive, _, _ = descriptor_matrix(sequences, descriptors=["gravy"])
    _, extensive, _, _ = descriptor_matrix(sequences, descriptors=["molecular_weight"])

    assert np.ptp(intensive[:, 0]) == pytest.approx(0.0, abs=1e-5)
    assert np.corrcoef(extensive[:, 0], lengths)[0, 1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


def test_defaults_are_the_intensive_descriptors_at_ph_seven():
    params = validate_params({})
    assert params["descriptors"] == list(DEFAULT_DESCRIPTORS)
    assert params["ph"] == 7.0


def test_an_unknown_descriptor_is_rejected_and_lists_the_valid_ones():
    with pytest.raises(BiophysError, match="not valid"):
        validate_descriptors(["gravy", "hydrophobic_moment"])


def test_a_repeated_descriptor_is_rejected():
    """Two identical columns would weight the descriptor twice."""
    with pytest.raises(BiophysError, match="more than once"):
        validate_descriptors(["gravy", "gravy"])


def test_an_empty_descriptor_list_is_rejected():
    with pytest.raises(BiophysError, match="no columns"):
        validate_descriptors([])


def test_a_bare_string_is_rejected_rather_than_iterated_as_characters():
    """`descriptors: gravy` in YAML is a string, and list('gravy') is five names."""
    with pytest.raises(BiophysError, match="not the single string"):
        validate_descriptors("gravy")


@pytest.mark.parametrize("ph", [-1.0, 14.5, "7", True, None])
def test_an_invalid_ph_is_rejected(ph):
    with pytest.raises(ValueError, match="biophys.ph"):
        validate_params({"ph": ph})


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
    """ADR 0006: a block in the default config works with nothing installed."""
    available, reason = BiophysProvider().is_available()
    assert available and reason == ""


def test_a_missing_features_file_names_both_modes_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_uniprot_metadata"):
        BiophysProvider().compute(Ctx(str(tmp_path)), {})


def test_the_block_carries_features_and_a_manifest(tmp_path):
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT), ("P2", "ACDEFGHIKL")))
    result = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.kind == "features"
    assert result.spec.provider == "biophys"
    assert result.spec.metric == "euclidean"
    assert result.protids == ["P1", "P2"]
    assert result.features.shape == (2, len(DEFAULT_DESCRIPTORS))
    assert result.manifest["extra"]["descriptors"] == list(DEFAULT_DESCRIPTORS)


def test_the_block_standardizes_its_columns_by_default(tmp_path):
    """Without it the euclidean distance is the isoelectric point and nothing else.

    This checks the provider in isolation, which is not by itself enough: the
    default only survives if `compute_block` refrains from filling the parameter
    in, and for a while it did not. `test_compute_block.py` covers that path.
    """
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT)))
    result = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.normalization == "zscore_within"


def test_the_default_block_records_no_length_proportional_descriptors(tmp_path):
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT)))
    result = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    assert result.manifest["extra"]["length_proportional_descriptors"] == []


def test_asking_for_molecular_weight_is_recorded_and_warned_about(tmp_path, capsys):
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT)))
    result = BiophysProvider().compute(
        Ctx(str(tmp_path)), {"descriptors": ["gravy", "molecular_weight"]}
    )
    assert result.manifest["extra"]["length_proportional_descriptors"] == ["molecular_weight"]
    assert "length" in capsys.readouterr().err


def test_the_block_is_fusable(tmp_path):
    """Unlike pLDDT, an intensive physicochemical descriptor is not circular.

    pLDDT is barred from geometries because it tracks length (ADR 0003). These
    descriptors are barred from tracking length instead, which is what makes
    them admissible -- see `test_every_default_descriptor_is_intensive`.
    """
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT)))
    result = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    assert result.spec.fusable is True
    assert result.spec.not_fusable_reason is None


def test_an_empty_features_file_is_an_error(tmp_path):
    write_features(tmp_path, "protid\tSequence\n")
    with pytest.raises(BiophysError, match="lists no proteins"):
        BiophysProvider().compute(Ctx(str(tmp_path)), {})


def test_two_computations_of_the_same_input_agree(tmp_path):
    """The block ends up in the output tree, so it must not be a source of churn."""
    write_features(tmp_path, features_file(("P1", ACTIN_FRAGMENT), ("P2", "ACDEFGHIKL")))
    first = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    second = BiophysProvider().compute(Ctx(str(tmp_path)), {})
    assert np.array_equal(first.features, second.features)
    assert first.manifest["cache_key"] == second.manifest["cache_key"]


def test_the_provider_registers_under_its_name():
    from blocks import biophys
    from spaces.registry import BLOCK_GROUP, get_provider

    biophys.register()
    assert isinstance(get_provider(BLOCK_GROUP, "biophys"), BiophysProvider)


# ---------------------------------------------------------------------------
# the constants, against the library they came from
# ---------------------------------------------------------------------------

CROSS_CHECK_SEQUENCES = [
    ACTIN_FRAGMENT,
    "ACDEFGHIKLMNPQRSTVWY",
    "MKKKKKAAAAAEEEEE",
    "GGGGGGGGGG",
]


@needs_biopython
def test_the_pka_tables_are_biopythons():
    """The tables themselves, not just numbers derived from them.

    This is the test that earns its keep. The first draft of the module used the
    EMBOSS pKa set from memory rather than the Bjellqvist set Biopython uses,
    and every derived check failed at once with no indication of which of the
    fourteen constants was responsible. Comparing the tables says so directly.
    """
    from Bio.SeqUtils import IsoelectricPoint as reference

    assert POSITIVE_PKS == reference.positive_pKs
    assert NEGATIVE_PKS == reference.negative_pKs
    assert NTERM_PKS == reference.pKnterminal
    assert CTERM_PKS == reference.pKcterminal


@needs_biopython
def test_the_weight_table_is_biopythons():
    from Bio.Data import IUPACData

    for residue, weight in AMINO_ACID_WEIGHTS.items():
        assert weight == pytest.approx(IUPACData.protein_weights[residue], abs=1e-4), residue


@needs_biopython
@pytest.mark.parametrize("sequence", CROSS_CHECK_SEQUENCES)
def test_gravy_matches_biopython(sequence):
    _, features, _, _ = descriptor_matrix({"P": sequence}, descriptors=["gravy"])
    assert features[0][0] == pytest.approx(ProteinAnalysis(sequence).gravy(), abs=1e-4)


@needs_biopython
@pytest.mark.parametrize("sequence", CROSS_CHECK_SEQUENCES)
def test_aromaticity_matches_biopython(sequence):
    _, features, _, _ = descriptor_matrix({"P": sequence}, descriptors=["aromaticity"])
    assert features[0][0] == pytest.approx(ProteinAnalysis(sequence).aromaticity(), abs=1e-4)


@needs_biopython
@pytest.mark.parametrize("sequence", CROSS_CHECK_SEQUENCES)
def test_molecular_weight_matches_biopython(sequence):
    _, features, _, _ = descriptor_matrix({"P": sequence}, descriptors=["molecular_weight"])
    expected = ProteinAnalysis(sequence).molecular_weight()
    assert features[0][0] == pytest.approx(expected, rel=1e-5)


@needs_biopython
@pytest.mark.parametrize("sequence", CROSS_CHECK_SEQUENCES)
def test_charge_matches_biopython(sequence):
    for ph in (4.0, 7.0, 10.0):
        expected = ProteinAnalysis(sequence).charge_at_pH(ph)
        assert charge_at_ph(sequence, ph) == pytest.approx(expected, abs=1e-4)


@needs_biopython
@pytest.mark.parametrize("sequence", CROSS_CHECK_SEQUENCES)
def test_the_isoelectric_point_matches_biopython_inside_its_search_window(sequence):
    """Only inside [4.05, 12]: outside it Biopython returns the window edge.

    `test_the_pi_search_is_not_clipped_to_a_narrow_window` covers the outside.
    """
    expected = ProteinAnalysis(sequence).isoelectric_point()
    if not 4.1 < expected < 11.9:
        pytest.skip(f"pI {expected:.2f} is at the edge of Biopython's search window")
    assert isoelectric_point(sequence) == pytest.approx(expected, abs=0.01)
