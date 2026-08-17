#!/usr/bin/env python
"""The biophys block: bulk physicochemical descriptors from the sequence.

Where `tmscore` is global shape and `threedi` is local shape, this block is not
shape at all. Hydropathy, charge and isoelectric point separate membrane
proteins from soluble ones and acidic from basic ones, and they do it for
proteins whose structures are too different for any superposition to relate.
That is the point of having it beside the structural blocks rather than instead
of them -- see ADR 0001 on co-registration.

**No new dependency and no new fetch.** The sequences are already in
`uniprot_features.tsv`, which both modes of the pipeline already produce, and
the descriptors are computed here from published constants rather than from
Biopython. Biopython is a fine library and is already in `envs/web_apis.yml`,
but `compute_block` runs in `envs/analysis.yml`, which does not have it. Adding
it there would change that environment's hash and force a fresh solve of the
environment whose pins exist precisely because a fresh solve once installed a
numpy-2-built matplotlib beside `numpy=1.23.5` (ADR 0006). Four hundred lines of
constants are not worth that risk, and forty are not worth it either.

The constants are Biopython's, so the two agree; `tests/test_biophys_block.py`
checks that against the real library whenever it happens to be installed, and
skips when it is not. That is the whole reason for choosing its constants over
anyone else's.

**Every default descriptor is intensive** -- a per-residue mean, a fraction, or
a pH. Molecular weight is available and is *not* a default, because MW is about
110 Da per residue and nothing else: fusing it makes protein length a principal
axis of a biology map, which is the exact argument ADR 0003 uses to keep pLDDT
out of the geometry. A config may still ask for it, and asking is recorded in
the manifest and warned about on stderr, because the choice should be visible
to whoever reads the map afterwards rather than buried in a config file.
"""

from __future__ import annotations
import csv
import os
import sys

import numpy as np
from spaces.base import BlockResult, BlockSpec
from spaces.manifest import Manifest, file_digest

__all__ = [
    "DEFAULT_DESCRIPTORS",
    "DESCRIPTORS",
    "BiophysError",
    "BiophysProvider",
    "charge_at_ph",
    "descriptor_matrix",
    "isoelectric_point",
    "read_sequences",
]

FEATURES_FILENAME = "uniprot_features.tsv"
FEATURES_SUBDIR = "protein_features"

PROTID_COLUMN = "protid"
DEFAULT_SEQUENCE_COLUMN = "Sequence"

#: The twenty residues every constant table below is defined over. A sequence
#: character outside this set (`X` for unknown, `U` for selenocysteine, `B` and
#: `Z` for the ambiguous pairs) has no published hydropathy or pKa, so it is
#: excluded from the sums and counted, rather than being guessed at or silently
#: treated as a zero -- a zero is a real hydropathy value, roughly glycine's.
STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: Kyte & Doolittle (1982) hydropathy index.
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

#: Average (not monoisotopic) molecular weights of the free amino acids, as
#: carried by ``Bio.Data.IUPACData.protein_weights``. A residue in a chain has
#: lost a water, so the chain weight is the sum of these minus one water per
#: peptide bond -- which is `sum - (n - 1) * WATER`, i.e. `sum - n * WATER +
#: WATER`. Stored as free-amino-acid weights rather than residue weights so the
#: arithmetic is the same as the reference implementation's and the two can be
#: compared without a fudge factor.
AMINO_ACID_WEIGHTS = {
    "A": 89.0932,
    "C": 121.1582,
    "D": 133.1027,
    "E": 147.1293,
    "F": 165.1891,
    "G": 75.0666,
    "H": 155.1546,
    "I": 131.1729,
    "K": 146.1876,
    "L": 131.1729,
    "M": 149.2113,
    "N": 132.1179,
    "P": 115.1305,
    "Q": 146.1445,
    "R": 174.201,
    "S": 105.0926,
    "T": 119.1192,
    "V": 117.1463,
    "W": 204.2252,
    "Y": 181.1885,
}
WATER = 18.0153

#: Residues that carry a charge, with the pKa of the protonated form. Bjellqvist
#: et al. (Electrophoresis 1993, 14, 1023-1031; 1994, 15, 529-539), which is the
#: set ``Bio.SeqUtils.IsoelectricPoint`` uses.
#:
#: **Not the EMBOSS set** (Nterm 9.69, K 10.5, R 12.4, Cterm 2.34, D 3.86,
#: E 4.25, C 8.33), which is the other table in wide circulation and is the one
#: a search engine offers first. The two give pI values that differ by a few
#: tenths and net charges that differ by whole units at the extremes, and
#: neither is wrong -- they are calibrated against different experiments. What
#: would be wrong is mixing them, or believing a cross-check against Biopython
#: that was never going to agree. The first draft of this file used the EMBOSS
#: numbers and `test_the_pka_tables_are_biopythons` is what caught it.
POSITIVE_PKS = {"Nterm": 7.5, "K": 10.0, "R": 12.0, "H": 5.98}
NEGATIVE_PKS = {"Cterm": 3.55, "D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}

#: The terminal pKa depends on which residue is at the terminus. Only these
#: residues shift it; everything else uses the generic value above.
NTERM_PKS = {"A": 7.59, "M": 7.00, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.70}
CTERM_PKS = {"D": 4.55, "E": 4.75}

AROMATIC_RESIDUES = "FWY"

DEFAULT_PH = 7.0

#: Widest pH the isoelectric point search will consider. Wider than any real
#: protein's pI, so the bisection's bracket is never the thing that determines
#: the answer.
PI_SEARCH_RANGE = (0.0, 14.0)
PI_TOLERANCE = 1e-4


class BiophysError(ValueError):
    """The features file is not shaped the way this block requires."""


def _composition(sequence: str) -> tuple:
    """Counts of the standard residues, and how many characters were not one.

    Returns:
        A ``(counts, n_unknown)`` pair. ``counts`` is a dict over
        :data:`STANDARD_RESIDUES` only.
    """
    counts = dict.fromkeys(STANDARD_RESIDUES, 0)
    unknown = 0
    for character in sequence:
        if character in counts:
            counts[character] += 1
        else:
            unknown += 1
    return counts, unknown


def _terminal_pks(sequence: str) -> tuple:
    """The N- and C-terminal pKa for this particular sequence."""
    positive = dict(POSITIVE_PKS)
    negative = dict(NEGATIVE_PKS)
    if sequence:
        positive["Nterm"] = NTERM_PKS.get(sequence[0], POSITIVE_PKS["Nterm"])
        negative["Cterm"] = CTERM_PKS.get(sequence[-1], NEGATIVE_PKS["Cterm"])
    return positive, negative


def charge_at_ph(sequence: str, ph: float) -> float:
    """Net charge of `sequence` at `ph`, by Henderson-Hasselbalch.

    Each ionizable group contributes its fractional protonation rather than a
    whole charge, so the result is continuous in pH -- which is what makes the
    bisection in :func:`isoelectric_point` well defined.
    """
    counts, _ = _composition(sequence)
    positive_pks, negative_pks = _terminal_pks(sequence)
    counts = dict(counts)
    counts["Nterm"] = 1 if sequence else 0
    counts["Cterm"] = 1 if sequence else 0

    positive = 0.0
    for group, pk in positive_pks.items():
        ratio = 10 ** (pk - ph)
        positive += counts.get(group, 0) * (ratio / (ratio + 1.0))
    negative = 0.0
    for group, pk in negative_pks.items():
        ratio = 10 ** (ph - pk)
        negative += counts.get(group, 0) * (ratio / (ratio + 1.0))
    return positive - negative


def isoelectric_point(sequence: str) -> float:
    """The pH at which `sequence` carries no net charge.

    Plain bisection. Net charge is strictly decreasing in pH -- every term is --
    so the root is unique and bracketing it is enough. The reference
    implementation searches only [4.05, 12] and returns the bracket edge for
    anything outside it; this one uses [0, 14] so that a genuinely basic protein
    gets its real pI rather than the edge of somebody's search window.
    """
    if not sequence:
        return float("nan")
    low, high = PI_SEARCH_RANGE
    while high - low > PI_TOLERANCE:
        middle = (low + high) / 2.0
        if charge_at_ph(sequence, middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _gravy(sequence: str) -> float:
    counts, _ = _composition(sequence)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return sum(HYDROPATHY[residue] * n for residue, n in counts.items()) / total


def _aromaticity(sequence: str) -> float:
    counts, _ = _composition(sequence)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return sum(counts[residue] for residue in AROMATIC_RESIDUES) / total


def _charge_per_residue(sequence: str, ph: float) -> float:
    counts, _ = _composition(sequence)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return charge_at_ph(sequence, ph) / total


def _molecular_weight(sequence: str) -> float:
    counts, _ = _composition(sequence)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    weight = sum(AMINO_ACID_WEIGHTS[residue] * n for residue, n in counts.items())
    return weight - (total - 1) * WATER


def _length(sequence: str) -> float:
    counts, _ = _composition(sequence)
    return float(sum(counts.values()))


#: Every descriptor this block can compute: how to compute it, whether it is
#: intensive, and what it means.
#:
#: ``intensive`` is the load-bearing field. An extensive descriptor grows with
#: the protein, so a space containing one is partly a map of protein length --
#: and because length correlates with almost everything, that axis will look
#: like signal. The flag is what lets the provider say so out loud instead of
#: leaving it for a reader to notice.
DESCRIPTORS = {
    "gravy": {
        "compute": lambda sequence, ph: _gravy(sequence),
        "intensive": True,
        "description": "mean Kyte-Doolittle hydropathy; high means hydrophobic",
    },
    "aromaticity": {
        "compute": lambda sequence, ph: _aromaticity(sequence),
        "intensive": True,
        "description": "fraction of F, W and Y (Lobry & Gautier 1994)",
    },
    "isoelectric_point": {
        "compute": lambda sequence, ph: isoelectric_point(sequence),
        "intensive": True,
        "description": "pH of zero net charge",
    },
    "charge_per_residue": {
        "compute": _charge_per_residue,
        "intensive": True,
        "description": "net charge at the configured pH, divided by length",
    },
    "molecular_weight": {
        "compute": lambda sequence, ph: _molecular_weight(sequence),
        "intensive": False,
        "description": "average molecular weight in daltons; grows with length",
    },
    "length": {
        "compute": lambda sequence, ph: _length(sequence),
        "intensive": False,
        "description": "number of standard residues; the length axis itself",
    },
}

#: The intensive descriptors, in a fixed order. Fixed rather than derived from
#: `DESCRIPTORS` at call time so that adding a descriptor to the table does not
#: silently change what every existing config computes.
DEFAULT_DESCRIPTORS = ("gravy", "aromaticity", "isoelectric_point", "charge_per_residue")


def read_sequences(text: str, sequence_column: str = DEFAULT_SEQUENCE_COLUMN) -> dict:
    """Parse `protid` and the sequence column out of a features TSV.

    Args:
        text: the file's contents, with a header row.
        sequence_column: which column holds the amino-acid sequence.

    Returns:
        A dict of protid -> sequence, in file order.
    """
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if reader.fieldnames is None:
        raise BiophysError("the features file is empty; it must have a header row.")
    for required in (PROTID_COLUMN, sequence_column):
        if required not in reader.fieldnames:
            raise BiophysError(
                f"the features file has no {required!r} column. Found: "
                f"{', '.join(reader.fieldnames)}. In cluster mode this file is "
                "supplied by the user, so a missing column means the file does not "
                "carry what the biophys block needs rather than that the pipeline "
                "went wrong."
            )

    sequences = {}
    for number, row in enumerate(reader, start=2):
        protid = (row.get(PROTID_COLUMN) or "").strip()
        if not protid:
            raise BiophysError(f"line {number} of the features file has an empty protid.")
        if protid in sequences:
            raise BiophysError(
                f"{protid} appears more than once in the features file. Which row "
                "describes it is ambiguous, so this is an error rather than a "
                "last-one-wins."
            )
        sequence = (row.get(sequence_column) or "").strip().upper()
        sequences[protid] = sequence
    return sequences


def descriptor_matrix(sequences, descriptors=DEFAULT_DESCRIPTORS, ph: float = DEFAULT_PH):
    """The (N, D) descriptor matrix, its protids, and what could not be computed.

    Returns:
        A ``(protids, features, unusable, n_unknown_residues)`` tuple.
        ``unusable`` lists protids whose sequence held no standard residue --
        their row is zeros, not NaN, so a reducer does not have to special-case
        them, and they are named so the zeros are not mistaken for measurements.
    """
    names = validate_descriptors(descriptors)
    protids = list(sequences)
    features = np.zeros((len(protids), len(names)), dtype=np.float64)
    unusable = []
    n_unknown = 0

    for row, protid in enumerate(protids):
        sequence = sequences[protid]
        counts, unknown = _composition(sequence)
        n_unknown += unknown
        if sum(counts.values()) == 0:
            unusable.append(protid)
            continue
        for column, name in enumerate(names):
            features[row, column] = DESCRIPTORS[name]["compute"](sequence, ph)

    return protids, features.astype(np.float32), unusable, n_unknown


def validate_descriptors(descriptors) -> list:
    """Check a descriptor list and return it as a list of names."""
    if isinstance(descriptors, str):
        raise BiophysError(
            f"biophys.descriptors must be a list of names, not the single string "
            f"{descriptors!r}."
        )
    names = list(descriptors)
    if not names:
        raise BiophysError("biophys.descriptors is empty; the block would have no columns.")
    unknown = [name for name in names if name not in DESCRIPTORS]
    if unknown:
        raise BiophysError(
            f"biophys.descriptors: {', '.join(repr(n) for n in unknown)} "
            f"{'is' if len(unknown) == 1 else 'are'} not valid. "
            f"Allowed: {', '.join(sorted(DESCRIPTORS))}."
        )
    duplicates = [name for index, name in enumerate(names) if name in names[:index]]
    if duplicates:
        raise BiophysError(
            f"biophys.descriptors lists {duplicates[0]!r} more than once, which would "
            "give the block two identical columns and weight it twice."
        )
    return names


def extensive_descriptors(names) -> list:
    """Which of `names` grow with protein length."""
    return [name for name in names if not DESCRIPTORS[name]["intensive"]]


def validate_params(params: dict) -> dict:
    """Validate and normalize this provider's parameters."""
    params = dict(params or {})
    descriptors = params.get("descriptors", DEFAULT_DESCRIPTORS)
    try:
        names = validate_descriptors(descriptors)
    except BiophysError as exc:
        raise ValueError(str(exc)) from None

    ph = params.get("ph", DEFAULT_PH)
    if isinstance(ph, bool) or not isinstance(ph, (int, float)):
        raise ValueError(f"biophys.ph must be a number, got {ph!r}.")
    ph = float(ph)
    if not 0.0 <= ph <= 14.0:
        raise ValueError(f"biophys.ph must be between 0 and 14, got {ph}.")

    params["descriptors"] = names
    params["ph"] = ph
    params.setdefault("sequence_column", DEFAULT_SEQUENCE_COLUMN)
    return params


class BiophysProvider:
    """Produces the `biophys` block from the sequences in the features file."""

    spec_schema = staticmethod(validate_params)

    #: Bumped when the block's meaning changes, so cached blocks invalidate.
    version = "1"

    def __init__(self, features_path: str | None = None):
        self._features_path = features_path

    def features_path_for(self, ctx, params: dict) -> str:
        if params.get("features_path"):
            return params["features_path"]
        if self._features_path:
            return self._features_path
        # Where the features table lives depends on the mode, not on the config:
        # search mode fetches it into the output directory, cluster mode takes
        # the user's file from the input directory. The Snakefile resolves that
        # and passes the path through `--provider-input features_file=...`,
        # which lands here. Not a param, because a machine-specific path in the
        # params would put a machine-specific value in the cache key.
        named = (getattr(ctx, "extras", None) or {}).get("features_file")
        if named:
            return named
        return ctx.path(FEATURES_SUBDIR, FEATURES_FILENAME)

    def is_available(self) -> tuple:
        """Always available: the constants are in this file and nothing is imported.

        ADR 0006 requires that every block named by the default config work with
        zero optional dependencies installed. This one is the reason the block
        does not use Biopython.
        """
        return True, ""

    def compute(self, ctx, params: dict) -> BlockResult:
        params = validate_params(params)
        path = self.features_path_for(ctx, params)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"the biophys block needs {path}. In search mode the "
                "`fetch_uniprot_metadata` rule produces it; in cluster mode it is "
                "the `features_file` named in the config."
            )
        with open(path) as handle:
            sequences = read_sequences(handle.read(), params["sequence_column"])
        if not sequences:
            raise BiophysError(f"{path} lists no proteins.")

        names = params["descriptors"]
        protids, features, unusable, n_unknown = descriptor_matrix(
            sequences, descriptors=names, ph=params["ph"]
        )

        extensive = extensive_descriptors(names)
        if extensive:
            # Loud, because the consequence shows up as a plausible-looking axis
            # rather than as an error. See the module docstring and ADR 0003.
            print(
                f"[biophys] {', '.join(extensive)} grow with protein length, so this "
                "block makes length a component of the geometry. Recorded in the "
                "manifest as length_proportional_descriptors.",
                file=sys.stderr,
            )

        manifest = Manifest.build(
            "block",
            params.get("block_id", "biophys"),
            provider="biophys",
            params=params,
            inputs={"features": file_digest(path)},
            protids=protids,
            seed=getattr(ctx, "seed", 123456),
            extra={
                "descriptors": names,
                "ph": params["ph"],
                "length_proportional_descriptors": extensive,
                "proteins_without_usable_sequence": unusable,
                "non_standard_residues": n_unknown,
            },
        )
        spec = BlockSpec(
            id=params.get("block_id", "biophys"),
            kind="features",
            fusable=True,
            metric="euclidean",
            # `zscore_within`, and here it is right where it was wrong for
            # `threedi`. These columns are a handful of dense physical
            # quantities on incomparable scales -- pI runs 4 to 12, charge per
            # residue runs about -0.1 to 0.1 -- so without standardizing, the
            # euclidean distance is the isoelectric point and nothing else.
            normalization=params.get("normalization", "zscore_within"),
            provider="biophys",
            params=params,
            version=self.version,
        )
        return BlockResult(
            spec=spec,
            protids=protids,
            features=features,
            manifest=manifest.to_dict(),
        )


def register() -> None:
    """Register this provider as a built-in."""
    from spaces.registry import BLOCK_GROUP, register_builtin

    register_builtin(BLOCK_GROUP, "biophys", BiophysProvider)
