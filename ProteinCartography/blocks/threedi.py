#!/usr/bin/env python
"""The 3Di block: local structural alphabet, as k-mer frequencies.

Foldseek encodes each residue's local tertiary environment as one letter of a
20-state alphabet, giving a string the same length as the amino-acid sequence.
This block turns that string into a k-mer profile, which is a *local* structural
description -- and that is the point, because the block it sits beside is not.

TM-score is a global superposition score. Two proteins that share a domain but
differ in how their domains are arranged score poorly, because no single rigid
superposition fits both. The 3Di profile has no such notion: a shared domain
contributes the same k-mers whatever the hinge angle between it and the rest of
the protein. So `threedi` and `tmscore` disagree in a specific, interpretable
way, and the disagreement is the signal -- see ADR 0001 on co-registration.

**This provider computes nothing.** Extraction needs the foldseek binary, which
lives in a different conda environment from the one `compute_block` runs in, so
`foldseek structureto3didescriptor` runs as its own rule and writes a TSV. The
provider reads that TSV, exactly as `tmscore` reads the pivoted matrix. The
arrangement also keeps this module free of subprocesses and therefore testable
from a string.

On the descriptor file's shape, and one hazard in it. Each row is four
tab-separated fields: name, amino-acid sequence, 3Di sequence, and a coordinate
descriptor. Fields 2 and 3 are **both uppercase letter strings of exactly the
same length**, which means reading the wrong one produces an amino-acid k-mer
profile labelled as a structural one, with nothing to indicate it. That is the
same shape of defect as the tmalign column collision recorded in ADR 0008, so
the reader asserts what it can: four fields, equal lengths, and the two
sequences not being identical.
"""

from __future__ import annotations
import os
from collections import Counter

import numpy as np
from spaces.base import BlockResult, BlockSpec
from spaces.manifest import Manifest, file_digest

__all__ = ["DESCRIPTORS_FILENAME", "ThreeDiProvider", "kmer_profile", "read_descriptors"]

DESCRIPTORS_FILENAME = "3di_descriptors.tsv"
FEATURES_SUBDIR = "protein_features"

#: Column index of the 3Di string in the descriptor file. Named rather than
#: inlined because field 2 is the amino-acid sequence and is indistinguishable
#: by shape -- see the module docstring.
THREEDI_COLUMN = 2
AMINO_ACID_COLUMN = 1
DESCRIPTOR_FIELDS = 4

#: Structure suffixes foldseek accepts, longest first so `.pdb.gz` is stripped
#: before `.gz`. Suffix removal is explicit because `str.rstrip(".pdb")` strips a
#: *character set* and would corrupt any protid ending in b, d, p or a dot --
#: a bug this pipeline has had before.
STRUCTURE_SUFFIXES = (
    ".pdb.gz",
    ".cif.gz",
    ".mmcif.gz",
    ".ent.gz",
    ".pdb",
    ".cif",
    ".mmcif",
    ".ent",
)

VALID_SCALINGS = ("frequency", "counts", "l2")
DEFAULT_K = 3


class DescriptorError(ValueError):
    """The 3Di descriptor file is not shaped the way this reader requires."""


def strip_structure_suffix(name: str) -> str:
    for suffix in STRUCTURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_descriptors(text: str) -> dict:
    """Parse ``foldseek structureto3didescriptor`` output into protid -> 3Di.

    Args:
        text: the file's contents. Four tab-separated fields per row, no header.

    Returns:
        A dict in file order, so downstream ordering is the caller's choice
        rather than an accident of dict construction.
    """
    descriptors = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != DESCRIPTOR_FIELDS:
            raise DescriptorError(
                f"line {number} of the 3Di descriptor file has {len(fields)} "
                f"tab-separated fields, expected {DESCRIPTOR_FIELDS} "
                "(name, amino acids, 3Di, coordinates). This reader depends on "
                "the field order, so it refuses a file it does not recognize."
            )
        # The name field is `<filename> <chain descriptor>`.
        protid = strip_structure_suffix(fields[0].split()[0]) if fields[0].strip() else ""
        amino_acids = fields[AMINO_ACID_COLUMN].strip()
        threedi = fields[THREEDI_COLUMN].strip()

        if not protid:
            raise DescriptorError(f"line {number} of the 3Di descriptor file has no name field.")
        if len(amino_acids) != len(threedi):
            raise DescriptorError(
                f"{protid}: the amino-acid sequence is {len(amino_acids)} residues and "
                f"the 3Di string is {len(threedi)}. Foldseek emits one 3Di letter per "
                "residue, so a mismatch means the fields are not what this reader "
                "thinks they are."
            )
        if amino_acids and amino_acids == threedi:
            raise DescriptorError(
                f"{protid}: the amino-acid and 3Di fields are identical. They encode "
                "different things, so this is a duplicated column rather than a "
                "protein -- reading it would produce a sequence profile labelled as "
                "a structural one."
            )
        if protid in descriptors:
            raise DescriptorError(
                f"{protid} appears more than once in the 3Di descriptor file. Which "
                "structure it refers to is ambiguous, so this is an error rather "
                "than a last-one-wins."
            )
        descriptors[protid] = threedi
    return descriptors


def kmer_vocabulary(descriptors, k: int) -> list:
    """Every k-mer present in the cohort, sorted.

    Restricted to observed k-mers rather than the full 20**k grid. At k=3 that
    grid is 8000 columns, nearly all of them zero, and the block's feature
    dimension is already cohort-scoped anyway -- `tmscore`'s profile
    representation has exactly N columns. The vocabulary goes in the manifest so
    the block stays interpretable after the fact.
    """
    observed = set()
    for sequence in descriptors.values():
        for start in range(len(sequence) - k + 1):
            observed.add(sequence[start : start + k])
    return sorted(observed)


def kmer_profile(descriptors, k: int = DEFAULT_K, scaling: str = "frequency", vocabulary=None):
    """The (N, V) k-mer matrix, its protids, and its vocabulary.

    ``frequency`` is the default because raw counts scale with protein length,
    which would make the dominant axis of any reduction "how long is this
    protein" -- a real property, and not the one this block exists to measure.
    """
    if scaling not in VALID_SCALINGS:
        raise DescriptorError(
            f"threedi.scaling: {scaling!r} is not valid. Allowed: {', '.join(VALID_SCALINGS)}."
        )
    if not isinstance(k, int) or k < 1:
        raise DescriptorError(f"threedi.k must be a positive integer, got {k!r}.")

    protids = list(descriptors)
    if vocabulary is None:
        vocabulary = kmer_vocabulary(descriptors, k)
    position = {kmer: index for index, kmer in enumerate(vocabulary)}

    features = np.zeros((len(protids), len(vocabulary)), dtype=np.float64)
    too_short = []
    for row, protid in enumerate(protids):
        sequence = descriptors[protid]
        if len(sequence) < k:
            too_short.append(protid)
            continue
        counts = Counter(sequence[start : start + k] for start in range(len(sequence) - k + 1))
        for kmer, count in counts.items():
            index = position.get(kmer)
            if index is not None:
                features[row, index] = count

    if scaling == "frequency":
        totals = features.sum(axis=1, keepdims=True)
        # A protein shorter than k contributes no k-mers. Its row stays zero
        # rather than becoming NaN, and it is reported separately.
        np.divide(features, totals, out=features, where=totals > 0)
    elif scaling == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        np.divide(features, norms, out=features, where=norms > 0)

    return protids, features.astype(np.float32), vocabulary, too_short


def validate_params(params: dict) -> dict:
    """Validate and normalize this provider's parameters."""
    params = dict(params or {})
    k = params.get("k", DEFAULT_K)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"threedi.k must be a positive integer, got {k!r}.")
    scaling = params.get("scaling", "frequency")
    if scaling not in VALID_SCALINGS:
        raise ValueError(
            f"threedi.scaling: {scaling!r} is not valid. Allowed: {', '.join(VALID_SCALINGS)}."
        )
    params["k"] = k
    params["scaling"] = scaling
    return params


class ThreeDiProvider:
    """Produces the `threedi` block from foldseek's 3Di descriptor output."""

    spec_schema = staticmethod(validate_params)

    #: Bumped when the block's meaning changes, so cached blocks invalidate.
    version = "1"

    def __init__(self, descriptors_path: str | None = None):
        self._descriptors_path = descriptors_path

    def descriptors_path_for(self, ctx, params: dict) -> str:
        if params.get("descriptors_path"):
            return params["descriptors_path"]
        if self._descriptors_path:
            return self._descriptors_path
        return ctx.path(FEATURES_SUBDIR, DESCRIPTORS_FILENAME)

    def is_available(self) -> tuple:
        """Always available: foldseek is already a pipeline dependency.

        The descriptor file's absence is a missing *input*, reported when the
        block is computed. Reporting it here would make a survey run before the
        extraction rule has run look like a broken installation.
        """
        return True, ""

    def compute(self, ctx, params: dict) -> BlockResult:
        params = validate_params(params)
        path = self.descriptors_path_for(ctx, params)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"the threedi block needs {path}, which the `extract_3di_descriptors` "
                "rule produces. Run the pipeline up to that rule first."
            )
        with open(path) as handle:
            descriptors = read_descriptors(handle.read())
        if not descriptors:
            raise DescriptorError(f"{path} contains no 3Di descriptors.")

        protids, features, vocabulary, too_short = kmer_profile(
            descriptors, k=params["k"], scaling=params["scaling"]
        )

        manifest = Manifest.build(
            "block",
            params.get("block_id", "threedi"),
            provider="threedi",
            params=params,
            inputs={"descriptors": file_digest(path)},
            protids=protids,
            seed=getattr(ctx, "seed", 123456),
            extra={
                "k": params["k"],
                "scaling": params["scaling"],
                "n_kmers": len(vocabulary),
                # Recorded so the columns remain interpretable, but only when the
                # vocabulary is small enough to be worth reading. At k=4 it can
                # run to tens of thousands of entries and would dwarf the rest.
                "kmers": vocabulary if len(vocabulary) <= 1024 else None,
                "proteins_shorter_than_k": too_short,
            },
        )
        spec = BlockSpec(
            id=params.get("block_id", "threedi"),
            kind="features",
            fusable=True,
            metric="euclidean",
            # `unit_mean_distance`, not `zscore_within`. Standardizing each k-mer
            # column would give a k-mer seen in three proteins the same weight as
            # one seen in all of them, which inflates noise in a sparse profile.
            # Scaling the block as a whole is what ADR 0002 actually needs.
            normalization=params.get("normalization", "unit_mean_distance"),
            provider="threedi",
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

    register_builtin(BLOCK_GROUP, "threedi", ThreeDiProvider)
