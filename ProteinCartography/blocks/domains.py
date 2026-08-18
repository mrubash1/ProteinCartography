#!/usr/bin/env python
"""The domains block: which curated families a protein belongs to.

The other three blocks measure the protein. This one reads what a curated
database says about it, which makes it different in kind and is the reason it
is worth having. Pfam and InterPro annotations come from HMMs built on
alignments far wider than any cohort here, so two proteins can share a family
assignment while being too distant for TM-score to relate them at all. Where
this space agrees with `structure` is confirmation; where it disagrees is
either a remote homolog the structural map missed, or an annotation artifact,
and both are findings (ADR 0001).

**A missing annotation is not an absent domain.** This is the same distinction
ADR 0009 draws for the similarity matrix's `0.0` fill, and it matters more here
because the bias is not random: well-studied proteins are better annotated, so
an unannotated row means "nobody has looked", not "nothing is there". The block
therefore reports how many proteins came back with no domains at all rather
than letting them sit in the matrix as origin points that appear to be similar
to one another. They are not similar; they are jointly uninformative.

On the metric. Jaccard is the natural distance between two sets and the reason
`jaccard` is in `spaces.base.METRICS`, but the reducer core is not metric-aware
-- `reduce_space` feeds features straight into a euclidean PCA -- so declaring
it here would write a claim into the manifest that nothing honors. Euclidean
distance between two binary presence vectors is the square root of the number
of families they differ on, which is a real and usable distance; it simply
weights a protein with many annotations more heavily than Jaccard would.
Asking for `jaccard` is an error that says so, rather than a setting that is
quietly ignored.
"""

from __future__ import annotations
import csv
import os
import re

import numpy as np
from spaces.base import BlockResult, BlockSpec
from spaces.manifest import Manifest, file_digest

__all__ = [
    "DOMAIN_SOURCES",
    "DomainsError",
    "DomainsProvider",
    "domain_matrix",
    "domain_vocabulary",
    "parse_accessions",
    "read_domains",
]

FEATURES_FILENAME = "uniprot_features.tsv"
FEATURES_SUBDIR = "protein_features"
PROTID_COLUMN = "protid"

#: Where domain assignments come from, and how to recognize one. The pattern is
#: the point: fields 21 and 22 of the features table are both semicolon-
#: separated lists of uppercase accessions and are indistinguishable by shape,
#: so reading the wrong one produces a plausible block of the wrong thing. The
#: same hazard as the 3Di descriptor file's two sequence columns, and it gets
#: the same treatment -- an assertion that fails when they are swapped.
DOMAIN_SOURCES = {
    "pfam": {"column": "Pfam", "pattern": re.compile(r"^PF\d{5}$")},
    "interpro": {"column": "InterPro", "pattern": re.compile(r"^IPR\d{6}$")},
}

DEFAULT_SOURCE = "pfam"

#: The only metric the reducer actually applies. See the module docstring.
SUPPORTED_METRICS = ("euclidean",)


class DomainsError(ValueError):
    """The features file's domain columns are not shaped the way this reader requires."""


def parse_accessions(cell: str) -> list:
    """Split a UniProt list cell into accessions, in order, without blanks.

    UniProt writes these as `PF00022;PF00125;` -- semicolon separated *and*
    semicolon terminated, so a naive split leaves an empty final element.
    """
    return [item.strip() for item in (cell or "").split(";") if item.strip()]


def read_domains(text: str, source: str = DEFAULT_SOURCE) -> dict:
    """Parse protid -> list of domain accessions from a features TSV.

    Args:
        text: the file's contents, with a header row.
        source: a key of :data:`DOMAIN_SOURCES`, or ``"both"``.

    Returns:
        A dict of protid -> accession list, in file order. The lists are
        deduplicated and sorted, so a block does not depend on the order
        UniProt happened to emit.
    """
    sources = list(DOMAIN_SOURCES) if source == "both" else [source]
    if any(name not in DOMAIN_SOURCES for name in sources):
        raise DomainsError(
            f"domains.source: {source!r} is not valid. "
            f"Allowed: {', '.join(sorted(DOMAIN_SOURCES))}, both."
        )

    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if reader.fieldnames is None:
        raise DomainsError("the features file is empty; it must have a header row.")
    columns = [DOMAIN_SOURCES[name]["column"] for name in sources]
    for column in [PROTID_COLUMN, *columns]:
        if column not in reader.fieldnames:
            raise DomainsError(
                f"the features file has no {column!r} column. Found: "
                f"{', '.join(reader.fieldnames)}. In cluster mode this file is "
                "supplied by the user, so a missing column means the file does not "
                "carry what the domains block needs rather than that the pipeline "
                "went wrong."
            )

    domains = {}
    for number, row in enumerate(reader, start=2):
        protid = (row.get(PROTID_COLUMN) or "").strip()
        if not protid:
            raise DomainsError(f"line {number} of the features file has an empty protid.")
        if protid in domains:
            raise DomainsError(
                f"{protid} appears more than once in the features file. Which row "
                "describes it is ambiguous, so this is an error rather than a "
                "last-one-wins."
            )
        found = []
        for name in sources:
            column = DOMAIN_SOURCES[name]["column"]
            pattern = DOMAIN_SOURCES[name]["pattern"]
            accessions = parse_accessions(row.get(column))
            wrong = [item for item in accessions if not pattern.match(item)]
            if wrong:
                raise DomainsError(
                    f"{protid}: the {column!r} column holds "
                    f"{', '.join(repr(w) for w in wrong[:3])}, which "
                    f"{'does' if len(wrong) == 1 else 'do'} not look like "
                    f"{name} accessions ({pattern.pattern}). The features table's "
                    "domain columns are adjacent lists of uppercase accessions and "
                    "are indistinguishable by shape, so this reader checks rather "
                    "than trusting the header."
                )
            found.extend(accessions)
        domains[protid] = sorted(set(found))
    return domains


def domain_vocabulary(domains) -> list:
    """Every accession present in the cohort, sorted.

    Observed only, for the same reason the 3Di block does not use the full
    k-mer grid: Pfam has over twenty thousand families and a cohort touches a
    handful of them. The vocabulary goes in the manifest so the columns stay
    interpretable afterwards.
    """
    observed = set()
    for accessions in domains.values():
        observed.update(accessions)
    return sorted(observed)


def domain_matrix(domains, vocabulary=None):
    """The (N, V) binary presence matrix, its protids, and its vocabulary.

    Returns:
        A ``(protids, features, vocabulary, without_domains)`` tuple.
        ``without_domains`` lists protids with no annotation at all. Their rows
        are zeros, which under any distance makes them look alike -- they are
        not alike, they are jointly unannotated, and the caller is told so.
    """
    protids = list(domains)
    if vocabulary is None:
        vocabulary = domain_vocabulary(domains)
    position = {accession: index for index, accession in enumerate(vocabulary)}

    features = np.zeros((len(protids), len(vocabulary)), dtype=np.float64)
    without_domains = []
    for row, protid in enumerate(protids):
        accessions = domains[protid]
        if not accessions:
            without_domains.append(protid)
            continue
        for accession in accessions:
            index = position.get(accession)
            if index is not None:
                features[row, index] = 1.0

    return protids, features.astype(np.float32), vocabulary, without_domains


def validate_params(params: dict) -> dict:
    """Validate and normalize this provider's parameters."""
    params = dict(params or {})
    source = params.get("source", DEFAULT_SOURCE)
    if source != "both" and source not in DOMAIN_SOURCES:
        raise ValueError(
            f"domains.source: {source!r} is not valid. "
            f"Allowed: {', '.join(sorted(DOMAIN_SOURCES))}, both."
        )

    metric = params.get("metric", "euclidean")
    if metric == "jaccard":
        raise ValueError(
            "domains.metric: 'jaccard' is the right distance for sets and is not "
            "available yet. The reducer core is not metric-aware -- `reduce_space` "
            "feeds features into a euclidean PCA -- so accepting it here would "
            "record a metric in the manifest that nothing applies. Euclidean "
            "distance on these binary vectors is the square root of the number of "
            "families two proteins differ on, which is usable; it weights a "
            "heavily annotated protein more than Jaccard would."
        )
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"domains.metric: {metric!r} is not valid. " f"Allowed: {', '.join(SUPPORTED_METRICS)}."
        )

    params["source"] = source
    params["metric"] = metric
    return params


class DomainsProvider:
    """Produces the `domains` block from the features table's Pfam/InterPro columns."""

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
        # Passed by the Snakefile, because the table is in the output directory
        # in search mode and the input directory in cluster mode. See
        # `compute_block --provider-input`.
        named = (getattr(ctx, "extras", None) or {}).get("features_file")
        if named:
            return named
        return ctx.path(FEATURES_SUBDIR, FEATURES_FILENAME)

    def is_available(self) -> tuple:
        """Always available: the annotations are already in the features table.

        ADR 0006 requires every block in the default config to work with zero
        optional dependencies. This one needs no fetch and no new package -- the
        Pfam and InterPro columns arrive with the UniProt metadata the pipeline
        already downloads.
        """
        return True, ""

    def compute(self, ctx, params: dict) -> BlockResult:
        params = validate_params(params)
        path = self.features_path_for(ctx, params)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"the domains block needs {path}. In search mode the "
                "`fetch_uniprot_metadata` rule produces it; in cluster mode it is "
                "the `features_file` named in the config."
            )
        with open(path) as handle:
            domains = read_domains(handle.read(), params["source"])
        if not domains:
            raise DomainsError(f"{path} lists no proteins.")

        protids, features, vocabulary, without_domains = domain_matrix(domains)
        if not vocabulary:
            raise DomainsError(
                f"none of the {len(protids)} proteins in {path} carry a "
                f"{params['source']} annotation, so the block would have no columns. "
                "This is a property of the cohort, not a bug: it means the domain "
                "space cannot be built for this run."
            )

        manifest = Manifest.build(
            "block",
            params.get("block_id", "domains"),
            provider="domains",
            params=params,
            inputs={"features": file_digest(path)},
            protids=protids,
            seed=getattr(ctx, "seed", 123456),
            extra={
                "source": params["source"],
                "n_families": len(vocabulary),
                # Small vocabularies are worth reading back; a large one would
                # dwarf the rest of the manifest. Same rule as the 3Di k-mers.
                "families": vocabulary if len(vocabulary) <= 1024 else None,
                # Not a diagnostic afterthought. An unannotated protein is an
                # origin point that resembles every other unannotated protein,
                # and the resemblance is an artifact of the annotation effort.
                "proteins_without_domains": without_domains,
                "annotated_fraction": round(1.0 - len(without_domains) / len(protids), 6),
            },
        )
        spec = BlockSpec(
            id=params.get("block_id", "domains"),
            kind="features",
            fusable=True,
            metric=params["metric"],
            # `unit_mean_distance`, for the reason `threedi` uses it and
            # `biophys` does not: this is a sparse indicator matrix, and
            # standardizing each column would give a family seen in two
            # proteins the same weight as one seen in all of them.
            normalization=params.get("normalization", "unit_mean_distance"),
            provider="domains",
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

    register_builtin(BLOCK_GROUP, "domains", DomainsProvider)
