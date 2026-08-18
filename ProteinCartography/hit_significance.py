#!/usr/bin/env python
"""Best significance per hit, across every query that found it.

``cohort.select`` can rank candidates by significance, which is the only one of
the three selection rules that is both reproducible and principled (ADR 0008).
It needs a score per accession, and this is what produces one: the strongest
evidence any input protein has for that hit.

**ADR 0008 asked for TM-score and TM-score is not available here.** The search
runs against the Foldseek *web API*, whose ``.m8`` output has 21 columns and no
alignment TM-score -- see ``constants.FOLDSEEK_COLUMN_NAMES``. The TM-score
matrix the pipeline is built around comes from the local all-versus-all run in
``foldseek_clustering``, which happens *after* the structures are downloaded. So
ranking the cohort by TM-score would need the structures whose download the
ranking is meant to decide. The measure used instead is the e-value the web API
does report, with the bit score alongside it. The ADR has been corrected.

Two properties matter for a cohort rule and both are deliberate:

*Best across queries, not per query.* A protein found weakly by four inputs and
strongly by one is a strong hit. Taking the best score means a hit is ranked by
the best evidence anyone has for it, and it makes the result independent of the
order the input proteins happen to be listed in.

*Absent is not zero.* An accession with no recorded evidence is left out of the
table entirely rather than given a worst-possible score. ``cohort.select`` ranks
those last and counts them, so "we never scored this" stays distinguishable from
"we scored this and it was bad".
"""

from __future__ import annotations
import argparse
import os
import re
import sys

import constants
import pandas as pd

__all__ = ["aggregate_significance", "blast_significance", "foldseek_significance"]

#: The AlphaFold model id embedded in a Foldseek target, e.g. `AF-P60709-F1-model_v4`.
_AF_MODEL = re.compile(r"AF-(.*)-F1-model")

#: Columns of the emitted table. `protid` first so the file reads like every
#: other per-protein table in the pipeline.
OUTPUT_COLUMNS = ["protid", "evalue", "bits", "n_queries", "sources"]


class TmalignOutputError(RuntimeError):
    """The .m8 came from ``--mode tmalign``, where the columns mean something else."""


#: Below this, a value is an e-value and not a TM-score. A TM-score is in [0, 1]
#: by construction and Foldseek does not report structurally meaningless ones, so
#: in practice they bottom out around 0.3; a useful 3Di-AA search reaches values
#: many orders of magnitude smaller than this.
_TMALIGN_EVALUE_FLOOR = 1e-3

#: In tmalign mode the bit-score column holds roughly TM-score times 100, so it
#: cannot exceed ~100. A 3Di-AA search puts real bit scores in the thousands.
_TMALIGN_MAX_BITS = 100


def _looks_like_tmalign(evalues, bits) -> bool:
    """Whether this file's `evalue` column is really a TM-score.

    ``foldseek_apiquery.py`` accepts ``--mode tmalign``, and the server returns
    the *same 21 columns in the same positions* with different meanings: the
    column ``constants.FOLDSEEK_COLUMN_NAMES`` calls ``evalue`` holds a TM-score,
    and the one it calls ``bits`` holds roughly that times 100.

    Nothing renames, so nothing errors -- the polarity simply inverts. Verified
    against a live tmalign query: 938 hits, e-value column spanning 0.402 to
    0.9999, top hit actin at 0.9999 and bottom hit an unrelated pyrophosphatase
    at 0.402. Ranking that column ascending selects the pyrophosphatase.

    Two conditions rather than one, because a weak 3Di-AA search really can
    return only e-values near 1. Requiring the bit scores to *also* be
    TM-score-shaped makes a false positive very unlikely, and the check is
    written to err toward not firing.
    """
    if evalues.empty or evalues.isna().all():
        return False
    finite = evalues.dropna()
    bounded = bool((finite >= 0).all() and (finite <= 1).all())
    never_small = bool(finite.min() > _TMALIGN_EVALUE_FLOOR)
    small_bits = bool(bits.dropna().empty or (bits.dropna() <= _TMALIGN_MAX_BITS).all())
    return bounded and never_small and small_bits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--foldseek-m8",
        nargs="*",
        default=[],
        help="Foldseek .m8 files from the web API, across all queries and databases.",
    )
    parser.add_argument(
        "-b",
        "--blast-results",
        nargs="*",
        default=[],
        help="BLAST results TSVs, one per query protein.",
    )
    parser.add_argument(
        "-m",
        "--refseq-mapping",
        nargs="*",
        default=[],
        help=(
            "RefSeq-to-UniProt mapping TSVs written by map_refseq_ids --mapping-output. "
            "Without these the BLAST e-values cannot be keyed to an accession and the "
            "BLAST evidence is skipped, with a warning."
        ),
    )
    parser.add_argument("-o", "--output", required=True, help="Destination TSV.")
    return parser.parse_args()


def foldseek_significance(m8_files) -> pd.DataFrame:
    """Best e-value and bit score per UniProt accession, across all .m8 files."""
    rows = []
    for path in m8_files:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        frame = pd.read_csv(path, sep="\t", names=constants.FOLDSEEK_COLUMN_NAMES)
        if frame.empty or "target" not in frame.columns:
            continue
        model = frame["target"].astype(str).str.split(" ", expand=True)[0]
        accession = model.str.extract(_AF_MODEL, expand=False)
        keep = accession.notna()
        if not keep.any():
            continue
        evalues = pd.to_numeric(frame.loc[keep, "evalue"], errors="coerce")
        bits = pd.to_numeric(frame.loc[keep, "bits"], errors="coerce")
        if _looks_like_tmalign(evalues, bits):
            raise TmalignOutputError(
                f"{path} looks like `foldseek_apiquery.py --mode tmalign` output: "
                f"its e-value column runs {evalues.min():.4g} to {evalues.max():.4g}, "
                "entirely inside [0, 1], with bit scores at TM-score scale. In that "
                "mode the server reuses the same column positions for different "
                "quantities -- the 'e-value' is a TM-score -- so ranking it "
                "ascending would select the *least* similar structures. Refusing "
                "rather than guessing: the mode is not recorded in the output, so "
                "this file cannot be interpreted with confidence. The pipeline runs "
                "3diaa mode, which this does support."
            )
        rows.append(
            pd.DataFrame(
                {
                    "protid": accession[keep].values,
                    "evalue": evalues.values,
                    "bits": bits.values,
                    # The query column is `job.pdb` for every web-API result, so
                    # the file itself is the only thing that identifies the query.
                    "query": os.path.abspath(path),
                }
            )
        )
    return _combine(rows, source="foldseek")


def _read_refseq_mapping(mapping_files) -> dict:
    """RefSeq or EMBL accession to UniProt accession.

    The mapping is many-to-one in practice. When two source accessions map to
    the same UniProt entry that is fine -- the scores are reduced by `min`/`max`
    afterwards, so it only means more evidence for the same protein.
    """
    mapping = {}
    for path in mapping_files:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        frame = pd.read_csv(path, sep="\t")
        if not {"from", "to"}.issubset(frame.columns):
            continue
        for source, target in zip(frame["from"], frame["to"]):
            if isinstance(source, str) and isinstance(target, str):
                mapping[source.strip()] = target.strip()
    return mapping


def blast_significance(blast_files, mapping_files) -> pd.DataFrame:
    """Best e-value and bit score per UniProt accession, across all queries.

    BLAST reports RefSeq accessions, so this needs the RefSeq-to-UniProt mapping
    that ``map_refseq_ids`` performs. Without it the evidence is dropped rather
    than guessed at, because a wrong key here silently ranks the wrong protein.
    """
    mapping = _read_refseq_mapping(mapping_files)
    if not mapping:
        if blast_files:
            print(
                "[hit_significance] no RefSeq-to-UniProt mapping given, so BLAST "
                "e-values cannot be keyed to an accession; using Foldseek evidence "
                "only. Pass --refseq-mapping to include BLAST.",
                file=sys.stderr,
            )
        return _combine([], source="blast")

    column_names = [name for name in constants.BLAST_OUTFMT.split(" ") if name != "6"]
    rows = []
    for path in blast_files:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        frame = pd.read_csv(path, sep="\t", names=column_names)
        if frame.empty or "sacc" not in frame.columns:
            continue
        accession = frame["sacc"].astype(str).map(mapping)
        keep = accession.notna()
        if not keep.any():
            continue
        rows.append(
            pd.DataFrame(
                {
                    "protid": accession[keep].values,
                    "evalue": pd.to_numeric(frame.loc[keep, "evalue"], errors="coerce").values,
                    "bits": pd.to_numeric(frame.loc[keep, "bitscore"], errors="coerce").values,
                    "query": os.path.abspath(path),
                }
            )
        )
    return _combine(rows, source="blast")


def _combine(rows, source: str) -> pd.DataFrame:
    """Reduce per-alignment rows to one row per accession."""
    empty = pd.DataFrame(columns=["protid", "evalue", "bits", "n_queries", "sources"])
    if not rows:
        return empty
    stacked = pd.concat(rows, axis=0, ignore_index=True)
    stacked = stacked.dropna(subset=["protid"])
    if stacked.empty:
        return empty
    grouped = stacked.groupby("protid", sort=True).agg(
        evalue=("evalue", "min"),  # lower is stronger
        bits=("bits", "max"),  # higher is stronger
        n_queries=("query", "nunique"),
    )
    grouped["sources"] = source
    return grouped.reset_index()


def aggregate_significance(m8_files, blast_files, mapping_files) -> pd.DataFrame:
    """One row per accession, holding the best evidence from either method."""
    foldseek = foldseek_significance(m8_files)
    blast = blast_significance(blast_files, mapping_files)
    both = pd.concat([foldseek, blast], axis=0, ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = both.groupby("protid", sort=True).agg(
        evalue=("evalue", "min"),
        bits=("bits", "max"),
        n_queries=("n_queries", "sum"),
        # Both methods finding a hit is worth seeing; it is the concordance
        # signal this pipeline is otherwise built to look for.
        sources=("sources", lambda values: "+".join(sorted(set(values)))),
    )
    return merged.reset_index()[OUTPUT_COLUMNS]


def main():
    args = parse_args()
    table = aggregate_significance(args.foldseek_m8, args.blast_results, args.refseq_mapping)
    table.to_csv(args.output, sep="\t", index=False)
    print(
        f"[hit_significance] wrote {len(table)} scored accessions to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
