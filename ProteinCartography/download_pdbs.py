#!/usr/bin/env python
"""Download the cohort's PDB files from AlphaFold.

This is where the cohort is decided. Everything downstream -- every block, every
space, every diagnostic -- is conditioned on the truncation that happens here,
and until now it happened silently. ``cohort.py`` makes the rule explicit and
writes a report; see ``docs/adr/0008-cohort-selection.md``.

The default rule, ``as_filtered``, is a plain prefix of the list in the order it
arrives, which is exactly what this script did before. The behavior change is
that the run now says how many candidates it discarded, that the discarded set
is not reproducible, and whether it was taxonomically different from the set
that was kept.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import sys
from pathlib import Path

import api_utils
import cohort
import fetch_accession
import tqdm
from ratelimiter import RateLimiter

#: The UniProt column the lineage comparison reads, and how to split it. Same
#: convention as ``fetch_uniprot_metadata``: ranks are comma-separated and each
#: carries a parenthesised rank name that is not part of the term.
LINEAGE_COLUMN = "Taxonomic lineage"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file path of a .txt file with one accession per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory in which to save the PDB files.",
    )
    parser.add_argument(
        "-M",
        "--max-structures",
        type=int,
        required=False,
        help="Maximum number of PDB files to download.",
    )
    parser.add_argument(
        "--selection",
        default="as_filtered",
        help=(
            "Which candidates to keep when truncating. 'as_filtered' (the default) "
            "keeps the list's own order, which is what the pipeline has always done "
            "and is not reproducible. 'accession' and 'significance' are both "
            "reproducible and both change which proteins reach the map."
        ),
    )
    parser.add_argument(
        "--significance-table",
        help=(
            "TSV of per-accession significance, used by --selection significance. "
            "Needs a 'protid' column and a column named by --significance-measure."
        ),
    )
    parser.add_argument(
        "--significance-measure",
        help=f"One of {', '.join(sorted(cohort.SIGNIFICANCE_MEASURES))}.",
    )
    parser.add_argument(
        "--uniprot-features",
        help=(
            "uniprot_features.tsv, read only for the lineage comparison between the "
            "retained and discarded sets. Optional; without it that comparison is "
            "omitted rather than guessed at."
        ),
    )
    parser.add_argument(
        "--candidates-before-filtering",
        help="aggregated_hits.txt, so the report can state the pre-filter count.",
    )
    parser.add_argument(
        "--cohort-report",
        help="Where to write the cohort report as JSON. Skipped if not given.",
    )
    args = parser.parse_args()
    return args


def _read_lineages(features_file) -> dict:
    """Accession to lineage terms, or an empty mapping if unavailable.

    Missing, malformed, or lineage-free metadata yields no comparison rather
    than a wrong one -- the diagnostic is not worth failing a download over.
    """
    if not features_file or not Path(features_file).exists():
        return {}
    import pandas as pd

    frame = pd.read_csv(features_file, sep="\t")
    if "protid" not in frame.columns or LINEAGE_COLUMN not in frame.columns:
        return {}
    lineages = {}
    for protid, raw in zip(frame["protid"], frame[LINEAGE_COLUMN]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        lineages[str(protid)] = [rank.split(" (")[0].strip() for rank in raw.split(", ")]
    return lineages


def _read_scores(table, measure) -> dict:
    """Accession to raw significance score, from the aggregated table."""
    if not table:
        return {}
    import pandas as pd

    frame = pd.read_csv(table, sep="\t")
    if "protid" not in frame.columns:
        raise SystemExit(f"significance table {table} has no 'protid' column.")
    if measure not in frame.columns:
        raise SystemExit(
            f"significance table {table} has no {measure!r} column; "
            f"it has {', '.join(frame.columns)}."
        )
    scores = {}
    for protid, value in zip(frame["protid"], frame[measure]):
        if value == value:  # NaN means the pipeline recorded no evidence.
            scores[str(protid)] = float(value)
    return scores


def _count_lines(path) -> int | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def select_cohort(
    accessions,
    maximum=None,
    selection="as_filtered",
    significance_table=None,
    significance_measure=None,
    uniprot_features=None,
    candidates_before_filtering=None,
    report_file=None,
):
    """Apply the selection rule, write the report, and return the retained ids."""
    scores = None
    if selection == "significance":
        scores = _read_scores(significance_table, significance_measure)

    try:
        chosen = cohort.select(
            accessions,
            max_structures=maximum,
            rule=selection,
            scores=scores,
            measure=significance_measure,
        )
    except cohort.CohortError as error:
        raise SystemExit(str(error)) from None

    report = cohort.CohortReport.build(
        chosen,
        accessions,
        n_candidates_before_filtering=_count_lines(candidates_before_filtering),
        lineages=_read_lineages(uniprot_features),
    )
    print(cohort.format_report(report), file=sys.stderr, flush=True)
    if report_file:
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        report.write(report_file)
    return list(chosen.retained)


def download_pdbs(
    input_file: str,
    output_dir: str,
    maximum=None,
    selection="as_filtered",
    significance_table=None,
    significance_measure=None,
    uniprot_features=None,
    candidates_before_filtering=None,
    report_file=None,
):
    """
    Download PDBs for the accessions listed in `input_file` from AlphaFold.

    Args:
        input_file (str): path to an text file containing one accession per line.
        output_dir (str): path to output directory in which to save the PDB files.
        maximum (int): maximum number of accessions to download. If None, downloads all.
        selection (str): which accessions to keep when `maximum` truncates the list.
        significance_table (str): TSV of per-accession scores, for `significance`.
        significance_measure (str): which column of that table to rank by.
        uniprot_features (str): metadata TSV, read only for the lineage comparison.
        candidates_before_filtering (str): hit list before filtering, for the report.
        report_file (str): where to write the cohort report as JSON.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(input_file) as file:
        accessions = file.read().splitlines()

    accessions = select_cohort(
        accessions,
        maximum=maximum,
        selection=selection,
        significance_table=significance_table,
        significance_measure=significance_measure,
        uniprot_features=uniprot_features,
        candidates_before_filtering=candidates_before_filtering,
        report_file=report_file,
    )

    session = api_utils.session_with_retry()
    rate_limiter = RateLimiter(max_calls=100, period=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures_to_accessions = {}
        for accession in accessions:
            future = executor.submit(
                rate_limiter(fetch_accession.fetch_pdb),
                accession=accession,
                output_dir=output_dir,
                session=session,
            )
            futures_to_accessions[future] = accession

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures_to_accessions),
            total=len(futures_to_accessions),
            desc="Downloading PDBs from AlphaFold",
        ):
            try:
                future.result()
            except Exception as exception:
                print(f"Error fetching PDB '{futures_to_accessions[future]}': {exception}")


def main():
    args = parse_args()
    download_pdbs(
        input_file=args.input,
        output_dir=args.output,
        maximum=args.max_structures,
        selection=args.selection,
        significance_table=args.significance_table,
        significance_measure=args.significance_measure,
        uniprot_features=args.uniprot_features,
        candidates_before_filtering=args.candidates_before_filtering,
        report_file=args.cohort_report,
    )


if __name__ == "__main__":
    main()
