#!/usr/bin/env python
"""What each cluster is made of, as a table.

The snakemake entry point for :mod:`enrichment`. Joins a cluster assignment to
an annotation table, tests every (cluster, annotation) pair, corrects for
multiplicity, and writes one tidy row per hypothesis.

Two files land under ``OUTPUT_DIR/enrichment/``:

``cluster_enrichment.tsv``
    One row per hypothesis: cluster, annotation column, term, the counts the
    test was computed from, effect, p, q, and -- when the test could not run --
    the reason. Sorted by q, with a full deterministic tiebreak, so it is
    readable at the top and byte-stable at the bottom.
``manifest.json``
    What was tested, what was not, and why: absent columns, dropped terms,
    universe sizes, and content digests of both inputs.

**Where the clusters come from, and what that means.** The pipeline has exactly
one clustering: Leiden over the TM-score matrix, written by
``leiden_clustering.py``. Spaces do not cluster -- ``reduce_space`` emits
coordinates and nothing else -- so an enrichment table today describes the
`structure` space, not the multi-space map, however many spaces the run built.
This module is deliberately indifferent to that: it takes a cluster table by
path and names the clustering in its output, so per-space clustering is a
wiring change rather than a rewrite. ADR 0012 §1 records the decision; it is
also what PLAN Phase 6's cross-space ARI needs and does not have.

Additive, like every other rule in this work: nothing existing consumes any of
it, and the rule stays out of the DAG unless a config asks for enrichment.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from config_io import load_config
from config_schema import from_legacy
from enrichment import (
    benjamini_hochberg,
    detect_encoding,
    hypergeometric_enrichment,
    mann_whitney_u,
    parse_terms,
    term_counts,
)
from spaces.manifest import Manifest, file_digest

ENRICHMENT_SUBDIR = "enrichment"
TABLE_FILENAME = "cluster_enrichment.tsv"

#: The tidy table's columns, in order. Fixed rather than derived, so a new
#: field cannot quietly reorder a file somebody is diffing.
TABLE_COLUMNS = (
    "clustering",
    "cluster",
    "annotation",
    "kind",
    "term",
    "n_cluster",
    "n_universe",
    "n_term_cluster",
    "n_term_universe",
    "effect",
    "effect_kind",
    "p_value",
    "q_value",
    "significant",
    "note",
)


class EnrichmentError(Exception):
    """The inputs cannot support the table that was asked for."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--configfile", required=True)
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument(
        "--clusters",
        required=True,
        metavar="PATH",
        help=(
            "a table with a protid column and a cluster column. The pipeline's "
            "leiden_features.tsv is the only one it currently produces; any table "
            "of the same shape works, and the clustering is named in the output."
        ),
    )
    parser.add_argument(
        "--annotations",
        required=True,
        metavar="PATH",
        help="a table with a protid column and the annotation columns to test.",
    )
    return parser.parse_args()


def read_keyed_table(path: str, what: str) -> pd.DataFrame:
    """A TSV keyed by protid, with the key checked rather than assumed.

    `protid` by name, never the first column by position: two of the tables
    this reads are written by different scripts and one of them (`leiden`)
    writes its key second on one code path.
    """
    frame = pd.read_csv(path, sep="\t")
    if "protid" not in frame.columns:
        raise EnrichmentError(
            f"{what} at {path} has no `protid` column. Columns: {list(frame.columns)[:10]}"
        )
    duplicated = frame["protid"][frame["protid"].duplicated()].unique()
    if len(duplicated):
        raise EnrichmentError(
            f"{what} at {path} names {len(duplicated)} protid(s) more than once, "
            f"starting with {list(duplicated[:5])}. A duplicated key would count "
            "one protein twice in every universe it appears in."
        )
    return frame.set_index("protid")


def join_on_protid(clusters: pd.DataFrame, annotations: pd.DataFrame, cluster_column: str):
    """Cluster labels beside annotations, over the proteins that have both.

    Returns the joined frame and the protids each side lost, because a table
    computed over an overlap nobody chose looks exactly like one that is not --
    the same argument as ADR 0011 §1, one layer out.
    """
    if cluster_column not in clusters.columns:
        raise EnrichmentError(
            f"the cluster table has no {cluster_column!r} column. Columns: "
            f"{list(clusters.columns)[:10]}. Set `enrichment.cluster_column` to one "
            "of these."
        )
    shared = [protid for protid in clusters.index if protid in annotations.index]
    if not shared:
        raise EnrichmentError(
            "the cluster table and the annotation table share no proteins, so there "
            "is nothing to enrich. Check that both describe the same run."
        )
    joined = annotations.loc[shared].copy()
    labels = clusters.loc[shared, cluster_column]

    # The annotation table usually carries this column already:
    # `aggregated_features.tsv` is built by joining `leiden_features.tsv` into
    # everything else, so both inputs name it. The cluster table is the
    # authority -- it is the one the caller pointed at -- but the two silently
    # disagreeing would mean the tables describe different runs, and an
    # enrichment computed across two runs is not wrong in any visible way.
    if cluster_column in joined.columns:
        disagree = joined.index[joined[cluster_column].values != labels.values]
        if len(disagree):
            raise EnrichmentError(
                f"the cluster table and the annotation table both have a "
                f"{cluster_column!r} column and they disagree for {len(disagree)} "
                f"protein(s), starting with {list(disagree[:5])}. They describe "
                "different runs; enriching across them would compare one run's "
                "clusters against another run's annotations."
            )
        joined = joined.drop(columns=[cluster_column])
    joined.insert(0, cluster_column, labels.values)

    dropped = {
        "unannotated": [p for p in clusters.index if p not in annotations.index],
        "unclustered": [p for p in annotations.index if p not in clusters.index],
    }
    return joined, dropped


def numeric_column(frame: pd.DataFrame, column: str) -> tuple:
    """A continuous column as floats, with unparseable entries named not hidden."""
    original = frame[column]
    values = pd.to_numeric(original, errors="coerce")
    unparseable = int((original.notna() & values.isna()).sum())
    if values.notna().sum() == 0 and original.notna().sum() > 0:
        raise EnrichmentError(
            f"{column!r} is listed under `enrichment.continuous` but holds no numbers "
            f"(for example {original.dropna().iloc[0]!r}). List it under "
            "`enrichment.categorical` instead."
        )
    return values.to_numpy(dtype=np.float64), unparseable


def categorical_rows(frame, column, encoding, clusters, cluster_column, min_term_count):
    """Every (term, cluster) hypothesis for one categorical column.

    The universe is the proteins whose cell is **present**, not every protein in
    the cohort. A protein whose annotation was never determined is not evidence
    that it lacks the term, and counting it as one biases every term toward
    whichever cluster happens to be better annotated -- annotation completeness
    correlates with taxonomy, which is one of the things being tested.

    This cuts harder than it looks, and the reason is the file format rather
    than the statistic. A protein with no Pfam families is written as an empty
    field, which `read_csv` returns as NaN, so after a round-trip through TSV
    "carries no families" and "was never annotated" are the same bytes and both
    leave the universe. On the demo cohort that takes Pfam's universe from 400
    to 232 and raises every fold enrichment accordingly. It is the conventional
    choice for an enrichment analysis -- the background is the annotated
    background -- but it is not free, so `n_universe` is on every row, the
    per-column universe is in the manifest, and any column that loses proteins
    says so on stderr. See ADR 0012 §3.
    """
    annotated = frame[frame[column].notna()]
    universe = len(annotated)
    counts = term_counts(annotated[column], encoding)
    tested = sorted(term for term, count in counts.items() if count >= min_term_count)
    dropped = sorted(term for term, count in counts.items() if count < min_term_count)

    carriers = {
        term: annotated[column].map(lambda value, t=term: t in parse_terms(value, encoding))
        for term in tested
    }
    rows = []
    for cluster in clusters:
        members = annotated[cluster_column] == cluster
        n_cluster = int(members.sum())
        for term in tested:
            carrying = carriers[term]
            k = int((carrying & members).sum())
            # `n_cluster == 0` means no member of this cluster is annotated for
            # this column at all. `hypergeometric_enrichment` returns an
            # untested result for that rather than a p-value, which is the same
            # branch and does not need its own.
            result = hypergeometric_enrichment(
                k=k, n=n_cluster, total_carrying=int(carrying.sum()), universe=universe
            )
            rows.append(
                _row(
                    cluster,
                    column,
                    "categorical",
                    term,
                    n_cluster=n_cluster,
                    n_universe=universe,
                    n_term_cluster=k,
                    n_term_universe=int(carrying.sum()),
                    result=result,
                )
            )
    return rows, {"universe": universe, "tested": tested, "dropped": dropped}


def continuous_rows(frame, column, values, clusters, cluster_column):
    """Every (cluster) hypothesis for one continuous column."""
    labels = frame[cluster_column].to_numpy()
    rows = []
    for cluster in clusters:
        members = labels == cluster
        result = mann_whitney_u(values[members], values[~members])
        rows.append(
            _row(
                cluster,
                column,
                "continuous",
                "",
                n_cluster=result.n_inside,
                n_universe=result.n_inside + result.n_outside,
                n_term_cluster="",
                n_term_universe="",
                result=result,
            )
        )
    return rows


def _row(
    cluster,
    column,
    kind,
    term,
    *,
    n_cluster,
    n_universe,
    n_term_cluster,
    n_term_universe,
    result,
):
    return {
        "cluster": cluster,
        "annotation": column,
        "kind": kind,
        "term": term,
        "n_cluster": n_cluster,
        "n_universe": n_universe,
        "n_term_cluster": n_term_cluster,
        "n_term_universe": n_term_universe,
        "effect": result.effect,
        "effect_kind": result.effect_kind,
        "p_value": result.p_value,
        "note": result.note,
    }


def correct_within_column(rows, fdr: float):
    """Benjamini-Hochberg per annotation column, not across the whole table.

    The columns have wildly different vocabulary sizes -- a lineage has a dozen
    terms, an InterPro column has thousands -- and pooling them lets the large
    vocabulary set the correction for the small one, so a finding's q-value
    would depend on which other columns happened to be configured. Every row
    carries `n_universe` and the family is one column, so a reader who wants a
    global correction can recompute it. ADR 0012 §4.
    """
    for column in sorted({row["annotation"] for row in rows}):
        family = [row for row in rows if row["annotation"] == column]
        q_values = benjamini_hochberg([row["p_value"] for row in family])
        for row, q_value in zip(family, q_values):
            row["q_value"] = float(q_value)
            row["significant"] = bool(q_value <= fdr) if not np.isnan(q_value) else False
    return rows


def to_frame(rows, clustering: str) -> pd.DataFrame:
    """The tidy table: readable at the top, byte-stable everywhere.

    Sorted by q ascending so the findings are first, with a full tiebreak on
    (annotation, cluster, term) so two runs over the same data produce the same
    bytes. Untested rows sort last -- they are the appendix, not the result.
    """
    if not rows:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    frame = pd.DataFrame(rows)
    frame.insert(0, "clustering", clustering)
    frame = frame.reindex(columns=list(TABLE_COLUMNS))
    frame = frame.sort_values(
        by=["q_value", "annotation", "cluster", "term"],
        ascending=[True, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return frame


def describe(report: dict) -> str:
    """One line for stderr. What was tested, and what was not."""
    parts = [
        f"{report['n_tested']} hypotheses over {report['n_proteins']} proteins "
        f"in {report['n_clusters']} clusters",
        f"{report['n_significant']} significant at q<={report['fdr']}",
    ]
    if report["n_untested"]:
        parts.append(f"{report['n_untested']} untested")
    if report["columns_absent"]:
        parts.append(f"absent from the table: {report['columns_absent']}")
    dropped = sum(len(entry["dropped"]) for entry in report["categorical"].values())
    if dropped:
        parts.append(f"{dropped} term(s) below min_term_count")
    return "; ".join(parts)


def main() -> int:
    args = parse_args()
    config = from_legacy(load_config(args.configfile))
    settings = config.enrichment

    if not settings.enabled:
        raise SystemExit(
            "enrichment names no columns. Set `enrichment.categorical` or "
            "`enrichment.continuous`, or remove the key entirely -- the rule does "
            "not enter the DAG when nothing asks for it."
        )

    output_dir = os.path.join(args.output_dir, ENRICHMENT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    clusters_table = read_keyed_table(args.clusters, "the cluster table")
    annotations = read_keyed_table(args.annotations, "the annotation table")
    frame, dropped_proteins = join_on_protid(clusters_table, annotations, settings.cluster_column)
    cluster_labels = sorted(frame[settings.cluster_column].dropna().unique())

    present = [column for column in settings.columns if column in frame.columns]
    absent = [column for column in settings.columns if column not in frame.columns]

    rows = []
    categorical_report, continuous_report = {}, {}
    for column in present:
        if settings.kind_of(column) == "categorical":
            encoding = settings.encodings.get(column) or detect_encoding(frame[column])
            column_rows, detail = categorical_rows(
                frame,
                column,
                encoding,
                cluster_labels,
                settings.cluster_column,
                settings.min_term_count,
            )
            detail["encoding"] = encoding
            categorical_report[column] = detail
            rows.extend(column_rows)
        else:
            values, unparseable = numeric_column(frame, column)
            continuous_report[column] = {
                "measured": int(np.count_nonzero(~np.isnan(values))),
                "unparseable": unparseable,
            }
            if unparseable:
                print(
                    f"[enrich_clusters] {column}: {unparseable} value(s) are not "
                    "numbers and are treated as missing",
                    file=sys.stderr,
                )
            rows.extend(
                continuous_rows(frame, column, values, cluster_labels, settings.cluster_column)
            )

    correct_within_column(rows, settings.fdr)
    table = to_frame(rows, settings.cluster_column)
    table.to_csv(os.path.join(output_dir, TABLE_FILENAME), sep="\t", index=False)

    report = {
        "clustering": settings.cluster_column,
        "clusters_path": args.clusters,
        "annotations_path": args.annotations,
        "n_proteins": len(frame),
        "n_clusters": len(cluster_labels),
        "n_tested": int(sum(1 for row in rows if not row["note"])),
        "n_untested": int(sum(1 for row in rows if row["note"])),
        "n_significant": int(sum(1 for row in rows if row["significant"])),
        "fdr": settings.fdr,
        "min_term_count": settings.min_term_count,
        "columns_requested": list(settings.columns),
        "columns_absent": absent,
        "categorical": categorical_report,
        "continuous": continuous_report,
        "proteins_dropped_by_the_join": dropped_proteins,
    }
    # Not a debug aid, for the reason `coregister` prints its index: a table
    # over an overlap, or one whose requested columns were not in the file,
    # looks exactly like a table that found nothing.
    print(f"[enrich_clusters] {describe(report)}", file=sys.stderr)
    if absent:
        print(
            f"[enrich_clusters] no column named {absent} in {args.annotations}; "
            "requested and not tested",
            file=sys.stderr,
        )
    for column, detail in sorted(categorical_report.items()):
        if detail["universe"] < len(frame):
            print(
                f"[enrich_clusters] {column}: tested against {detail['universe']} of "
                f"{len(frame)} proteins -- the rest have no value in that column, and "
                "an unannotated protein is not evidence that it lacks a term",
                file=sys.stderr,
            )

    manifest = Manifest.build(
        "enrichment",
        "enrichment",
        provider="enrich_clusters",
        params={
            "cluster_column": settings.cluster_column,
            "categorical": list(settings.categorical),
            "continuous": list(settings.continuous),
            "min_term_count": settings.min_term_count,
            "fdr": settings.fdr,
        },
        inputs={
            "clusters": file_digest(args.clusters),
            "annotations": file_digest(args.annotations),
        },
        protids=list(frame.index),
        extra=report,
    )
    manifest.write(os.path.join(output_dir, "manifest.json"))

    print(f"[enrich_clusters] {len(table)} row(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
