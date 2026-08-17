#!/usr/bin/env python
"""Censoring diagnostics for the all-versus-all similarity matrix.

A zero in that matrix means "this pair lost the per-query top-1000 cut", not
"these two proteins are dissimilar" -- see ``docs/adr/0009-censoring-semantics.md``.
On a production 2530-protein run, 60.5% of cells are that fill and not one is a
measured zero.

This module reports what that costs. Four of the five reports are bookkeeping.
The fifth, :func:`cross_cluster_edge_retention`, is the one worth reading:

    Foldseek reports the top N partners per query. Between-cluster pairs carry
    the weakest scores, so they are the first to fall off that list. At high
    censoring almost every surviving edge is *within* a cluster. The clusters
    therefore look crisper as censoring worsens, while their arrangement
    relative to one another decays into noise -- the map gets more convincing
    and less true at the same time.

That failure mode is invisible in the map itself, which is why it needs a number.
"""

from __future__ import annotations
from collections.abc import Mapping

import numpy as np
import pandas as pd
from matrix_io import LabeledMatrix

__all__ = [
    "asymmetry_report",
    "censoring_report",
    "cross_cluster_edge_retention",
    "per_protein_censoring",
]


def per_protein_censoring(matrix: LabeledMatrix) -> pd.DataFrame:
    """Per-protein censoring, as a tidy table.

    ``censoring_rate`` is a property of how well a protein was measured, not of
    the protein, so it is overlay-only -- see ADR 0003.
    """
    censored = matrix.censored
    n_measured = (~censored).sum(axis=1)
    return pd.DataFrame(
        {
            "protid": list(matrix.protids),
            "n_measured": n_measured.astype(int),
            "n_censored": censored.sum(axis=1).astype(int),
            "censoring_rate": censored.mean(axis=1).astype(float),
        }
    )


def _pair_state(matrix: LabeledMatrix):
    """Row/column index arrays and the label-aligned censoring mask.

    Everything here goes through labels. The matrix's column order is not
    guaranteed to be its row order (ADR 0007), so a positional read would be
    comparing the wrong pairs.
    """
    col_pos = {c: j for j, c in enumerate(matrix.columns)}
    missing = [p for p in matrix.protids if p not in col_pos]
    if missing:
        raise ValueError(
            f"{len(missing)} row label(s) have no matching column, so pairwise "
            f"statistics are undefined; first few: {missing[:5]}"
        )
    order = np.fromiter(
        (col_pos[p] for p in matrix.protids), dtype=np.intp, count=len(matrix.protids)
    )
    # Square, label-aligned views. Rows and columns now index the same proteins.
    censored = matrix.censored[:, order]
    values = matrix.values[:, order]
    return values, censored


def asymmetry_report(matrix: LabeledMatrix) -> dict:
    """How asymmetric the matrix is, under denominators that are stated.

    Three different numbers are defensible here and they differ by a factor of
    nearly two, so each is reported with its denominator named rather than one
    being picked and called "the" asymmetry:

    * ``one_way_fraction_of_pairs`` -- of unordered pairs reported at all, how
      many were reported in only one direction. This is the censoring statistic.
    * ``one_way_fraction_of_measured_cells`` -- the same numerator over ordered
      non-zero cells. Smaller, and easy to quote by accident.
    * ``max_abs_difference_measured`` -- the largest disagreement between two
      directions that were *both* measured. This is the only one of the three
      that is about asymmetry rather than about censoring; comparing a measured
      value against a fill measures the fill.

    On production data the last distinction matters: including fills reports a
    maximum difference of 0.99, and excluding them reports 0.67.
    """
    values, censored = _pair_state(matrix)
    n = values.shape[0]
    if n < 2:
        return {
            "n_proteins": int(n),
            "pairs_reported": 0,
            "pairs_one_way": 0,
            "one_way_fraction_of_pairs": 0.0,
            "one_way_fraction_of_measured_cells": 0.0,
            "max_abs_difference_measured": 0.0,
            "median_abs_difference_measured": 0.0,
            "fraction_exactly_equal_measured": 1.0,
        }

    upper = np.triu_indices(n, k=1)
    a_measured = ~censored[upper]
    b_measured = ~censored.T[upper]

    both = a_measured & b_measured
    one_way = a_measured ^ b_measured
    reported = int((a_measured | b_measured).sum())

    diffs = np.abs(values[upper][both] - values.T[upper][both])
    measured_cells = int((~censored).sum())

    return {
        "n_proteins": int(n),
        "pairs_reported": reported,
        "pairs_both_directions": int(both.sum()),
        "pairs_one_way": int(one_way.sum()),
        "one_way_fraction_of_pairs": float(one_way.sum() / reported) if reported else 0.0,
        "one_way_fraction_of_measured_cells": (
            float(one_way.sum() / measured_cells) if measured_cells else 0.0
        ),
        "max_abs_difference_measured": float(diffs.max()) if diffs.size else 0.0,
        "median_abs_difference_measured": float(np.median(diffs)) if diffs.size else 0.0,
        "fraction_exactly_equal_measured": (float((diffs == 0).mean()) if diffs.size else 1.0),
    }


def cross_cluster_edge_retention(matrix: LabeledMatrix, clusters: Mapping) -> tuple:
    """Measured-pair retention within versus between clusters.

    Returns ``(table, summary)``. The table has one row per ordered cluster pair
    with ``possible``, ``measured`` and ``retention``; the summary reduces it to
    the comparison that matters.

    Read ``between_over_within`` as: how much less likely a cross-cluster
    relationship was to survive the per-query cut than a within-cluster one. As
    it approaches zero, the clusters are still real but their *positions
    relative to each other* are increasingly determined by the handful of
    cross-cluster edges that happened to survive.
    """
    _values, censored = _pair_state(matrix)
    protids = list(matrix.protids)
    missing = [p for p in protids if p not in clusters]
    if missing:
        raise ValueError(
            f"{len(missing)} protein(s) have no cluster assignment; first few: " f"{missing[:5]}"
        )

    labels = [clusters[p] for p in protids]
    unique = sorted(set(labels))
    code_of = {label: i for i, label in enumerate(unique)}
    codes = np.fromiter((code_of[label] for label in labels), dtype=np.intp, count=len(labels))

    measured = ~censored
    np.fill_diagonal(measured, False)  # a protein against itself is not a relationship

    k = len(unique)
    possible = np.zeros((k, k), dtype=np.int64)
    observed = np.zeros((k, k), dtype=np.int64)
    counts = np.bincount(codes, minlength=k)

    for i in range(k):
        rows = codes == i
        for j in range(k):
            cols = codes == j
            n_possible = counts[i] * counts[j] - (counts[i] if i == j else 0)
            possible[i, j] = n_possible
            observed[i, j] = int(measured[np.ix_(rows, cols)].sum())

    rows_out = []
    for i in range(k):
        for j in range(k):
            if possible[i, j] == 0:
                continue
            rows_out.append(
                {
                    "cluster_a": unique[i],
                    "cluster_b": unique[j],
                    "within": i == j,
                    "possible": int(possible[i, j]),
                    "measured": int(observed[i, j]),
                    "retention": float(observed[i, j] / possible[i, j]),
                }
            )
    table = pd.DataFrame(rows_out)

    within_possible = int(np.trace(possible))
    within_observed = int(np.trace(observed))
    between_possible = int(possible.sum() - within_possible)
    between_observed = int(observed.sum() - within_observed)

    within_retention = within_observed / within_possible if within_possible else float("nan")
    between_retention = between_observed / between_possible if between_possible else float("nan")
    ratio = (
        between_retention / within_retention
        if within_retention and within_retention == within_retention and within_retention > 0
        else float("nan")
    )

    summary = {
        "n_clusters": k,
        "within_possible": within_possible,
        "within_measured": within_observed,
        "within_retention": within_retention,
        "between_possible": between_possible,
        "between_measured": between_observed,
        "between_retention": between_retention,
        "between_over_within": ratio,
    }
    return table, summary


def censoring_report(matrix: LabeledMatrix, clusters: Mapping | None = None) -> dict:
    """The full censoring picture for one matrix.

    ``clusters`` is optional; supply it to get the cross-cluster edge-retention
    analysis, which is the part that says whether the *arrangement* of the map
    is trustworthy.
    """
    from matrix_io import summarize_censoring

    report = {"matrix": summarize_censoring(matrix)}
    report["asymmetry"] = asymmetry_report(matrix)

    per_protein = per_protein_censoring(matrix)
    rates = per_protein["censoring_rate"]
    report["per_protein"] = {
        "min": float(rates.min()),
        "median": float(rates.median()),
        "max": float(rates.max()),
        "mean": float(rates.mean()),
    }

    if clusters is not None:
        table, summary = cross_cluster_edge_retention(matrix, clusters)
        report["cross_cluster_edge_retention"] = summary
        report["cross_cluster_table"] = table.to_dict(orient="records")

    report["interpretation"] = _interpret(report)
    return report


def _interpret(report: dict) -> list:
    """Plain-language warnings. Numbers nobody reads are numbers nobody acts on."""
    notes = []
    matrix = report["matrix"]
    rate = matrix["censoring_rate"]

    if matrix.get("cap_detected"):
        cap = matrix.get("inferred_cap")
        notes.append(
            f"A per-query cap of {cap} reported partners was detected "
            f"({matrix['rows_at_max_fraction']:.1%} of rows sit exactly on it). "
            f"This accounts for a predicted {matrix['censoring_predicted_by_cap']:.1%} "
            f"of cells being empty against {rate:.1%} observed. Raising Foldseek's "
            "--max-seqs is the only way to reduce it at the source."
        )
    if rate >= 0.5:
        notes.append(
            f"{rate:.1%} of the matrix is missing rather than measured. Any statistic "
            "that averages over raw cells -- including the cluster-similarity "
            "heatmaps -- is dominated by absent pairs, not by low scores."
        )
    if matrix.get("measured_zero_count", 0) > 0:
        notes.append(
            f"{matrix['measured_zero_count']} cell(s) parsed to zero without being the "
            "fill token, so `values == 0` is NOT a valid censoring test for this "
            "matrix. Use the mask."
        )

    asym = report.get("asymmetry", {})
    if asym.get("one_way_fraction_of_pairs", 0) > 0.25:
        notes.append(
            f"{asym['one_way_fraction_of_pairs']:.1%} of reported pairs were reported "
            "in only one direction, so the matrix is far from symmetric. Any metric "
            "that assumes symmetry needs an explicit symmetrization rule."
        )

    retention = report.get("cross_cluster_edge_retention")
    if retention and retention["between_over_within"] == retention["between_over_within"]:
        ratio = retention["between_over_within"]
        if ratio < 0.5:
            notes.append(
                f"Cross-cluster relationships survived the per-query cut at "
                f"{ratio:.2f}x the rate of within-cluster ones "
                f"({retention['between_retention']:.1%} vs "
                f"{retention['within_retention']:.1%}). The clusters are better "
                "supported than the distances between them; treat the arrangement of "
                "clusters in the map with more caution than the clusters themselves."
            )
    if not notes:
        notes.append("No censoring problems detected in this matrix.")
    return notes
