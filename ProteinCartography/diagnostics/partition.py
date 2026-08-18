#!/usr/bin/env python
"""Is this partition a property of the proteins, or of the resolution knob?

Phase 5 items 7 and 8. Both ask the same uncomfortable question from opposite
directions, and neither can be answered by looking at a partition on its own --
a clustering algorithm returns clusters whatever it is given, and they always
look like clusters.

**Item 7, the cluster stability tree.** Leiden takes a resolution parameter and
returns more clusters as it rises. Sweeping it and measuring the adjusted Rand
index between adjacent resolutions distinguishes two very different pictures
that a single run cannot: a *plateau*, where a range of resolutions all recover
the same grouping, means the grouping is in the data; uniformly low agreement
means the number of clusters is a property of the parameter and nothing else.

**Item 8, negative controls.** "If the pipeline produces attractive clusters on
noise, and it will, everyone should see that first." Two controls, and they fail
in different ways on purpose. Permuting the cluster labels over the same
distances destroys the correspondence while preserving every marginal, so it
says how much of the observed separation is arithmetic rather than structure --
and it needs no clustering, so it is computed here. Clustering a *random*
distance matrix is the more alarming one, because it produces a genuinely
positive silhouette; the caller supplies it, because only the caller has a
clusterer.

Everything in this module is numpy over a distance matrix and a label vector.
The clustering itself lives in ``ProteinCartography/clustering.py``, so that
these statistics are testable in an environment with no scanpy in it, which is
the environment the unit suite runs in.

Both statistics are cross-checked against scikit-learn behind ``importorskip``,
including the degenerate cases where its answer is a convention rather than a
formula: ``adjusted_rand_index`` of two all-in-one-cluster partitions is 1.0 by
definition and 0/0 by the formula, and a singleton cluster's silhouette is 0 for
the same reason.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CONTROL_DESCRIPTIONS",
    "MEANINGFUL_MARGIN",
    "PLATEAU_THRESHOLD",
    "ControlResult",
    "NegativeControlReport",
    "PartitionError",
    "ResolutionStep",
    "ResolutionSweep",
    "adjusted_rand_index",
    "contingency_table",
    "negative_controls",
    "resolution_sweep",
    "silhouette",
]

#: What ``diagnostics.negative_controls`` may name, and what each one holds
#: fixed. Enumerated here rather than in the config schema for the reason
#: ``fusion.STRATEGY_PARAMS`` is: the module that implements a thing owns the
#: list of what it implements, and the validator imports it, so a control
#: cannot be named in a config without existing.
CONTROL_DESCRIPTIONS = {
    "shuffled_labels": (
        "the same clusters of the same sizes, assigned to proteins at random. "
        "Needs no clusterer, and isolates correspondence from every marginal."
    ),
    "random_distances": (
        "a clustering fitted to a random matrix of the same shape. The "
        "alarming one: it returns a genuinely positive silhouette."
    ),
}

#: Adjacent resolutions agreeing at or above this are treated as one level of
#: the tree. A reporting band, like ``embedding``'s: no literature fixes it.
PLATEAU_THRESHOLD = 0.80

#: How far the observed silhouette must exceed a control's before the partition
#: is described as carrying structure the control does not.
MEANINGFUL_MARGIN = 0.05


class PartitionError(ValueError):
    """Raised when two labelings cannot be compared at all."""


def _as_codes(labels) -> np.ndarray:
    """Labels of any hashable type to contiguous integer codes.

    Leiden hands back strings like ``LC03`` and the legacy path prefixes them
    again; comparing partitions must not depend on either. Sorted for
    determinism -- ``np.unique`` sorts, so the same labeling always produces
    the same codes regardless of the order the labels first appear in.
    """
    values = np.asarray(list(labels))
    if values.ndim != 1:
        raise PartitionError(f"labels must be one-dimensional, got shape {values.shape}")
    _, codes = np.unique(values, return_inverse=True)
    return codes


def contingency_table(left, right) -> np.ndarray:
    """Counts of proteins by (left cluster, right cluster)."""
    a, b = _as_codes(left), _as_codes(right)
    if a.shape != b.shape:
        raise PartitionError(
            f"the two labelings cover different numbers of proteins, {a.size} and {b.size}. "
            "Align them to a shared protein index before comparing."
        )
    if a.size == 0:
        raise PartitionError("cannot compare empty labelings")
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def _pairs(counts: np.ndarray) -> float:
    """Sum of ``C(n, 2)`` over an array of counts."""
    counts = counts.astype(np.float64)
    return float((counts * (counts - 1) / 2).sum())


def adjusted_rand_index(left, right) -> float:
    """Agreement between two partitions, chance-corrected. 0 is chance, 1 exact.

    The Hubert-Arabie form: the pair-counting Rand index minus its expectation
    under the hypergeometric null, over the same difference at the maximum. It
    can go negative, and a negative value is meaningful -- two partitions can
    agree *less* than independent ones, which is what a systematically crossed
    labeling does.
    """
    table = contingency_table(left, right)
    n = float(table.sum())
    index = _pairs(table)
    rows = _pairs(table.sum(axis=1))
    columns = _pairs(table.sum(axis=0))
    total_pairs = n * (n - 1) / 2

    # Both partitions trivial and identical -- every protein alone, or all of
    # them together. The formula is 0/0 there; scikit-learn's convention is
    # 1.0, and matching it matters because the pipeline hits the all-in-one
    # case whenever a space has fewer than three proteins (leiden_clustering
    # short-circuits at N < 3).
    if table.shape[0] == table.shape[1] and table.shape[0] in (1, int(n)):
        return 1.0
    expected = rows * columns / total_pairs if total_pairs else 0.0
    maximum = (rows + columns) / 2
    if maximum == expected:
        return 0.0
    return float((index - expected) / (maximum - expected))


def silhouette(distances: np.ndarray, labels) -> np.ndarray:
    """Per-protein silhouette from a precomputed distance matrix.

    ``(b - a) / max(a, b)``, where ``a`` is the mean distance to the protein's
    own cluster and ``b`` the smallest mean distance to any other. A protein
    alone in its cluster scores 0: ``a`` is undefined for it, and scoring it 1
    would reward a clustering for producing singletons.
    """
    distances = np.asarray(distances, dtype=np.float64)
    codes = _as_codes(labels)
    n = codes.size
    if distances.shape != (n, n):
        raise PartitionError(
            f"distances must be square over the {n} labeled proteins, got {distances.shape}"
        )
    from diagnostics.embedding import require_finite

    require_finite(distances, "the distances the silhouette is computed over")
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise PartitionError(
            "the silhouette needs at least two clusters; this labeling has one. "
            "A single-cluster partition has nothing to be separated from."
        )

    # Mean distance from every protein to every cluster, including its own.
    sums = np.zeros((n, n_clusters), dtype=np.float64)
    np.add.at(sums.T, codes, distances.T)
    sizes = np.bincount(codes, minlength=n_clusters).astype(np.float64)

    own = codes
    own_sizes = sizes[own]
    # Excluding self costs nothing extra: the diagonal is zero, so the sum is
    # already the sum over "others", and only the divisor changes.
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(own_sizes > 1, sums[np.arange(n), own] / (own_sizes - 1), 0.0)
        others = sums / sizes[None, :]
    others[np.arange(n), own] = np.inf
    b = others.min(axis=1)

    scores = np.zeros(n, dtype=np.float64)
    real = own_sizes > 1
    denominator = np.maximum(a, b)
    valid = real & (denominator > 0)
    scores[valid] = (b[valid] - a[valid]) / denominator[valid]
    return scores


@dataclass(frozen=True)
class ResolutionStep:
    """One rung of the sweep."""

    resolution: float
    n_clusters: int
    labels: list
    silhouette_mean: float

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "n_clusters": self.n_clusters,
            "silhouette_mean": self.silhouette_mean,
        }


@dataclass(frozen=True)
class ResolutionSweep:
    """The sweep, and where in it the partition stopped moving."""

    space_id: str
    steps: list
    #: ``(low, high, ari)`` for each adjacent pair, in sweep order.
    adjacent: list = field(default_factory=list)

    def plateau(self) -> tuple:
        """The widest run of adjacent resolutions agreeing at the threshold.

        Returned as ``(low, high, width)`` where width counts resolutions, so a
        lone resolution agreeing with nothing gives width 1. The widest such run
        is the level of the tree worth reporting; if it is 1 everywhere, the
        number of clusters is a property of the parameter.
        """
        if not self.steps:
            return (None, None, 0)
        best = (self.steps[0].resolution, self.steps[0].resolution, 1)
        start, width = self.steps[0].resolution, 1
        for _, high, ari in self.adjacent:
            if ari >= PLATEAU_THRESHOLD:
                width += 1
            else:
                start, width = high, 1
            if width > best[2]:
                best = (start, high, width)
        return best

    def to_dict(self) -> dict:
        low, high, width = self.plateau()
        return {
            "space_id": self.space_id,
            "steps": [step.to_dict() for step in self.steps],
            "adjacent_ari": [
                {"low": low_r, "high": high_r, "ari": ari} for low_r, high_r, ari in self.adjacent
            ],
            "plateau": {"low": low, "high": high, "n_resolutions": width},
            "warnings": self.warnings(),
        }

    def warnings(self) -> list:
        notes = []
        if len(self.steps) < 2:
            return notes
        _, _, width = self.plateau()
        if width < 2:
            notes.append(
                "no two adjacent resolutions agree at "
                f"{PLATEAU_THRESHOLD:.2f} ARI. The number of clusters is tracking the "
                "resolution parameter rather than any level in the data, so no particular "
                "partition of this space should be reported as the partition."
            )
        counts = {step.n_clusters for step in self.steps}
        if len(counts) == 1:
            notes.append(
                f"every resolution returned {counts.pop()} clusters. Either the structure is "
                "very strong, or the sweep is too narrow to have moved anything."
            )
        return notes


def resolution_sweep(space_id: str, distances: np.ndarray, partitions) -> ResolutionSweep:
    """Assemble the sweep from partitions the caller has already computed.

    Args:
        space_id: for the report only.
        distances: square pairwise distances, in the partitions' protein order.
        partitions: mapping of resolution to that resolution's labels.
    """
    steps = []
    for resolution in sorted(partitions):
        labels = list(partitions[resolution])
        codes = _as_codes(labels)
        n_clusters = int(codes.max()) + 1
        mean = float(silhouette(distances, labels).mean()) if n_clusters > 1 else 0.0
        steps.append(
            ResolutionStep(
                resolution=float(resolution),
                n_clusters=n_clusters,
                labels=labels,
                silhouette_mean=mean,
            )
        )
    adjacent = [
        (
            steps[i].resolution,
            steps[i + 1].resolution,
            adjusted_rand_index(steps[i].labels, steps[i + 1].labels),
        )
        for i in range(len(steps) - 1)
    ]
    return ResolutionSweep(space_id=space_id, steps=steps, adjacent=adjacent)


@dataclass(frozen=True)
class ControlResult:
    """One partition's separation, observed or control."""

    name: str
    description: str
    n_clusters: int
    silhouette_mean: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_clusters": self.n_clusters,
            "silhouette_mean": self.silhouette_mean,
        }


@dataclass(frozen=True)
class NegativeControlReport:
    """What the observed partition scores, beside what nothing scores."""

    space_id: str
    observed: ControlResult
    controls: list
    #: Controls the caller asked for and could not produce, name to reason. A
    #: control that silently vanishes is worse than one that fails: the reader
    #: sees a shorter list and cannot tell "ran and found nothing" from "never
    #: ran". The demo produced exactly that -- one space's random-distance
    #: control returned a single cluster and disappeared from the report while
    #: six other spaces kept theirs.
    skipped: dict = field(default_factory=dict)

    def margin(self, name: str) -> float:
        for control in self.controls:
            if control.name == name:
                return self.observed.silhouette_mean - control.silhouette_mean
        raise KeyError(f"no control named {name!r}; have {[c.name for c in self.controls]}")

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "observed": self.observed.to_dict(),
            "controls": [control.to_dict() for control in self.controls],
            "margins": {control.name: self.margin(control.name) for control in self.controls},
            "skipped": dict(self.skipped),
            "warnings": self.warnings(),
        }

    def warnings(self) -> list:
        notes = [
            f"the '{name}' control was requested and not produced: {reason}"
            for name, reason in sorted(self.skipped.items())
        ]
        for control in self.controls:
            margin = self.margin(control.name)
            if margin <= 0:
                notes.append(
                    f"the '{control.name}' control scores {control.silhouette_mean:.3f} against "
                    f"the observed partition's {self.observed.silhouette_mean:.3f}. This space's "
                    "clusters are no better separated than ones found in "
                    f"{control.description}, and should not be reported as findings."
                )
            elif margin < MEANINGFUL_MARGIN:
                notes.append(
                    f"the observed partition beats the '{control.name}' control by only "
                    f"{margin:.3f} silhouette. That is within the range this control varies "
                    "over, so the separation is weak evidence at best."
                )
        return notes


def negative_controls(
    space_id: str,
    distances: np.ndarray,
    labels,
    *,
    extra=(),
    skipped=None,
    permutations: int = 20,
    seed: int = 0,
) -> NegativeControlReport:
    """The observed partition against a permutation null and any extra controls.

    The permutation control is built here because it needs no clusterer: the
    cluster sizes, the distance matrix and the number of clusters are all held
    fixed and only the assignment is shuffled, so it isolates correspondence
    from every marginal that could produce separation by itself.

    Args:
        space_id: for the report only.
        distances: square pairwise distances over the labeled proteins.
        labels: the observed cluster assignment.
        extra: ``(name, description, distances, labels)`` tuples the caller
            computed, which is where "cluster a random matrix" goes.
        skipped: name to reason, for controls the caller could not produce.
        permutations: how many shuffles to average the permutation null over.
        seed: makes the permutation null reproducible.
    """
    distances = np.asarray(distances, dtype=np.float64)
    codes = _as_codes(labels)
    observed = ControlResult(
        name="observed",
        description="the partition this space actually produced",
        n_clusters=int(codes.max()) + 1,
        silhouette_mean=float(silhouette(distances, codes).mean()),
    )

    rng = np.random.RandomState(seed)
    shuffled = [
        float(silhouette(distances, rng.permutation(codes)).mean()) for _ in range(permutations)
    ]
    controls = [
        ControlResult(
            name="shuffled_labels",
            description=(
                f"the same {observed.n_clusters} clusters of the same sizes, assigned to "
                f"proteins at random ({permutations} permutations)"
            ),
            n_clusters=observed.n_clusters,
            silhouette_mean=float(np.mean(shuffled)),
        )
    ]
    for name, description, control_distances, control_labels in extra:
        control_codes = _as_codes(control_labels)
        controls.append(
            ControlResult(
                name=name,
                description=description,
                n_clusters=int(control_codes.max()) + 1,
                silhouette_mean=float(
                    silhouette(
                        np.asarray(control_distances, dtype=np.float64), control_codes
                    ).mean()
                ),
            )
        )
    return NegativeControlReport(
        space_id=space_id, observed=observed, controls=controls, skipped=dict(skipped or {})
    )
