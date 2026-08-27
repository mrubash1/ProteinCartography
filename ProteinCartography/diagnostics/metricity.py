#!/usr/bin/env python
"""Is this distance matrix a distance matrix?

``representation: direct`` reads the all-versus-all similarity matrix as a
matrix and hands ``1 - TM`` downstream as a distance. The config gate for that
mode (``config_schema._validate_representations``) asks whether the rows and
columns line up. That is a necessary check and it is not a sufficient one:
alignment makes the matrix mean its labels, and says nothing about whether the
numbers in it can be laid out in a Euclidean space at all.

They largely cannot. Double-centre the squared ``1 - TM`` distances of the
published 160-protein actin cohort and the negative eigenvalue mass is 44.6% of
the positive mass -- so roughly a third of the structure in that matrix is not
embeddable in any number of Euclidean dimensions, and every reducer that
consumes it is discarding it. This module reports that number.

**It reports; it does not gate.** Measured across dense cohorts the fraction
spans 32% to 93%, and the widest gap is between two *cohorts of the same query*,
not between families: at a matched core of 100 proteins, two archived runs of
one S1 peptidase query return 31.8% and 57.5%. Cohort composition moves this
further than family identity does. The number also climbs with core size within
a single cohort -- 79.1% to 92.9% between n=100 and n=600 on one actin cohort --
so it is not comparable across cohorts even at a matched family, and a threshold
fitted to any one of them would be a constant derived from that cohort (see
``docs/FOLLOWUPS.md`` #49). Note that *repeated runs* are not a source of
variation here at all: three archived runs of one identical query differ in none
of their 4.7 million cells. Emitting the number without a verdict is deliberate,
and matches how ``docs/INTERPRETING.md`` handles the other statistics whose
thresholds are not yet earned.

**READ THIS BEFORE QUOTING ANY NUMBER FROM HERE.** A matrix of true Euclidean
distances double-centres to a positive semi-definite Gram matrix and returns
under 1e-9. But that is NOT the input this pipeline supplies. `blocks/tmscore.py`
forms distances as ``1 - TM``, and that affine transform is not the Euclidean
one: by Schoenberg's criterion, if ``S`` is positive semi-definite with a unit
diagonal then ``sqrt(2(1 - S))`` is Euclidean and ``1 - S`` in general is not.
Measured here on cosine similarities of random unit vectors -- data that is
Euclidean by construction -- ``1 - S`` returns **41.8% to 43.4%** at n=100 to
600, while ``sqrt(2(1 - S))`` returns ~1e-15. So roughly forty points of the
fraction reported on any ``1 - TM`` matrix are manufactured by the transform,
not measured in the data, and a cohort scoring 31.8% is scoring BELOW what
exactly-Euclidean data scores in this convention. The comparison to zero is
therefore only meaningful at ``square_first=False`` or after the Schoenberg
transform. `docs/FOLLOWUPS.md` #59 records this; it is not fixed here because
changing the transform changes every manifest that carries the number.

**Every number here is meaningless without its convention, so the convention
travels with it.** "Negative eigenvalue mass" has appeared in this project's
notes as 44.6%, 30.8%, 24.8% and 19.9% -- four figures for one name, and all
four are correct. They are the four cells of a two-by-two: double-centre the
*squared* distances or the raw ones, and divide the negative mass by the
*positive* mass or by the total. Nobody had computed it wrongly; nobody had
written down which of the four they meant. So :func:`metricity_report` takes
both axes as arguments and records them in its own output, and a caller cannot
obtain a bare number that a later reader could misinterpret.

Cost is one symmetric eigendecomposition, O(n^3): a few seconds at n=2500.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_DENOMINATOR",
    "DEFAULT_SQUARE_FIRST",
    "DENOMINATORS",
    "metricity_report",
]

#: How to normalize the negative mass. ``positive`` is the ratio classical-MDS
#: treatments usually mean; ``total`` is the share of all spectral mass.
DENOMINATORS = ("positive", "total")

#: Torgerson's classical scaling double-centres *squared* distances, so that is
#: the default. Raw distances are offered because the notes contain figures
#: computed that way and they have to remain reproducible.
DEFAULT_SQUARE_FIRST = True

DEFAULT_DENOMINATOR = "positive"


def _gram(distances: np.ndarray, square_first: bool) -> np.ndarray:
    """Torgerson double-centring: B = -1/2 * J D J."""
    matrix = distances.astype(np.float64, copy=True)
    np.fill_diagonal(matrix, 0.0)
    if square_first:
        matrix = matrix**2
    n = matrix.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ matrix @ centering
    # Double-centring is exactly symmetric in exact arithmetic and only nearly
    # symmetric in floating point. eigvalsh reads one triangle, so an asymmetric
    # last bit would silently pick a side; average instead.
    return (gram + gram.T) / 2.0


def metricity_report(
    distances,
    *,
    square_first: bool = DEFAULT_SQUARE_FIRST,
    denominator: str = DEFAULT_DENOMINATOR,
    top_k: int = 10,
    censoring_rate: float | None = None,
    n_proteins: int | None = None,
) -> dict:
    """How far this square distance matrix is from being Euclidean.

    Parameters
    ----------
    distances:
        Square, symmetric, zero-diagonal. This is the matrix as the consumer
        will use it -- if censored pairs were filled with a sentinel distance,
        they are part of what is measured here, which is why `censoring_rate`
        is recorded next to the answer.
    square_first, denominator:
        The two convention axes. See the module docstring; both are echoed back
        in the result so the number is never separable from its definition.
    top_k:
        How many leading positive eigenvalues to report. The spectrum says how
        many dimensions carry the embeddable part, which is the other half of
        the question and is not answerable from the fraction alone.
    censoring_rate, n_proteins:
        Cohort context. A metricity figure without the cohort that produced it
        is not comparable to another one: measured on sub-cohorts of a single
        family this statistic runs from 15.3% to 49.3%, a wider spread than
        censoring produces. Recorded, never used in the computation.

    Returns
    -------
    dict
        JSON-serializable, shaped for a manifest's ``extra``.
    """
    if denominator not in DENOMINATORS:
        raise ValueError(
            f"denominator: {denominator!r} is not valid. Allowed: {', '.join(DENOMINATORS)}. "
            "The choice changes the answer by nearly a factor of two, so it has no default "
            "that can be left implicit."
        )

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"metricity needs a square distance matrix, got shape {values.shape}. "
            "A condensed vector must be expanded first."
        )
    if values.shape[0] < 3:
        # Report the refusal, do not raise it. This is called unconditionally
        # from the tmscore block, nothing gates on the value, and the report
        # already ships `verdict: None` saying so -- so raising failed a whole
        # block build over a provenance number that no consumer reads. The
        # shape and symmetry checks above stay hard: those are malformed input.
        return {
            "negative_mass_fraction": None,
            "convention": {
                "double_centered": "squared_distances" if square_first else "raw_distances",
                "denominator": denominator,
                "formula": "not computed",
            },
            "positive_mass": None,
            "negative_mass": None,
            "n": int(values.shape[0]),
            "top_positive_eigenvalues": [],
            "cohort": {
                "n_proteins": int(n_proteins) if n_proteins is not None else int(values.shape[0]),
                "censoring_rate": float(censoring_rate) if censoring_rate is not None else None,
            },
            "verdict": None,
            "verdict_note": (
                f"not computed: {values.shape[0]} proteins cannot support a spectrum, "
                "which needs at least 3."
            ),
        }
    if not np.allclose(values, values.T, atol=1e-8):
        raise ValueError(
            "metricity needs a symmetric distance matrix. TM-score is normalized per "
            "query, so the raw matrix is not symmetric; symmetrize it explicitly and "
            "record which rule was used."
        )

    eigenvalues = np.linalg.eigvalsh(_gram(values, square_first))
    positive_mass = float(eigenvalues[eigenvalues > 0].sum())
    negative_mass = float(-eigenvalues[eigenvalues < 0].sum())
    base = positive_mass if denominator == "positive" else positive_mass + negative_mass
    fraction = float(negative_mass / base) if base > 0 else 0.0

    leading = np.sort(eigenvalues[eigenvalues > 0])[::-1][:top_k]
    return {
        "negative_mass_fraction": fraction,
        "convention": {
            "double_centered": "squared_distances" if square_first else "raw_distances",
            "denominator": denominator,
            "formula": (
                "sum(|negative eigenvalues|) / sum(positive eigenvalues)"
                if denominator == "positive"
                else "sum(|negative eigenvalues|) / sum(|all eigenvalues|)"
            ),
        },
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "n": int(values.shape[0]),
        "top_positive_eigenvalues": [float(v) for v in leading],
        "cohort": {
            "n_proteins": int(n_proteins) if n_proteins is not None else int(values.shape[0]),
            "censoring_rate": float(censoring_rate) if censoring_rate is not None else None,
        },
        "verdict": None,
        "verdict_note": (
            "reported, not gated. Do not compare this to another cohort's figure, and "
            "do not read it as the fraction of structure that is non-Euclidean: on a "
            "`1 - TM` matrix roughly 40 points of it are manufactured by that transform "
            "rather than measured, since `1 - S` is not the Euclidean transform of a "
            "similarity (`sqrt(2(1-S))` is). It is also sample-size dependent and varies "
            "more with which subset is analysed than with protein family. See "
            "docs/FOLLOWUPS.md #59, #49 and #51."
        ),
    }
