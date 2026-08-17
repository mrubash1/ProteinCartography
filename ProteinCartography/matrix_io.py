#!/usr/bin/env python
"""The single loader for labeled similarity matrices.

Every consumer of an all-versus-all matrix should read it through
:func:`load_labeled_matrix` rather than calling ``pandas.read_csv`` directly.
The loader exists to enforce two properties that the raw file does not.

**Index alignment.** ``foldseek_clustering.py`` writes the header from one
iteration and the rows from another. Before PR #106 the header came from an
unsorted ``set``, so the column order was a per-process permutation of the row
order. On a production run of 2530 proteins, only 2 columns sat in their row
position: reading that matrix positionally returns the wrong cell 99.92% of the
time, while reading it by label is exact. PR #106 fixes the writer. This loader
is what catches the next occurrence, and what protects analyses pointed at
output written before the fix.

**Censoring.** ``get_line_for_protid`` writes the literal string ``"0.0"`` for
any pair absent from the Foldseek output, and Foldseek's own scores arrive in
``%.3E`` form. The two are therefore perfectly separable *in the file* and
indistinguishable *after* ``pandas.read_csv`` coerces both to float. On the same
production run, 60.48% of cells were the fill and not one was a measured zero.
Those absences are not measurements of dissimilarity -- the lowest score Foldseek
reports anywhere in that matrix is 0.0549, so ``0.0`` is a value it never
produces. This loader recovers the distinction while it still exists, by
comparing the raw token before conversion, and hands back an explicit mask.

See ``docs/adr/0007-matrix-index-alignment.md`` and
``docs/adr/0009-censoring-semantics.md``.
"""

from __future__ import annotations
import csv
import gzip
import io
import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CENSORED_FILL_TOKEN",
    "LabeledMatrix",
    "MatrixAlignmentError",
    "assert_aligned",
    "load_labeled_matrix",
]

# The exact token `foldseek_clustering.get_line_for_protid` writes for a pair that
# is absent from the Foldseek output. Compared as a string, deliberately: the
# whole point is to catch it before float conversion makes it indistinguishable
# from a measured zero.
CENSORED_FILL_TOKEN = "0.0"

# The index column written by `pivot_foldseek_results`.
PROTID_COLUMN = "protid"


class MatrixAlignmentError(Exception):
    """Raised when a matrix's column labels are not its row labels, in order."""


@dataclass(frozen=True)
class LabeledMatrix:
    """A matrix that carries its own labels, so it cannot be indexed blindly.

    There is deliberately no attribute that hands back a bare array alongside no
    labels. Positional indexing of this data is what ADR 0007 exists to prevent,
    so discarding the labels has to be something a caller does on purpose.

    Attributes:
        protids: row labels, in file order. This is the canonical index.
        columns: column labels, in file order, with any prefix already stripped.
        values: float32, shape ``(len(protids), len(columns))``.
        censored: bool, same shape. True where the source token was the fill
            rather than a measured score.
        source: the path this was read from, for error messages.
        column_prefix: the prefix stripped from the column labels, if any.
    """

    protids: list[str]
    columns: list[str]
    values: np.ndarray
    censored: np.ndarray
    source: str = ""
    column_prefix: str = ""
    _measured_zero_tokens: dict = field(default_factory=dict, repr=False)
    _precision_loss: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple:
        return self.values.shape

    @property
    def is_square(self) -> bool:
        return len(self.protids) == len(self.columns)

    @property
    def is_aligned(self) -> bool:
        """True when the column labels are the row labels, in the same order."""
        return self.protids == self.columns

    @property
    def censoring_rate(self) -> float:
        """Fraction of all cells that were written as the fill token."""
        if self.censored.size == 0:
            return 0.0
        return float(self.censored.mean())

    def censoring_rate_per_protid(self) -> pd.Series:
        """Per-row censoring rate, indexed by protid.

        This is the per-protein diagnostic. It is deliberately not fusable --
        it is a property of how well a protein was measured, not of the protein,
        and it correlates with length. See ADR 0003.
        """
        if self.censored.shape[1] == 0:
            rates = np.zeros(len(self.protids), dtype=np.float64)
        else:
            rates = self.censored.mean(axis=1)
        return pd.Series(rates, index=pd.Index(self.protids, name=PROTID_COLUMN))

    def measured_zero_count(self) -> int:
        """Cells that parsed to zero but were *not* written as the fill token.

        A genuinely measured TM-score of zero would appear as ``0.000E+00``. On
        the production runs examined for ADR 0009 this is zero everywhere, which
        is what licenses treating every zero as censoring. A nonzero count here
        means that assumption does not hold for this file, and the mask -- not
        ``values == 0`` -- is the only correct way to identify censoring.
        """
        return int(sum(self._measured_zero_tokens.values()))

    def to_frame(self) -> pd.DataFrame:
        """The values as a labeled DataFrame, matching the legacy read.

        Equivalent to ``pd.read_csv(path, sep="\\t", index_col="protid")`` for a
        well-formed file, but reached through the alignment check.
        """
        return pd.DataFrame(
            self.values,
            index=pd.Index(self.protids, name=PROTID_COLUMN),
            columns=self.columns,
        )

    def aligned_diagonal(self) -> np.ndarray:
        """The diagonal read by *label*, not by position.

        Raises if the matrix is not square-by-label. On a permuted matrix this
        returns all 1.0 where the positional diagonal is almost entirely wrong,
        which is the cleanest demonstration of why ADR 0007 exists.
        """
        missing = [p for p in self.protids if p not in set(self.columns)]
        if missing:
            raise MatrixAlignmentError(
                f"{len(missing)} row label(s) have no matching column in "
                f"{self.source or '<matrix>'}; first few: {missing[:5]}"
            )
        col_pos = {c: j for j, c in enumerate(self.columns)}
        idx = np.fromiter(
            (col_pos[p] for p in self.protids), dtype=np.intp, count=len(self.protids)
        )
        return self.values[np.arange(len(self.protids)), idx]

    def reindexed_to_rows(self) -> LabeledMatrix:
        """A copy whose columns are reordered to match the row order.

        Only valid when the label sets are equal. This is the repair path; it is
        never applied silently.
        """
        assert_same_label_set(self.protids, self.columns, self.source)
        col_pos = {c: j for j, c in enumerate(self.columns)}
        order = np.fromiter(
            (col_pos[p] for p in self.protids), dtype=np.intp, count=len(self.protids)
        )
        return LabeledMatrix(
            protids=list(self.protids),
            columns=list(self.protids),
            values=self.values[:, order],
            censored=self.censored[:, order],
            source=self.source,
            column_prefix=self.column_prefix,
            _measured_zero_tokens=dict(self._measured_zero_tokens),
            _precision_loss=dict(self._precision_loss),
        )


def _duplicates(labels: Sequence[str]) -> list:
    seen, dupes = set(), []
    for label in labels:
        if label in seen and label not in dupes:
            dupes.append(label)
        seen.add(label)
    return dupes


def assert_no_duplicate_labels(
    protids: Sequence[str], columns: Sequence[str], source: str = ""
) -> None:
    """Raise if either axis repeats a label.

    A duplicate label makes the matrix ambiguous: two columns both claim to be
    the same protein, so "the column for P" has no single answer. Any lookup
    built as ``{label: position}`` silently resolves to the last one, which is a
    wrong cell rather than an error. Checked before anything reorders.
    """
    for axis, labels in (("row", protids), ("column", columns)):
        dupes = _duplicates(labels)
        if dupes:
            raise MatrixAlignmentError(
                "\n".join(
                    [
                        f"Duplicate {axis} label(s) in {source or '<matrix>'}: "
                        f"{dupes[:5]}" + (f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else ""),
                        f"  {len(labels)} {axis}s, {len(set(labels))} distinct.",
                        "",
                        f"A repeated {axis} label makes the matrix ambiguous -- two "
                        f"{axis}s both claim to be the same protein, so there is no "
                        "single correct cell for that pair. This cannot be repaired "
                        "by reordering, and reordering it anyway would silently keep "
                        "one arbitrary copy and discard the rest.",
                    ]
                )
            )


def assert_same_label_set(protids: Sequence[str], columns: Sequence[str], source: str = "") -> None:
    """Raise unless the row and column labels are the same set, one each.

    Set equality alone is not enough: ``[A, B]`` and ``[A, B, B]`` have equal
    sets but describe different matrices, and reordering by a label lookup would
    quietly drop a column and mis-assign the rest. Duplicates and length are
    checked first, for that reason.
    """
    assert_no_duplicate_labels(protids, columns, source)

    rows, cols = set(protids), set(columns)
    if rows == cols and len(protids) == len(columns):
        return
    only_rows = sorted(rows - cols)
    only_cols = sorted(cols - rows)
    raise MatrixAlignmentError(
        "\n".join(
            [
                f"Row and column labels differ in {source or '<matrix>'}.",
                f"  {len(protids)} rows, {len(columns)} columns.",
                f"  {len(only_rows)} label(s) appear only as rows: {only_rows[:5]}",
                f"  {len(only_cols)} label(s) appear only as columns: {only_cols[:5]}",
                "This is not a column-ordering problem and cannot be repaired by "
                "reordering. The matrix is malformed.",
            ]
        )
    )


def assert_aligned(protids: Sequence[str], columns: Sequence[str], source: str = "") -> None:
    """Raise unless the column labels are the row labels, in the same order.

    The error names the first divergent position and quantifies how bad the
    permutation is, because "some columns are out of order" and "the column
    order is essentially random" call for different responses.
    """
    if list(protids) == list(columns):
        return

    assert_same_label_set(protids, columns, source)

    # Same set, same length, wrong order -- the PR #106 defect. Quantify it.
    # `assert_same_label_set` has already rejected a length mismatch, so `zip`
    # cannot truncate away the divergence and `first_bad` is always found.
    first_bad = next((i for i, (r, c) in enumerate(zip(protids, columns)) if r != c), None)
    if first_bad is None:  # pragma: no cover - unreachable after the checks above
        raise MatrixAlignmentError(
            f"{source or '<matrix>'}: row and column labels agree at every position "
            f"but the sequences differ ({len(protids)} rows, {len(columns)} columns). "
            "This should be unreachable; please report it."
        )
    in_place = sum(1 for r, c in zip(protids, columns) if r == c)
    n = len(protids)
    pct = (100.0 * in_place / n) if n else 0.0

    raise MatrixAlignmentError(
        "\n".join(
            [
                f"Column order does not match row order in {source or '<matrix>'}.",
                f"  First divergence at position {first_bad}: "
                f"row label {protids[first_bad]!r}, column label {columns[first_bad]!r}.",
                f"  Only {in_place} of {n} columns ({pct:.2f}%) sit in their row position.",
                "",
                "The label sets are identical, so this is the PR #106 defect: "
                "`reading_data` returned `targets` as a set, and Python salts string "
                "hashing per process, so the header was written in a different order "
                "from the rows. Reading this matrix positionally returns the wrong "
                "cell for almost every entry; reading it by label is exact.",
                "",
                "Fixes, in order of preference:",
                "  1. Regenerate the matrix with a build that includes PR #106.",
                "  2. Pass repair=True to load_labeled_matrix() to reorder the "
                "columns to row order in memory. The data is recoverable -- only "
                "the ordering is wrong.",
            ]
        )
    )


def _open_text(path: str) -> Iterator[io.TextIOBase]:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def _as_float(token: str):
    """Parse a token in double precision, or None if it is not a number.

    Used to compare against the value after casting to the storage dtype, so
    that a token which underflows or overflows in float32 can be spotted.
    """
    try:
        return float(token)
    except ValueError:
        return None


def load_labeled_matrix(
    path: str,
    *,
    require_alignment: bool = True,
    repair: bool = False,
    column_prefix: str = "",
    censored_fill_token: str = CENSORED_FILL_TOKEN,
    dtype: type = np.float32,
) -> LabeledMatrix:
    """Read a labeled similarity matrix, checking alignment and recovering the mask.

    Args:
        path: the TSV to read. ``.gz`` is handled transparently.
        require_alignment: assert that the column labels are the row labels in
            the same order. Turn this off only for a deliberately rectangular
            matrix, such as ``key_protid_tmscore_features.tsv``, whose columns
            are a subset of the proteins.
        repair: if the labels are the same set in a different order, reorder the
            columns to match the rows and warn loudly, instead of raising.
        column_prefix: stripped from each column label before comparison.
            ``calculate_key_protid_tmscores.py`` writes columns prefixed
            ``TMscore_v_``.
        censored_fill_token: the exact string treated as "not measured".
        dtype: float dtype for the values. float32 is lossless with respect to
            Foldseek's four-significant-figure output and halves memory
            (ADR 0004).

    Returns:
        A :class:`LabeledMatrix`.

    Raises:
        MatrixAlignmentError: on a label-set mismatch, or on a label-order
            mismatch when ``require_alignment`` is set and ``repair`` is not.
    """
    source = str(path)
    if not os.path.exists(source):
        raise FileNotFoundError(f"No matrix file at {source}")
    if repair and not require_alignment:
        raise ValueError(
            "repair=True has no effect when require_alignment=False, and returning a "
            "still-permuted matrix from a call that asked for a repair is worse than "
            "refusing. Choose one: require_alignment=True with repair=True to reorder, "
            "or require_alignment=False to accept the matrix as written."
        )

    protids: list[str] = []
    rows: list[np.ndarray] = []
    censored_rows: list[np.ndarray] = []
    measured_zero_tokens: dict = {}
    underflowed: dict = {}
    overflowed: dict = {}

    handle = _open_text(source)
    try:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise MatrixAlignmentError(f"{source} is empty.") from None

        if not header or header[0] != PROTID_COLUMN:
            raise MatrixAlignmentError(
                f"{source} does not start with a {PROTID_COLUMN!r} column; "
                f"first header field is {(header or [''])[0]!r}. "
                "This loader is for labeled matrices written by "
                "`pivot_foldseek_results`."
            )

        columns = [
            c[len(column_prefix) :] if column_prefix and c.startswith(column_prefix) else c
            for c in header[1:]
        ]
        n_cols = len(columns)

        for lineno, parts in enumerate(reader, start=2):
            if not parts or (len(parts) == 1 and not parts[0].strip()):
                continue  # tolerate a trailing blank line
            if len(parts) != n_cols + 1:
                raise MatrixAlignmentError(
                    f"{source} line {lineno}: expected {n_cols + 1} fields "
                    f"(protid + {n_cols} columns) but found {len(parts)}."
                )
            protids.append(parts[0])
            tokens = parts[1:]

            # The mask is built from the raw token, before any float conversion.
            # This is the whole reason the loader parses by hand rather than
            # handing the file to pandas: after conversion the distinction is
            # gone. See ADR 0009.
            mask = np.fromiter((t == censored_fill_token for t in tokens), dtype=bool, count=n_cols)
            try:
                values = np.asarray(tokens, dtype=dtype)
            except ValueError as exc:
                raise MatrixAlignmentError(
                    f"{source} line {lineno}: could not parse a numeric value ({exc})."
                ) from None

            # Any cell that reads as zero but was not the fill is a *measured*
            # zero, and its existence invalidates `values == 0` as a censoring
            # test for this file. Record it rather than assuming it away.
            #
            # The comparison is done in double, not in `dtype`. A token like
            # 1e-46 is nonzero in double and underflows to 0.0 in float32; if we
            # only looked at the cast value we would see a zero, find that it is
            # not the fill token, and then -- because `float(token) != 0.0` --
            # fail to count it either. It would be a zero that is neither
            # censored nor flagged, which is exactly the ambiguity this loader
            # exists to remove.
            suspicious = np.flatnonzero((values == 0) & ~mask)
            for j in suspicious:
                token = tokens[int(j)]
                exact = _as_float(token)
                if exact == 0.0:
                    measured_zero_tokens[token] = measured_zero_tokens.get(token, 0) + 1
                elif exact is not None:
                    underflowed[token] = underflowed.get(token, 0) + 1

            # Overflow is the same class of silent corruption in the other
            # direction: a finite token becoming inf.
            if not np.all(np.isfinite(values)):
                bad = np.flatnonzero(~np.isfinite(values))
                for j in bad:
                    token = tokens[int(j)]
                    exact = _as_float(token)
                    if exact is not None and np.isfinite(exact):
                        overflowed[token] = overflowed.get(token, 0) + 1

            rows.append(values)
            censored_rows.append(mask)
    finally:
        handle.close()

    if not protids:
        raise MatrixAlignmentError(f"{source} has a header but no data rows.")

    values_arr = np.vstack(rows)
    censored_arr = np.vstack(censored_rows)

    matrix = LabeledMatrix(
        protids=protids,
        columns=columns,
        values=values_arr,
        censored=censored_arr,
        source=source,
        column_prefix=column_prefix,
        _measured_zero_tokens=measured_zero_tokens,
        _precision_loss={"underflowed": underflowed, "overflowed": overflowed},
    )

    if underflowed or overflowed:
        warnings.warn(
            f"{source}: {sum(underflowed.values())} value(s) underflowed and "
            f"{sum(overflowed.values())} overflowed when cast to "
            f"{np.dtype(dtype).name}. An underflowed value becomes a zero that is "
            "neither the censoring fill nor a measured zero, so it would be "
            "invisible to both checks. Examples: "
            f"{sorted(underflowed)[:3] + sorted(overflowed)[:3]}",
            RuntimeWarning,
            stacklevel=2,
        )

    if require_alignment:
        if repair and matrix.protids != matrix.columns:
            assert_same_label_set(matrix.protids, matrix.columns, source)
            in_place = sum(1 for r, c in zip(matrix.protids, matrix.columns) if r == c)
            warnings.warn(
                f"Reordering the columns of {source} to match its row order. "
                f"Only {in_place} of {len(matrix.protids)} columns were in their row "
                "position. This is the PR #106 defect: the matrix header was written "
                "from an unsorted set, so the column order is a permutation of the "
                "row order. The data is sound and the repair is exact, but the file "
                "on disk is still wrong -- regenerate it with a build that includes "
                "PR #106.",
                RuntimeWarning,
                stacklevel=2,
            )
            matrix = matrix.reindexed_to_rows()
        else:
            assert_aligned(matrix.protids, matrix.columns, source)

    return matrix


def load_matrix_frame(
    path: str,
    *,
    require_alignment: bool = True,
    repair: bool = False,
    column_prefix: str = "",
) -> pd.DataFrame:
    """Convenience wrapper returning just the DataFrame, after the checks.

    For callers that want the legacy shape but not the legacy silence. The mask
    is discarded, so prefer :func:`load_labeled_matrix` for anything that should
    be censoring-aware.
    """
    return load_labeled_matrix(
        path,
        require_alignment=require_alignment,
        repair=repair,
        column_prefix=column_prefix,
    ).to_frame()


def summarize_censoring(matrix: LabeledMatrix) -> dict:
    """Diagnostics for a loaded matrix. Feeds the manifest and the QC panel.

    The per-row/per-column asymmetry is the tell for a per-query cap: Foldseek's
    ``--max-seqs`` bounds how many partners each *query* reports, so rows pile up
    at the cap while columns are free to vary. See ADR 0009.
    """
    censored = matrix.censored
    n_rows, n_cols = censored.shape
    per_row_measured = (~censored).sum(axis=1)
    per_col_measured = (~censored).sum(axis=0)

    summary = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_cells": int(censored.size),
        "n_censored": int(censored.sum()),
        "censoring_rate": matrix.censoring_rate,
        "measured_per_row": {
            "min": int(per_row_measured.min()) if n_rows else 0,
            "median": float(np.median(per_row_measured)) if n_rows else 0.0,
            "max": int(per_row_measured.max()) if n_rows else 0,
        },
        "measured_per_col": {
            "min": int(per_col_measured.min()) if n_cols else 0,
            "median": float(np.median(per_col_measured)) if n_cols else 0.0,
            "max": int(per_col_measured.max()) if n_cols else 0,
        },
        "measured_zero_count": matrix.measured_zero_count(),
        "is_aligned": matrix.is_aligned,
    }

    # A hard per-query cap shows up as a large share of rows sitting at exactly
    # the maximum, *with no corresponding pile-up in the columns*. Both halves
    # matter: a matrix that simply has one censored cell per row also puts every
    # row on the same count, and calling that a per-query cap would be wrong.
    # Foldseek's --max-seqs bounds each query's partner count, so rows pile up
    # while columns are free to vary. See ADR 0009.
    if n_rows:
        row_max = int(per_row_measured.max())
        at_cap = int((per_row_measured == row_max).sum())
        summary["rows_at_max"] = at_cap
        summary["rows_at_max_fraction"] = at_cap / n_rows
        col_max = int(per_col_measured.max()) if n_cols else 0
        cols_at_max = int((per_col_measured == col_max).sum()) if n_cols else 0
        summary["cols_at_max_fraction"] = (cols_at_max / n_cols) if n_cols else 0.0
        summary["cap_detected"] = bool(
            row_max < n_cols
            and at_cap / n_rows >= 0.5
            # Columns must NOT show the same pile-up. If they do, the uniformity
            # is a property of the whole matrix rather than of each query, and a
            # per-query cap is the wrong explanation.
            and (cols_at_max / n_cols) < 0.5
        )
        if summary["cap_detected"]:
            summary["inferred_cap"] = row_max
            summary["censoring_predicted_by_cap"] = 1.0 - (row_max / n_cols)
    return summary


def detect_permutation(matrix: LabeledMatrix) -> dict | None:
    """Describe the row/column permutation, or None if the matrix is aligned.

    Reported rather than raised, so a caller can log the state of an archived
    matrix without failing.
    """
    if matrix.is_aligned:
        return None
    if set(matrix.protids) != set(matrix.columns):
        return {"same_label_set": False}
    n = len(matrix.protids)
    in_place = sum(1 for r, c in zip(matrix.protids, matrix.columns) if r == c)
    return {
        "same_label_set": True,
        "n": n,
        "columns_in_row_position": in_place,
        "fraction_in_place": (in_place / n) if n else 0.0,
    }
