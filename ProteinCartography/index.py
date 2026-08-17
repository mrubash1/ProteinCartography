#!/usr/bin/env python
"""The canonical protein index, and alignment that refuses to guess.

Every block in a space is a table of numbers whose rows mean something only
because of the protids beside them. Getting two blocks out of step by one row is
undetectable downstream: the shapes still match, the reducer still runs, and the
map is wrong.

pandas makes this easy to do by accident. ``df.reindex(other_index)`` fills
missing labels with NaN and returns happily. This module exists so that the
pipeline never takes that path: :meth:`ProteinIndex.align` raises on a missing
protid instead of inventing a row for it.

See ``docs/adr/0001-block-space-view.md``.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["IndexAlignmentError", "ProteinIndex"]


class IndexAlignmentError(Exception):
    """Raised when data cannot be aligned to an index without inventing rows."""


def _describe(labels: Sequence[str], limit: int = 5) -> str:
    shown = list(labels)[:limit]
    suffix = f" (+{len(labels) - limit} more)" if len(labels) > limit else ""
    return f"{shown}{suffix}"


def _find_duplicates(labels: Sequence[str]) -> list:
    seen, dupes = set(), []
    for label in labels:
        if label in seen and label not in dupes:
            dupes.append(label)
        seen.add(label)
    return dupes


@dataclass(frozen=True)
class ProteinIndex:
    """An ordered, duplicate-free set of protids shared by every block.

    Frozen and hashable, so it can be stored on a manifest and compared cheaply
    between blocks.
    """

    protids: tuple

    def __post_init__(self):
        if not self.protids:
            raise IndexAlignmentError("A ProteinIndex cannot be empty.")
        seen, duplicates = set(), []
        for p in self.protids:
            if p in seen:
                duplicates.append(p)
            seen.add(p)
        if duplicates:
            raise IndexAlignmentError(
                f"Duplicate protids in index: {_describe(sorted(set(duplicates)))}. "
                "A protein must appear exactly once; a duplicate silently doubles "
                "its weight in every geometry."
            )
        object.__setattr__(self, "_positions", {p: i for i, p in enumerate(self.protids)})

    # -- construction ------------------------------------------------------

    @classmethod
    def from_iterable(cls, protids: Iterable) -> ProteinIndex:
        return cls(tuple(str(p) for p in protids))

    @classmethod
    def from_matrix(cls, matrix) -> ProteinIndex:
        """Build from a :class:`matrix_io.LabeledMatrix`'s row labels."""
        return cls.from_iterable(matrix.protids)

    # -- basics ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.protids)

    def __iter__(self):
        return iter(self.protids)

    def __contains__(self, protid) -> bool:
        return protid in self._positions

    def __getitem__(self, i):
        return self.protids[i]

    @property
    def as_list(self) -> list:
        return list(self.protids)

    def to_pandas(self) -> pd.Index:
        return pd.Index(self.protids, name="protid")

    def position(self, protid: str) -> int:
        try:
            return self._positions[protid]
        except KeyError:
            raise IndexAlignmentError(
                f"{protid!r} is not in this index of {len(self)} proteins."
            ) from None

    # -- alignment ---------------------------------------------------------

    def positions_of(self, protids: Sequence[str]) -> np.ndarray:
        """Positions of `protids` within this index. Raises on any absentee."""
        missing = [p for p in protids if p not in self._positions]
        if missing:
            raise IndexAlignmentError(
                f"{len(missing)} protid(s) are not in the index: {_describe(missing)}"
            )
        return np.fromiter((self._positions[p] for p in protids), dtype=np.intp, count=len(protids))

    def align(
        self,
        protids: Sequence[str],
        values: np.ndarray,
        *,
        axis: int = 0,
        what: str = "data",
    ) -> np.ndarray:
        """Reorder `values` along `axis` so its rows follow this index.

        Raises rather than filling. If `protids` is missing anything the index
        has, that is an error: the alternative is a NaN row that survives all the
        way to a coordinate.

        Extra protids not in the index are dropped, which is safe and is the
        common case when a block was computed over a superset.
        """
        protids = list(protids)
        if values.shape[axis] != len(protids):
            raise IndexAlignmentError(
                f"{what}: {len(protids)} protids but axis {axis} has length "
                f"{values.shape[axis]}."
            )

        # A duplicate in the *source* labels is as dangerous as one in the index,
        # and easier to miss because the source comes from a data file rather
        # than from code. Building `{label: position}` over it keeps only the
        # last occurrence, so the aligned row is one arbitrary copy and the
        # others vanish -- silently, because the length check above still passes.
        source_duplicates = _find_duplicates(protids)
        available = set(protids)
        if source_duplicates:
            also_missing = [p for p in self.protids if p not in available]
            lines = [
                f"{what}: duplicate protid(s) in the source labels: "
                f"{_describe(source_duplicates)}",
                "",
                "Which row belongs to a repeated protid is ambiguous, and resolving "
                "it by position would silently keep one copy and discard the rest. "
                "De-duplicate the source before aligning.",
            ]
            if also_missing:
                # Very often the same mistake: a row was duplicated in place of
                # the one that is missing. Say so rather than making the caller
                # fix one error to discover the next.
                lines += [
                    "",
                    f"Separately, {len(also_missing)} of the index's {len(self)} "
                    f"proteins are missing from the source: {_describe(also_missing)}",
                ]
            raise IndexAlignmentError("\n".join(lines))
        missing = [p for p in self.protids if p not in available]
        if missing:
            raise IndexAlignmentError(
                "\n".join(
                    [
                        f"{what} is missing {len(missing)} of the index's "
                        f"{len(self)} proteins: {_describe(missing)}",
                        "",
                        "Refusing to reindex, because filling these with NaN would "
                        "produce a row of coordinates for a protein that was never "
                        "measured. Either compute the block over the full index, or "
                        "narrow the index with ProteinIndex.intersection().",
                    ]
                )
            )

        source_pos = {p: i for i, p in enumerate(protids)}
        order = np.fromiter((source_pos[p] for p in self.protids), dtype=np.intp, count=len(self))
        return np.take(values, order, axis=axis)

    def align_frame(self, frame: pd.DataFrame, *, what: str = "features") -> pd.DataFrame:
        """`align` for a DataFrame indexed by protid."""
        aligned = self.align(list(frame.index), frame.to_numpy(), axis=0, what=what)
        return pd.DataFrame(aligned, index=self.to_pandas(), columns=frame.columns)

    # -- set operations ----------------------------------------------------

    def intersection(self, other: Iterable) -> ProteinIndex:
        """The proteins in both, in *this* index's order.

        Order is taken from self deliberately: intersection must be a
        deterministic function of its inputs, and set iteration order is not.
        """
        other_set = set(other)
        kept = tuple(p for p in self.protids if p in other_set)
        if not kept:
            raise IndexAlignmentError(
                "Intersection is empty: the two protid sets share no members."
            )
        return ProteinIndex(kept)

    def missing_from(self, protids: Iterable) -> list:
        available = set(protids)
        return [p for p in self.protids if p not in available]

    def equals(self, other: ProteinIndex) -> bool:
        return tuple(self.protids) == tuple(other.protids)

    def __repr__(self) -> str:
        head = ", ".join(repr(p) for p in self.protids[:3])
        tail = ", ..." if len(self) > 3 else ""
        return f"ProteinIndex({len(self)} proteins: [{head}{tail}])"
