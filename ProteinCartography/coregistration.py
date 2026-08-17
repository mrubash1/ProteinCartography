#!/usr/bin/env python
"""Co-registration: one protein index, several independent geometries.

Co-registration is the default product of this work (ADR 0001). Fusion mixes
several kinds of evidence into one set of coordinates and is an explicitly
invoked analysis; co-registration leaves each kind of evidence in its own space
and puts the *same* proteins in all of them, so that where two spaces disagree
is visible rather than averaged away.

That guarantee has a precondition nothing enforced until now. The four blocks
draw their protein sets from three different files -- `tmscore` from the
similarity matrix, `threedi` from the descriptor table, `biophys` and `domains`
from the UniProt features table -- and those files are produced by different
rules at different points in the pipeline. Two spaces built over slightly
different sets still reduce cleanly, still plot, and still look co-registered.
Every per-protein comparison between them would then be quietly conditioned on
an overlap nobody chose. `docs/FOLLOWUPS.md` #30 recorded this against group 7;
this module is the answer to it.

**The index is the intersection, and the loss is reported rather than hidden.**
Refusing outright to co-register spaces whose sets differ would be wrong: a
provider legitimately drops a protein it has no data for -- a structure with no
UniProt record, a sequence Foldseek could not fold -- and that is a fact about
the cohort, not a failure. What must never happen is losing those proteins
without saying so. So the intersection is taken, in the reference space's order,
and every dropped protein is named in the report.

The one hard error is an empty intersection, which means the spaces have no
protein in common and there is nothing to co-register.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from index import IndexAlignmentError, ProteinIndex

__all__ = [
    "CoregistrationError",
    "CoregistrationReport",
    "shared_index",
]


class CoregistrationError(Exception):
    """Raised when a set of spaces cannot be co-registered at all."""


@dataclass(frozen=True)
class SpaceContribution:
    """What one space brought to the shared index, and what it lost joining it.

    `dropped` is the interesting field. It is the proteins this space measured
    that at least one other space did not, so they cannot appear in any
    cross-space comparison. An empty tuple is the healthy case and the one the
    demo produces; a long tuple means two stages of the pipeline disagree about
    the cohort, which is worth knowing before reading any of the maps.
    """

    space_id: str
    n_own: int
    dropped: tuple

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def is_complete(self) -> bool:
        """True when this space contributed every protein it had."""
        return not self.dropped

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "n_own": self.n_own,
            "n_dropped": self.n_dropped,
            "dropped": list(self.dropped),
        }


@dataclass(frozen=True)
class CoregistrationReport:
    """The shared index, plus an account of how each space reached it."""

    index: ProteinIndex
    reference: str
    contributions: tuple

    @property
    def n_shared(self) -> int:
        return len(self.index)

    @property
    def is_exact(self) -> bool:
        """True when every space had exactly the shared set, so nothing was lost."""
        return all(c.is_complete for c in self.contributions)

    def contribution(self, space_id: str) -> SpaceContribution:
        for c in self.contributions:
            if c.space_id == space_id:
                return c
        raise KeyError(space_id)

    def describe(self) -> str:
        """A short human-readable account, for stderr and for the review log."""
        if self.is_exact:
            return (
                f"{len(self.contributions)} spaces co-registered over "
                f"{self.n_shared} proteins; every space had the full set."
            )
        lines = [
            f"{len(self.contributions)} spaces co-registered over {self.n_shared} "
            f"proteins (reference: {self.reference!r}).",
            "",
            "Some spaces measured proteins the others did not. Those proteins are "
            "absent from every cross-space comparison below, so the comparison is "
            "conditioned on the overlap rather than on the cohort:",
        ]
        for c in self.contributions:
            if c.is_complete:
                continue
            shown = list(c.dropped)[:5]
            suffix = f" (+{c.n_dropped - 5} more)" if c.n_dropped > 5 else ""
            lines.append(f"  {c.space_id}: {c.n_own} own, {c.n_dropped} dropped: {shown}{suffix}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "n_shared": self.n_shared,
            "is_exact": self.is_exact,
            "protids": self.index.as_list,
            "spaces": [c.to_dict() for c in self.contributions],
        }


def shared_index(
    space_protids: Mapping[str, Sequence[str]],
    reference: str | None = None,
) -> CoregistrationReport:
    """The protein index every named space can be aligned to.

    Args:
        space_protids: space id -> that space's protids, in its own order.
        reference: which space's ordering the shared index inherits. Defaults to
            the first key. Order has to come from somewhere nameable: set
            iteration order is not reproducible, and sorting would silently
            reorder every space away from the order its blocks were computed in.

    Returns:
        A :class:`CoregistrationReport` whose `index` is the intersection in the
        reference space's order.

    Raises:
        CoregistrationError: no spaces given, an unknown reference, or an empty
            intersection.
        IndexAlignmentError: a space's own protids contain a duplicate.
    """
    if not space_protids:
        raise CoregistrationError(
            "no spaces to co-register. `coregistration.compare` needs at least "
            "one space, and comparing spaces needs at least two."
        )

    space_ids = list(space_protids)
    if reference is None:
        reference = space_ids[0]
    elif reference not in space_protids:
        raise CoregistrationError(
            f"coregistration reference {reference!r} is not among the spaces being "
            f"co-registered: {space_ids}."
        )

    # Building a ProteinIndex per space is not ceremony: it is where a duplicate
    # protid is caught. A repeated row doubles that protein's weight in a
    # geometry, and an intersection computed over sets would have discarded the
    # evidence that it happened.
    indexes = {}
    for space_id in space_ids:
        try:
            indexes[space_id] = ProteinIndex.from_iterable(space_protids[space_id])
        except IndexAlignmentError as error:
            raise CoregistrationError(f"space {space_id!r}: {error}") from error

    shared = indexes[reference]
    for space_id in space_ids:
        if space_id == reference:
            continue
        try:
            shared = shared.intersection(indexes[space_id])
        except IndexAlignmentError as error:
            raise CoregistrationError(
                f"spaces {reference!r} and {space_id!r} share no proteins, so there "
                f"is nothing to co-register: {error}"
            ) from error

    kept = set(shared.protids)
    contributions = tuple(
        SpaceContribution(
            space_id=space_id,
            n_own=len(indexes[space_id]),
            dropped=tuple(p for p in indexes[space_id] if p not in kept),
        )
        for space_id in space_ids
    )
    return CoregistrationReport(index=shared, reference=reference, contributions=contributions)
