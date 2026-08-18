"""The names of the files a space run writes, owned in one place.

Every name here has exactly one writer and at least one reader, and until this
module existed each side spelled the name out for itself. That has produced four
defects in the explorer alone, every one of them silent:

- the readable mask read faithfulness from the wrong file,
- the aggregated-features path was wrong,
- the provenance footer read ``manifest.json``, which nothing writes, so every
  explorer ever built shipped an empty provenance section (GE.5),
- disagreement mode read per-protein Jaccard off ``summary.tsv``, which is
  aggregate only, so it coloured every protein ``null`` (GE.17).

None of them raised. A reader that cannot find a file returns "no data", which
renders as a blank panel rather than as an error, and no test noticed because
the tests built their fixtures with the *reader's* spelling rather than by
running the writer.

So the rule this module enforces is mechanical rather than advisory: **a name is
defined once and imported by both sides.** Drift stops being a thing that can be
reviewed for and becomes a thing that cannot be expressed.

This module must stay import-light. The explorer payload runs in the bare
environment (ADR 0006) and `test_optional_dependencies.py` asserts it pulls in
neither plotly nor the reducer stack, so nothing here may import numpy, pandas,
scikit-learn or any sibling pipeline module.
"""

from __future__ import annotations
import os

#: Directory under the run's output root holding the co-registration outputs.
COREGISTRATION_DIRNAME = "coregistration"

#: Aggregate co-registration table: one row per compared pair.
SUMMARY_FILENAME = "summary.tsv"

#: Separates the two space ids in a per-pair filename.
PAIR_SEPARATOR = "__vs__"

#: Per-space diagnostics, as written by ``diagnose_space``.
DIAGNOSTICS_FILENAME = "diagnostics.json"

#: A space's own partition, as written by ``diagnose_space``.
CLUSTERS_FILENAME = "clusters.tsv"

#: The pseudo-reducer under which ``diagnose_space`` writes its manifest.
DIAGNOSTICS_MANIFEST_KEY = "diagnostics"


def pair_filename(space_a: str, space_b: str) -> str:
    """The per-protein comparison file for one pair of spaces.

    The two ids are written in the order the comparison was made, and the
    summary row for that pair carries the same order, so a reader that has a
    summary row can always rebuild the filename from it.
    """
    return f"{space_a}{PAIR_SEPARATOR}{space_b}.tsv"


def manifest_filename(reducer: str) -> str:
    """A space's manifest for one reducer.

    There is no bare ``manifest.json`` at the space level -- that name belongs
    to the *block* store, which is a different directory and a different thing.
    Reading it at the space level is GE.5, and it fails by returning nothing.
    """
    return f"manifest_{reducer}.json"


def embedding_filename(reducer: str) -> str:
    """A space's 2-D embedding under one reducer."""
    return f"embedding_{reducer}.tsv"


def faithfulness_filename(reducer: str) -> str:
    """Per-protein trustworthiness and continuity under one reducer."""
    return f"faithfulness_{reducer}.tsv"


def aggregated_features_filename(analysis_name: str) -> str:
    """The legacy pipeline's aggregated feature table.

    Written by ``aggregate_features.py`` on the default path, which this module
    deliberately does not touch: that writer is upstream code under the
    byte-identical requirement. The name is recorded here so the *readers* added
    by this work share one spelling of it.
    """
    return f"{analysis_name}_aggregated_features.tsv"


def coregistration_dir(output_dir: str) -> str:
    """The co-registration output directory for a run."""
    return os.path.join(output_dir, COREGISTRATION_DIRNAME)


def summary_path(output_dir: str) -> str:
    """The co-registration summary table for a run."""
    return os.path.join(coregistration_dir(output_dir), SUMMARY_FILENAME)
