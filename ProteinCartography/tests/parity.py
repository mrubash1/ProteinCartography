#!/usr/bin/env python
"""Machinery for comparing two pipeline runs file by file.

The promise this work makes is that the default configuration produces the same
output it always has. That promise is only worth something if it is checked
mechanically, so this module runs the pipeline twice -- from two checkouts, or
twice from one -- and compares every file that lands in the output tree.

**It compares everything and excludes by explicit rule**, rather than comparing
an allowlist. An allowlist silently ignores files nobody thought to list, which
is the wrong failure direction: a new output that differs should break the test
and make someone justify it.

Three kinds of difference are known and are handled openly:

*Normalized.* Plotly stamps one random ``<div>`` uuid into each HTML figure.
Strip it and three of the four HTML outputs are byte-identical, so they are
compared after normalization rather than skipped.

*Excluded by rule.* Clock readings, and foldseek's internal working directories.
Each exclusion carries its reason, and every one was arrived at by measurement:
the first version of this harness excluded far more, and the first version after
that excluded far less and failed. Foldseek's `temp/` in particular cannot be
compared at all -- its scratch subdirectory is named with a random integer, so
the *set* of files differs between runs, and which shard a record lands in
depends on thread scheduling.

*Genuinely nondeterministic.* The semantic-analysis wordcloud has a stochastic
layout. Rather than assert in advance that it does not matter, the harness runs
the pipeline twice from the *same* code and records what differs -- a self-diff.
Anything that differs there cannot be evidence in the cross-version comparison;
anything that does not must be identical there.

Two things keep this honest. The self-diff establishes the floor empirically
instead of trusting the exclusion list, and
:func:`assert_critical_outputs_compared` names the artifacts that carry the
promise and checks they were actually reached -- because a parity test can be
hollowed out one reasonable-looking exclusion at a time.

Run it directly for a report:

    python ProteinCartography/tests/parity.py --baseline ../pc-baseline
"""

from __future__ import annotations
import argparse
import filecmp
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CRITICAL_OUTPUTS",
    "EXCLUSIONS",
    "assert_critical_outputs_compared",
    "ParityReport",
    "compare_trees",
    "normalize_bytes",
    "run_pipeline",
]

# One random uuid per Plotly figure, in the div id and the matching script call.
_PLOTLY_UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

#: Glob patterns excluded from comparison, each with the reason it is excluded.
#: An exclusion silences a file forever, so each one below is a claim -- checked
#: against two real runs -- that the difference is about representation or
#: scratch state rather than about the science.
EXCLUSIONS = (
    ("benchmarks/*", "snakemake benchmark files record wall-clock time"),
    (
        "*/temp/*",
        "foldseek's internal working directory. Its split databases distribute "
        "records across shards by thread scheduling, its scratch subdirectory is "
        "named with a random integer, and its .lookup/.source files hold internal "
        "numeric ids assigned in directory-scan order. None of it is a pipeline "
        "output; the derived TSVs are compared and are what PR #106 made "
        "deterministic. (Verified: including it produced 3 differing shards and 4 "
        "randomly-named files per run, and *which* shard differed varied between "
        "runs, so the self-diff floor could not stabilize it either.)",
    ),
    ("*/temp_tm/*", "as above, foldseek's working directory for key-protid TM-scores"),
    (
        "foldseek_clustering_results/all_by_all_tmscore.tsv",
        "the raw pair list, emitted in foldseek shard order. Its sorted derivative "
        "all_by_all_tmscore_pivoted.tsv IS compared and IS byte-identical, which is "
        "precisely the guarantee PR #106 added.",
    ),
    (
        "key_protid_tmscores_results/key_protid_tmscores.tsv",
        "raw pair list, as above; the pivoted key_protid_tmscore_features.tsv IS "
        "compared and IS byte-identical",
    ),
    ("*.pdf", "matplotlib embeds a creation timestamp in the PDF metadata"),
    ("*.svg", "matplotlib embeds a creation timestamp"),
    ("*.snakemake_timestamp", "snakemake bookkeeping, holds an mtime"),
)

#: The artifacts whose byte-identity *is* the backwards-compatibility promise.
#: `compare_trees` reports whether each was actually compared, so a future
#: exclusion cannot quietly remove one of them from the test's reach.
CRITICAL_OUTPUTS = (
    "final_results/{name}_aggregated_features.tsv",
    "final_results/{name}_leiden_similarity.tsv",
    "final_results/{name}_strucluster_similarity.tsv",
    "foldseek_clustering_results/all_by_all_tmscore_pivoted.tsv",
    "foldseek_clustering_results/all_by_all_tmscore_pivoted_pca_umap.tsv",
    "foldseek_clustering_results/leiden_features.tsv",
    "foldseek_clustering_results/struclusters_features.tsv",
    "protein_features/key_protid_tmscore_features.tsv",
    "protein_features/pdb_features.tsv",
    "protein_features/source_features.tsv",
    "protein_features/uniprot_features.tsv",
)


def _excluded_by(relpath: str):
    for pattern, reason in EXCLUSIONS:
        if fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch(f"/{relpath}", f"*/{pattern}"):
            return reason
    return None


def normalize_bytes(relpath: str, data: bytes) -> bytes:
    """Strip known-random, semantically empty content before comparing."""
    if relpath.endswith(".html"):
        # Plotly generates a fresh uuid for each figure's container div.
        return _PLOTLY_UUID.sub(b"PLOTLY-DIV-UUID", data)
    return data


@dataclass
class ParityReport:
    identical: list = field(default_factory=list)
    normalized_identical: list = field(default_factory=list)
    differing: list = field(default_factory=list)
    only_in_a: list = field(default_factory=list)
    only_in_b: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.differing or self.only_in_a or self.only_in_b)

    @property
    def n_compared(self) -> int:
        return len(self.identical) + len(self.normalized_identical) + len(self.differing)

    def describe(self, limit: int = 40) -> str:
        lines = [
            f"compared           : {self.n_compared}",
            f"  byte-identical   : {len(self.identical)}",
            f"  identical after normalization: {len(self.normalized_identical)}",
            f"  DIFFERING        : {len(self.differing)}",
            f"excluded by rule   : {len(self.excluded)}",
            f"only in A          : {len(self.only_in_a)}",
            f"only in B          : {len(self.only_in_b)}",
        ]
        for rel in self.differing[:limit]:
            lines.append(f"  ! {rel}")
        if len(self.differing) > limit:
            lines.append(f"  ... and {len(self.differing) - limit} more")
        for rel in self.only_in_a[:limit]:
            lines.append(f"  + only in A: {rel}")
        for rel in self.only_in_b[:limit]:
            lines.append(f"  - only in B: {rel}")
        return "\n".join(lines)


def _relative_files(root: Path) -> set:
    return {
        str(Path(dirpath, name).relative_to(root))
        for dirpath, _dirs, names in os.walk(root)
        for name in names
    }


def compare_trees(a: Path, b: Path, *, ignore: set = frozenset()) -> ParityReport:
    """Compare two output trees. `ignore` is a set of relpaths to skip entirely.

    `ignore` is how the self-diff result is fed back in: paths already shown to
    differ between two runs of identical code cannot be evidence about a code
    change.
    """
    a, b = Path(a), Path(b)
    report = ParityReport()
    files_a, files_b = _relative_files(a), _relative_files(b)

    for rel in sorted(files_a | files_b):
        reason = _excluded_by(rel)
        if reason is not None:
            report.excluded[rel] = reason
            continue
        if rel in ignore:
            report.excluded[rel] = "known nondeterministic (established by self-diff)"
            continue
        if rel not in files_b:
            report.only_in_a.append(rel)
            continue
        if rel not in files_a:
            report.only_in_b.append(rel)
            continue

        if filecmp.cmp(a / rel, b / rel, shallow=False):
            report.identical.append(rel)
            continue
        da = normalize_bytes(rel, (a / rel).read_bytes())
        db = normalize_bytes(rel, (b / rel).read_bytes())
        if da == db:
            report.normalized_identical.append(rel)
        else:
            report.differing.append(rel)
    return report


def run_pipeline(
    repo: Path,
    workdir: Path,
    *,
    conda_prefix: Path | None = None,
    cores: int = 8,
    extra_config: dict | None = None,
    dataset: str = "actin",
) -> Path:
    """Run the mocked search-mode pipeline from `repo` into `workdir`.

    Mocked, so it needs no network. `conda_prefix` lets two checkouts share one
    set of built conda environments, which is what makes running the baseline
    affordable.
    """
    repo, workdir = Path(repo), Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    input_dir, output_dir = workdir / "input", workdir / "output"
    shutil.copytree(
        repo
        / "ProteinCartography"
        / "tests"
        / "integration-test-artifacts"
        / "search-mode"
        / dataset
        / "input",
        input_dir,
    )

    config = {
        "mode": "search",
        "analysis_name": "parity",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "plotting_modes": ["pca_umap"],
        "max_blast_hits": 10,
        "max_foldseek_hits": 10,
        "max_structures": 10,
    }
    config.update(extra_config or {})

    import yaml

    config_path = workdir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    env = dict(os.environ)
    env["PROTEINCARTOGRAPHY_SHOULD_USE_MOCKS"] = "true"
    env.pop("PROTEINCARTOGRAPHY_SHOULD_LOG_API_REQUESTS", None)

    cmd = [
        sys.executable,
        "-m",
        "snakemake",
        "--snakefile",
        str(repo / "Snakefile"),
        "--configfile",
        str(config_path),
        "--use-conda",
        "--conda-frontend",
        "conda",
        "--cores",
        str(cores),
    ]
    if conda_prefix is not None:
        cmd += ["--conda-prefix", str(conda_prefix)]

    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pipeline run from {repo} failed ({proc.returncode})\n"
            f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-5000:]}"
        )
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="path to the baseline checkout")
    parser.add_argument("--repo", default=".", help="path to the checkout under test")
    parser.add_argument("--workdir", default="/tmp/pc-parity")
    parser.add_argument(
        "--conda-prefix",
        default=None,
        help="shared conda env directory, so the baseline does not rebuild them",
    )
    args = parser.parse_args(argv)

    repo, baseline = Path(args.repo).resolve(), Path(args.baseline).resolve()
    work = Path(args.workdir)
    prefix = Path(args.conda_prefix) if args.conda_prefix else repo / ".snakemake" / "conda"

    print("running HEAD twice to establish the nondeterminism floor ...")
    head_1 = run_pipeline(repo, work / "head1", conda_prefix=prefix)
    head_2 = run_pipeline(repo, work / "head2", conda_prefix=prefix)
    self_diff = compare_trees(head_1, head_2)
    print("self-diff:\n" + self_diff.describe())

    print("\nrunning the baseline ...")
    base = run_pipeline(baseline, work / "base", conda_prefix=prefix)
    parity = compare_trees(head_1, base, ignore=set(self_diff.differing))
    print("parity vs baseline:\n" + parity.describe())

    return 0 if parity.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


def assert_critical_outputs_compared(report: ParityReport, analysis_name: str) -> None:
    """Raise unless every artifact in :data:`CRITICAL_OUTPUTS` was really compared.

    A parity test can be hollowed out one exclusion at a time without anyone
    noticing, because each individual exclusion looks reasonable. This makes the
    hollowing-out fail loudly: these specific files must appear in the compared
    set, not in the excluded set.
    """
    compared = set(report.identical) | set(report.normalized_identical) | set(report.differing)
    expected = {path.format(name=analysis_name) for path in CRITICAL_OUTPUTS}
    missing = sorted(expected - compared)
    if missing:
        reasons = {m: report.excluded.get(m, "not produced by the run") for m in missing}
        raise AssertionError(
            "these outputs carry the backwards-compatibility promise but were not "
            "compared:\n" + "\n".join(f"  {path}: {why}" for path, why in reasons.items())
        )


# ---------------------------------------------------------------------------
# component-level parity at N > 500
# ---------------------------------------------------------------------------
#
# The end-to-end fixture is 11 proteins, and mutation testing showed that is
# structurally unable to see several realistic refactor errors: at N=11 the PCA
# component count, the UMAP neighbour count and Leiden's n_pcs are all clamped
# to the same value whatever the config says, and the censoring fill token never
# appears because all 121 pairs are measured.
#
# Running the whole pipeline at N>500 would mean synthesizing 750 PDB files and
# a Foldseek run. But the port only touched the reduction step, and that step
# takes a matrix -- not structures. So the matrix is generated directly and the
# reducers are run on it from both checkouts. This reaches the regime the small
# fixture cannot: above 500 rows, `svd_solver="auto"` would switch to the
# randomized solver, which is the defect PR #106 fixed.
#
# Procedural, seeded, and not committed as data: the seed is the fixture.

CENSORED_FILL = "0.0"

DEFAULT_FIXTURE_N = 750
DEFAULT_FIXTURE_SEED = 0
DEFAULT_FIXTURE_CAP = 300


def synthetic_matrix(
    path: Path,
    *,
    n: int = DEFAULT_FIXTURE_N,
    seed: int = DEFAULT_FIXTURE_SEED,
    cap: int = DEFAULT_FIXTURE_CAP,
    permute_columns: bool = False,
) -> Path:
    """Write a similarity matrix with the statistical shape of a real one.

    Reproduces the three properties measured on production output that matter
    for the code under test:

    * a per-query cap, so most rows report exactly `cap` partners and the rest
      are the literal ``"0.0"`` fill -- the censoring of ADR 0009;
    * scores in Foldseek's ``%.3E`` form with no low tail, so a fill is a value
      the generator never otherwise emits;
    * an exact 1.0 on the *label* diagonal.

    `permute_columns` reproduces the PR #106 defect for tests that need it.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    labels = [f"P{i:05d}" for i in range(n)]
    columns = list(labels)
    if permute_columns:
        columns = [labels[i] for i in rng.permutation(n)]

    # Column label -> the protein's canonical index, so a permuted header still
    # writes each score into the cell its labels claim. Writing by position here
    # would generate a matrix that is wrong rather than merely permuted.
    index_of = {label: i for i, label in enumerate(labels)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\t".join(["protid"] + columns) + "\n")
        for i, row_label in enumerate(labels):
            # Each query reports its own capped set of partners, chosen
            # independently, so every row sits at exactly the cap while the
            # columns vary -- the signature of a per-query limit rather than a
            # score threshold. The self-hit is always reported and is counted
            # against the cap, so the row totals are exactly uniform.
            others = [j for j in range(n) if j != i]
            k = max(0, min(cap, n) - 1)
            partners = set(rng.choice(others, size=k, replace=False).tolist())
            partners.add(i)
            scores = rng.uniform(0.05, 0.99, size=n)
            cells = []
            for column_label in columns:
                j = index_of[column_label]
                if j == i:
                    cells.append("1.000E+00")
                elif j in partners:
                    cells.append(f"{scores[j]:.3E}")
                else:
                    cells.append(CENSORED_FILL)
            fh.write("\t".join([row_label] + cells) + "\n")
    return path


def run_reducer(
    repo: Path,
    matrix_path: Path,
    workdir: Path,
    *,
    mode: str = "pca_umap",
    python: str | None = None,
) -> Path:
    """Run `dim_reduction.py` from `repo` on `matrix_path`, into `workdir`.

    The matrix is copied in first, because the script derives its output paths
    from its input path.
    """
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    local_matrix = workdir / Path(matrix_path).name
    shutil.copy(matrix_path, local_matrix)

    cmd = [
        python or sys.executable,
        str(Path(repo) / "ProteinCartography" / "dim_reduction.py"),
        "--input",
        str(local_matrix),
        "--mode",
        mode,
    ]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dim_reduction from {repo} failed ({proc.returncode})\n{proc.stderr[-4000:]}"
        )
    return workdir
