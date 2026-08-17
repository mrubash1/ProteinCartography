#!/usr/bin/env python
"""Mutation testing for the parity test.

A parity test that passes because it is not really comparing anything is worse
than no test: it manufactures confidence in the one place this work most needs
real confidence. The only way to know the test detects change is to change
something on purpose and watch it fail.

Each mutation below is a small, plausible edit to a pipeline script -- the kind
of thing a refactor gets wrong. For each one this harness patches the file, runs
the pipeline, compares against an unmutated reference run, and records whether
parity noticed. A mutation that survives is a hole in the test, and the fix is a
new assertion, never a relaxed one.

Run it directly:

    python ProteinCartography/tests/mutation_check.py --workdir /tmp/pc-mutation

It restores every file it touches, including on failure.
"""

from __future__ import annotations
import argparse
import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from parity import compare_trees, run_pipeline

__all__ = ["MUTATIONS", "Mutation", "run_mutation_suite"]


@dataclass(frozen=True)
class Mutation:
    """A deliberate defect, and what it is meant to prove the test can see."""

    name: str
    path: str
    old: str
    new: str
    detects: str


MUTATIONS = (
    Mutation(
        name="pca_components",
        path="ProteinCartography/dim_reduction.py",
        old="n_components=30,\n            prep_step=True,",
        new="n_components=20,\n            prep_step=True,",
        detects="a changed PCA dimensionality, which moves every coordinate",
    ),
    Mutation(
        name="umap_neighbors",
        path="ProteinCartography/dim_reduction.py",
        old="    n_neighbors=80,",
        new="    n_neighbors=40,",
        detects="a changed UMAP parameter, which changes the layout but not the matrix",
    ),
    Mutation(
        name="unsorted_matrix_columns",
        path="ProteinCartography/foldseek_clustering.py",
        old="    return entries, sorted(targets)",
        new="    return entries, list(targets)",
        detects=(
            "reintroducing PR #106's column-order defect -- the single most "
            "important regression this test exists to catch"
        ),
    ),
    Mutation(
        name="censoring_fill_value",
        path="ProteinCartography/foldseek_clustering.py",
        old='scores.append(targets_to_scores.get(target, "0.0"))',
        new='scores.append(targets_to_scores.get(target, "0.00"))',
        detects="a changed fill token, which would break every censoring mask downstream",
    ),
    Mutation(
        name="leiden_n_pcs",
        path="ProteinCartography/leiden_clustering.py",
        old="    n_pcs=30,",
        new="    n_pcs=10,",
        detects="a changed clustering parameter, on the branch that forks from the matrix",
    ),
    Mutation(
        name="feature_join_semantics",
        path="ProteinCartography/aggregate_features.py",
        old='agg_df = agg_df.merge(df, on="protid", how="outer")',
        new='agg_df = agg_df.merge(df, on="protid", how="inner")',
        detects="a changed join, which silently drops proteins from the feature table",
    ),
    Mutation(
        name="pca_solver",
        path="ProteinCartography/dim_reduction.py",
        old='svd_solver="full", random_state=random_state',
        new='svd_solver="auto", random_state=random_state',
        detects=(
            "undoing PR #106's determinism fix. At N=11 the solver choice is a "
            "no-op, so this one is EXPECTED TO SURVIVE and is the evidence that "
            "the small fixture cannot stand in for the N>500 one"
        ),
    ),
    Mutation(
        name="transposed_matrix",
        path="ProteinCartography/foldseek_clustering.py",
        old="            csv_writer.writerow(get_line_for_protid(entry, targets))",
        new="            csv_writer.writerow(get_line_for_protid(entry, targets)[:1] "
        "+ list(reversed(get_line_for_protid(entry, targets)[1:])))",
        detects="reversed column order within each row, with the header left alone",
    ),
)


@contextlib.contextmanager
def _patched(repo: Path, mutation: Mutation):
    """Apply a mutation, guaranteeing restoration."""
    target = repo / mutation.path
    original = target.read_text()
    if mutation.old not in original:
        raise RuntimeError(
            f"mutation {mutation.name!r} does not apply: its anchor text is not in "
            f"{mutation.path}. The file has changed and the mutation needs updating."
        )
    try:
        target.write_text(original.replace(mutation.old, mutation.new, 1))
        yield
    finally:
        target.write_text(original)


def run_mutation_suite(repo: Path, workdir: Path, conda_prefix: Path) -> list:
    """Run every mutation and report which ones parity detected."""
    repo, workdir = Path(repo), Path(workdir)

    print("=== reference run (unmutated) ===", flush=True)
    reference = run_pipeline(repo, workdir / "reference", conda_prefix=conda_prefix)
    print("=== self-diff, to establish the nondeterminism floor ===", flush=True)
    reference_b = run_pipeline(repo, workdir / "reference_b", conda_prefix=conda_prefix)
    floor = set(compare_trees(reference, reference_b).differing)
    print(f"floor: {sorted(floor) or 'nothing -- fully deterministic'}\n", flush=True)

    results = []
    for mutation in MUTATIONS:
        print(f"=== mutation: {mutation.name} ===", flush=True)
        run_dir = workdir / f"mut_{mutation.name}"
        try:
            with _patched(repo, mutation):
                mutated = run_pipeline(repo, run_dir, conda_prefix=conda_prefix)
                report = compare_trees(reference, mutated, ignore=floor)
                detected, note = (not report.ok), ""
                changed = report.differing[:6]
        except RuntimeError as exc:
            # A mutation that makes the pipeline crash is also detected -- the
            # run failing is a louder signal than a diff.
            detected, note, changed = True, f"pipeline failed: {str(exc)[:120]}", []
        results.append(
            {
                "name": mutation.name,
                "detected": detected,
                "detects": mutation.detects,
                "changed_files": changed,
                "note": note,
            }
        )
        verdict = "DETECTED" if detected else "SURVIVED"
        print(f"  -> {verdict}  {note}", flush=True)
        for rel in changed:
            print(f"     ! {rel}", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        print(flush=True)
    return results


def format_report(results: list) -> str:
    lines = ["", "=" * 74, "MUTATION TESTING OF THE PARITY TEST", "=" * 74]
    detected = [r for r in results if r["detected"]]
    survived = [r for r in results if not r["detected"]]
    lines.append(f"{len(detected)}/{len(results)} mutations detected")
    lines.append("")
    for r in results:
        mark = "detected" if r["detected"] else "SURVIVED"
        lines.append(f"  [{mark:8s}] {r['name']}")
        lines.append(f"              {r['detects']}")
        if r["note"]:
            lines.append(f"              {r['note']}")
        for rel in r["changed_files"][:4]:
            lines.append(f"              ! {rel}")
    if survived:
        lines += [
            "",
            "Surviving mutations are holes in the parity test unless the entry's",
            "`detects` field explains why survival is expected.",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workdir", default="/tmp/pc-mutation")
    parser.add_argument("--conda-prefix", default=None)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    prefix = Path(args.conda_prefix) if args.conda_prefix else repo / ".snakemake" / "conda"
    results = run_mutation_suite(repo, Path(args.workdir), prefix)
    print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
