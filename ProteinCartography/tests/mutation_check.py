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

from parity import compare_trees, run_pipeline, run_reducer, synthetic_matrix

__all__ = [
    "MUTATIONS",
    "REDUCER_MUTATIONS",
    "Mutation",
    "run_mutation_suite",
    "run_reducer_mutation_suite",
]


class MutationDidNotApply(RuntimeError):
    """The anchor text was not found, so nothing was mutated.

    Distinguished from every other failure because it is the one that must never
    be scored as a pass: no mutation was applied, so the run could not have
    differed, and calling that "detected" turns the suite into a no-op.
    """


@dataclass(frozen=True)
class Mutation:
    """A deliberate defect, and what it is meant to prove the test can see."""

    name: str
    path: str
    old: str
    new: str
    detects: str
    #: Set when the fixture is known to be incapable of showing this change, with
    #: the reason. Survival is then the expected result and is itself evidence.
    expected_to_survive: str = ""
    #: How many times the anchor is expected to appear. Every occurrence is
    #: replaced; a mismatch is an error rather than a partial mutation.
    occurrences: int = 1
    #: For the reducer suite: which `dim_reduction.py --mode` exercises this code.
    mode: str = "pca_umap"


MUTATIONS = (
    Mutation(
        name="pca_components",
        path="ProteinCartography/dim_reduction.py",
        old="n_components=30,\n            prep_step=True,",
        new="n_components=20,\n            prep_step=True,",
        occurrences=2,  # the pca_tsne and pca_umap branches of main()
        detects="a changed PCA dimensionality, which moves every coordinate",
        expected_to_survive=(
            "at N=11 both 30 and 20 clamp to min(matrix.shape)=11, so the two "
            "configurations are the same computation. Covered by the N=750 "
            "reducer suite instead."
        ),
    ),
    Mutation(
        name="umap_neighbors",
        path="ProteinCartography/dim_reduction.py",
        old="    n_neighbors=80,",
        new="    n_neighbors=40,",
        detects="a changed UMAP parameter, which changes the layout but not the matrix",
        expected_to_survive=(
            "at N=11 both 80 and 40 clamp to n-1=10. Covered by the N=750 " "reducer suite instead."
        ),
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
        expected_to_survive=(
            "the 11-protein fixture has all 121 pairs measured, so the fill "
            "token is never emitted and cannot differ. Only a fixture above the "
            "per-query cap can show this."
        ),
    ),
    Mutation(
        name="leiden_n_pcs",
        path="ProteinCartography/leiden_clustering.py",
        old="    n_pcs=30,",
        new="    n_pcs=10,",
        detects="a changed clustering parameter, on the branch that forks from the matrix",
        expected_to_survive=("at N=11 both 30 and 10 clamp to min(n-1, n_vars-1)=10."),
    ),
    Mutation(
        name="feature_join_semantics",
        path="ProteinCartography/aggregate_features.py",
        old='agg_df = agg_df.merge(df, on="protid", how="outer")',
        new='agg_df = agg_df.merge(df, on="protid", how="inner")',
        detects="a changed join, which silently drops proteins from the feature table",
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
    """Apply a mutation, guaranteeing restoration.

    The occurrence count is checked rather than assumed. Patching only the first
    of several matches is how a mutation ends up applied to a code path the run
    never executes, which then reports as "survived" and looks like a hole in
    the test rather than a mistake in the mutation.
    """
    target = repo / mutation.path
    original = target.read_text()
    found = original.count(mutation.old)
    if found == 0:
        raise MutationDidNotApply(
            f"mutation {mutation.name!r} does not apply: its anchor text is not in "
            f"{mutation.path}. The file has changed and the mutation needs updating."
        )
    if found != mutation.occurrences:
        raise MutationDidNotApply(
            f"mutation {mutation.name!r} expected its anchor {mutation.occurrences} "
            f"time(s) in {mutation.path} but found it {found} time(s). An ambiguous "
            "anchor silently mutates the wrong call site; make it specific or set "
            "`occurrences` to the real count."
        )
    try:
        target.write_text(original.replace(mutation.old, mutation.new))
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
        outcome, note, changed = "survived", "", []
        try:
            with _patched(repo, mutation):
                mutated = run_pipeline(repo, run_dir, conda_prefix=conda_prefix)
                report = compare_trees(reference, mutated, ignore=floor)
                outcome = "detected" if not report.ok else "survived"
                changed = report.differing[:6]
        except MutationDidNotApply as exc:
            # NOT a detection. The anchor text moved, so nothing was mutated and
            # the run was never going to differ. Counting this as a pass is how
            # a mutation suite quietly stops testing anything.
            outcome, note = "error", str(exc)
        except RuntimeError as exc:
            # A crash IS a detection: the run failing is a louder signal than a
            # diff, and a refactor that breaks the pipeline is caught either way.
            outcome, note = "detected", f"pipeline failed: {str(exc)[:160]}"
        results.append(
            {
                "name": mutation.name,
                "outcome": outcome,
                "detected": outcome == "detected",
                "detects": mutation.detects,
                "expected_to_survive": mutation.expected_to_survive,
                "changed_files": changed,
                "note": note,
            }
        )
        verdict = outcome.upper()
        print(f"  -> {verdict}  {note}", flush=True)
        for rel in changed:
            print(f"     ! {rel}", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        print(flush=True)
    return results


def format_report(results: list) -> str:
    lines = ["", "=" * 74, "MUTATION TESTING OF THE PARITY TEST", "=" * 74]
    detected = [r for r in results if r["outcome"] == "detected"]
    expected = [r for r in results if r["outcome"] == "survived" and r["expected_to_survive"]]
    holes = [r for r in results if r["outcome"] == "survived" and not r["expected_to_survive"]]
    errors = [r for r in results if r["outcome"] == "error"]

    lines.append(
        f"{len(detected)} detected, {len(expected)} survived as expected, "
        f"{len(holes)} UNEXPLAINED HOLES, {len(errors)} did not apply"
    )
    lines.append("")
    for r in results:
        mark = {"detected": "detected", "survived": "SURVIVED", "error": "DID NOT APPLY"}[
            r["outcome"]
        ]
        lines.append(f"  [{mark:13s}] {r['name']}")
        lines.append(f"                  {r['detects']}")
        if r["expected_to_survive"]:
            lines.append(f"                  expected to survive: {r['expected_to_survive']}")
        if r["note"]:
            lines.append(f"                  {r['note']}")
        for rel in r["changed_files"][:4]:
            lines.append(f"                  ! {rel}")

    if errors:
        lines += [
            "",
            "A mutation that DID NOT APPLY tested nothing -- its anchor text has",
            "moved. Update the anchor; do not read it as a pass.",
        ]
    if holes:
        lines += [
            "",
            "UNEXPLAINED HOLES are places the parity test cannot see a real change.",
            "Each needs either a fixture that can show it or a written reason why",
            "survival is acceptable. Never relax the test to make one go away.",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workdir", default="/tmp/pc-mutation")
    parser.add_argument("--conda-prefix", default=None)
    parser.add_argument("--analysis-python", default=None)
    parser.add_argument(
        "--reducer-only",
        action="store_true",
        help="skip the slow end-to-end suite and run only the N=750 reducer mutations",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    prefix = Path(args.conda_prefix) if args.conda_prefix else repo / ".snakemake" / "conda"

    results = []
    if not args.reducer_only:
        results += run_mutation_suite(repo, Path(args.workdir), prefix)

    python = args.analysis_python or _find_analysis_python(prefix)
    if python:
        results += run_reducer_mutation_suite(repo, Path(args.workdir) / "reducer", python)
    else:
        print(
            "skipping the N=750 reducer suite: no environment with scikit-learn "
            f"and umap-learn found under {prefix}",
            flush=True,
        )

    print(format_report(results))
    holes = [r for r in results if r["outcome"] == "survived" and not r["expected_to_survive"]]
    errors = [r for r in results if r["outcome"] == "error"]
    return 1 if (holes or errors) else 0


def _find_analysis_python(conda_prefix: Path):
    import subprocess

    for candidate in sorted(Path(conda_prefix).glob("*/bin/python")):
        probe = subprocess.run([str(candidate), "-c", "import sklearn, umap"], capture_output=True)
        if probe.returncode == 0:
            return str(candidate)
    return None


#: Mutations aimed at the reduction step, exercised on a 750-protein matrix.
#:
#: These exist because the 11-protein end-to-end fixture cannot see them: every
#: parameter they change is clamped to the same value at that size. Running the
#: reducer alone on a generated matrix reaches N>500 without needing 750 PDB
#: files and a Foldseek run.
REDUCER_MUTATIONS = (
    Mutation(
        name="reducer_pca_components",
        path="ProteinCartography/dim_reduction.py",
        old="n_components=30,\n            prep_step=True,",
        new="n_components=20,\n            prep_step=True,",
        occurrences=2,  # both branches of main(); patching one would miss the mode under test
        detects="a changed PCA dimensionality, at a size where it is not clamped away",
    ),
    Mutation(
        name="reducer_umap_neighbors",
        # The caller passes this explicitly, so mutating the reducer's own
        # default would be invisible -- correctly, since nothing reads it.
        path="ProteinCartography/dim_reduction.py",
        old="    n_neighbors=80,",
        new="    n_neighbors=40,",
        detects="a changed UMAP neighbour count, at a size where it is not clamped away",
    ),
    Mutation(
        name="reducer_pca_solver",
        path="ProteinCartography/spaces/reducers/core.py",
        old='n_components=n_components, svd_solver="full", random_state=random_state',
        new='n_components=n_components, svd_solver="auto", random_state=random_state',
        detects=(
            "undoing PR #106's determinism fix. Above 500 rows 'auto' selects the "
            "randomized solver, which is the whole reason that PR exists"
        ),
    ),
    Mutation(
        name="reducer_pca_column_naming",
        path="ProteinCartography/spaces/reducers/core.py",
        old='column_names=[f"PC{i}" for i in range(coordinates.shape[1])],',
        new='column_names=[f"PC{i + 1}" for i in range(coordinates.shape[1])],',
        detects="renaming a column that ships in aggregated_features.tsv",
    ),
    Mutation(
        name="reducer_tsne_perplexity",
        path="ProteinCartography/dim_reduction.py",
        old="    perplexity=50,",
        new="    perplexity=30,",
        mode="pca_tsne",  # pca_umap never reaches t-SNE at all
        detects="a changed t-SNE perplexity, at a size where it is not clamped away",
    ),
)


def run_reducer_mutation_suite(
    repo: Path, workdir: Path, python: str, *, n: int = 750, mode: str = "pca_umap"
) -> list:
    """Mutate the reduction step and check the N>500 comparison notices.

    `python` must be an interpreter with scikit-learn and umap-learn -- the
    reduction step runs in `envs/analysis.yml`, not in the test environment.
    """
    repo, workdir = Path(repo), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_matrix(workdir / "fixture" / "all_by_all_tmscore_pivoted.tsv", n=n)

    # One reference per mode, since a mutation is only visible in the mode that
    # executes its code path.
    references, floors = {}, {}
    for needed in sorted({m.mode for m in REDUCER_MUTATIONS} | {mode}):
        print(f"=== reference reduction at N={n}, mode={needed} ===", flush=True)
        a = run_reducer(repo, matrix, workdir / f"reference_{needed}", mode=needed, python=python)
        b = run_reducer(repo, matrix, workdir / f"reference_{needed}_b", mode=needed, python=python)
        references[needed] = a
        floors[needed] = set(compare_trees(a, b).differing)
        print(
            f"floor({needed}): {sorted(floors[needed]) or 'nothing -- fully deterministic'}\n",
            flush=True,
        )

    results = []
    for mutation in REDUCER_MUTATIONS:
        print(f"=== reducer mutation: {mutation.name} (mode={mutation.mode}) ===", flush=True)
        reference = references[mutation.mode]
        floor = floors[mutation.mode]
        outcome, note, changed = "survived", "", []
        try:
            with _patched(repo, mutation):
                mutated = run_reducer(
                    repo,
                    matrix,
                    workdir / f"mut_{mutation.name}",
                    mode=mutation.mode,
                    python=python,
                )
                report = compare_trees(reference, mutated, ignore=floor)
                outcome = "detected" if not report.ok else "survived"
                changed = report.differing[:6]
        except MutationDidNotApply as exc:
            outcome, note = "error", str(exc)
        except RuntimeError as exc:
            outcome, note = "detected", f"reduction failed: {str(exc)[:160]}"
        results.append(
            {
                "name": mutation.name,
                "outcome": outcome,
                "detected": outcome == "detected",
                "detects": mutation.detects,
                "expected_to_survive": mutation.expected_to_survive,
                "changed_files": changed,
                "note": note,
            }
        )
        print(f"  -> {outcome.upper()}  {note}", flush=True)
        for rel in changed:
            print(f"     ! {rel}", flush=True)
        shutil.rmtree(workdir / f"mut_{mutation.name}", ignore_errors=True)
        print(flush=True)
    return results


if __name__ == "__main__":
    raise SystemExit(main())
