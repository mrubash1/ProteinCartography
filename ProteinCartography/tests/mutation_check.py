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

from parity import (
    FUSED_SPACES,
    SPACE_OUTPUT_PREFIXES,
    compare_trees,
    run_leiden,
    run_pipeline,
    run_pivot,
    run_reducer,
    synthetic_matrix,
    synthetic_pair_list,
)

__all__ = [
    "MUTATIONS",
    "COMPONENT_MUTATIONS",
    "REDUCER_MUTATIONS",
    "SPACE_MUTATIONS",
    "Mutation",
    "run_mutation_suite",
    "run_component_mutation_suite",
    "run_reducer_mutation_suite",
    "run_space_mutation_suite",
]


class MutationDidNotApply(RuntimeError):
    """The anchor text was not found, so nothing was mutated.

    Distinguished from every other failure because it is the one that must never
    be scored as a pass: no mutation was applied, so the run could not have
    differed, and calling that "detected" turns the suite into a no-op.
    """


@dataclass(frozen=True)
class Mutation:
    """A deliberate defect, and what it is meant to prove the test can see.

    One trap is worth stating, because it caught me three times: **do not
    anchor on a default that the caller overrides.** Mutating
    ``def f(..., n_pcs=30)`` does nothing when ``main()`` passes ``n_pcs`` from
    an argparse default, and the run then reports "survived" -- indistinguishable
    from a genuine hole in the test. Anchor on the value the executed path
    actually reads.
    """

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
            "configurations are the same computation. Covered by "
            "reducer_pca_components at N=750."
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
            "token is never emitted and cannot differ. Covered by "
            "component_censoring_fill, which drives the pivot step from a "
            "capped pair list."
        ),
    ),
    Mutation(
        name="leiden_n_pcs",
        path="ProteinCartography/leiden_clustering.py",
        old="    n_pcs=30,",
        new="    n_pcs=10,",
        detects="a changed clustering parameter, on the branch that forks from the matrix",
        expected_to_survive=(
            "at N=11 both 30 and 10 clamp to min(n-1, n_vars-1)=10. Covered by "
            "component_leiden_n_pcs at N=750."
        ),
    ),
    Mutation(
        name="feature_join_semantics",
        path="ProteinCartography/aggregate_features.py",
        old='agg_df = agg_df.merge(df, on="protid", how="outer")',
        new='agg_df = agg_df.merge(df, on="protid", how="inner")',
        detects="a changed join, which silently drops proteins from the feature table",
    ),
    Mutation(
        name="cohort_selection_order",
        path="ProteinCartography/cohort.py",
        old="        ordered = list(candidates)",
        new="        ordered = sorted(candidates)",
        detects=(
            "the default cohort rule quietly sorting. This is the exact change "
            "ADR 0008 rejected: it looks like tidying up and it selects a "
            "different set of proteins, so the map is different and nothing "
            "downstream would say so"
        ),
    ),
    Mutation(
        name="cohort_truncation_boundary",
        path="ProteinCartography/cohort.py",
        old="        retained, discarded = ordered[:max_structures], ordered[max_structures:]",
        new=(
            "        retained, discarded = ordered[: max_structures - 1], "
            "ordered[max_structures - 1 :]"
        ),
        detects="an off-by-one at the truncation point, which drops a protein from the map",
    ),
    Mutation(
        name="cohort_significance_polarity",
        path="ProteinCartography/cohort.py",
        old='    sign = 1.0 if better == "lower" else -1.0',
        new='    sign = -1.0 if better == "lower" else 1.0',
        detects=(
            "an inverted significance ranking, which would select the *worst* "
            "hits while looking entirely normal"
        ),
        expected_to_survive=(
            "the default config uses selection: as_filtered, so this code path "
            "never runs in the parity fixture. It is a real hole in the *parity* "
            "test and a deliberate one: making the default exercise significance "
            "ranking would mean changing the default cohort, which is the one "
            "thing this work promises not to do. Covered instead by the unit "
            "tests test_evalue_ranks_lowest_first and test_tmscore_ranks_highest_first, "
            "which assert the two directions against each other."
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


#: Mutants in the four modules THIS BRANCH ADDED, which `MUTATIONS` above
#: cannot reach.
#:
#: FOLLOWUPS #44 is the reason this exists and its own text is why it took so
#: long. All ten of `MUTATIONS` name six files -- `dim_reduction.py`,
#: `foldseek_clustering.py`, `leiden_clustering.py`, `aggregate_features.py`,
#: `cohort.py`, `spaces/reducers/core.py` -- and none of them is `fusion.py`,
#: `diagnose_space.py`, `enrich_clusters.py` or anything under `diagnostics/`.
#: So "12 detected / 5 survived as expected / 0 unexplained holes" was, for
#: those four modules, guaranteed by construction rather than measured.
#:
#: #44 then said the fix "needs a different harness, comparing HEAD against HEAD
#: with the mutation applied, not HEAD against the baseline". THAT WAS WRONG
#: ABOUT THE HARNESS IT WAS DESCRIBING: `run_mutation_suite` above already runs
#: an unmutated reference and a mutated run FROM THE SAME CHECKOUT and diffs
#: them. Nothing about the baseline enters it. What was actually missing was a
#: CONFIG -- the default one puts no space in the DAG, so the four modules never
#: execute. `parity.FUSED_SPACES` is that config, and this suite is the rest.
#:
#: Anchoring follows `Mutation`'s own warning: every `old` below is a value the
#: executed path reads, not an overridable default.
SPACE_MUTATIONS = (
    Mutation(
        name="fused_distance_not_squared",
        path="ProteinCartography/fusion.py",
        old="    fused = np.sqrt(squared)",
        new="    fused = squared",
        detects=(
            "late fusion returning SQUARED distances as distances. The classic "
            "form of this refactor error, and it is not a rescaling: squaring "
            "changes the ratio of every pair of distances, so the map moves."
        ),
    ),
    Mutation(
        name="distorted_threshold",
        path="ProteinCartography/diagnostics/embedding.py",
        old="DISTORTED_THRESHOLD = 0.70",
        new="DISTORTED_THRESHOLD = 0.20",
        detects=(
            "the faithfulness band moving. Every space's `diagnostics.json` "
            "records which proteins are unreliable at this cutoff, and the "
            "explorer draws them differently, so a moved threshold is a moved "
            "refusal rather than a moved number."
        ),
        expected_to_survive=(
            "MEASURED, NOT ASSUMED, AND IT IS NOT COVERAGE. On this fixture every "
            "protein scores trustworthiness 1.0 and continuity 1.0 in both spaces "
            "-- 10 and 11 rows of `1.0\t1.0` in `faithfulness_pca.tsv` -- because "
            "a cohort this small clamps k far enough that the PCA layout is "
            "perfectly faithful. NO value of this threshold between 0 and 1 "
            "changes a byte, so the mutant cannot be detected here and its "
            "survival says something about the fixture rather than about the "
            "test. The faithfulness BAND therefore has no end-to-end coverage "
            "anywhere: the N=750 reducer suite is the only large fixture and it "
            "does not run `diagnose_space`. Stated rather than counted."
        ),
    ),
    Mutation(
        name="pearson_denominator",
        path="ProteinCartography/diagnostics/redundancy.py",
        old="    denominator = np.sqrt(float(a @ a) * float(b @ b))",
        new="    denominator = float(a @ a) * float(b @ b)",
        detects=(
            "the block-agreement correlation losing its normalisation. Chosen "
            "after measuring what this fixture can actually show: stability is "
            "1.0 by construction at n=10, the partition is one cluster, the "
            "sweep's ARI is 1.0 and the silhouette is 0.0 -- `redundancy` "
            "carries the only live numbers in the whole diagnostics report "
            "(pearson 0.6907, spearman 0.8781 over 45 pairs), so it is the only "
            "place a `diagnostics/` mutant can be falsified end to end here."
        ),
    ),
    Mutation(
        name="own_partition_ignored",
        path="ProteinCartography/diagnose_space.py",
        old="    clusters = own.as_mapping() if own is not None else legacy_clusters",
        new="    clusters = legacy_clusters",
        detects=(
            "every space being diagnosed against the STRUCTURE space's Leiden "
            "partition instead of its own -- which is precisely the defect "
            "FOLLOWUPS #41 described, and which the code no longer has. A "
            "mutant that reintroduces a fixed historical bug is the strongest "
            "kind there is: it proves the fix is load-bearing rather than "
            "merely present."
        ),
    ),
    Mutation(
        name="enrichment_rank_base",
        path="ProteinCartography/enrichment.py",
        old="            ranks[order[start:stop]] = 0.5 * (start + stop + 1)",
        new="            ranks[order[start:stop]] = 0.5 * (start + stop)",
        detects=(
            "the Mann-Whitney ranks going 0-based. Not invented: "
            "`coregistration.average_ranks` computes the same statistic 0-based "
            "for a different purpose, so this is the exact edit a consolidation "
            "of the two would make, and it shifts every U and every p-value in "
            "`cluster_enrichment.tsv`."
        ),
    ),
)


def run_space_mutation_suite(repo: Path, workdir: Path, conda_prefix: Path) -> list:
    """`SPACE_MUTATIONS`, end to end, under a config that reaches the four modules.

    The same shape as `run_mutation_suite` with two differences, and both are
    guards rather than conveniences.

    **The floor is checked, not just printed.** If any output under
    `spaces/`, `coregistration/` or `enrichment/` turns up in the self-diff
    floor, every mutant below becomes unfalsifiable in exactly the way that
    reports as "survived", so this RAISES instead of running them. A suite whose
    negative result cannot be told from a broken one is worse than no suite.

    **Each result records whether it was detected on the NEW path.** A mutant
    can change a legacy output too -- `enrich_clusters` reads
    `leiden_features.tsv` -- and being detected there says nothing about the
    module it mutated. `on_space_path` is False when none of the differing files
    is under one of those prefixes, and the report prints it.
    """
    repo, workdir = Path(repo), Path(workdir)

    print("=== spaces reference run (unmutated) ===", flush=True)
    reference = run_pipeline(
        repo, workdir / "reference", conda_prefix=conda_prefix, extra_config=FUSED_SPACES
    )
    print("=== spaces self-diff, to establish the nondeterminism floor ===", flush=True)
    reference_b = run_pipeline(
        repo, workdir / "reference_b", conda_prefix=conda_prefix, extra_config=FUSED_SPACES
    )
    floor = set(compare_trees(reference, reference_b).differing)
    print(f"floor: {sorted(floor) or 'nothing -- fully deterministic'}\n", flush=True)

    swallowed = sorted(f for f in floor if f.startswith(SPACE_OUTPUT_PREFIXES))
    if swallowed:
        raise RuntimeError(
            "the nondeterminism floor contains output this suite mutates: "
            f"{swallowed}. Every space mutant would be ignorable on those files "
            "and would report 'survived' for a reason that is not a hole in the "
            "test. Fix the nondeterminism before trusting any result below."
        )

    results = []
    for mutation in SPACE_MUTATIONS:
        print(f"=== space mutation: {mutation.name} ===", flush=True)
        run_dir = workdir / f"mut_{mutation.name}"
        outcome, note, changed, on_path = "survived", "", [], False
        try:
            with _patched(repo, mutation):
                mutated = run_pipeline(
                    repo, run_dir, conda_prefix=conda_prefix, extra_config=FUSED_SPACES
                )
                report = compare_trees(reference, mutated, ignore=floor)
                outcome = "detected" if not report.ok else "survived"
                changed = report.differing[:6]
                on_path = any(f.startswith(SPACE_OUTPUT_PREFIXES) for f in report.differing)
        except MutationDidNotApply as exc:
            outcome, note = "error", str(exc)
        except RuntimeError as exc:
            outcome, note = "detected", f"pipeline failed: {str(exc)[:160]}"
            on_path = True
        results.append(
            {
                "name": f"space:{mutation.name}",
                "outcome": outcome,
                "detected": outcome == "detected",
                "detects": mutation.detects,
                "expected_to_survive": mutation.expected_to_survive,
                "changed_files": changed,
                "on_space_path": on_path,
                "note": note,
            }
        )
        where = "" if on_path else "  (NOT on the spaces path)"
        print(f"  -> {outcome.upper()}{where}  {note}", flush=True)
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
        # A space mutant detected only through a legacy output says nothing
        # about the module it mutated, and the count alone cannot show that.
        if r["outcome"] == "detected" and r.get("on_space_path") is False:
            lines.append(
                "                  DETECTED, BUT NOT ON THE SPACES PATH -- no file "
                "under spaces/, coregistration/ or enrichment/ differed"
            )
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
    parser.add_argument(
        "--spaces-only",
        action="store_true",
        help="run only SPACE_MUTATIONS, which is the affordable half to iterate on",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    prefix = Path(args.conda_prefix) if args.conda_prefix else repo / ".snakemake" / "conda"

    results = []
    if not args.reducer_only and not args.spaces_only:
        results += run_mutation_suite(repo, Path(args.workdir), prefix)

    if not args.reducer_only:
        # Its own workdir: this suite runs a different CONFIG through the same
        # pipeline, and sharing a directory would have one reference run
        # standing in for the other.
        results += run_space_mutation_suite(repo, Path(args.workdir) / "spaces", prefix)

    if args.spaces_only:
        print(format_report(results))
        holes = [r for r in results if r["outcome"] == "survived" and not r["expected_to_survive"]]
        return 1 if (holes or [r for r in results if r["outcome"] == "error"]) else 0

    python = args.analysis_python or _find_analysis_python(prefix)
    if python:
        results += run_reducer_mutation_suite(repo, Path(args.workdir) / "reducer", python)
        results += run_component_mutation_suite(repo, Path(args.workdir) / "component", python)
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


#: Mutations aimed at the two steps neither other suite reaches.
#:
#: The reducer suite starts from a matrix, so it never exercises the pivot that
#: *creates* the censoring fill; and it never touches Leiden, which forks from
#: the matrix independently. Both were unreachable by any test until these
#: existed -- the earlier claim that every N=11 survivor had an N=750
#: counterpart was wrong for exactly these two.
COMPONENT_MUTATIONS = (
    Mutation(
        name="component_censoring_fill",
        path="ProteinCartography/foldseek_clustering.py",
        old='scores.append(targets_to_scores.get(target, "0.0"))',
        new='scores.append(targets_to_scores.get(target, "0.00"))',
        detects=(
            "a changed censoring fill token. Every downstream mask keys on the "
            "exact string, so this would silently disable censoring detection"
        ),
    ),
    Mutation(
        name="component_leiden_n_pcs",
        # The CLI default, not the function default. `main()` always passes
        # `n_pcs` explicitly from `args.n_pcs`, so mutating the function
        # signature changes nothing -- a mistake made three times while building
        # this suite, each time looking exactly like a hole in the test.
        path="ProteinCartography/leiden_clustering.py",
        old='        default="30",',
        new='        default="10",',
        occurrences=1,
        detects="a changed Leiden parameter, at a size where it is not clamped away",
    ),
)


def run_component_mutation_suite(repo: Path, workdir: Path, python: str) -> list:
    """Mutate the pivot and Leiden steps and check the comparison notices.

    Two runners rather than one, because the two steps take different inputs:
    the pivot consumes a raw pair list, Leiden consumes a matrix.
    """
    repo, workdir = Path(repo), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    pair_list = synthetic_pair_list(workdir / "fixture" / "pairs.tsv", n=200, cap=80)
    matrix = synthetic_matrix(workdir / "fixture" / "all_by_all_tmscore_pivoted.tsv", n=750)

    runners = {
        "component_censoring_fill": lambda out: run_pivot(repo, pair_list, out, python=python),
        "component_leiden_n_pcs": lambda out: run_leiden(repo, matrix, out, python=python),
    }

    references, floors = {}, {}
    for name, run in runners.items():
        print(f"=== reference for {name} ===", flush=True)
        a = run(workdir / ("reference_" + name))
        b = run(workdir / ("reference_" + name + "_b"))
        references[name] = a
        floors[name] = set(compare_trees(a, b).differing)
        print(f"floor: {sorted(floors[name]) or 'nothing -- fully deterministic'}\n", flush=True)

    results = []
    for mutation in COMPONENT_MUTATIONS:
        print(f"=== component mutation: {mutation.name} ===", flush=True)
        outcome, note, changed = "survived", "", []
        try:
            with _patched(repo, mutation):
                mutated = runners[mutation.name](workdir / ("mut_" + mutation.name))
                report = compare_trees(
                    references[mutation.name], mutated, ignore=floors[mutation.name]
                )
                outcome = "detected" if not report.ok else "survived"
                changed = report.differing[:6]
        except MutationDidNotApply as exc:
            outcome, note = "error", str(exc)
        except RuntimeError as exc:
            outcome, note = "detected", f"step failed: {str(exc)[:160]}"
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
        shutil.rmtree(workdir / ("mut_" + mutation.name), ignore_errors=True)
        print(flush=True)
    return results


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
