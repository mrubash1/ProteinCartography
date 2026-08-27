"""Wirings the Snakefiles depend on that no other test observes.

PC-037's audit read every hunk of the +778/-8 Snakefile diff and asked, for each
line that modifies something upstream already had, which existing test goes red
if it is wrong. Seven had no answer. This file is that answer.

These are TEXT invariants for six of the seven, and the sixth is behavioural.
A text invariant is a weak test in general and the right one here, because the
risk these lines carry is not that they compute something wrong -- it is that
they are DELETED as dead and nothing notices. The audit measured exactly that:
removing any of U1-U5 changes no output byte under any config in this
repository, so every test in the suite stays green.

WHY THEY ARE NOT DEAD, WHICH IS THE PART THAT MATTERS:

`--mode {FOLDSEEK_MODE}` is DEFAULT-VALUED, not dead. `config.yml` sets
`foldseek_mode: "3diaa"`, `foldseek_apiquery.py` defaults to the same, and
`extract_foldseek_hits.py` defaults to None whose only use is a `!= "3diaa"`
guard -- so today every pass is a no-op. Under a different `foldseek_mode` they
are live, and the four are TWO DIFFERENT FLAGS: `Snakefile:403` and
`Snakefile_domain:155` set the API search mode on `foldseek_apiquery.py`, while
`Snakefile:421` and `Snakefile_domain:174` arm the REFUSAL GUARD in
`extract_foldseek_hits.py:86`, which raises `TmalignOutputError` rather than
rank a tmalign file backwards. Deleting the second pair removes a check.

Since commit 233 `foldseek_mode` has exactly one runnable value, which makes the
passes look even more deletable and makes this file more necessary rather than
less: the day a second mode becomes runnable, these are the lines that carry it.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _why_no_snakemake():
    """``None`` when snakemake actually RUNS, else the reason it does not.

    The same probe as `test_snakefile_parses_without_optional_dependencies.py`,
    and for the reason recorded there: `shutil.which` finds a pyenv shim that
    then exits non-zero, so a `which`-based guard runs the test and fails it for
    a reason it does not exist to catch.
    """
    exe = shutil.which("snakemake")
    if exe is None:
        return "needs snakemake"
    try:
        probe = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    except OSError as exc:  # pragma: no cover - depends on the machine
        return f"snakemake at {exe} is not executable: {exc}"
    if probe.returncode != 0:
        return f"snakemake at {exe} does not run: {(probe.stderr or probe.stdout).strip()[:120]}"
    return None


_WHY_NO_SNAKEMAKE = _why_no_snakemake()

#: ``(row, file, needle, why it must stay)``. The rows are PC-037-AUDIT's.
WIRINGS = (
    (
        "U1",
        "Snakefile",
        "--mode {FOLDSEEK_MODE} \\",
        "run_foldseek: the API search mode. Without it foldseek_apiquery falls back to its own "
        "default and the configured mode is silently ignored.",
    ),
    (
        # U1 and U2 share a needle by construction: `Snakefile` passes the flag
        # twice and the two passes go to different scripts. The count check
        # below is what tells them apart; these two rows carry the two reasons.
        "U2",
        "Snakefile",
        "--mode {FOLDSEEK_MODE} \\",
        "extract_foldseek_hits: arms the refusal guard at extract_foldseek_hits.py:86. Without "
        "it the mode arrives as None and a tmalign-shaped file is ranked instead of refused.",
    ),
    (
        "U3",
        "Snakefile_domain",
        "--mode {FOLDSEEK_MODE} \\",
        "domain_run_foldseek: the same API mode on the domain path.",
    ),
    (
        "U4",
        "Snakefile_domain",
        "--mode {FOLDSEEK_MODE} \\",
        "domain_extract_foldseek_hits: the same refusal guard on the domain path.",
    ),
    (
        "U5",
        "Snakefile",
        '" --foldseek-mode " + FOLDSEEK_MODE',
        "download_pdbs records the search mode in the cohort report, which is the only "
        "provenance of what search produced a cohort.",
    ),
    (
        "U6",
        "Snakefile",
        'DOMAIN_COHORT_SELECTION = "accession"',
        "the domain path's stated substitute for `significance`, which it cannot honor. "
        "Deleting it makes the domain map fall through to the script's own default while the "
        "protein map is cut by significance, with nothing to say the two used different rules.",
    ),
    (
        "U7",
        "Snakefile",
        'block_id="|".join(MULTISPACE_CONFIG.block_ids()) if MULTISPACE_ENABLED else "^$"',
        'the "^$" fallback: with multispace off there are no block ids, and an empty '
        "alternation would match anything rather than nothing.",
    ),
    (
        "U7b",
        "Snakefile",
        'space_id="|".join(MULTISPACE_CONFIG.space_ids()) if MULTISPACE_ENABLED else "^$"',
        "the same fallback for space_id.",
    ),
)


def _text(name):
    return (REPO_ROOT / name).read_text()


@pytest.mark.parametrize("row,name,needle,why", WIRINGS, ids=[w[0] for w in WIRINGS])
def test_the_wiring_is_still_there(row, name, needle, why):
    assert needle in _text(name), f"{row}: {name} no longer contains {needle!r}. {why}"


def test_the_two_mode_flags_are_both_present_and_are_not_one_flag():
    """U1-U4 are two flags, not four copies of one.

    A reader deleting "the duplicated `--mode`" would take the refusal guard
    with the search mode. Each Snakefile must pass it exactly twice.
    """
    for name in ("Snakefile", "Snakefile_domain"):
        assert _text(name).count("--mode {FOLDSEEK_MODE}") == 2, (
            f"{name} should pass --mode twice -- once to foldseek_apiquery (the search mode) "
            "and once to extract_foldseek_hits (the refusal guard)"
        )


@pytest.mark.parametrize("row,name,needle,why", WIRINGS, ids=[w[0] for w in WIRINGS])
def test_the_guard_would_catch_a_removed_wiring(row, name, needle, why):
    """The planted defect, against the real file rather than a fixture.

    Without this, every assertion above could be checking a string that is
    trivially present -- and the audit's whole finding is that these lines look
    deletable, so a guard that cannot see a deletion guards nothing.
    """
    original = _text(name)
    stripped = original.replace(needle, "")
    assert (
        stripped != original
    ), f"{row}: the anchor did not match the real file, so the check above is vacuous"
    assert (
        needle not in stripped
    ), f"{row}: removing the wiring left it present, so the check above cannot see a deletion"


@pytest.mark.skipif(_WHY_NO_SNAKEMAKE is not None, reason=_WHY_NO_SNAKEMAKE or "")
def test_a_significance_config_gives_the_domain_map_a_stated_rule(tmp_path):
    """U6, and this one is behavioural rather than textual.

    `DOMAIN_COHORT_SELECTION` substitutes `accession` for `significance` on the
    domain path, because ranking by significance needs a per-accession table the
    domain side does not build. The substitution EXECUTES on any significance
    config with the domain map on, and until now nothing observed it -- so the
    branch could have silently dropped the substitution, or the message, and
    every test would have stayed green.

    Reached with a merged JSON configfile because `cohort.selection` is nested
    and `--config key=value` cannot address it. JSON rather than YAML because
    no environment here has PyYAML and scikit-learn together, and JSON is what
    the Snakefile is handed anyway.

    WHAT THIS CANNOT SEE, stated rather than implied. The domain side's
    `--selection accession` never appears in a dry run on a fresh tree: the
    domain rules sit behind the `domain_query_gate` checkpoint, and `-n`
    resolves the DAG only as far as the first unresolved checkpoint -- the same
    limit `test_review_regressions._checkpoint_orphans` documents. So this
    asserts the two halves that ARE observable, the announcement and the
    protein path keeping its own rule, and the row `U6` in `WIRINGS` above
    covers deletion of the substitution itself. The first version of this test
    asserted only the message, and passed against a tree with the substitution
    removed, because the announcement sits inside the same `if` and still
    printed.
    """
    extra = tmp_path / "significance.json"
    extra.write_text(json.dumps({"cohort": {"selection": "significance"}, "domain_map": "auto"}))
    result = subprocess.run(
        [
            "snakemake",
            "--configfile",
            "demo/search-mode/config_actin_small.yml",
            str(extra),
            "-n",
            "--quiet",
            "--rerun-incomplete",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    combined = result.stdout + result.stderr
    assert "[cohort] selection 'significance' cannot be applied to the domain map" in combined, (
        "the domain path no longer states that it substituted a different cohort rule; "
        "a config that truncates the protein map by significance and the domain map by "
        "something else must say so"
    )
    assert "'accession'" in combined, "the message must name the rule it substituted"

    # The substitution must not leak to the PROTEIN path, which keeps the rule
    # the config asked for. This is the half a dry run can actually observe.
    shell = subprocess.run(
        [
            "snakemake",
            "--configfile",
            "demo/search-mode/config_actin_small.yml",
            str(extra),
            "-n",
            "-p",
            "--rerun-incomplete",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert (
        "--selection significance" in shell.stdout + shell.stderr
    ), "the protein path must still be cut by the rule the config named"
