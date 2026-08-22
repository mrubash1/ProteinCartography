"""Tests for `extract_foldseek_hits`, which had none.

This module decides WHICH hits reach the map: it concatenates one frame per
Foldseek database, drops non-AlphaFold targets, filters on e-value, and then
truncates. The truncation is the part worth pinning, and it is pinned here as a
DEFECT rather than as behaviour worth keeping -- see the docstring on
`test_truncation_keeps_the_first_n_in_database_order`.

`pd.read_csv` is real here rather than mocked. The module's whole job is what
pandas does to a `.m8`, and a mocked reader would test the mock.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import constants  # noqa: E402
from extract_foldseek_hits import extract_foldseekhits  # noqa: E402


def m8_row(accession: str, evalue: float, bits: int = 500) -> str:
    """One `.m8` line, with everything but the fields under test left as filler."""
    row = ["-"] * len(constants.FOLDSEEK_COLUMN_NAMES)
    index = {name: i for i, name in enumerate(constants.FOLDSEEK_COLUMN_NAMES)}
    row[index["query"]] = "job.pdb"
    row[index["target"]] = f"AF-{accession}-F1-model_v4 Some protein"
    row[index["evalue"]] = str(evalue)
    row[index["bits"]] = str(bits)
    return "\t".join(row)


def write_m8(path, rows) -> str:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return str(path)


def hits_in(path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the empty-file guard (FOLLOWUPS #24)
# ---------------------------------------------------------------------------


def test_a_zero_byte_m8_is_skipped_and_produces_an_empty_output(tmp_path, capsys):
    """The guard's own case, now that it runs before the read rather than after.

    Inert under the pinned pandas 2.0.1 either way -- reading an empty file with
    `names=` supplied returns a 0-row frame instead of raising -- so this test
    passes before and after the move. It is here to say what the behaviour IS,
    because #24 asserted `EmptyDataError` and that is not what happens.
    """
    empty = write_m8(tmp_path / "alis_afdb50.m8", [])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([empty], str(out))
    assert hits_in(out) == []
    assert "No matching foldseek hits found" in capsys.readouterr().out


def test_an_empty_file_beside_a_real_one_does_not_lose_the_real_one(tmp_path):
    """The case the guard exists for: skipping one input must not skip the rest,
    including when the empty one comes FIRST and the accumulator is still
    unset."""
    empty = write_m8(tmp_path / "alis_afdb50.m8", [])
    real = write_m8(tmp_path / "alis_afdb-swissprot.m8", [m8_row("P00001", 1e-40)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([empty, real], str(out))
    assert hits_in(out) == ["P00001"]


def test_a_file_whose_rows_are_all_filtered_out_yields_an_empty_output(tmp_path, capsys):
    """Distinct from the empty-file case: the file has rows, and none survive."""
    weak = write_m8(tmp_path / "alis_afdb50.m8", [m8_row("P00001", 10.0)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([weak], str(out), evalue=1e-3)
    assert hits_in(out) == []
    assert "No matching foldseek hits found" in capsys.readouterr().out


def test_a_non_alphafold_target_is_dropped(tmp_path):
    """Only AlphaFold models are downloadable, so only they are candidates."""
    row = m8_row("P00001", 1e-40).replace("AF-P00001-F1-model_v4", "1abc_A")
    path = write_m8(tmp_path / "alis_afdb50.m8", [row, m8_row("P00002", 1e-40)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00002"]


def test_a_repeated_accession_is_written_once(tmp_path):
    """`unique()` across the concatenated frames, so one protein is one hit."""
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 1e-40), m8_row("P00001", 1e-90), m8_row("P00002", 1e-40)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00001", "P00002"]


# ---------------------------------------------------------------------------
# what truncation does today (FOLLOWUPS #23)
# ---------------------------------------------------------------------------


def test_truncation_keeps_the_first_n_in_database_order(tmp_path):
    """THIS PINS A DEFECT. It is not a description of behaviour worth keeping.

    `max_foldseek_hits` keeps the first N accessions in the order the frames
    were concatenated, which is the order the Snakefile expands
    `db=FOLDSEEK_DATABASES` -- so it is the order of `foldseek_databases` in
    `config.yml`, today `afdb50`, `afdb-swissprot`, `afdb-proteome`. It is NOT
    the N most significant hits, and the e-values needed to rank them are in the
    frame the whole time.

    The fixture makes the two rules disagree on purpose: the weakest hit is
    first in database order and the strongest is last, so a run that kept the
    top 2 by e-value would return `{P00003, P00004}` and this returns
    `{P00001, P00002}`.

    **PC-014 phase 4 is what changes this**, and it cannot land under the
    parity requirement without a decision, because it changes which proteins
    reach the map. Until then the rule is at least written down and a change to
    it cannot be silent. A test that blesses a defect without saying so is worse
    than no test.
    """
    first = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 1e-5), m8_row("P00002", 1e-6)],
    )
    second = write_m8(
        tmp_path / "alis_afdb-swissprot.m8",
        [m8_row("P00003", 1e-90), m8_row("P00004", 1e-80)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([first, second], str(out), max_num_hits=2)

    assert hits_in(out) == ["P00001", "P00002"], "database order, not significance"
    # State the counterfactual, so the defect is legible from the test alone.
    by_significance = ["P00003", "P00004"]
    assert hits_in(out) != by_significance


def test_truncation_is_not_applied_when_no_maximum_is_given(tmp_path):
    """The other half: the default keeps everything that survived the filters."""
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row(f"P{i:05d}", 1e-40) for i in range(5)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert len(hits_in(out)) == 5


# ---------------------------------------------------------------------------
# the tmalign refusal (PC-014 phase 2)
#
# `foldseek_apiquery.py --mode tmalign` returns the SAME 21 columns in the same
# positions with different meanings: `evalue` holds a TM-score and `bits` holds
# roughly that times 100. Nothing renames, so nothing errors -- the filter's
# polarity simply inverts, and `evalue < 0.01` keeps the LEAST similar
# structures. `hit_significance` has refused this since it was written; this
# module, which is the one that actually decides the cohort, did not.
# ---------------------------------------------------------------------------


def tmalign_row(accession: str, tm_score: float) -> str:
    """A row as the server returns it in tmalign mode: TM-score in the `evalue`
    column, roughly TM-score x 100 in `bits`."""
    return m8_row(accession, tm_score, bits=int(round(tm_score * 100)))


def test_a_tmalign_shaped_file_is_refused_rather_than_filtered(tmp_path):
    """The whole point: filtering this file would keep the worst hits.

    Shaped on the live tmalign query recorded in `hit_significance`'s own
    docstring -- e-values spanning 0.402 to 0.9999 with the best hit at the TOP
    of the range, which is the opposite of an e-value.
    """
    from hit_significance import TmalignOutputError

    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [tmalign_row("P00001", 0.9999), tmalign_row("P00002", 0.402)],
    )
    with pytest.raises(TmalignOutputError, match="tmalign"):
        extract_foldseekhits([path], str(tmp_path / "hits.txt"))


def test_the_refusal_names_the_file_and_the_range_it_saw(tmp_path):
    """A refusal a reader cannot check is a refusal they will override."""
    from hit_significance import TmalignOutputError

    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [tmalign_row("P00001", 0.9999), tmalign_row("P00002", 0.402)],
    )
    with pytest.raises(TmalignOutputError) as caught:
        extract_foldseekhits([path], str(tmp_path / "hits.txt"))
    message = str(caught.value)
    assert "alis_afdb50.m8" in message
    assert "0.402" in message and "0.9999" in message
    # It must say why refusing beats guessing, and what the pipeline does run.
    assert "the mode is not recorded in the output" in message
    assert "3diaa" in message


def test_one_tmalign_file_among_real_ones_fails_the_whole_run(tmp_path):
    """Silently dropping the bad file would produce a cohort from two databases
    of three, which is a different cohort reported as the same one."""
    from hit_significance import TmalignOutputError

    good = write_m8(tmp_path / "alis_afdb50.m8", [m8_row("P00001", 1e-40)])
    bad = write_m8(
        tmp_path / "alis_afdb-swissprot.m8",
        [tmalign_row("P00002", 0.91), tmalign_row("P00003", 0.55)],
    )
    with pytest.raises(TmalignOutputError):
        extract_foldseekhits([good, bad], str(tmp_path / "hits.txt"))


def test_the_guard_does_not_fire_on_the_real_fixture_ranges(tmp_path):
    """MEASURED on the repo's own `demo/search-mode` fixture, not quoted from
    the ticket, whose numbers are from an older snapshot and do not match:

        alis_afdb50.m8          evalue 2.862e-76 .. 0.0009674   bits max 3145
        alis_afdb-swissprot.m8  evalue 9.138e-80 .. 5.252       bits max 3290
        alis_afdb-proteome.m8   evalue 1.525e-79 .. 9.829       bits max 3282

    afdb50 is the one worth pinning: every one of its e-values is <= 1, so the
    guard's `bounded` condition is TRUE for it and only the other two keep it
    from firing. A guard written with `bounded` alone would refuse the
    pipeline's own fixture.
    """
    import pandas as pd
    from hit_significance import looks_like_tmalign

    cases = {
        "afdb50": ([2.862e-76, 1e-20, 0.0009674], [3145, 900, 40]),
        "afdb-swissprot": ([9.138e-80, 1.0, 5.252], [3290, 200, 30]),
        "afdb-proteome": ([1.525e-79, 2.0, 9.829], [3282, 150, 25]),
    }
    for name, (evalues, bits) in cases.items():
        assert not looks_like_tmalign(pd.Series(evalues), pd.Series(bits)), name

    # And the fixture that IS bounded still does not fire, which is the point.
    bounded = pd.Series(cases["afdb50"][0])
    assert bool((bounded >= 0).all() and (bounded <= 1).all()), "afdb50 is bounded in [0, 1]"


def test_a_3diaa_file_is_still_processed_normally(tmp_path):
    """The refusal must not cost the normal path anything."""
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 2.862e-76, bits=3145), m8_row("P00002", 1e-20, bits=900)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00001", "P00002"]


def test_the_evalue_default_and_filter_are_unchanged(tmp_path):
    """This phase adds a refusal and nothing else. Pinned, because the tempting
    next edit is to 'tidy' the threshold while the file is open."""
    from extract_foldseek_hits import DEFAULT_EVALUE

    assert DEFAULT_EVALUE == 0.01
    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [m8_row("P00001", 1e-40, bits=3000), m8_row("P00002", 0.5, bits=200)],
    )
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out))
    assert hits_in(out) == ["P00001"], "0.5 is above the 0.01 default and must be dropped"


def test_the_module_imports_nothing_its_rule_environment_lacks():
    """`extract_foldseek_hits` runs under `envs/pandas.yml`, whose only
    dependency is `pandas=2.0.1`.

    Phase 2 added an import of `hit_significance` to reuse the tmalign guard.
    That is safe only because `hit_significance` itself imports nothing beyond
    the standard library, `constants` and pandas -- and an import added to it
    later would break `extract_foldseek_hits` at RUNTIME, inside a snakemake
    rule, where the failure is a red job rather than a red test. Pinned here
    because the coupling is invisible from either file.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {"argparse", "os", "re", "sys", "constants", "pandas", "__future__"}
    for name in ("extract_foldseek_hits.py", "hit_significance.py"):
        tree = ast.parse((root / name).read_text())
        # MODULE LEVEL ONLY, which is what runs when the rule imports the file.
        # `extract_foldseek_hits.main` defers `tests.mock_domain_hits` inside an
        # `if PROTEINCARTOGRAPHY_SHOULD_USE_MOCKS` branch that production never
        # takes; a deferred import under a env-var guard is not a dependency of
        # the rule environment, and this test said it was until it was run.
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            # The two modules import each other, which is the coupling itself.
            unexpected = roots - allowed - {"extract_foldseek_hits", "hit_significance"}
            assert not unexpected, f"{name} imports {unexpected}, which envs/pandas.yml lacks"

    # Not vacuous: the rule above must actually reject something.
    probe = ast.parse("import scipy\n")
    top = probe.body[0]
    assert {alias.name.split(".")[0] for alias in top.names} - allowed == {"scipy"}


# ---------------------------------------------------------------------------
# the mode as a stated value (PC-014 phase 3)
# ---------------------------------------------------------------------------


def test_a_stated_tmalign_is_refused_before_anything_is_read(tmp_path):
    """A stated mode beats a guess.

    The column-shape heuristic can only ever say "this looks wrong", and it
    cannot see an empty or fully-filtered tmalign file at all. When the pipeline
    knows what it asked for -- and it does, from `foldseek_mode` -- saying so is
    better evidence than inspecting the result.
    """
    from hit_significance import TmalignOutputError

    empty = write_m8(tmp_path / "alis_afdb50.m8", [])
    with pytest.raises(TmalignOutputError, match="only interpret '3diaa'"):
        extract_foldseekhits([empty], str(tmp_path / "hits.txt"), mode="tmalign")


def test_a_stated_3diaa_is_processed_normally(tmp_path):
    """The mode the pipeline actually passes must cost the normal path nothing."""
    path = write_m8(tmp_path / "alis_afdb50.m8", [m8_row("P00001", 1e-40, bits=3000)])
    out = tmp_path / "hits.txt"
    extract_foldseekhits([path], str(out), mode="3diaa")
    assert hits_in(out) == ["P00001"]


def test_an_unstated_mode_still_falls_back_to_the_heuristic(tmp_path):
    """Direct invocation, where nobody passed a mode. The shape check is what is
    left, and it must still fire."""
    from hit_significance import TmalignOutputError

    path = write_m8(
        tmp_path / "alis_afdb50.m8",
        [tmalign_row("P00001", 0.9999), tmalign_row("P00002", 0.402)],
    )
    with pytest.raises(TmalignOutputError, match="tmalign"):
        extract_foldseekhits([path], str(tmp_path / "hits.txt"), mode=None)


def test_the_command_line_parser_builds_without_a_flag_collision():
    """`-m` was already `--max-num-hits` and phase 3 briefly gave it to `--mode`
    as well.

    argparse raises at PARSE-BUILD time, so the collision took down every
    `extract_foldseek_hits` job in the pipeline while every unit test here
    still passed -- nothing in this file had ever built the parser. Only the
    end-to-end domain test caught it. This builds the parser, which is the
    cheap half of what that test does.
    """
    import sys
    from unittest import mock

    import extract_foldseek_hits

    argv = [
        "extract_foldseek_hits.py",
        "--input",
        "a.m8",
        "--output",
        "out.txt",
        "--mode",
        "3diaa",
        "--max-num-hits",
        "5",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = extract_foldseek_hits.parse_args()
    assert args.mode == "3diaa"
    assert args.max_num_hits == 5
