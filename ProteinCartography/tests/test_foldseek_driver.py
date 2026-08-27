#!/usr/bin/env python
"""The exact foldseek command lines the two TM-score drivers run.

`foldseek_clustering.run_foldseek_clustering` and
`calculate_key_protid_tmscores.run_foldseek_clustering` shared three of their
foldseek calls, with a `TODO (KC)` in the second saying so. They now share
`run_tmscore_pass`, and this file is what makes that safe to have done: nothing
else in the suite runs either function -- foldseek is not installed in any test
environment -- so a wrong argument order would have surfaced only in a pipeline
run, on the default output path.

The assertions are on the argv `subprocess.run` receives, not on any file, so
they hold with no foldseek anywhere. They pin the three differences the
consolidation had to preserve: which database is the target, `--exhaustive-search`
on the query-vs-target pass only, and `clust` on the all-vs-all pass only.
"""

from __future__ import annotations
import subprocess

import calculate_key_protid_tmscores
import foldseek_clustering
import pytest


@pytest.fixture()
def foldseek_calls(monkeypatch, tmp_path):
    """Every foldseek argv a driver runs, as strings, with tmp_path elided."""
    recorded = []

    def record(argv, *args, **kwargs):
        recorded.append([str(part).replace(str(tmp_path), "<ROOT>") for part in argv])

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(foldseek_clustering.subprocess, "run", record)
    monkeypatch.setattr(calculate_key_protid_tmscores.subprocess, "run", record)
    return recorded


def test_the_all_by_all_pass_runs_six_foldseek_calls_in_this_order(foldseek_calls, tmp_path):
    (tmp_path / "q").mkdir()
    (tmp_path / "r").mkdir()
    foldseek_clustering.run_foldseek_clustering(str(tmp_path / "q"), str(tmp_path / "r"))
    assert [argv[1] for argv in foldseek_calls] == [
        "createdb",
        "search",
        "aln2tmscore",
        "createtsv",
        "clust",
        "createtsv",
    ]
    # Query and target are the same database: that is what "all by all" means,
    # and it is the difference from the query-vs-target driver below.
    search = foldseek_calls[1]
    assert search[2] == search[3] == "<ROOT>/r/temp/temp_db"
    assert search[-1] == "-a"
    assert "--exhaustive-search" not in search, (
        "the all-vs-all pass must NOT be exhaustive; it is the query-vs-target "
        "pass that is, and the two share a helper now"
    )
    assert foldseek_calls[3][-1] == "<ROOT>/r/all_by_all_tmscore.tsv"
    assert foldseek_calls[4][-4:] == ["--cluster-mode", "0", "--similarity-type", "2"]
    assert foldseek_calls[5][-1] == "<ROOT>/r/struclusters.tsv"


def test_the_query_vs_target_pass_runs_four_and_is_exhaustive(foldseek_calls, tmp_path):
    (tmp_path / "q").mkdir()
    (tmp_path / "t").mkdir()
    (tmp_path / "r").mkdir()
    calculate_key_protid_tmscores.run_foldseek_clustering(
        str(tmp_path / "q" / "db"), str(tmp_path / "t"), str(tmp_path / "r")
    )
    assert [argv[1] for argv in foldseek_calls] == [
        "createdb",
        "search",
        "aln2tmscore",
        "createtsv",
    ], "this driver must not cluster; only the all-vs-all one does"
    # The target database is built from the target FOLDER, and the query is the
    # database the pipeline already made. Swapping them would silently score
    # every pair the wrong way round.
    assert foldseek_calls[0][2:] == ["<ROOT>/t", "<ROOT>/r/temp_tm/temp_db_target"]
    search = foldseek_calls[1]
    assert search[2] == "<ROOT>/q/db"
    assert search[3] == "<ROOT>/r/temp_tm/temp_db_target"
    assert search[-2:] == ["-a", "--exhaustive-search"]
    assert foldseek_calls[3][-1] == "<ROOT>/r/key_protid_tmscores.tsv"


def test_the_shared_pass_is_what_both_drivers_call(foldseek_calls, tmp_path):
    """A guard on the consolidation itself.

    If either driver grew its own copy of the three calls back, this file's
    other two tests would still pass -- they assert on argv, which a copy would
    reproduce. This one asserts the sharing.
    """
    import inspect

    assert "run_tmscore_pass" in inspect.getsource(
        calculate_key_protid_tmscores.run_foldseek_clustering
    )
    assert "run_tmscore_pass" in inspect.getsource(foldseek_clustering.run_foldseek_clustering)
