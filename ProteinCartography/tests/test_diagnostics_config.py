#!/usr/bin/env python
"""Every DiagnosticsConfig field is read by something, or says why not.

A value written into a config and honored by nothing is the most persistent
defect shape in this work. `spec.metric` is recorded and never consulted
(FOLLOWUPS #29); `spec.normalization` is recorded and applied nowhere (#32);
every provider's `normalization` default was unreachable for three commit
groups because its caller always passed the parameter (REVIEW_LOG G6.4). Each
looked like evidence, which is why each survived so long.

Group 8a's fix for the fusion half of this was a validator that enumerates the
parameters each strategy will read and rejects the rest. It found a dead
``params: {K: 20}`` in the repository's own test suite within a second of
existing (G8.8). This file is the same idea for diagnostics, in the one form
available here: consumption cannot be checked by a validator, because it is a
property of the code rather than of the config, so it is checked by looking.

The exemption list is the point. A field that no diagnostic reads must be named
here with the Phase 5 item that will read it, and the test fails both when an
unlisted field is dead *and* when a listed one comes alive -- so the list
shrinks as group 8c lands rather than being left behind as a stale comment.
"""

from __future__ import annotations
import ast
import dataclasses
from pathlib import Path

import pytest
from config_schema import ConfigError, DiagnosticsConfig, from_legacy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Empty as of group 8c, which is what finishes FOLLOWUPS #36. Every field is
#: now read by `diagnose_space.py`, and both tests below stay in place: the
#: first fails if a future field arrives dead, the second if this list is ever
#: refilled with something that is not.
NOT_YET_CONSUMED: dict = {}


def _consumers(field_name: str) -> list:
    """Modules outside `config_schema` that *read* ``....diagnostics.<field>``.

    Parsed rather than grepped. The first version of this searched for the
    literal string, and it fired twice in one commit group on prose: a
    docstring saying which config key to raise, and a comment naming the key a
    validator enumerates. Both were false positives, and the pressure they
    create is to word documentation around the test, which is exactly
    backwards -- the fields that most need explaining are the ones this list
    covers.

    So it looks for the attribute access itself: an ``Attribute`` node named
    for the field whose own value is an ``Attribute`` named ``diagnostics``.
    That is narrower than a grep, and deliberately: it cannot see
    ``getattr(config.diagnostics, name)``. It still catches the case that has
    actually recurred three times, which is a field no code anywhere reaches.
    """
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name in ("config_schema.py",) or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == field_name
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "diagnostics"
            ):
                found.append(path.relative_to(PACKAGE_ROOT).as_posix())
                break
    return found


def _fields() -> list:
    return [f.name for f in dataclasses.fields(DiagnosticsConfig)]


def test_every_field_is_either_consumed_or_explained():
    for name in _fields():
        if name in NOT_YET_CONSUMED:
            continue
        assert _consumers(name), (
            f"DiagnosticsConfig.{name} is read by nothing outside config_schema.py. "
            "Either add the code that reads it, or add it to NOT_YET_CONSUMED with "
            "the Phase 5 item that will. A field honored by nothing is the defect "
            "shape of FOLLOWUPS #29, #32 and REVIEW_LOG G6.4."
        )


def test_the_exemption_list_shrinks_rather_than_going_stale():
    """The other direction, and the reason this is a test rather than a note.

    It did its job: implementing neighborhood stability made
    `bootstrap_replicates` live and this failed until the entry point read it,
    forcing the list to be trimmed in the same commit rather than left behind
    as a stale comment. The list is empty now and the test stays.
    """
    for name, reason in NOT_YET_CONSUMED.items():
        assert name in _fields(), f"NOT_YET_CONSUMED names {name!r}, which is not a field"
        consumers = _consumers(name)
        assert not consumers, (
            f"DiagnosticsConfig.{name} is now read by {consumers}, so it is no longer "
            f"pending on {reason!r}. Remove it from NOT_YET_CONSUMED."
        )


def test_the_exemption_list_covers_no_field_that_does_not_exist():
    assert set(NOT_YET_CONSUMED) <= set(_fields())


# --- the one live field, proved live ----------------------------------------


def test_k_is_carried_from_the_config():
    config = from_legacy({"diagnostics": {"k": 4}})
    assert config.diagnostics.k == 4


def test_k_defaults_to_the_module_default():
    from diagnostics.embedding import DEFAULT_K

    assert from_legacy({}).diagnostics.k == DEFAULT_K


def test_the_neighborhood_size_actually_reaches_the_statistic(tmp_path):
    """The test that makes `k` a live field rather than a recorded one.

    Two runs of the real entry point over the same blocks and the same
    embedding, differing only in `diagnostics.k`, must produce different
    numbers. A config value that changes nothing is exactly what this file
    exists to prevent, and asserting only that the value round-trips through
    `from_legacy` would not notice.
    """
    import json
    import sys

    from embedding_cohort import FOLD, embedding_cohort
    from spaces.store import BlockStore

    cohort = embedding_cohort(n=60)
    output_dir = tmp_path / "output"
    BlockStore(str(output_dir)).write_block(cohort.block_result("truth"))
    embedding = tmp_path / "fold.tsv"
    cohort.write_embedding(embedding, FOLD)

    def run(k):
        config = tmp_path / f"config_{k}.json"
        config.write_text(
            json.dumps(
                {
                    "blocks": {"truth": {"provider": "tmscore"}},
                    "spaces": {"s": {"blocks": ["truth"], "strategy": "none", "reducers": ["pca"]}},
                    "diagnostics": {"k": k},
                }
            )
        )
        import diagnose_space

        original = sys.argv
        try:
            sys.argv = [
                "diagnose_space.py",
                "-c",
                str(config),
                "-s",
                "s",
                "-o",
                str(output_dir),
                "--embedding",
                f"pca={embedding}",
            ]
            diagnose_space.main()
        finally:
            sys.argv = original
        with open(output_dir / "spaces" / "s" / "diagnostics.json") as handle:
            return json.load(handle)["faithfulness"][0]

    small, large = run(3), run(20)
    assert small["k"] == 3
    assert large["k"] == 20
    assert small["trustworthiness_mean"] != large["trustworthiness_mean"]


def test_a_k_below_one_is_refused():
    with pytest.raises(ConfigError, match="diagnostics.k"):
        from_legacy({"diagnostics": {"k": 0}})


def test_an_unknown_diagnostics_key_is_refused():
    """The same strictness `STRATEGY_PARAMS` applies to fusion params: a
    misspelled key is silently ignored otherwise, which is indistinguishable
    from a key that is read and does nothing."""
    with pytest.raises(ConfigError, match="diagnostics"):
        from_legacy({"diagnostics": {"kk": 5}})
