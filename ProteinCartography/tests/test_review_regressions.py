"""Regression tests for defects found by the adversarial review gates.

Every test here corresponds to a finding from one of those gates. They are
kept together rather than filed under the module they exercise because the
useful property is the provenance: each one is a bug that was actually present
and actually produced a wrong answer, so a failure here means a real regression
rather than a changed opinion about design.

Two of these (B1, B2) were "blocks" severity -- they returned wrong numbers and
reported success.
"""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from config_schema import ConfigError, MultispaceConfig
from index import IndexAlignmentError, ProteinIndex
from spaces.base import BlockResult, BlockSpec, BlockSpecError, NotFusableError
from spaces.manifest import Manifest
from spaces.store import BlockStore

REPO_ROOT = Path(__file__).resolve().parents[2]

FILL = "0.0"


def fmt(v):
    return f"{v:.3E}"


def write(path, row_labels, col_labels, cells):
    lines = ["\t".join(["protid"] + list(col_labels))]
    for label, row in zip(row_labels, cells):
        lines.append("\t".join([label] + list(row)))
    path.write_text("\n".join(lines) + "\n")
    return path


# ==========================================================================
# B2 (blocks) -- align silently took the last duplicate from the SOURCE labels
# ==========================================================================


def test_b2_duplicate_source_labels_are_refused():
    """Was: `{p: i for i, p in enumerate(protids)}` kept the last occurrence, so
    aligning source ['A','B','A'] to index ('A','B') returned source row 2 for
    'A' -- silently, because the length check still passed.
    """
    idx = ProteinIndex.from_iterable(["A", "B"])
    values = np.array([[10.0, 10.0], [20.0, 20.0], [99.0, 99.0]])
    with pytest.raises(IndexAlignmentError) as excinfo:
        idx.align(["A", "B", "A"], values)
    message = str(excinfo.value)
    # The two discriminating facts: it refused, and it said which protid. The
    # remedy sentence ("...would silently keep one copy and discard the rest")
    # was asserted here too, which coupled the test to prose that carries no
    # information the exception type and the named protid do not already carry.
    assert "duplicate protid" in message
    assert "'A'" in message


def test_b2_align_frame_inherits_the_guard():
    import pandas as pd

    idx = ProteinIndex.from_iterable(["A", "B"])
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.Index(["A", "B", "A"]))
    with pytest.raises(IndexAlignmentError, match="duplicate protid"):
        idx.align_frame(frame)


def test_b2_a_duplicate_that_also_hides_an_absence_reports_both():
    """A row duplicated in place of a missing one is the common shape of this
    mistake, so the error names both problems rather than making the caller fix
    one to discover the next.
    """
    idx = ProteinIndex.from_iterable(["A", "B", "C"])
    values = np.zeros((3, 1))
    with pytest.raises(IndexAlignmentError) as excinfo:
        idx.align(["A", "B", "B"], values)
    message = str(excinfo.value)
    # Both problems named: the repeated 'B', and the 'C' it displaced.
    #
    # This used to read `"3 proteins are missing" in message or "missing from
    # the source" in message`, which was wrong twice over. The phrase only ever
    # matched by accident -- the 3 is the *index* size, and exactly one protein
    # is missing -- and the `or` was added to absorb a rewording, so a further
    # rewording of either clause would have been absorbed the same way. The
    # protids are the fact; the sentence around them is not.
    assert "duplicate protid" in message
    assert "'B'" in message
    assert "'C'" in message


# ==========================================================================
# B3 -- is_fresh could never return True across processes
# ==========================================================================


def pairwise_spec(**overrides):
    kwargs = {
        "id": "tmscore",
        "kind": "pairwise",
        "fusable": True,
        "metric": "precomputed",
        "normalization": "unit_mean_distance",
        "provider": "tmscore",
        "symmetrization": "mean",
    }
    kwargs.update(overrides)
    return BlockSpec(**kwargs)


def test_b3_cache_hits_for_an_independently_rebuilt_manifest(tmp_path):
    """Was: write_block injected values_digest/spec into manifest.extra, which
    feeds cache_key, so only a caller who already had the output could match --
    i.e. never the caller the cache exists for.
    """
    store = BlockStore(str(tmp_path))
    built = Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B", "C"])
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3)),
        built,
    )
    # Rebuilt from the same inputs by a fresh process that has NOT computed the
    # block. This is the only case that matters.
    rebuilt = Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B", "C"])
    assert store.is_fresh("tmscore", rebuilt)


def test_b3_write_block_does_not_mutate_the_callers_manifest(tmp_path):
    """Was: the passing freshness tests passed *because* of this mutation."""
    store = BlockStore(str(tmp_path))
    manifest = Manifest.build("block", "tmscore", protids=["A", "B"])
    before = copy.deepcopy(manifest)
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B"], distances=np.zeros(1)),
        manifest,
    )
    assert manifest.extra == before.extra
    assert manifest.derived == before.derived
    assert manifest.cache_key == before.cache_key


def test_b3_changed_inputs_still_miss(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(
        BlockResult(spec=pairwise_spec(), protids=["A", "B"], distances=np.zeros(1)),
        Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B"]),
    )
    assert not store.is_fresh(
        "tmscore", Manifest.build("block", "tmscore", params={"v": 2}, protids=["A", "B"])
    )


def test_b3_values_digest_describes_the_file_as_stored(tmp_path):
    """Was: the digest hashed the pre-cast float64 array while the file on disk
    was float32, so it could never verify its own file.
    """
    from spaces.manifest import values_digest

    store = BlockStore(str(tmp_path))
    store.write_block(
        BlockResult(
            spec=pairwise_spec(),
            protids=["A", "B", "C"],
            distances=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    )
    stored = store.read_block("tmscore")
    recorded = store.read_manifest("tmscore").derived["values_digest"]
    assert values_digest(stored.distances) == recorded


# ==========================================================================
# B6 -- the fusable protection was keyed on a user-chosen block id
# ==========================================================================


def _fuse(block_id, block_body):
    return MultispaceConfig.from_dict(
        {
            "blocks": {"tmscore": {"provider": "tmscore"}, block_id: block_body},
            "spaces": {"s": {"blocks": ["tmscore", block_id], "strategy": "late"}},
        }
    )


@pytest.mark.parametrize("block_id", ["tax", "lineage", "my_taxonomy_block"])
def test_b6_protection_follows_the_provider_not_the_block_name(block_id):
    """Was: keyed on the block id, so it protected `taxonomy:` and missed
    `tax:` -- protecting exactly the users who already knew about it.
    """
    with pytest.raises(NotFusableError, match="circular"):
        _fuse(block_id, {"provider": "uniprot_lineage"})


def test_b6_protection_is_case_insensitive_on_the_block_id():
    with pytest.raises(NotFusableError):
        _fuse("Taxonomy", {"provider": "some_unknown_provider"})


def test_b6_explicit_override_requires_a_written_justification():
    """Was: `fusable: true` silently defeated the protection."""
    with pytest.raises(ConfigError) as excinfo:
        _fuse("taxonomy", {"provider": "uniprot_lineage", "fusable": True})
    message = str(excinfo.value)
    assert "known overlay-only signal" in message
    assert "fusable_override_reason" in message


def test_b6_override_with_a_justification_is_allowed_and_recorded():
    """A maintainer may have a reason; it just has to be written down."""
    config = _fuse(
        "taxonomy",
        {
            "provider": "uniprot_lineage",
            "fusable": True,
            "fusable_override_reason": "deliberate positive control for the circularity check",
        },
    )
    block = config.blocks["taxonomy"]
    assert block.fusable
    assert "positive control" in block.params["fusable_override_reason"]


def test_b6_struclusters_provider_is_covered():
    with pytest.raises(NotFusableError, match="amino-acid identity"):
        _fuse("sc", {"provider": "struclusters"})


# ==========================================================================
# B7 -- the alignment_verified gate accepted any truthy value
# ==========================================================================


@pytest.mark.parametrize("value", ["false", "no", "0", 1, []])
def test_b7_non_boolean_alignment_verified_is_rejected(value):
    """Was: the string "false" is truthy, so it switched the gate OFF -- the
    opposite of what it says, on the gate standing between the user and a
    99.92%-wrong direct read.
    """
    with pytest.raises(ConfigError):
        MultispaceConfig.from_dict(
            {
                "blocks": {
                    "tmscore": {
                        "provider": "tmscore",
                        "representation": "direct",
                        "alignment_verified": value,
                    }
                },
                "spaces": {"structure": {"blocks": ["tmscore"]}},
            }
        )


def test_b7_real_true_still_opens_the_gate():
    config = MultispaceConfig.from_dict(
        {
            "blocks": {
                "tmscore": {
                    "provider": "tmscore",
                    "representation": "direct",
                    "alignment_verified": True,
                }
            },
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert config.blocks["tmscore"].representation == "direct"


# ==========================================================================
# Notes -- lower severity, same origin
# ==========================================================================


def test_note_with_protids_refuses_to_relabel_with_a_different_length():
    result = BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3))
    with pytest.raises(BlockSpecError, match="renames rows in place"):
        result.with_protids(["A", "B"])


def test_note_with_protids_refuses_duplicates():
    result = BlockResult(spec=pairwise_spec(), protids=["A", "B", "C"], distances=np.zeros(3))
    with pytest.raises(BlockSpecError, match="doubles that protein's weight"):
        result.with_protids(["A", "A", "C"])


# ==========================================================================
# Design findings from the API-probe pass
# ==========================================================================


def test_asymmetric_pairwise_block_can_be_represented():
    """Was: the condensed-triangle shape check made symmetry a type-level
    assertion, so half of every directed measurement had to be discarded with
    no record that it had been.
    """
    values = np.array([[1.0, 0.8, 0.5], [0.7, 1.0, 0.4], [0.5, 0.45, 1.0]])
    result = BlockResult(
        spec=pairwise_spec(kind="pairwise_directed", symmetrization=None),
        protids=["A", "B", "C"],
        distances=values,
    )
    assert result.distances.shape == (3, 3)
    assert result.distances[0][1] != result.distances[1][0]


def test_symmetric_pairwise_must_declare_its_symmetrization():
    with pytest.raises(BlockSpecError) as excinfo:
        pairwise_spec(symmetrization=None)
    assert "modelling choice" in str(excinfo.value)


def test_directed_block_may_not_claim_a_symmetrization():
    with pytest.raises(BlockSpecError, match="contradictory"):
        pairwise_spec(kind="pairwise_directed", symmetrization="mean")


def test_absence_and_censoring_are_different_channels():
    """A predictor that has no value is not the same as a pair that lost a cut,
    and one anonymous mask could not tell them apart.
    """
    features = np.zeros((3, 2))
    absent = np.zeros((3, 2), dtype=bool)
    absent[2, :] = True
    result = BlockResult(
        spec=BlockSpec(
            id="deeploc",
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="zscore_within",
            provider="deeploc",
        ),
        protids=["A", "B", "C"],
        features=features,
        channels={"absent": absent},
    )
    assert result.absent_rate == pytest.approx(1 / 3)
    assert result.censoring_rate is None  # no claim made about censoring


def test_confidence_is_a_channel_not_a_feature_column():
    """Encoding confidence into the value would let low confidence move a point."""
    result = BlockResult(
        spec=BlockSpec(
            id="clean_ec",
            kind="features",
            fusable=True,
            metric="cosine",
            normalization="none",
            provider="clean",
        ),
        protids=["A", "B"],
        features=np.zeros((2, 4)),
        channels={"confidence": np.full((2, 4), 0.8)},
    )
    assert result.channel("confidence").mean() == pytest.approx(0.8)


def test_unknown_channel_names_are_rejected():
    with pytest.raises(BlockSpecError, match="unknown channel"):
        BlockResult(
            spec=pairwise_spec(),
            protids=["A", "B"],
            distances=np.zeros(1),
            channels={"vibes": np.zeros(1, dtype=bool)},
        )


def test_object_dtype_features_are_rejected_at_construction():
    """Was: accepted here and failed much later inside np.save."""
    ragged = np.empty((2, 1), dtype=object)
    ragged[0, 0] = ["PF00069", "PF00017"]
    ragged[1, 0] = ["PF00069"]
    with pytest.raises(BlockSpecError, match="floating-point array"):
        BlockResult(
            spec=BlockSpec(
                id="domains",
                kind="features",
                fusable=True,
                metric="jaccard",
                normalization="none",
                provider="domains",
            ),
            protids=["A", "B"],
            features=ragged,
        )


def test_unexplained_nan_is_rejected():
    features = np.zeros((2, 2))
    features[0, 0] = np.nan
    with pytest.raises(BlockSpecError, match="not covered by a 'censored' or 'absent'"):
        BlockResult(
            spec=BlockSpec(
                id="x",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="x",
            ),
            protids=["A", "B"],
            features=features,
        )


def test_declared_nan_is_accepted():
    features = np.zeros((2, 2))
    features[0, 0] = np.nan
    absent = np.zeros((2, 2), dtype=bool)
    absent[0, 0] = True
    result = BlockResult(
        spec=BlockSpec(
            id="x",
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="none",
            provider="x",
        ),
        protids=["A", "B"],
        features=features,
        channels={"absent": absent},
    )
    assert np.isnan(result.features[0, 0])


def test_channels_round_trip_through_the_store_with_their_names(tmp_path):
    store = BlockStore(str(tmp_path))
    censored = np.zeros((2, 2), dtype=bool)
    censored[0, 0] = True
    absent = np.zeros((2, 2), dtype=bool)
    absent[1, 1] = True
    store.write_block(
        BlockResult(
            spec=BlockSpec(
                id="multi",
                kind="features",
                fusable=True,
                metric="euclidean",
                normalization="none",
                provider="x",
            ),
            protids=["A", "B"],
            features=np.zeros((2, 2)),
            channels={"censored": censored, "absent": absent},
        )
    )
    loaded = store.read_block("multi")
    assert set(loaded.channels) == {"censored", "absent"}
    np.testing.assert_array_equal(loaded.channels["censored"], censored)
    np.testing.assert_array_equal(loaded.channels["absent"], absent)


# ==========================================================================
# GE.2 (blocks) -- a checkpoint output nothing consumes silently drops a rule
# ==========================================================================
#
# `download_pdbs` is a checkpoint. Adding `cohort_report.json` to its outputs
# looked additive and was not: on an output tree produced before this branch the
# structures directory exists and the report does not, and because *no job
# requests the report*, snakemake never re-runs the checkpoint to produce it.
# `checkpoints.download_pdbs.get()` then raises, `get_pdb_filepaths` contributes
# no `copy_pdb` job, and the run proceeds without the query proteins -- silently,
# not with an error. Reproduced in isolation against snakemake 7.25.3: the same
# tree plans 3 jobs including `copy_pdb` with a single-output checkpoint and 2
# jobs without, and the surviving job receives the directory rather than the
# file list.
#
# The fix is to declare each new output only when the rule that reads it is in
# the DAG, so it is never an orphan. These tests assert that, not the prose.


def _planned_outputs(config_path):
    """The output-file set snakemake plans for a config, via `--summary`."""
    completed = subprocess.run(
        [
            "snakemake",
            "--configfile",
            str(config_path),
            "--summary",
            "--rerun-incomplete",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        pytest.fail(f"snakemake --summary failed:\n{completed.stdout}\n{completed.stderr}")
    rows = completed.stdout.splitlines()[1:]
    return {row.split("\t")[0] for row in rows if "\t" in row}


def _write_config(tmp_path, name, extra):
    body = {
        "mode": "search",
        "analysis_name": "ge2",
        "input_dir": "demo/search-mode/input/",
        "output_dir": str(tmp_path / "out") + "/",
        "max_blast_hits": 10,
        "max_foldseek_hits": 10,
        "max_structures": 10,
        "plotting_modes": ["pca_umap"],
    }
    body.update(extra)
    path = tmp_path / name
    # JSON is valid YAML, and no environment here has PyYAML alongside the rest
    # of the stack -- the Snakefile reads either.
    path.write_text(json.dumps(body))
    return path


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_default_search_path_declares_neither_new_output(tmp_path):
    """The regression itself. Both files are absent from the legacy plan."""
    planned = _planned_outputs(_write_config(tmp_path, "legacy.json", {}))
    assert planned, "the summary returned no rows, so this test proves nothing"
    orphans = [p for p in planned if p.endswith(("cohort_report.json", ".mapping.tsv"))]
    assert orphans == [], (
        f"{orphans} are declared on the default path but no rule consumes them. "
        "An unconsumed output on the `download_pdbs` checkpoint drops `copy_pdb` "
        "on any pre-existing output tree."
    )


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_cohort_report_appears_exactly_when_a_space_reads_it(tmp_path):
    spaces = {
        "blocks": {"tmscore": {"provider": "tmscore", "representation": "profile"}},
        "spaces": {"legacy": {"blocks": ["tmscore"], "strategy": "none", "reducers": ["pca"]}},
    }
    planned = _planned_outputs(_write_config(tmp_path, "spaces.json", spaces))
    assert any(p.endswith("cohort_report.json") for p in planned), (
        "`diagnose_space` reads the cohort report, so enabling spaces must declare it. "
        "If this fails the report is never written and the diagnostic loses its input."
    )


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_refseq_mapping_appears_exactly_when_significance_reads_it(tmp_path):
    significance = {"cohort": {"selection": "significance"}}
    planned = _planned_outputs(_write_config(tmp_path, "significance.json", significance))
    assert any(p.endswith(".mapping.tsv") for p in planned), (
        "`aggregate_hit_significance` reads the RefSeq mapping, so significance "
        "selection must declare it."
    )


# ==========================================================================
# The checkpoint-output class, rather than the two filenames that happened to
# hit it. A checkpoint output that no job in the plan asks for is never a
# reason to re-run the checkpoint, so on a pre-existing output tree the
# checkpoint is left alone, `checkpoints.<name>.get()` raises, and everything
# that function contributes silently disappears from the DAG (FOLLOWUPS #47).
#
# The supply side comes from snakemake's own parse -- `rule.is_checkpoint` over
# the built workflow -- so a fourth checkpoint is covered the day it is added.
# The demand side has two routes and needs both:
#
#   * another rule naming the path in its `input`, which the workflow gives us;
#   * a `checkpoints.<name>.get(...).output.<attr>` reference inside an input
#     function, which the workflow CANNOT give us because it lives in a Python
#     body that has not run. That half is a source scan, and it is a scan
#     rather than a hand-written table so that a fourth consumer is covered
#     too -- but it is a scan, and this comment is where that is admitted.
#
# NOT USED: `snakemake --detailed-summary`. Its `input-file(s)` column is
# populated only for files that already exist; on a fresh tree 1 of 39 rows
# carries inputs and the rest are "-", so it cannot answer the demand question
# this check is about.
# ==========================================================================

_CHECKPOINT_GET_CONSUMER_RE = re.compile(r"checkpoints\.(\w+)\.get\([^)]*\)\.output\.(\w+)")

#: Both Snakefiles, because `Snakefile` includes the domain one and two of the
#: three checkpoints live there.
_SNAKEFILES = ("Snakefile", "Snakefile_domain")


def _checkpoint_get_consumers() -> set:
    """(checkpoint, output attribute) pairs consumed through `.get()`."""
    pairs = set()
    for name in _SNAKEFILES:
        pairs |= set(_CHECKPOINT_GET_CONSUMER_RE.findall((REPO_ROOT / name).read_text()))
    return pairs


def _demand_from_input_functions(workflow) -> set:
    """Paths named by an input FUNCTION rather than by a literal.

    The third demand route, and the one that is easy to miss: `diagnose_space`
    reads the cohort report through `cohort=get_cohort_report_input`, a
    callable, so `rule.input` holds the function and not the path. Without this
    the check calls a consumed output an orphan -- which it did, on the first
    run, against the very file GE.2 was about.

    Called with empty wildcards, which is enough for the functions that do not
    branch on one. The rest raise -- `IncompleteCheckpointException` for
    anything that reaches through a checkpoint, `AttributeError` for anything
    that needs a wildcard -- and are skipped here because the `.get()` scan
    already covers the checkpoint-dependent ones.
    """
    from snakemake.io import Wildcards

    resolved = set()
    for rule in workflow.rules:
        for item in rule.input:
            if not callable(item):
                continue
            try:
                produced = item(Wildcards(fromdict={}))
            except Exception:
                continue
            if isinstance(produced, (str, Path)):
                values = [produced]
            else:
                try:
                    values = list(produced or [])
                except TypeError:
                    continue
            resolved.update(str(value) for value in values if isinstance(value, (str, Path)))
    return resolved


def _checkpoint_orphans(config_path):
    """Checkpoint outputs that are in the plan and that nothing in it asks for.

    Returns a list of ``"<rule>.<attr> -> <path>"`` strings, empty when every
    planned checkpoint output has a consumer.

    LIMIT, stated rather than implied: `--summary` on a fresh tree resolves the
    DAG only as far as the first unresolved checkpoint, so this covers
    CHECKPOINT outputs and says nothing about jobs that appear only after a
    checkpoint has run. It is not general DAG coverage.
    """
    from snakemake.workflow import Workflow

    planned = _planned_outputs(config_path)
    config = json.loads(Path(config_path).read_text())
    workflow = Workflow(
        snakefile=str(REPO_ROOT / "Snakefile"), overwrite_config=config, use_conda=True
    )
    workflow.include(str(REPO_ROOT / "Snakefile"), overwrite_default_target=True)

    demanded = {item for rule in workflow.rules for item in rule.input if isinstance(item, str)}
    demanded |= _demand_from_input_functions(workflow)
    through_get = _checkpoint_get_consumers()

    orphans = []
    for rule in workflow.rules:
        if not getattr(rule, "is_checkpoint", False):
            continue
        by_attr = dict(zip(rule.output.keys(), [str(path) for path in rule.output]))
        for path in (str(item) for item in rule.output):
            if path not in planned:
                continue
            attr = next((name for name, value in by_attr.items() if value == path), None)
            if path in demanded or (rule.name, attr) in through_get:
                continue
            orphans.append(f"{rule.name}.{attr} -> {path}")
    return orphans


_ORPHAN_MESSAGE = (
    "these checkpoint outputs are planned and no job asks for them. An unconsumed "
    "checkpoint output is never a reason to re-run the checkpoint, so on a pre-existing "
    "output tree `checkpoints.<name>.get()` raises and every job the input function "
    "would have contributed vanishes from the DAG without an error"
)


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_no_checkpoint_declares_an_output_the_legacy_plan_ignores(tmp_path):
    orphans = _checkpoint_orphans(_write_config(tmp_path, "legacy.json", {}))
    assert orphans == [], f"{orphans}: {_ORPHAN_MESSAGE}"


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_no_checkpoint_declares_an_output_the_spaces_plan_ignores(tmp_path):
    """The config that turns `cohort_report.json` on. It is declared here
    because `diagnose_space` reads it, which is the fix GE.2 landed."""
    spaces = {
        "blocks": {"tmscore": {"provider": "tmscore", "representation": "profile"}},
        "spaces": {"legacy": {"blocks": ["tmscore"], "strategy": "none", "reducers": ["pca"]}},
    }
    orphans = _checkpoint_orphans(_write_config(tmp_path, "spaces.json", spaces))
    assert orphans == [], f"{orphans}: {_ORPHAN_MESSAGE}"


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_no_checkpoint_declares_an_output_the_significance_plan_ignores(tmp_path):
    """Significance adds an input to the checkpoint rather than an output, so
    this config is here to prove the check is not fooled by the rule changing
    shape underneath it."""
    significance = {
        "blocks": {"tmscore": {"provider": "tmscore", "representation": "profile"}},
        "spaces": {"legacy": {"blocks": ["tmscore"], "strategy": "none", "reducers": ["pca"]}},
        "cohort": {"selection": "significance", "significance_rule": {"measure": "evalue"}},
    }
    orphans = _checkpoint_orphans(_write_config(tmp_path, "significance.json", significance))
    assert orphans == [], f"{orphans}: {_ORPHAN_MESSAGE}"


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_no_checkpoint_declares_an_output_the_cluster_plan_ignores(tmp_path):
    """Cluster mode does not run `download_pdbs` at all -- the user supplies
    the structures -- so this covers the branch where the checkpoint's outputs
    are absent from the plan rather than present and consumed."""
    body = {
        "mode": "cluster",
        "analysis_name": "cluster-mode-demo",
        "input_dir": "demo/cluster-mode/input",
        "output_dir": str(tmp_path / "out") + "/",
        "plotting_modes": ["pca_umap"],
        "features_file": "uniprot_features.tsv",
        "key_protids": ["P60709"],
    }
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(body))
    orphans = _checkpoint_orphans(path)
    assert orphans == [], f"{orphans}: {_ORPHAN_MESSAGE}"


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="needs snakemake")
def test_the_orphan_check_finds_a_checkpoint_output_nobody_reads(tmp_path, monkeypatch):
    """A check that has never been seen to fail is not a check.

    `download_pdbs.protein_structures_dir` reaches the plan through exactly one
    route -- `checkpoints.download_pdbs.get(...).output.protein_structures_dir`
    inside `get_pdb_filepaths`. Removing that route is the whole of the GE.2
    state: a planned checkpoint output with no consumer. So the probe empties
    the `.get()` scan and requires the check to report it.
    """
    import sys

    config_path = _write_config(tmp_path, "legacy.json", {})
    assert _checkpoint_orphans(config_path) == [], "the tree must be clean before the probe"

    monkeypatch.setattr(sys.modules[__name__], "_checkpoint_get_consumers", set)
    orphans = _checkpoint_orphans(config_path)
    assert any(o.startswith("download_pdbs.protein_structures_dir") for o in orphans), orphans
