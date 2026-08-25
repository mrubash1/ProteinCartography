"""Tests for the `compute_block` entry point.

This file exists because the entry point had none, and the gap had already cost
something. `compute_block` sits between the config and the provider and fills
in parameters on the way through, and one of those fills silently overrode
every provider's own default. That went unnoticed for two blocks because both
happened to want the same value; the third one did not, and the wrong value was
already written into its manifest before anyone looked.

The lesson generalizes, and it is the same one that makes a mutation survive
when it is anchored on a default the caller overrides: **a default is dead code
if its caller always passes the parameter.** A test that calls the provider
directly cannot see that. So the tests here drive `main()` with an argv, which
is the path the Snakefile actually takes.
"""

import json
import sys

import compute_block
import pytest


@pytest.fixture
def run_dir(tmp_path):
    """An output directory with the inputs a `biophys` block needs."""
    features = tmp_path / "output" / "protein_features"
    features.mkdir(parents=True)
    (features / "uniprot_features.tsv").write_text(
        "protid\tSequence\n"
        "P1\tMDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQ\n"
        "P2\tMKKKKKAAAAAEEEEEGGGGGWWWWW\n"
    )
    return tmp_path


def write_config(tmp_path, blocks):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"blocks": blocks, "spaces": {}}))
    return path


def run(monkeypatch, config_path, block_id, output_dir, *extra):
    argv = [
        "compute_block.py",
        "--configfile",
        str(config_path),
        "--block-id",
        block_id,
        "--output-dir",
        str(output_dir),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return compute_block.main()


def manifest_of(output_dir, block_id):
    path = output_dir / "blocks" / block_id / "manifest.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# the defect this file was written for
# ---------------------------------------------------------------------------


def test_a_provider_default_survives_when_the_config_is_silent(monkeypatch, run_dir):
    """The regression. `biophys` asks for `zscore_within` and must get it.

    Was: `BlockConfig.normalization` defaulted to `unit_mean_distance` and
    `compute_block` passed it down as though the user had asked for it, so the
    provider's `params.get("normalization", ...)` never saw its own default.
    The manifest recorded `unit_mean_distance` for a block whose columns are a
    pH beside a per-residue charge, where the unnormalized euclidean distance
    is the isoelectric point and nothing else.
    """
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    assert run(monkeypatch, config, "biophys", run_dir / "output") == 0
    spec = manifest_of(run_dir / "output", "biophys")["derived"]["spec"]
    assert spec["normalization"] == "zscore_within"


def test_an_explicit_normalization_still_wins(monkeypatch, run_dir):
    """The config remains the authority when it says something."""
    config = write_config(
        run_dir, {"biophys": {"provider": "biophys", "normalization": "unit_mean_distance"}}
    )
    assert run(monkeypatch, config, "biophys", run_dir / "output") == 0
    spec = manifest_of(run_dir / "output", "biophys")["derived"]["spec"]
    assert spec["normalization"] == "unit_mean_distance"


def test_an_invalid_normalization_is_still_rejected(monkeypatch, run_dir):
    """Making the field optional must not make it unvalidated."""
    config = write_config(run_dir, {"biophys": {"provider": "biophys", "normalization": "zscore"}})
    with pytest.raises(Exception, match="normalization"):
        run(monkeypatch, config, "biophys", run_dir / "output")


# ---------------------------------------------------------------------------
# named provider inputs
# ---------------------------------------------------------------------------


def test_a_named_input_reaches_the_provider(monkeypatch, tmp_path):
    """Cluster mode's features file is outside the output directory."""
    elsewhere = tmp_path / "user_inputs"
    elsewhere.mkdir()
    features = elsewhere / "my_features.tsv"
    features.write_text("protid\tSequence\nP1\tMKKKAAAEEE\n")

    output = tmp_path / "output"
    output.mkdir()
    config = write_config(tmp_path, {"biophys": {"provider": "biophys"}})

    code = run(
        monkeypatch,
        config,
        "biophys",
        output,
        "--provider-input",
        f"features_file={features}",
    )
    assert code == 0
    assert (output / "blocks" / "biophys" / "protids.txt").read_text().split() == ["P1"]


def test_a_named_input_does_not_enter_the_cache_key(monkeypatch, tmp_path):
    """Otherwise the same run from two directories caches differently.

    The path is a fact about this machine; the file's content is the fact about
    the block. Only the digest belongs in the manifest.
    """
    keys = []
    for name in ("run_one", "run_two"):
        root = tmp_path / name
        (root / "elsewhere").mkdir(parents=True)
        features = root / "elsewhere" / "features.tsv"
        features.write_text("protid\tSequence\nP1\tMKKKAAAEEE\n")
        output = root / "output"
        output.mkdir()
        config = write_config(root, {"biophys": {"provider": "biophys"}})
        run(monkeypatch, config, "biophys", output, "--provider-input", f"features_file={features}")
        keys.append(manifest_of(output, "biophys")["cache_key"])
    assert keys[0] == keys[1]


@pytest.mark.parametrize("bad", ["features_file", "=/path", "  =/path"])
def test_a_malformed_provider_input_is_rejected(bad):
    with pytest.raises(SystemExit, match="NAME=PATH"):
        compute_block.parse_provider_inputs([bad])


def test_a_path_containing_an_equals_sign_is_kept_whole():
    parsed = compute_block.parse_provider_inputs(["features_file=/tmp/a=b/f.tsv"])
    assert parsed == {"features_file": "/tmp/a=b/f.tsv"}


# ---------------------------------------------------------------------------
# the rest of the entry point
# ---------------------------------------------------------------------------


def test_an_undefined_block_id_lists_the_defined_ones(monkeypatch, run_dir):
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    with pytest.raises(SystemExit, match="biophys"):
        run(monkeypatch, config, "nonesuch", run_dir / "output")


def test_an_unavailable_provider_records_a_skip_rather_than_failing(monkeypatch, run_dir, capsys):
    """ADR 0006: a missing optional dependency costs you the block and no more."""
    from spaces.registry import BLOCK_GROUP, register_builtin

    class Unavailable:
        def is_available(self):
            return False, "pip install nothing"

    register_builtin(BLOCK_GROUP, "unavailable_for_test", Unavailable)
    config = write_config(run_dir, {"gone": {"provider": "unavailable_for_test"}})

    assert run(monkeypatch, config, "gone", run_dir / "output") == 0
    skip = json.loads((run_dir / "output" / "blocks" / "gone" / "SKIPPED.json").read_text())
    assert skip["reason"].endswith("pip install nothing") or "pip install nothing" in skip["reason"]
    assert "skipping" in capsys.readouterr().err


def test_the_builtin_providers_all_register(monkeypatch, run_dir):
    """A provider added to `blocks/` and not to `_register_builtins` is invisible."""
    from spaces.registry import BLOCK_GROUP, list_providers

    compute_block._register_builtins()
    registered = set(list_providers(BLOCK_GROUP))
    assert {"tmscore", "threedi", "biophys", "domains"} <= registered


def test_a_providers_spec_schema_is_called_before_anything_is_computed(monkeypatch, run_dir):
    """ADR 0010: `spec_schema` is the parameter contract and the FRAMEWORK calls it.

    It was bound by all four built-ins and called by nothing, and that was
    invisible because each of the four calls its own `validate_params` at the
    top of `compute`. So the provider here validates ONLY through `spec_schema`
    and asserts inside `compute` -- which is the third-party case the contract
    exists for, and the only one that can tell the two arrangements apart.
    """
    from spaces.registry import BLOCK_GROUP, register_builtin

    class Fussy:
        @staticmethod
        def spec_schema(params):
            if params.get("mode") != "ok":
                raise ValueError("fussy.mode must be 'ok'")
            return dict(params)

        def is_available(self):
            return True, ""

        def compute(self, ctx, params):
            raise AssertionError("compute ran on parameters spec_schema rejects")

    register_builtin(BLOCK_GROUP, "fussy_for_test", Fussy)
    config = write_config(run_dir, {"f": {"provider": "fussy_for_test", "mode": "no"}})
    with pytest.raises(ValueError, match="fussy.mode"):
        run(monkeypatch, config, "f", run_dir / "output")


def test_a_provider_without_a_spec_schema_still_computes(monkeypatch, run_dir):
    """The other half of ADR 0006: the hook is optional, like `plan`."""
    from spaces.base import BlockResult, BlockSpec
    from spaces.registry import BLOCK_GROUP, register_builtin

    class Bare:
        def is_available(self):
            return True, ""

        def compute(self, ctx, params):
            import numpy as np

            return BlockResult(
                spec=BlockSpec(
                    id=params["block_id"],
                    kind="features",
                    fusable=True,
                    metric="euclidean",
                    normalization="zscore_within",
                    provider="bare_for_test",
                ),
                protids=["P1", "P2"],
                features=np.zeros((2, 1), dtype=float),
            )

    register_builtin(BLOCK_GROUP, "bare_for_test", Bare)
    config = write_config(run_dir, {"bare": {"provider": "bare_for_test"}})
    assert run(monkeypatch, config, "bare", run_dir / "output") == 0


# ---------------------------------------------------------------------------
# PC-012 phase 1 -- the freshness check that could never fire (FOLLOWUPS #27)
# ---------------------------------------------------------------------------


def test_a_second_run_over_unchanged_inputs_skips_and_says_so(monkeypatch, run_dir, capsys):
    """The defect, end to end.

    `expected` used to be built with `protids=[]` and no `inputs`, so its
    `cache_key` could not equal any written one and `is_fresh` was always False.
    Every block recomputed on every invocation snakemake allowed through.
    """
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    output = run_dir / "output"

    assert run(monkeypatch, config, "biophys", output) == 0
    assert "wrote 'biophys'" in capsys.readouterr().err

    assert run(monkeypatch, config, "biophys", output) == 0
    assert "is up to date" in capsys.readouterr().err


def test_force_still_recomputes(monkeypatch, run_dir, capsys):
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    output = run_dir / "output"
    run(monkeypatch, config, "biophys", output)
    capsys.readouterr()
    run(monkeypatch, config, "biophys", output, "--force")
    assert "wrote 'biophys'" in capsys.readouterr().err


def test_a_changed_input_is_not_skipped(monkeypatch, run_dir, capsys):
    """Content hashes, never mtimes -- so this has to be a real edit."""
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    output = run_dir / "output"
    run(monkeypatch, config, "biophys", output)
    capsys.readouterr()

    features = run_dir / "output" / "protein_features" / "uniprot_features.tsv"
    features.write_text(features.read_text() + "P3\tMKKAAAEEEGGGWWW\n")
    run(monkeypatch, config, "biophys", output)
    assert "wrote 'biophys'" in capsys.readouterr().err


def test_a_changed_param_is_not_skipped(monkeypatch, run_dir, capsys):
    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    output = run_dir / "output"
    run(monkeypatch, config, "biophys", output)
    capsys.readouterr()

    changed = write_config(run_dir, {"biophys": {"provider": "biophys", "ph": 6.5}})
    run(monkeypatch, changed, "biophys", output)
    assert "wrote 'biophys'" in capsys.readouterr().err


@pytest.mark.parametrize("provider_name", ["biophys", "threedi", "domains", "tmscore"])
def test_every_builtin_providers_plan_agrees_with_what_it_writes(
    monkeypatch, tmp_path, provider_name
):
    """The exit criterion, and the thing that makes the skip safe.

    `plan()` and `compute()` must agree on `Manifest.input_key`. They come
    through one `_manifest` per provider so they cannot drift, and this asserts
    it for each of the four rather than trusting the shape.

    They must ALSO disagree on `cache_key`, which is the fact the whole design
    rests on: `cache_key` folds in `extra`, `plan` cannot know `extra`, and that
    is why `is_fresh` compares `input_key` instead.
    """
    import compute_block as cb
    from blocks.tmscore import PipelineContext
    from spaces.manifest import Manifest
    from spaces.registry import BLOCK_GROUP, get_provider

    cb._register_builtins()
    output = tmp_path / "output"
    features = output / "protein_features"
    features.mkdir(parents=True)
    (features / "uniprot_features.tsv").write_text(
        "protid\tSequence\tPfam\n"
        "P1\tMDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQ\tPF00022\n"
        "P2\tMKKKKKAAAAAEEEEEGGGGGWWWWW\tPF00022;PF00023\n"
    )
    # `foldseek structureto3didescriptor` output: four tab-separated fields,
    # no header -- name, amino acids, 3Di, coordinates. `read_descriptors`
    # refuses anything else rather than guessing, which is how the first draft
    # of this fixture made `threedi.plan` return None.
    (features / "3di_descriptors.tsv").write_text(
        "P1.pdb\tMDDDIAALV\tDVQAVCVVD\t0,0,0\nP2.pdb\tMKKKKKAAA\tQQVVDDAAC\t0,0,0\n"
    )
    clustering = output / "foldseek_clustering_results"
    clustering.mkdir(parents=True)
    (clustering / "all_by_all_tmscore_pivoted.tsv").write_text(
        "protid\tP1\tP2\nP1\t1.000E+00\t7.000E-01\nP2\t7.000E-01\t1.000E+00\n"
    )

    provider = get_provider(BLOCK_GROUP, provider_name)
    ctx = PipelineContext(output_dir=str(output))
    params = provider.spec_schema({})
    params["block_id"] = provider_name

    planned = provider.plan(ctx, params)
    assert planned is not None, f"{provider_name}.plan returned None on a complete input tree"
    written = provider.compute(ctx, params).manifest
    if isinstance(written, dict):
        written = Manifest.from_dict(written)

    assert (
        planned.input_key == written.input_key
    ), f"{provider_name}: plan and compute disagree about their inputs"
    assert planned.protids_digest == written.protids_digest
    assert planned.inputs == written.inputs
    assert planned.cache_key != written.cache_key, (
        f"{provider_name}: plan and compute agree on cache_key, which would mean this "
        "provider records nothing it learned while computing -- check `extra`"
    )


def test_a_provider_without_a_plan_still_works(monkeypatch, run_dir, capsys):
    """ADR 0006: a third-party provider that never heard of `plan` must keep
    working, and must simply recompute every time as it did before."""
    import compute_block as cb
    from spaces.registry import BLOCK_GROUP, get_provider

    config = write_config(run_dir, {"biophys": {"provider": "biophys"}})
    output = run_dir / "output"
    run(monkeypatch, config, "biophys", output)
    capsys.readouterr()

    cb._register_builtins()
    provider = get_provider(BLOCK_GROUP, "biophys")
    monkeypatch.delattr(type(provider), "plan")
    run(monkeypatch, config, "biophys", output)
    assert "wrote 'biophys'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# PC-012 phase 2 -- a pinned vocabulary arrives as an input, not as a param
# ---------------------------------------------------------------------------


def _vocabulary_run(tmp_path, families=("PF00022",)):
    output = tmp_path / "output"
    features = output / "protein_features"
    features.mkdir(parents=True)
    (features / "uniprot_features.tsv").write_text("protid\tPfam\nP1\tPF00022\nP2\tPF99999\nP3\t\n")
    vocabulary = tmp_path / "families.txt"
    vocabulary.write_text(
        "# a pinned vocabulary can carry its own provenance\n" + "\n".join(families) + "\n"
    )
    return output, vocabulary


def test_a_pinned_vocabulary_reaches_the_provider_and_is_not_a_param(monkeypatch, tmp_path):
    """The rule `test_a_named_input_does_not_enter_the_cache_key` already pins
    for `features_file`, asserted again for the one that decides a block's
    COLUMNS: identity rests on the file's digest, never on where it sits."""
    output, vocabulary = _vocabulary_run(tmp_path)
    config = write_config(
        tmp_path, {"domains": {"provider": "domains", "vocabulary_file": str(vocabulary)}}
    )
    assert run(monkeypatch, config, "domains", output) == 0

    manifest = manifest_of(output, "domains")
    assert manifest["inputs"]["vocabulary"].startswith("sha256:")
    assert str(vocabulary) not in json.dumps(manifest["params"]), (
        "the vocabulary PATH reached params, which is hashed into the cache key -- "
        "the same vocabulary in two directories would be two different blocks"
    )
    assert manifest["extra"]["vocabulary_pinned"] == {"name": "families.txt", "n_tokens": 1}
    # And the block was actually built over the pinned column set.
    assert manifest["extra"]["n_families"] == 1
    assert manifest["extra"]["proteins_annotated_outside_vocabulary"] == ["P2"]


def test_moving_the_vocabulary_file_does_not_change_the_cache_key(monkeypatch, tmp_path):
    """Identity is the digest. Two runs of the same tokens from two paths must
    be the same block."""
    import shutil

    output, vocabulary = _vocabulary_run(tmp_path)
    config = write_config(
        tmp_path, {"domains": {"provider": "domains", "vocabulary_file": str(vocabulary)}}
    )
    run(monkeypatch, config, "domains", output)
    first = manifest_of(output, "domains")["cache_key"]

    elsewhere = tmp_path / "somewhere else" / "families.txt"
    elsewhere.parent.mkdir()
    shutil.copy(vocabulary, elsewhere)
    moved = write_config(
        tmp_path, {"domains": {"provider": "domains", "vocabulary_file": str(elsewhere)}}
    )
    second_out = tmp_path / "output2"
    (second_out / "protein_features").mkdir(parents=True)
    shutil.copy(
        output / "protein_features" / "uniprot_features.tsv",
        second_out / "protein_features" / "uniprot_features.tsv",
    )
    run(monkeypatch, moved, "domains", second_out)
    assert manifest_of(second_out, "domains")["cache_key"] == first


def test_changing_the_vocabulary_contents_does_change_the_cache_key(monkeypatch, tmp_path):
    """The other half. A different column set is a different block, and the
    freshness check added in phase 1 has to see that."""
    output, vocabulary = _vocabulary_run(tmp_path)
    config = write_config(
        tmp_path, {"domains": {"provider": "domains", "vocabulary_file": str(vocabulary)}}
    )
    run(monkeypatch, config, "domains", output)
    first = manifest_of(output, "domains")["cache_key"]

    vocabulary.write_text("PF00022\nPF99999\n")
    run(monkeypatch, config, "domains", output, "--force")
    assert manifest_of(output, "domains")["cache_key"] != first


def test_pinning_nothing_leaves_the_manifest_exactly_as_it_was(monkeypatch, tmp_path):
    """The keys are ABSENT rather than null when no vocabulary is named, so a
    run that pins nothing has the manifest -- and therefore the cache key -- it
    had before any of this existed."""
    output, _ = _vocabulary_run(tmp_path)
    config = write_config(tmp_path, {"domains": {"provider": "domains"}})
    run(monkeypatch, config, "domains", output)
    manifest = manifest_of(output, "domains")
    assert "vocabulary" not in manifest["inputs"]
    assert "vocabulary_pinned" not in manifest["extra"]


@pytest.mark.parametrize(
    ("contents", "match"),
    [("", "no tokens in it"), ("# only a comment\n\n", "no tokens in it")],
)
def test_an_unusable_vocabulary_is_refused_by_name(monkeypatch, tmp_path, contents, match):
    """A vocabulary that silently came back empty would give every protein an
    all-zero row, which is a result that looks like a finding."""
    from spaces.base import VocabularyError, read_vocabulary

    path = tmp_path / "empty.txt"
    path.write_text(contents)
    with pytest.raises(VocabularyError, match=match) as excinfo:
        read_vocabulary(path)
    assert str(path) in str(excinfo.value), "the message must name the file"


def test_a_missing_vocabulary_file_is_refused_by_name(tmp_path):
    from spaces.base import VocabularyError, read_vocabulary

    with pytest.raises(VocabularyError, match="does not exist"):
        read_vocabulary(tmp_path / "nope.txt")


def test_a_vocabulary_keeps_file_order_and_drops_duplicates(tmp_path):
    """File ORDER, not sorted: the vocabulary is the block's column order, and
    sorting it here would reorder every column against the file the user wrote.
    Deduplicated because a repeated token would give one feature two columns."""
    from spaces.base import read_vocabulary

    path = tmp_path / "v.txt"
    path.write_text("PF00099\n# comment\n\nPF00011\nPF00099\n")
    assert read_vocabulary(path) == ["PF00099", "PF00011"]


def test_a_kmer_vocabulary_that_does_not_match_k_is_refused(monkeypatch, tmp_path):
    """Every token would miss, every protein would get an all-zero row, and the
    only signal would be an out-of-vocabulary fraction of 1.0 that nobody reads."""
    output = tmp_path / "output"
    features = output / "protein_features"
    features.mkdir(parents=True)
    (features / "3di_descriptors.tsv").write_text(
        "P1.pdb\tMDDDIAALV\tABCABCABC\t0,0,0\nP2.pdb\tMKKKKKAAA\tAABBCCAAB\t0,0,0\n"
    )
    vocabulary = tmp_path / "kmers.txt"
    vocabulary.write_text("ABC\nBCA\n")
    config = write_config(
        tmp_path,
        {"threedi": {"provider": "threedi", "k": 2, "vocabulary_file": str(vocabulary)}},
    )
    with pytest.raises(Exception, match="are not 2 characters long"):
        run(monkeypatch, config, "threedi", output)
