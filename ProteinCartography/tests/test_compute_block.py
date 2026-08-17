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
    assert {"tmscore", "threedi", "biophys"} <= registered
