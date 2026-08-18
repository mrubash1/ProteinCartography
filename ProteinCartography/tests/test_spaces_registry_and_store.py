"""Tests for provider discovery and the on-disk block store."""

import json
import os

import numpy as np
import pytest
from spaces import registry
from spaces.base import BlockResult, BlockSpec
from spaces.manifest import Manifest, hash_params, protids_digest, values_digest
from spaces.store import BlockStore, StoreError


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear_builtins()
    yield
    registry.clear_builtins()


class FakeProvider:
    spec_schema = dict

    def __init__(self, available=True, reason=""):
        self._available = available
        self._reason = reason

    def is_available(self):
        return self._available, self._reason

    def compute(self, ctx, params):  # pragma: no cover - not exercised here
        raise NotImplementedError


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_builtin_registration_and_lookup():
    registry.register_builtin(registry.BLOCK_GROUP, "fake", FakeProvider)
    provider = registry.get_provider(registry.BLOCK_GROUP, "fake")
    assert isinstance(provider, FakeProvider)


def test_list_providers_reports_source():
    registry.register_builtin(registry.BLOCK_GROUP, "fake", FakeProvider)
    listing = registry.list_providers(registry.BLOCK_GROUP)
    assert listing["fake"] == "builtin"


def test_unknown_provider_lists_the_alternatives():
    registry.register_builtin(registry.BLOCK_GROUP, "tmscore", FakeProvider)
    registry.register_builtin(registry.BLOCK_GROUP, "biophys", FakeProvider)
    with pytest.raises(registry.ProviderNotFoundError) as excinfo:
        registry.get_provider(registry.BLOCK_GROUP, "tmscroe")
    message = str(excinfo.value)
    assert "Unknown provider 'tmscroe'" in message
    # A typo should be obvious from the error alone.
    assert "tmscore" in message
    assert "biophys" in message


def test_empty_group_gives_a_diagnostic_not_a_bare_keyerror():
    with pytest.raises(registry.ProviderNotFoundError) as excinfo:
        registry.get_provider(registry.BLOCK_GROUP, "anything")
    assert "No providers are registered" in str(excinfo.value)


def test_unavailable_provider_raises_with_its_own_reason():
    registry.register_builtin(
        registry.BLOCK_GROUP,
        "plm",
        lambda: FakeProvider(available=False, reason="pip install fair-esm"),
    )
    with pytest.raises(registry.ProviderUnavailableError) as excinfo:
        registry.get_provider(registry.BLOCK_GROUP, "plm")
    assert "pip install fair-esm" in str(excinfo.value)


def test_require_available_false_returns_the_provider_anyway():
    """The Snakefile surveys before it computes; a survey must not raise."""
    registry.register_builtin(
        registry.BLOCK_GROUP, "plm", lambda: FakeProvider(available=False, reason="no torch")
    )
    provider = registry.get_provider(registry.BLOCK_GROUP, "plm", require_available=False)
    assert provider.is_available() == (False, "no torch")


def test_available_providers_survey_does_not_raise():
    registry.register_builtin(registry.BLOCK_GROUP, "ok", FakeProvider)
    registry.register_builtin(
        registry.BLOCK_GROUP, "nope", lambda: FakeProvider(False, "missing weights")
    )
    infos = {i.name: i for i in registry.available_providers(registry.BLOCK_GROUP)}
    assert infos["ok"].available
    assert not infos["nope"].available
    assert "missing weights" in infos["nope"].reason


def test_survey_survives_a_provider_whose_check_explodes():
    class Exploding:
        def is_available(self):
            raise RuntimeError("boom")

    registry.register_builtin(registry.BLOCK_GROUP, "bad", Exploding)
    infos = {i.name: i for i in registry.available_providers(registry.BLOCK_GROUP)}
    assert not infos["bad"].available
    assert "boom" in infos["bad"].reason


def test_unknown_group_rejected():
    with pytest.raises(ValueError, match="unknown provider group"):
        registry.list_providers("proteincartography.nonsense")


def test_entry_point_discovery_shadows_builtins(monkeypatch):
    """Entry points win, so a downstream package can replace a default."""

    class FakeEntryPoint:
        name = "tmscore"

        def load(self):
            return lambda: "from-entry-point"

    registry.register_builtin(registry.BLOCK_GROUP, "tmscore", FakeProvider)
    monkeypatch.setattr(registry, "_iter_entry_points", lambda group: [FakeEntryPoint()])

    assert registry.list_providers(registry.BLOCK_GROUP)["tmscore"] == "entry_point"
    assert registry.get_provider(registry.BLOCK_GROUP, "tmscore") == "from-entry-point"


def test_entry_point_that_fails_to_import_is_explained(monkeypatch):
    class BrokenEntryPoint:
        name = "plm"

        def load(self):
            raise ImportError("No module named 'torch'")

    monkeypatch.setattr(registry, "_iter_entry_points", lambda group: [BrokenEntryPoint()])
    with pytest.raises(registry.ProviderUnavailableError) as excinfo:
        registry.get_provider(registry.BLOCK_GROUP, "plm")
    message = str(excinfo.value)
    assert "failed to import" in message
    assert "torch" in message


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def test_manifest_cache_key_is_stable_across_key_order():
    a = hash_params({"x": 1, "y": 2})
    b = hash_params({"y": 2, "x": 1})
    assert a == b


def test_manifest_cache_key_changes_with_params():
    m1 = Manifest.build("block", "b", params={"k": 1}, protids=["A"])
    m2 = Manifest.build("block", "b", params={"k": 2}, protids=["A"])
    assert m1.cache_key != m2.cache_key


def test_manifest_cache_key_changes_with_cohort():
    m1 = Manifest.build("block", "b", protids=["A", "B"])
    m2 = Manifest.build("block", "b", protids=["A", "B", "C"])
    assert m1.cache_key != m2.cache_key


def test_protids_digest_is_order_sensitive():
    """The protid list *is* the index, so its order is part of its identity."""
    assert protids_digest(["A", "B"]) != protids_digest(["B", "A"])


def test_values_digest_includes_dtype():
    a = np.zeros(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float64)
    assert values_digest(a) != values_digest(b)


def test_values_digest_includes_shape():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros((4,), dtype=np.float32)
    assert values_digest(a) != values_digest(b)


def test_manifest_round_trip(tmp_path):
    manifest = Manifest.build(
        "block", "tmscore", provider="tmscore", params={"rep": "profile"}, protids=["A", "B"]
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = Manifest.read(path)
    assert loaded.matches(manifest)
    assert loaded.id == "tmscore"
    assert loaded.n_proteins == 2


def test_manifest_records_versions_and_environment():
    manifest = Manifest.build("block", "b", protids=["A"])
    assert "numpy" in manifest.versions
    assert manifest.versions["numpy"] is not None
    assert "python" in manifest.environment


def test_cache_key_ignores_environment_but_not_versions():
    """Same versions on a different machine should hit the cache."""
    m1 = Manifest.build("block", "b", protids=["A"])
    m2 = Manifest.build("block", "b", protids=["A"])
    m2.environment = {"python": "9.9.9", "platform": "elsewhere"}
    assert m1.cache_key == m2.cache_key

    m2.versions = dict(m2.versions, numpy="0.0.0")
    assert m1.cache_key != m2.cache_key


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def spec(**overrides):
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


def test_block_round_trip_pairwise(tmp_path):
    store = BlockStore(str(tmp_path))
    protids = ["A", "B", "C"]
    distances = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.write_block(BlockResult(spec=spec(), protids=protids, distances=distances))

    loaded = store.read_block("tmscore")
    assert loaded.protids == protids
    np.testing.assert_allclose(loaded.distances, distances)
    assert loaded.spec.id == "tmscore"
    assert loaded.spec.metric == "precomputed"


def test_block_round_trip_features_with_mask(tmp_path):
    store = BlockStore(str(tmp_path))
    features = np.arange(6, dtype=np.float64).reshape(3, 2)
    mask = np.zeros((3, 2), dtype=bool)
    mask[0, 0] = True
    store.write_block(
        BlockResult(
            spec=spec(id="biophys", kind="features", metric="euclidean", symmetrization=None),
            protids=["A", "B", "C"],
            features=features,
            mask=mask,
        )
    )
    loaded = store.read_block("biophys")
    # Stored as float32 per ADR 0004.
    assert loaded.features.dtype == np.float32
    np.testing.assert_allclose(loaded.features, features, rtol=1e-6)
    np.testing.assert_array_equal(loaded.mask, mask)
    assert loaded.censoring_rate == pytest.approx(1 / 6)


def test_protids_file_is_plain_text(tmp_path):
    """A human should be able to see what a block covers without loading numpy."""
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B", "C"], distances=np.zeros(3)))
    text = (tmp_path / "blocks" / "tmscore" / "protids.txt").read_text()
    assert text == "A\nB\nC\n"


def test_manifest_written_beside_the_arrays(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)))
    data = json.loads((tmp_path / "blocks" / "tmscore" / "manifest.json").read_text())
    assert data["kind"] == "block"
    assert data["id"] == "tmscore"
    assert "cache_key" in data
    assert "values_digest" in data["derived"]
    assert data["derived"]["spec"]["metric"] == "precomputed"


def test_the_providers_manifest_reaches_disk(tmp_path):
    """Regression: `write_block` used to rebuild a minimal manifest and drop it.

    Everything a provider records about what it computed *from* -- input
    digests, the seed, its own notes -- lived only on `result.manifest`, and
    `write_block` replaced it with one built from the spec alone. Every block on
    disk therefore had `inputs: {}`, which meant a changed input produced an
    unchanged `cache_key` and a stale block looked fresh.
    """
    store = BlockStore(str(tmp_path))
    provider_manifest = Manifest.build(
        "block",
        "tmscore",
        provider="tmscore",
        params={"representation": "profile"},
        inputs={"similarity_matrix": "sha256:abc"},
        protids=["A", "B"],
        seed=99,
        extra={"censoring": {"rate": 0.6}},
    )
    store.write_block(
        BlockResult(
            spec=spec(),
            protids=["A", "B"],
            distances=np.zeros(1),
            manifest=provider_manifest.to_dict(),
        )
    )
    data = json.loads((tmp_path / "blocks" / "tmscore" / "manifest.json").read_text())
    assert data["inputs"] == {"similarity_matrix": "sha256:abc"}
    assert data["seed"] == 99
    assert data["extra"] == {"censoring": {"rate": 0.6}}
    # Still stamped with the output-derived facts, which the provider cannot know.
    assert "values_digest" in data["derived"]


def test_an_input_change_changes_the_cache_key_of_a_written_block(tmp_path):
    """The consequence of the above, stated as the property that was broken."""
    store = BlockStore(str(tmp_path))
    keys = []
    for digest in ("sha256:aaa", "sha256:bbb"):
        manifest = Manifest.build(
            "block", "tmscore", provider="tmscore", inputs={"m": digest}, protids=["A", "B"]
        )
        store.write_block(
            BlockResult(
                spec=spec(), protids=["A", "B"], distances=np.zeros(1), manifest=manifest.to_dict()
            )
        )
        keys.append(
            json.loads((tmp_path / "blocks" / "tmscore" / "manifest.json").read_text())["cache_key"]
        )
    assert keys[0] != keys[1]


def test_an_explicit_manifest_still_wins_over_the_results_own(tmp_path):
    store = BlockStore(str(tmp_path))
    explicit = Manifest.build("block", "tmscore", protids=["A", "B"], extra={"from": "caller"})
    carried = Manifest.build("block", "tmscore", protids=["A", "B"], extra={"from": "provider"})
    store.write_block(
        BlockResult(
            spec=spec(), protids=["A", "B"], distances=np.zeros(1), manifest=carried.to_dict()
        ),
        explicit,
    )
    data = json.loads((tmp_path / "blocks" / "tmscore" / "manifest.json").read_text())
    assert data["extra"] == {"from": "caller"}


def test_manifest_from_dict_round_trips_and_ignores_the_cache_key():
    manifest = Manifest.build("block", "x", provider="p", inputs={"a": "1"}, protids=["A"])
    restored = Manifest.from_dict(manifest.to_dict())
    assert restored.cache_key == manifest.cache_key
    assert restored.inputs == {"a": "1"}


def test_manifest_from_dict_drops_unknown_keys():
    """So a manifest written by a newer version stays readable."""
    data = Manifest.build("block", "x", protids=["A"]).to_dict()
    data["a_field_from_the_future"] = 1
    assert Manifest.from_dict(data).id == "x"


def test_freshness_is_content_based(tmp_path):
    store = BlockStore(str(tmp_path))
    manifest = Manifest.build("block", "tmscore", params={"v": 1}, protids=["A", "B"])
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)), manifest)

    assert store.is_fresh("tmscore", manifest)

    changed = Manifest.build("block", "tmscore", params={"v": 2}, protids=["A", "B"])
    assert not store.is_fresh("tmscore", changed)


def test_freshness_ignores_mtime(tmp_path):
    store = BlockStore(str(tmp_path))
    manifest = Manifest.build("block", "tmscore", protids=["A", "B"])
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)), manifest)
    target = tmp_path / "blocks" / "tmscore" / "distances.npy"
    os.utime(target, (0, 0))
    assert store.is_fresh("tmscore", manifest)


def test_missing_block_is_not_fresh(tmp_path):
    store = BlockStore(str(tmp_path))
    assert not store.is_fresh("nope", Manifest.build("block", "nope", protids=["A"]))


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)))
    assert not os.path.exists(str(tmp_path / "blocks" / "tmscore.tmp"))
    assert store.list_blocks() == ["tmscore"]


def test_rewrite_replaces_cleanly(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B", "C"], distances=np.zeros(3)))
    loaded = store.read_block("tmscore")
    assert loaded.protids == ["A", "B", "C"]
    assert loaded.distances.shape == (3,)


def test_reading_an_absent_block_raises(tmp_path):
    store = BlockStore(str(tmp_path))
    with pytest.raises(StoreError, match="No stored block"):
        store.read_block("nope")


def test_block_without_protids_is_refused(tmp_path):
    """Without labels the rows have no meaning, so this must not load."""
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)))
    os.remove(str(tmp_path / "blocks" / "tmscore" / "protids.txt"))
    with pytest.raises(StoreError, match="no meaning"):
        store.read_block("tmscore")


def test_invalidate_removes_the_block(tmp_path):
    store = BlockStore(str(tmp_path))
    store.write_block(BlockResult(spec=spec(), protids=["A", "B"], distances=np.zeros(1)))
    store.invalidate("tmscore")
    assert not store.has_block("tmscore")
    assert store.list_blocks() == []


def test_list_blocks_on_empty_store(tmp_path):
    assert BlockStore(str(tmp_path)).list_blocks() == []


def test_two_manifests_do_not_share_one_versions_dict():
    """The safeguard `package_versions` argues for at length, actually tested.

    Gate E replaced the immutable cached tuple with a single shared mutable
    dict and the whole suite stayed green -- the existing test rebinds
    (`m2.versions = dict(...)`) rather than mutating, so it could not see it.
    `Manifest.versions` is a plain mutable field, and one caller mutating its
    own manifest in place would otherwise rewrite the provenance recorded by
    every other manifest in the process.
    """
    from spaces.manifest import package_versions

    first, second = package_versions(), package_versions()
    assert first == second
    assert first is not second
    first["numpy"] = "0.0.0-mutated"
    assert second.get("numpy") != "0.0.0-mutated"
    assert package_versions().get("numpy") != "0.0.0-mutated"
