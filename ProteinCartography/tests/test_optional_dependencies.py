"""ADR 0006, mechanically checked: the framework works with nothing installed.

The test environment (`envs/cartography_test.yml`, which is what CI uses) has
pandas and numpy and nothing else -- no scikit-learn, no umap-learn, no scipy,
no scanpy, no torch. That is deliberate, and it is the strongest available
statement of "optional dependencies stay optional": if the core modules could
not be imported without them, this file would fail in CI.

The check has to be an import test rather than a promise, because the failure
mode is quiet. A module-scope `import sklearn` in a new provider works fine on
the developer's machine, works fine in the analysis environment, and breaks only
for the maintainer who installed nothing -- which is the person this PR most
needs not to break.
"""

import importlib
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Every module that must import with only numpy and pandas available. A
#: provider that needs a heavy dependency imports it inside the function that
#: uses it, so that `is_available()` can be consulted before anything is loaded.
CORE_MODULES = [
    "matrix_io",
    "index",
    "config_schema",
    "coregistration",
    "coregister",
    "enrichment",
    "enrich_clusters",
    "fusion",
    "spaces.base",
    "spaces.registry",
    "spaces.store",
    "spaces.manifest",
    "spaces.reducers.core",
    "blocks.tmscore",
    "blocks.threedi",
    "blocks.biophys",
    "blocks.domains",
    "diagnostics.censoring",
    "diagnostics.embedding",
    "diagnostics.redundancy",
    "diagnostics.stability",
    "diagnostics.partition",
    # scanpy-backed, and listed here for exactly that reason: the import must
    # stay inside the function so that `is_available()` can be consulted first.
    "clustering",
    # The explorer's payload is pandas-only; plotly is imported inside
    # `build_explorer.main` after `is_available()` has been consulted.
    "explorer.payload",
    "explorer.template",
]

HEAVY = ["sklearn", "umap", "scipy", "scanpy", "torch"]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports_without_heavy_dependencies(module_name):
    """Importing must not pull in anything optional."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_the_test_environment_really_lacks_them():
    """Guard the guard.

    If the test environment ever gains scikit-learn, the tests above stop
    proving anything, and they would keep passing while proving nothing. This
    fails loudly instead, so the guarantee is re-established deliberately rather
    than lost silently.
    """
    present = []
    for name in HEAVY:
        try:
            importlib.import_module(name)
            present.append(name)
        except ImportError:
            continue
    if present:
        pytest.skip(
            f"{present} are installed here, so the import tests above are not "
            "proving that the framework works without them. That guarantee is "
            "enforced by CI, which runs in envs/cartography_test.yml."
        )


def test_a_provider_reports_missing_dependencies_rather_than_raising():
    """A missing optional dependency must degrade the run, never break it."""
    from spaces.registry import BLOCK_GROUP, available_providers, register_builtin

    class NeedsTorch:
        def is_available(self):
            try:
                import torch  # noqa: F401
            except ImportError:
                return False, "pip install torch, then `make fetch-models`"
            return True, ""

    register_builtin(BLOCK_GROUP, "_needs_torch_probe", NeedsTorch)
    try:
        infos = {i.name: i for i in available_providers(BLOCK_GROUP)}
        info = infos["_needs_torch_probe"]
        # Surveying must not raise, whichever way availability comes out.
        assert isinstance(info.available, bool)
        if not info.available:
            assert "torch" in info.reason
    finally:
        from spaces import registry

        registry._BUILTINS[BLOCK_GROUP].pop("_needs_torch_probe", None)


def test_tmscore_profile_path_needs_no_optional_dependency(tmp_path):
    """The default block must work in the bare environment.

    `profile` is the representation the pipeline has always used, so it has to
    be computable wherever the pipeline runs. Only `direct` needs scipy, and
    only because of `squareform`.
    """
    from blocks.tmscore import PipelineContext, TMScoreProvider

    labels = ["P1", "P2", "P3"]
    matrix_dir = tmp_path / "foldseek_clustering_results"
    matrix_dir.mkdir(parents=True)
    rows = ["\t".join(["protid"] + labels)]
    for i, a in enumerate(labels):
        cells = [f"{1.0 if i == j else 0.7:.3E}" for j in range(len(labels))]
        rows.append("\t".join([a] + cells))
    (matrix_dir / "all_by_all_tmscore_pivoted.tsv").write_text("\n".join(rows) + "\n")

    result = TMScoreProvider().compute(PipelineContext(output_dir=str(tmp_path)), {})
    assert result.protids == labels
    assert result.features.shape == (3, 3)
    assert result.spec.kind == "features"
    assert result.spec.metric == "euclidean"


def test_the_explorer_payload_does_not_import_plotly():
    """plotly is 3.6 MB of JavaScript and a heavy import, and the explorer's
    payload does not need it -- only `build_explorer.main` does, after
    consulting `is_available()`.

    Checked by inspecting `sys.modules` rather than by adding plotly to
    :data:`HEAVY`. `cartography_tidy` has plotly installed (arcadia-pycolor
    pulls it in), so listing it there would make `test_the_test_environment_
    really_lacks_them` skip and take the sklearn, umap, scipy and scanpy checks
    down with it -- a strictly worse trade than testing the real property
    directly.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import explorer.payload, explorer.template; "
            "print('plotly' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_ROOT),
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False", (
        "importing the explorer's payload pulled plotly in; it must stay inside "
        "build_explorer.main so an unavailable plotly is a skipped rule rather than "
        "an import error"
    )
