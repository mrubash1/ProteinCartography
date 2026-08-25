"""The Snakefile must PARSE in an environment with none of the heavy stack.

This exists because prose did not hold the invariant. `config_utils` carried a
docstring stating that `SET_MODES` was imported inside a function so that the
Snakefile kept no parse-time dependency on the HTTP stack. The Snakefile calls
that function while it is being parsed, so the deferral achieved nothing, and
`snakemake -n` failed outright in any environment without `bioservices`:

    ModuleNotFoundError in file Snakefile, line 167: No module named 'bioservices'
      Snakefile:167 -> config_utils.py:162 -> foldseek_apiquery.py:7 -> api_utils.py:4

Measured in an environment built from `envs/cartography_test.yml`, which is the
one BOTH CI workflows use: HEAD exited 1, the fork point exited 0. Every DAG dry
run this branch had ever reported was green because the local driver environment
happens to carry `bioservices`, so nothing in the suite could see it.

Two things follow about how this is written.

**It runs in a SUBPROCESS.** The failure is snakemake parsing a file, not pytest
importing a module, and `conftest.py`'s `sys.modules` stub -- which fixes the
collection half of the same problem -- cannot reach a subprocess. A test that
checked the import inside this process would pass while the real thing stayed
broken.

**It blocks the modules rather than requiring a bare environment.** A `skipif`
on "is sklearn absent" would skip in every environment anyone actually develops
in, and a guard that never executes is not a guard. `sitecustomize.py` is
imported automatically by any Python that starts with it on `PYTHONPATH`, so the
child sees the modules as missing however rich the parent environment is.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _why_no_snakemake():
    """``None`` when snakemake actually RUNS, else the reason it does not.

    Not `shutil.which("snakemake") is None`, which is the idiom the rest of the
    suite uses and which is wrong on any machine with pyenv: `which` finds
    `~/.pyenv/shims/snakemake`, the shim then prints "pyenv: snakemake: command
    not found" and exits non-zero, and the guard concludes snakemake is present.
    The test then fails for a reason it does not exist to catch. That happened
    twice on this branch in one day -- once producing a report of "10 failures"
    against a tree whose battery was green -- so this probe runs the thing.
    """
    exe = shutil.which("snakemake")
    if exe is None:
        return "needs snakemake"
    try:
        probe = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    except OSError as exc:  # pragma: no cover - depends on the machine
        return f"snakemake at {exe} is not executable: {exc}"
    if probe.returncode != 0:
        return (
            f"snakemake at {exe} does not run (exit {probe.returncode}): "
            f"{(probe.stderr or probe.stdout).strip()[:120]}"
        )
    return None


_WHY_NO_SNAKEMAKE = _why_no_snakemake()

#: What `envs/cartography_test.yml` does not ship. `bioservices` is the module
#: that actually broke; the rest are here because ADR 0006 rule 3 says the
#: framework must work with zero optional dependencies, and a parse that reached
#: any of them would break CI in the same way for a different reason.
BLOCKED = ("bioservices", "sklearn", "umap", "scipy", "scanpy")

_SITECUSTOMIZE = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}


    class _Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in BLOCKED else None

        def load_module(self, name):
            raise ImportError(f"blocked by the test: {{name}}")

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"blocked by the test: {{name}}")
            return None


    sys.meta_path.insert(0, _Blocker())
    for _name in list(sys.modules):
        if _name.split(".")[0] in BLOCKED:
            del sys.modules[_name]
    """
)


def _env_with_blocker(tmp_path):
    """A child environment in which `BLOCKED` cannot be imported.

    `ProteinCartography/` goes on the path too, because the package's modules
    import each other flat (`import config_utils`, not
    `ProteinCartography.config_utils`) and that is normally supplied by
    `pyproject.toml`'s `pythonpath`, which only applies under pytest. A
    subprocess gets nothing from it.
    """
    (tmp_path / "sitecustomize.py").write_text(_SITECUSTOMIZE.format(blocked=BLOCKED))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(tmp_path), str(REPO_ROOT / "ProteinCartography"), env.get("PYTHONPATH"))
        if part
    )
    return env


def test_the_blocker_actually_blocks(tmp_path):
    """The guard's own guard. If `sitecustomize` silently failed to load -- a
    typo, a PYTHONPATH that did not take -- every assertion below would pass by
    proving nothing, which is the failure mode this whole file exists to close.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import bioservices"],
        cwd=str(REPO_ROOT),
        env=_env_with_blocker(tmp_path),
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0, "sitecustomize did not load; the rest of this file is vacuous"
    assert "blocked by the test" in probe.stderr


def test_config_utils_imports_without_the_heavy_stack(tmp_path):
    """The module the Snakefile reaches at parse time, on its own.

    Cheaper than the dry run below and it fails first, so a regression here is
    read as what it is rather than as a snakemake problem.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config_utils; print(config_utils._get_foldseek_mode({}))",
        ],
        cwd=str(REPO_ROOT),
        env=_env_with_blocker(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3diaa"


@pytest.mark.skipif(_WHY_NO_SNAKEMAKE is not None, reason=_WHY_NO_SNAKEMAKE or "")
@pytest.mark.parametrize(
    "configfile",
    ["demo/cluster-mode/config.yml", "demo/search-mode/config_actin_small.yml"],
)
def test_the_default_dags_resolve_without_the_heavy_stack(tmp_path, configfile):
    """The end-to-end claim, in the shape the failure actually took.

    Both DEFAULT configs, because these are the two upstream CI resolves and the
    two the "default DAGs are unchanged" job compares against the fork point.
    """
    result = subprocess.run(
        ["snakemake", "--configfile", configfile, "-n", "--quiet", "--rerun-incomplete"],
        cwd=str(REPO_ROOT),
        env=_env_with_blocker(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`snakemake -n` could not resolve {configfile} without "
        f"{', '.join(BLOCKED)}:\n{result.stderr[-2000:]}"
    )
    assert "ModuleNotFoundError" not in result.stderr
