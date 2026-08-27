import sys
from unittest.mock import MagicMock

import pytest

from ProteinCartography import file_utils

# `api_utils.py:4` is `from bioservices import UniProt`, at module scope, and
# `envs/cartography_test.yml` -- the environment BOTH CI workflows use -- does
# not ship bioservices. `test_packaging.py:158` records upstream's own awareness
# of that, and stubs the module out for the one subprocess that needs it.
#
# `make test` is `pytest -vv -s .` from the repo root, so pytest imports test
# modules in sorted order. Upstream survived because the only module reaching
# that import was `test_fetch_accession.py`, which stubs bioservices itself at
# its line 8, before importing anything. This branch added
# `test_config_schema.py`, which reaches the same import through
# `foldseek_apiquery` -- and sorts BEFORE `test_fetch_accession`. Collection
# therefore died on the real import: measured in an environment built from
# `envs/cartography_test.yml`, HEAD exited 2 with "Interrupted: 1 error during
# collection" and ran zero tests, while the same command in `../pc-baseline` at
# the fork point exited 0. A green suite on a developer machine that happens to
# have bioservices installed says nothing about either.
#
# The stub lives HERE rather than at the top of `test_config_schema.py` because
# a per-file stub fixes the one file and not the class: the next test module
# that sorts early and reaches `api_utils` reintroduces it. `setdefault` leaves
# a real bioservices alone where one is installed, so this changes nothing for
# anybody who has the package.
sys.modules.setdefault("bioservices", MagicMock())


@pytest.fixture(scope="session", autouse=True)
def repo_dirpath():
    return file_utils.find_repo_dirpath()


@pytest.fixture
def integration_test_artifacts_dirpath(repo_dirpath):
    return repo_dirpath / "ProteinCartography" / "tests" / "integration-test-artifacts"


def pytest_addoption(parser):
    """
    Add custom CLI options for pytest
    """
    parser.addoption(
        "--no-mocks",
        action="store_true",
        default=False,
        help="Run tests without mocks",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help=(
            "Run tests marked `slow`. These run the pipeline end to end, several "
            "times, and take minutes rather than seconds."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests marked `slow` unless `--runslow` was given.

    The parity tests run the whole pipeline four times to compare this branch
    against the baseline, which is far too slow for an edit-test loop but is the
    evidence behind the backwards-compatibility claim.

    That evidence is only worth something if something runs it. `make test` does
    not pass `--runslow`, so the job that does is the `parity` job in
    `.github/workflows/multispace.yml`, which also creates the baseline worktree
    the fixtures need and fails if they skipped for want of it. This docstring
    previously claimed CI ran the suite on every pull request while no workflow
    passed the flag at all.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
