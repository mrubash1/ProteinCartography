import pytest

from ProteinCartography import file_utils


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
