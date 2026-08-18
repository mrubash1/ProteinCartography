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
    evidence behind the backwards-compatibility claim, so CI runs it on every
    pull request.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def synthetic_matrix_750(tmp_path_factory):
    """The default 750-protein similarity matrix, generated once per session.

    `test_determinism.py`, `test_clustering.py` and `test_parity.py` each build
    an identical copy under ``--runslow``, at 203.8 ms a time; copying the file
    is 0.9 ms. The bytes are a pure function of the generator's seed, so one
    copy is the same fixture three times.

    Deliberately a fixture rather than a memo inside `parity.synthetic_matrix`:
    `test_parity.py::test_the_synthetic_fixture_is_reproducible` compares the
    bytes of two calls with the same arguments, and memoising the function would
    reduce that to comparing one cached object with itself. Same trap as
    `docs/FOLLOWUPS.md`-adjacent R12 in the test-speed plan -- a cache that
    makes a test pass by making it vacuous.

    The import is inside the body because the tests directory is not on
    ``sys.path`` when this root-level conftest is first imported.
    """
    from parity import synthetic_matrix

    return synthetic_matrix(
        tmp_path_factory.mktemp("synthetic_750") / "all_by_all_tmscore_pivoted.tsv"
    )
