import os
import pathlib
import shutil

import pandas as pd
import pytest
import snakemake
import yaml


def _load_config(filepath):
    """
    Convenience function to load a yaml config file.
    """
    with open(filepath) as file:
        config = yaml.safe_load(file)
    return config


@pytest.fixture
def config_filepath(tmp_path):
    """
    Generate a config file for testing the pipeline.
    """
    config = {
        "mode": "search",
        "analysis_name": "test",
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
        "plotting_modes": ["pca_umap"],
        "max_blast_hits": 10,
        "max_foldseek_hits": 10,
        "max_structures": 10,
    }

    filepath = tmp_path / "config.yaml"
    with open(filepath, "w") as file:
        yaml.dump(config, file)

    return str(filepath)


@pytest.fixture
def stage_inputs(integration_test_artifacts_dirpath, config_filepath):
    """
    Create the input directory and input files for the pipeline.
    """

    config = _load_config(config_filepath)

    # For now, hard-code the dataset name.
    dataset_name = "actin"
    shutil.copytree(
        integration_test_artifacts_dirpath / "search-mode" / dataset_name / "input",
        config["input_dir"],
    )


@pytest.fixture
def set_env_variables(pytestconfig):
    """
    Set the env variable used to mock the API responses
    made by the python scripts that are called by the snakemake rules.

    Note: this works because the rule environments inherit their env variables from the environment
    in which `snakemake` was called (which is this pytest python process).
    """

    # The names of the proteincartography-specific env variables.
    should_use_mocks = "PROTEINCARTOGRAPHY_SHOULD_USE_MOCKS"
    should_log_api_requests = "PROTEINCARTOGRAPHY_SHOULD_LOG_API_REQUESTS"

    using_mocks = not pytestconfig.getoption("no_mocks")
    if using_mocks:
        os.environ[should_use_mocks] = "true"

    # Don't log API requests during the tests.
    should_log_api_requests_value = os.environ.pop(should_log_api_requests, None)

    # `foldseek_apiquery.py` waits 30 s between polls of the public Foldseek
    # server, and this test's DAG runs it. Under mocks the ticket is answered
    # instantly, so that is pure wall clock -- about 30 s of this file's ~47 s.
    # `parity.py` removes it for the parity runs without editing any source
    # file, by handing the child processes a `usercustomize` module; the same
    # mechanism works here, and for the same reason the comment above gives:
    # rule environments inherit this process's environment.
    #
    # **Gated on mocks, and that is not a detail.** With `--no-mocks` this test
    # polls the real server, where the 30 s wait is politeness toward a shared
    # public resource rather than dead time. The cluster-mode test is left alone
    # entirely for the same reason: it has no mocking fixture at all, and its
    # `key_protids` pulls `run_foldseek` into the DAG, so its sleep is always
    # against the real server.
    # Both halves are inside the gate. An earlier version popped
    # PYTHONNOUSERSITE unconditionally and gated only the redirect -- three
    # lines below the paragraph above -- which under `--no-mocks` re-enabled the
    # machine's real ~/.local site-packages on sys.path for every rule
    # interpreter, and any usercustomize.py living there auto-executes. That is
    # the isolation "optional deps stay optional" depends on, in the run closest
    # to production. Fourth instance of a comment stating an invariant it does
    # not enforce; Gate E's adversarial pass found it.
    previous_user_base = os.environ.get("PYTHONUSERBASE")
    previous_no_user_site = None
    if using_mocks:
        from parity import foldseek_sleep_user_base

        previous_no_user_site = os.environ.pop("PYTHONNOUSERSITE", None)
        os.environ["PYTHONUSERBASE"] = str(foldseek_sleep_user_base())

    yield

    os.environ.pop(should_use_mocks, None)
    if previous_user_base is None:
        os.environ.pop("PYTHONUSERBASE", None)
    else:
        os.environ["PYTHONUSERBASE"] = previous_user_base
    if previous_no_user_site is not None:
        os.environ["PYTHONNOUSERSITE"] = previous_no_user_site

    # As a convenience, restore the logging env variable to its original value.
    if should_log_api_requests_value is not None:
        os.environ[should_log_api_requests] = should_log_api_requests_value


@pytest.mark.usefixtures("stage_inputs")
@pytest.mark.usefixtures("set_env_variables")
def test_pipeline_in_search_mode_with_mocked_api_calls(repo_dirpath, config_filepath):
    """
    Run the pipeline in "search" mode with the test config file, the temporary snakefile,
    and mocked API calls.
    """
    snakemake.snakemake(
        snakefile=(repo_dirpath / "Snakefile"),
        configfiles=[config_filepath],
        use_conda=True,
        cores=8,
        verbose=True,
    )

    config = _load_config(config_filepath)
    input_dirpath = pathlib.Path(config["input_dir"])
    output_dirpath = pathlib.Path(config["output_dir"])

    # Check (some of) the expected output files.
    expected_output_filepaths = [
        output_dirpath / "final_results" / f"{config['analysis_name']}_{appendix}"
        for appendix in [
            "leiden_similarity.html",
            "strucluster_similarity.html",
            "semantic_analysis.pdf",
            "semantic_analysis.html",
        ]
    ]
    for filepath in expected_output_filepaths:
        # TODO (KC): Check that the content of the files looks correct.
        # (not sure we can do a literal comparison because of timestamps, umap stochasticity, etc.)
        assert filepath.exists()

    # Check that the shape of the all-by-all similarity matrix is correct: the
    # structures foldseek clustered are the `max_structures` the cohort admits
    # plus the query structures staged as input.
    #
    # Both terms are derived, not written out. They were `11` and `12` here and
    # a bare `+ 1` in `test_parity.py`, so adding a second query structure to
    # the fixture, or changing `max_structures` in the config above, would have
    # failed this test in a way that says nothing about the pipeline.
    # `test_pipeline_in_cluster_mode.py` already counts its inputs this way.
    num_structures = config["max_structures"] + len(list(input_dirpath.glob("*.pdb")))
    similarity_matrix_filepath = (
        output_dirpath / "foldseek_clustering_results" / "all_by_all_tmscore_pivoted.tsv"
    )
    similarity_matrix = pd.read_csv(similarity_matrix_filepath, sep="\t")
    # One row and one column per structure, plus one column for the index.
    assert similarity_matrix.shape == (num_structures, num_structures + 1)

    domain_html_name = f"{config['analysis_name']}_leiden_similarity_domain.html"
    domain_html = output_dirpath / "final_results" / domain_html_name
    assert not domain_html.exists()
    domain_structs = output_dirpath / "domain_path" / "domain_structures"
    assert not domain_structs.exists() or not any(domain_structs.glob("*.pdb"))
