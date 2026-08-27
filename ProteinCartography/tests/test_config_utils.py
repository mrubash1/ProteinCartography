import pytest

from ProteinCartography import config_utils


def make_input_dir(tmp_path, num_pdb_files, protid_prefix="protein"):
    """
    Create an input directory containing `num_pdb_files` (empty) PDB files
    and return the protids of those files.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)

    protids = [f"{protid_prefix}{index}" for index in range(num_pdb_files)]
    for protid in protids:
        (input_dir / f"{protid}.pdb").touch()
    return protids


def make_cluster_mode_config(tmp_path, key_protids=None):
    return {
        "mode": "cluster",
        "input_dir": str(tmp_path / "input"),
        "key_protids": key_protids if key_protids is not None else [],
    }


@pytest.mark.parametrize("num_pdb_files", [0, 1])
def test_cluster_mode_requires_at_least_one_pair(tmp_path, num_pdb_files):
    """
    A cohort too small to contain a single pair is not clusterable, so it must fail.
    """
    make_input_dir(tmp_path, num_pdb_files)
    config = make_cluster_mode_config(tmp_path)

    with pytest.raises(config_utils.ProteinCartographyInputError) as exc_info:
        config_utils._get_protids(config)

    # the error must name the real constraint (pairwise comparison) and the actual count,
    # rather than an arbitrary minimum.
    message = str(exc_info.value)
    assert "pairwise" in message
    assert f"found {num_pdb_files}" in message


@pytest.mark.parametrize("num_pdb_files", [2, 3, 9, 11])
def test_cluster_mode_allows_small_cohorts(tmp_path, num_pdb_files):
    """
    Cohorts smaller than the old hard-coded minimum of ten must be allowed to run;
    the components with real floors (leiden clustering, dimensionality reduction)
    degrade on their own terms for N < 3.
    """
    make_input_dir(tmp_path, num_pdb_files)
    config = make_cluster_mode_config(tmp_path)

    search_mode_input_protids, key_protids = config_utils._get_protids(config)
    assert search_mode_input_protids == []
    assert key_protids == []


@pytest.mark.parametrize("num_pdb_files", [2, 9])
def test_cluster_mode_warns_about_small_cohorts(tmp_path, capsys, num_pdb_files):
    """
    A cohort below the advisory size runs, but says so on stderr.
    """
    make_input_dir(tmp_path, num_pdb_files)
    config = make_cluster_mode_config(tmp_path)

    config_utils._get_protids(config)

    captured = capsys.readouterr()
    assert f"only {num_pdb_files} PDB files" in captured.err
    assert captured.out == ""


def test_cluster_mode_does_not_warn_about_large_cohorts(tmp_path, capsys):
    make_input_dir(tmp_path, 10)
    config = make_cluster_mode_config(tmp_path)

    config_utils._get_protids(config)

    captured = capsys.readouterr()
    assert captured.err == ""


def test_cluster_mode_key_protids_must_be_a_subset(tmp_path):
    """
    The key-protid check must still apply to small cohorts.
    """
    make_input_dir(tmp_path, 3)
    config = make_cluster_mode_config(tmp_path, key_protids=["not-an-input-protein"])

    with pytest.raises(config_utils.ProteinCartographyInputError):
        config_utils._get_protids(config)


def test_cluster_mode_returns_key_protids(tmp_path):
    protids = make_input_dir(tmp_path, 3)
    config = make_cluster_mode_config(tmp_path, key_protids=[protids[0]])

    search_mode_input_protids, key_protids = config_utils._get_protids(config)
    assert search_mode_input_protids == []
    assert key_protids == [protids[0]]


def test_search_mode_requires_a_fasta_file(tmp_path):
    """
    Search mode is unaffected by the cluster-mode floor: PDB files alone are not enough.
    """
    make_input_dir(tmp_path, 3)
    config = {"mode": "search", "input_dir": str(tmp_path / "input")}

    with pytest.raises(config_utils.ProteinCartographyInputError):
        config_utils._get_protids(config)


def test_search_mode_accepts_a_single_fasta_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "P60709.fasta").touch()
    config = {"mode": "search", "input_dir": str(input_dir)}

    search_mode_input_protids, key_protids = config_utils._get_protids(config)
    assert search_mode_input_protids == ["P60709"]
    assert key_protids == ["P60709"]
