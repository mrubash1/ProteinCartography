#!/usr/bin/env python
"""Leiden clustering over a ProteinCartography TM-score matrix.

For N < 3, assign every protein to a single cluster and skip Scanpy. Otherwise
clamp ``n_neighbors`` / ``n_pcs`` to valid ranges for the matrix size.

**Optionally records what it ran with.** ``--manifest`` writes a sidecar JSON
naming the input and its digest, the parameters as REQUESTED and as USED after
clamping, the library versions, and the resolution -- which is scanpy's default
because this module never passes one, and is therefore the parameter most likely
to differ between two runs that look identical from the command line.

That option exists because four archived ``leiden_features.tsv`` files in this
project could not be regenerated from their own inputs, and none of them carried
any record of the settings that produced them, so the disagreement could not be
diagnosed -- only observed (FOLLOWUPS #78).

It is OPT-IN, and that is deliberate rather than timid: this script runs on the
default output path, so writing a file unconditionally would add an output to
the default DAG and the parity suite would see it as a differing file.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

__all__ = ["scanpy_leiden_cluster"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input file path of a similarity matrix."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path to a file, usually leiden_features.tsv",
    )
    parser.add_argument(
        "-n",
        "--n-neighbors",
        default="10",
        help="Number of n_neighbors to pass to sc.pp.neighbors().",
    )
    parser.add_argument(
        "-c",
        "--n-pcs",
        default="30",
        help="Number of n_pcs to pass to sc.pp.neighbors().",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional path for a sidecar JSON recording the input digest and the "
            "parameters actually used. Omitted by the default pipeline, so the "
            "default DAG's outputs are unchanged."
        ),
    )
    parser.add_argument(
        "-l",
        "--cluster-name",
        default="LeidenCluster",
        help="Name of cluster column. Defaults to 'LeidenCluster'.",
    )
    parser.add_argument(
        "-a",
        "--cluster-abbrev",
        default="LC",
        help="Abbreviation to add as prefix for cluster labels. Defaults to 'LC'.",
    )
    args = parser.parse_args()
    return args


def _singleton_membership(
    protids: list[str], cluster_name: str, cluster_abbrev: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protid": protids,
            cluster_name: [f"{cluster_abbrev}0"] * len(protids),
        }
    )


def _module_version(name: str):
    """The version of an installed library, however that library spells it.

    `leidenalg` carries its version as `leidenalg.version` and has no
    `__version__` at all, so asking only for the dunder records null for the
    library that chose the partition. `scipy` and `igraph` are read for the same
    reason: FOLLOWUPS #42 names scipy as the leading hypothesis for two
    environments returning different Leiden partitions from one matrix, and a
    manifest that omits it cannot settle that.
    """
    try:
        module = __import__(name)
    except Exception:
        return None
    for attr in ("__version__", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return None


def _write_manifest(path, *, input_file, requested, used, cluster_name, n_proteins, n_clusters):
    """What this run actually did, beside its own output.

    The USED values matter more than the requested ones: `n_neighbors` is raised
    to `round(n/10)` and both are clamped to the matrix, so two runs invoked
    identically on different-sized cohorts do different things (FOLLOWUPS #79).
    `resolution` is recorded as scanpy's default because this module never
    passes one -- and it is the parameter that most plausibly explains an
    archive nobody can reproduce.
    """
    import json
    import platform

    versions = {"python": platform.python_version()}
    for name in ("scanpy", "leidenalg", "igraph", "scipy", "sklearn", "numpy", "pandas"):
        versions[name] = _module_version(name)
    try:
        from spaces.manifest import file_digest

        digest = file_digest(input_file)
    except Exception:
        digest = None
    payload = {
        "input": {"path": os.path.abspath(input_file), "sha256": digest},
        "requested": requested,
        "used": used,
        "resolution": "scanpy default (this module passes none)",
        "cluster_column": cluster_name,
        "n_proteins": n_proteins,
        "n_clusters": n_clusters,
        "versions": versions,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[leiden_clustering] wrote {path}", flush=True)


def scanpy_leiden_cluster(
    input_file: str,
    savefile=None,
    n_neighbors=10,
    n_pcs=30,
    cluster_name="LeidenCluster",
    cluster_abbrev="LC",
    manifest_path=None,
    **kwargs,
):
    """
    Uses Scanpy's Leiden clustering implementation to perform clustering.

    Args:
        input_file (str): path of input distances matrix.
        savefile (str): path of destination file.
        n_neighbors (int): number of neighbors for clustering. Defaults to 10.
        n_pcs (int): number of PCs to use for initial PCA.
        **kwargs are passed to `sc.pp.neighbors()`.
    """
    adata = sc.read_csv(input_file, delimiter="\t")
    n = int(adata.n_obs)
    if n < 3:
        print(
            f"[leiden_clustering] N={n} too small for Leiden; "
            f"assigning all proteins to {cluster_abbrev}0"
        )
        membership = _singleton_membership(list(adata.obs_names), cluster_name, cluster_abbrev)
        if savefile is not None:
            membership.to_csv(savefile, sep="\t", index=None)
        if manifest_path is not None:
            _write_manifest(
                manifest_path,
                input_file=input_file,
                requested={"n_neighbors": int(n_neighbors), "n_pcs": int(n_pcs)},
                used={"n_neighbors": None, "n_pcs": None, "path": "singleton, N<3"},
                cluster_name=cluster_name,
                n_proteins=n,
                n_clusters=1,
            )
        return membership

    sc.tl.pca(adata, svd_solver="arpack")

    n_neighbors = int(n_neighbors)
    n_pcs = int(n_pcs)
    n_neighbors_recommended = int(np.round(n / 10))
    n_neighbors_used = max(n_neighbors, n_neighbors_recommended)
    # Scanpy requires 1 < n_neighbors <= N - 1 for a connected graph.
    n_neighbors_used = max(2, min(n_neighbors_used, n - 1))
    max_pcs = max(1, min(n - 1, adata.n_vars - 1 if adata.n_vars > 1 else 1))
    n_pcs_used = max(1, min(n_pcs, max_pcs))

    sc.pp.neighbors(adata, n_neighbors=n_neighbors_used, n_pcs=n_pcs_used, **kwargs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    membership = pd.DataFrame(adata.obs["leiden"]).reset_index()
    membership.rename(columns={"index": "protid", "leiden": cluster_name}, inplace=True)
    max_chars = len(str(membership[cluster_name].astype(int).max()))
    membership[cluster_name] = cluster_abbrev + membership[cluster_name].apply(
        lambda x: str(x).zfill(max_chars)
    ).astype(str)

    if savefile is not None:
        membership.to_csv(savefile, sep="\t", index=None)

    if manifest_path is not None:
        _write_manifest(
            manifest_path,
            input_file=input_file,
            requested={"n_neighbors": int(n_neighbors), "n_pcs": int(n_pcs)},
            used={"n_neighbors": n_neighbors_used, "n_pcs": n_pcs_used},
            cluster_name=cluster_name,
            n_proteins=n,
            n_clusters=int(membership[cluster_name].nunique()),
        )

    return membership


def main():
    args = parse_args()
    n_neighbors = int(args.n_neighbors)
    n_pcs = int(args.n_pcs)

    scanpy_leiden_cluster(
        args.input,
        args.output,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        cluster_name=args.cluster_name,
        cluster_abbrev=args.cluster_abbrev,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
