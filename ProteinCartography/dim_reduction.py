#!/usr/bin/env python
"""Dimensionality reduction for ProteinCartography similarity matrices.

Clamps UMAP/t-SNE parameters for small N and falls back to a PCA/identity
layout when N < 3 so tiny maps still produce coordinates.

The numerical work lives in :mod:`spaces.reducers.core`, which the multi-space
machinery also calls. This module keeps its CLI, its filenames and its return
types exactly as they were and is otherwise a thin layer over that shared core.

The sharing is the point. Two implementations of "the pipeline's PCA" -- one for
the legacy map and one for a new space -- would eventually disagree, and the
disagreement would surface as a scientific difference with no obvious cause.
`ProteinCartography/tests/test_parity.py` checks byte-for-byte that this
refactor changed nothing.
"""

from __future__ import annotations
import argparse

import pandas as pd

# Snakemake runs this as `python ProteinCartography/dim_reduction.py`, which puts
# the package directory on the path and makes the flat import the working one.
# The fallback is for `from ProteinCartography import dim_reduction`, which the
# README advertises and which worked at the commit this branch forked from: the
# flat form raises `ModuleNotFoundError: spaces` there, and this module -- with
# its `__all__` of calculate_PCA/TSNE/UMAP -- is the one most likely to be used
# as a library. It was the only import regression on the branch.
try:
    from spaces.reducers.core import reduce_pca, reduce_tsne, reduce_umap
except ModuleNotFoundError:  # pragma: no cover - exercised by test_packaging
    from ProteinCartography.spaces.reducers.core import (
        reduce_pca,
        reduce_tsne,
        reduce_umap,
    )

__all__ = ["calculate_PCA", "calculate_TSNE", "calculate_UMAP"]

MODES = ["pca", "tsne", "umap", "pca_tsne", "pca_umap"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--output-prefix", help="Prefix for resulting .tsv files.")
    parser.add_argument(
        "-m",
        "--mode",
        default="pca",
        help=f"Mode of dimensionality reduction.\nValid arguments are {MODES}.",
    )
    parser.add_argument(
        "-r",
        "--random-state",
        default="123456",
        help="Random state for umap and tsne modes.",
    )
    return parser.parse_args()


def _save_path(pivot_file: str, saveprefix: str | None, dimtype: str) -> str:
    if saveprefix is not None:
        return "_".join([saveprefix, dimtype + ".tsv"])
    return pivot_file.replace(".tsv", "_" + dimtype + ".tsv")


def calculate_PCA(
    pivot_file: str,
    random_state: int,
    n_components=2,
    save=False,
    saveprefix=None,
    dimtype="pca",
    prep_step=False,
    **kwargs,
):
    pivoted_df = pd.read_csv(pivot_file, sep="\t", index_col="protid")
    result = reduce_pca(
        pivoted_df.to_numpy(),
        list(pivoted_df.index),
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )
    pca_results_df = pd.DataFrame(
        result.coordinates,
        columns=result.column_names,
        index=pivoted_df.index,
    )

    savefile = _save_path(pivot_file, saveprefix, dimtype)
    if save:
        pca_results_df.to_csv(savefile, sep="\t")
    if prep_step:
        return savefile
    return pca_results_df


def calculate_TSNE(
    pivot_file: str,
    random_state: int,
    n_components=2,
    perplexity=50,
    n_iter=2000,
    save=False,
    saveprefix=None,
    dimtype="tsne",
    **kwargs,
):
    pivoted_df = pd.read_csv(pivot_file, sep="\t", index_col="protid")
    result = reduce_tsne(
        pivoted_df.to_numpy(),
        list(pivoted_df.index),
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        **kwargs,
    )
    tsne_results_df = pd.DataFrame(
        result.coordinates,
        columns=result.column_names,
        index=pivoted_df.index,
    )

    savefile = _save_path(pivot_file, saveprefix, dimtype)
    if save:
        tsne_results_df.to_csv(savefile, sep="\t")
    return tsne_results_df


def calculate_UMAP(
    pivot_file: str,
    random_state: int,
    n_components=2,
    n_neighbors=80,
    min_dist=0.5,
    save=False,
    saveprefix=None,
    dimtype="umap",
    **kwargs,
):
    pivoted_df = pd.read_csv(pivot_file, sep="\t", index_col="protid")
    result = reduce_umap(
        pivoted_df.to_numpy(),
        list(pivoted_df.index),
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        # Below N=3 the fallback reuses this matrix's existing PC columns rather
        # than running PCA on PCA output, which is what the pre-port code did.
        input_column_names=list(pivoted_df.columns),
        **kwargs,
    )
    umap_results_df = pd.DataFrame(
        result.coordinates,
        columns=result.column_names,
        index=pivoted_df.index,
    )

    savefile = _save_path(pivot_file, saveprefix, dimtype)
    if save:
        umap_results_df.to_csv(savefile, sep="\t")
    return umap_results_df


def main():
    args = parse_args()
    pivot_file = args.input
    saveprefix = args.output_prefix
    mode = args.mode.lower()

    try:
        random_state = int(args.random_state)
    except Exception:
        random_state = 123456

    if mode not in MODES:
        raise Exception(f"{mode} provided is not valid.\nValid modes include {MODES}.")

    if mode == "pca":
        calculate_PCA(pivot_file, random_state, save=True, saveprefix=saveprefix)
    elif mode == "tsne":
        calculate_TSNE(pivot_file, random_state, save=True, saveprefix=saveprefix)
    elif mode == "umap":
        calculate_UMAP(pivot_file, random_state, save=True, saveprefix=saveprefix)
    elif mode == "pca_tsne":
        saveprefix1 = pivot_file.replace(".tsv", "temp1")
        pca_results_file = calculate_PCA(
            pivot_file,
            random_state,
            save=True,
            saveprefix=saveprefix1,
            n_components=30,
            prep_step=True,
        )
        saveprefix2 = pca_results_file.replace("temp1", "").replace(".tsv", "")
        calculate_TSNE(pca_results_file, random_state, save=True, saveprefix=saveprefix2)
    elif mode == "pca_umap":
        saveprefix1 = pivot_file.replace(".tsv", "temp2")
        pca_results_file = calculate_PCA(
            pivot_file,
            random_state,
            save=True,
            saveprefix=saveprefix1,
            n_components=30,
            prep_step=True,
        )
        saveprefix2 = pca_results_file.replace("temp2", "").replace(".tsv", "")
        calculate_UMAP(pca_results_file, random_state, save=True, saveprefix=saveprefix2)


if __name__ == "__main__":
    main()
