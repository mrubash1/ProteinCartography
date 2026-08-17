#!/usr/bin/env python
"""Reduce one space to 2-D coordinates.

The snakemake entry point for a space. Reads the space's block(s) from the
store, derives a feature matrix, and runs the requested reducer through
:mod:`spaces.reducers.core` -- the same code the legacy ``dim_reduction.py``
path calls, so the two cannot drift.

Only ``strategy: none`` is implemented here. Multi-block fusion is a later
commit; a space that asks for it fails with a message saying so rather than
silently reducing one of its blocks and producing a map that looks right.
"""

from __future__ import annotations
import argparse
import json
import os

from config_io import load_config
from config_schema import from_legacy
from index import ProteinIndex
from spaces.manifest import Manifest
from spaces.reducers.core import reduce_pca, reduce_tsne, reduce_umap
from spaces.store import BlockStore

#: The legacy `plotting_modes` vocabulary, as reducer pipelines. Each entry is
#: the sequence of steps to apply; `pca_umap` means PCA-30 then UMAP, which is
#: what the pipeline has always done.
REDUCER_PIPELINES = {
    "pca": ("pca",),
    "tsne": ("tsne",),
    "umap": ("umap",),
    "pca_tsne": ("pca30", "tsne"),
    "pca_umap": ("pca30", "umap"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--configfile", required=True)
    parser.add_argument("-s", "--space-id", required=True)
    parser.add_argument(
        "-r", "--reducer", required=True, help=f"one of {sorted(REDUCER_PIPELINES)}"
    )
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument("--random-state", default="123456")
    return parser.parse_args()


def _features_for(space, store: BlockStore, config):
    """The feature matrix a space reduces, plus its protid index."""
    if len(space.blocks) != 1 or space.strategy != "none":
        raise SystemExit(
            f"space {space.id!r} uses strategy {space.strategy!r} over "
            f"{len(space.blocks)} blocks, and multi-block fusion is not implemented "
            "yet. Reducing one of its blocks and calling the result the space would "
            "produce a map that looks correct and is not, so this is an error rather "
            "than a fallback."
        )

    block_id = space.blocks[0]
    directory = store.block_dir(block_id)
    if os.path.exists(os.path.join(directory, "SKIPPED.json")):
        with open(os.path.join(directory, "SKIPPED.json")) as fh:
            skip = json.load(fh)
        raise SystemExit(
            f"space {space.id!r} needs block {block_id!r}, which was skipped: "
            f"{skip.get('reason', 'no reason recorded')}"
        )

    result = store.read_block(block_id)
    if result.features is None:
        raise SystemExit(
            f"block {block_id!r} is a pairwise block. Reducing a precomputed "
            "distance needs a metric-aware reducer, which is not implemented yet; "
            "use representation: profile."
        )
    index = ProteinIndex.from_iterable(result.protids)
    return result, index


def main() -> int:
    args = parse_args()
    config = from_legacy(load_config(args.configfile))

    if args.space_id not in config.spaces:
        raise SystemExit(
            f"space {args.space_id!r} is not defined. "
            f"Defined spaces: {config.space_ids() or '(none)'}"
        )
    if args.reducer not in REDUCER_PIPELINES:
        raise SystemExit(
            f"reducer {args.reducer!r} is not valid. "
            f"Allowed: {', '.join(sorted(REDUCER_PIPELINES))}."
        )

    space = config.spaces[args.space_id]
    store = BlockStore(args.output_dir)
    block, index = _features_for(space, store, config)

    try:
        random_state = int(args.random_state)
    except ValueError:
        random_state = 123456

    values = block.features
    protids = list(index)
    steps = REDUCER_PIPELINES[args.reducer]
    provenance = []

    for step in steps:
        if step == "pca30":
            result = reduce_pca(values, protids, n_components=30, random_state=random_state)
        elif step == "pca":
            result = reduce_pca(values, protids, random_state=random_state)
        elif step == "umap":
            result = reduce_umap(
                values,
                protids,
                random_state=random_state,
                input_column_names=provenance[-1]["column_names"] if provenance else None,
            )
        elif step == "tsne":
            result = reduce_tsne(values, protids, random_state=random_state)
        else:  # pragma: no cover - REDUCER_PIPELINES is closed
            raise SystemExit(f"unknown reduction step {step!r}")
        values = result.coordinates
        provenance.append({**result.params_used, "column_names": list(result.column_names)})

    space_dir = os.path.join(args.output_dir, "spaces", args.space_id)
    os.makedirs(space_dir, exist_ok=True)

    embedding_path = os.path.join(space_dir, "embedding_" + args.reducer + ".tsv")
    result.to_frame().to_csv(embedding_path, sep="\t")

    manifest = Manifest.build(
        "space",
        args.space_id,
        provider="reduce_space",
        params={
            "reducer": args.reducer,
            "strategy": space.strategy,
            "blocks": list(space.blocks),
        },
        inputs={"block:" + block.spec.id: block.manifest.get("cache_key", "")},
        protids=protids,
        seed=random_state,
        extra={"steps": provenance},
    )
    manifest.write(os.path.join(space_dir, "manifest_" + args.reducer + ".json"))

    print(
        f"[reduce_space] wrote {embedding_path} ({len(protids)} proteins, "
        f"{len(result.column_names)} dimensions)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
