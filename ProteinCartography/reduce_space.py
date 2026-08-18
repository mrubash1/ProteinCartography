#!/usr/bin/env python
"""Reduce one space to 2-D coordinates.

The snakemake entry point for a space. Reads the space's block(s) from the
store, derives a feature matrix, and runs the requested reducer through
:mod:`spaces.reducers.core` -- the same code the legacy ``dim_reduction.py``
path calls, so the two cannot drift.

All four strategies of ADR 0002 run through here: `none` for a single block and
`early`, `late` and `graph` for several. The combining itself is
:mod:`fusion`, which is pure computation; this module reads the blocks, puts
them on one protein index, hands them over, and records what came back.

**A multi-block space is co-registered before it is fused.** The blocks are
produced by different rules from different files, so their protein sets can
differ -- a structure with no UniProt record, a sequence Foldseek could not
fold. Fusion combines row `i` of each block as one protein and cannot detect
that they are not the same protein, so the index is the intersection and every
dropped protein is named. Same reasoning, and the same refusal to hide the loss,
as :mod:`coregistration`.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

from config_io import load_config
from config_schema import from_legacy
from fusion import FusionError, FusionInput, fuse
from index import IndexAlignmentError, ProteinIndex
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


def read_blocks(space, store: BlockStore) -> list:
    """Every block a space needs, or a message saying which one it cannot have."""
    results = []
    for block_id in space.blocks:
        directory = store.block_dir(block_id)
        skipped = os.path.join(directory, "SKIPPED.json")
        if os.path.exists(skipped):
            with open(skipped) as fh:
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
        results.append(result)
    return results


def shared_index(space, blocks) -> tuple:
    """The proteins every block in this space has, in the first block's order.

    Returns ``(index, dropped)``, where `dropped` maps a block id to the
    proteins it measured that some other block did not. The order comes from the
    first block listed, because it has to come from somewhere nameable -- set
    iteration order is not reproducible, and sorting would reorder the space
    away from the order its blocks were computed in.
    """
    index = ProteinIndex.from_iterable(blocks[0].protids)
    for block in blocks[1:]:
        try:
            index = index.intersection(block.protids)
        except IndexAlignmentError as error:
            raise SystemExit(
                f"space {space.id!r}: blocks {blocks[0].spec.id!r} and "
                f"{block.spec.id!r} share no proteins, so there is nothing to fuse. "
                f"{error}"
            ) from error
    kept = set(index.protids)
    dropped = {
        block.spec.id: [p for p in block.protids if p not in kept]
        for block in blocks
        if any(p not in kept for p in block.protids)
    }
    return index, dropped


def fuse_blocks(space, blocks, index: ProteinIndex):
    """Align every block to the shared index and combine them."""
    inputs = []
    for block in blocks:
        values = index.align(
            block.protids,
            block.features,
            what=f"space {space.id!r} block {block.spec.id!r}",
        )
        inputs.append(FusionInput(block.spec.id, values, space.weight_for(block.spec.id)))
    try:
        return fuse(space.strategy, inputs, dict(space.params))
    except FusionError as error:
        raise SystemExit(f"space {space.id!r}: {error}") from error


def features_for(space, store: BlockStore, config=None) -> tuple:
    """The matrix a space reduces, its protid index, and how it was combined."""
    blocks = read_blocks(space, store)
    index, dropped = shared_index(space, blocks)
    for block_id, missing in sorted(dropped.items()):
        shown = missing[:5]
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(
            f"[reduce_space] space {space.id!r}: block {block_id!r} measured "
            f"{len(missing)} protein(s) that another block in this space did not, "
            f"so they are absent from the fused geometry: {shown}{suffix}",
            file=sys.stderr,
        )
    fused = fuse_blocks(space, blocks, index)
    if space.strategy != "none":
        print(f"[reduce_space] space {space.id!r}: {fused.describe()}", file=sys.stderr)
    return fused, index, blocks


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
    fused, index, blocks = features_for(space, store, config)

    try:
        random_state = int(args.random_state)
    except ValueError:
        random_state = 123456

    values = fused.values
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
            "weights": {b: space.weight_for(b) for b in space.blocks},
        },
        inputs={"block:" + block.spec.id: block.manifest.get("cache_key", "") for block in blocks},
        protids=protids,
        seed=random_state,
        # The fusion record is not decoration. ADR 0002 requires that no fused
        # map exist without its weight vector and contribution shares beside it,
        # and this file is the only place they are written down.
        extra={"steps": provenance, "fusion": fused.to_dict()},
    )
    manifest.write(os.path.join(space_dir, "manifest_" + args.reducer + ".json"))

    print(
        f"[reduce_space] wrote {embedding_path} ({len(protids)} proteins, "
        f"{len(result.column_names)} dimensions)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
