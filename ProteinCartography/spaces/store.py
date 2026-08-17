#!/usr/bin/env python
"""On-disk storage for blocks and spaces.

Layout, strictly additive to the existing output tree:

    OUTPUT_DIR/blocks/{block_id}/
    ├── features.npy | distances.npy    # float32; condensed for pairwise
    ├── mask.npy                        # bool, censoring
    ├── protids.txt                     # canonical order, one per line
    └── manifest.json

Arrays are ``float32``. Foldseek reports four significant figures, so float32 is
lossless with respect to the source and halves memory. Pairwise blocks store the
condensed upper triangle, halving it again.

The protid list is a separate plain-text file rather than a numpy array of
strings, so a human can check what a block was computed over without loading
anything.

Everything goes through this module so that a sparse or out-of-core backend can
be added later without touching a single provider.

See ``docs/adr/0004-storage-and-scale.md``.
"""

from __future__ import annotations
import os
import shutil
from dataclasses import dataclass, replace

import numpy as np
from spaces.base import CHANNEL_SEMANTICS, BlockResult, BlockSpec
from spaces.manifest import Manifest, values_digest

__all__ = ["BlockStore", "StoreError"]

FEATURES_FILE = "features.npy"
DISTANCES_FILE = "distances.npy"
MASK_FILE = "mask.npy"  # legacy name, still read for old blocks
CHANNEL_PREFIX = "channel_"
PROTIDS_FILE = "protids.txt"
MANIFEST_FILE = "manifest.json"
SPEC_FILE = "spec.json"

STORAGE_DTYPE = np.float32


class StoreError(RuntimeError):
    """Raised when an artifact on disk is missing, malformed, or inconsistent."""


def _write_protids(path, protids) -> None:
    with open(path, "w") as fh:
        for p in protids:
            fh.write(f"{p}\n")


def _read_protids(path) -> list:
    with open(path) as fh:
        return [line.rstrip("\n") for line in fh if line.rstrip("\n")]


@dataclass(frozen=True)
class BlockStore:
    """Reads and writes blocks under ``root/blocks/{block_id}/``."""

    root: str

    @property
    def blocks_dir(self) -> str:
        return os.path.join(self.root, "blocks")

    def block_dir(self, block_id: str) -> str:
        return os.path.join(self.blocks_dir, block_id)

    # -- write -------------------------------------------------------------

    def write_block(self, result: BlockResult, manifest: Manifest | None = None) -> str:
        """Persist a block. Returns the directory written.

        The write is atomic at the directory level: everything lands in a
        sibling ``.tmp`` directory which is swapped in at the end, so an
        interrupted run cannot leave a half-written block that a later run
        mistakes for a cache hit.
        """
        target = self.block_dir(result.spec.id)
        staging = target + ".tmp"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)

        if result.features is not None:
            stored_payload = np.ascontiguousarray(result.features, dtype=STORAGE_DTYPE)
            np.save(os.path.join(staging, FEATURES_FILE), stored_payload)
        else:
            stored_payload = np.ascontiguousarray(result.distances, dtype=STORAGE_DTYPE)
            np.save(os.path.join(staging, DISTANCES_FILE), stored_payload)

        # Channels are written under their own names, so their meaning survives
        # the trip to disk. A single anonymous mask.npy would not: three
        # providers would each mean something different by it and the file
        # could not say which. See spaces.base.CHANNEL_SEMANTICS.
        for name, array in result.channels.items():
            np.save(
                os.path.join(staging, f"{CHANNEL_PREFIX}{name}.npy"), np.ascontiguousarray(array)
            )

        _write_protids(os.path.join(staging, PROTIDS_FILE), result.protids)

        manifest = manifest or Manifest.build(
            "block",
            result.spec.id,
            provider=result.spec.provider,
            params=result.spec.params,
            protids=result.protids,
        )
        # Copy rather than mutate: the caller built this manifest to describe the
        # inputs and will very likely reuse it to ask `is_fresh`. Writing our
        # output-derived fields onto their object would make that question answer
        # itself, which is how the cache appeared to work while being unable to
        # hit across processes.
        manifest = replace(manifest, extra=dict(manifest.extra), derived=dict(manifest.derived))
        manifest.derived.update(
            {
                # Digest the array as it is actually stored, not as it arrived --
                # a digest that cannot verify the file it describes is not doing
                # anything.
                "values_digest": values_digest(stored_payload),
                "spec": _spec_to_dict(result.spec),
                "censoring_rate": result.censoring_rate,
                "absent_rate": result.absent_rate,
                "channels": sorted(result.channels),
            }
        )
        manifest.write(os.path.join(staging, MANIFEST_FILE))

        # Swap the new directory in before removing the old one, so the window in
        # which neither exists is a rename rather than a delete-then-rename.
        retired = target + ".old"
        if os.path.exists(retired):
            shutil.rmtree(retired)
        if os.path.exists(target):
            os.replace(target, retired)
        os.replace(staging, target)
        if os.path.exists(retired):
            shutil.rmtree(retired)
        return target

    # -- read --------------------------------------------------------------

    def has_block(self, block_id: str) -> bool:
        d = self.block_dir(block_id)
        return os.path.isdir(d) and os.path.exists(os.path.join(d, MANIFEST_FILE))

    def read_manifest(self, block_id: str) -> Manifest:
        path = os.path.join(self.block_dir(block_id), MANIFEST_FILE)
        if not os.path.exists(path):
            raise StoreError(f"No manifest for block {block_id!r} at {path}")
        return Manifest.read(path)

    def read_block(self, block_id: str, spec: BlockSpec | None = None) -> BlockResult:
        d = self.block_dir(block_id)
        if not os.path.isdir(d):
            raise StoreError(f"No stored block {block_id!r} under {self.blocks_dir}")

        protids_path = os.path.join(d, PROTIDS_FILE)
        if not os.path.exists(protids_path):
            raise StoreError(
                f"Block {block_id!r} has no {PROTIDS_FILE}. Without it the rows have "
                "no meaning and the block cannot be used."
            )
        protids = _read_protids(protids_path)

        manifest = self.read_manifest(block_id)
        spec = spec or _spec_from_dict(manifest.derived.get("spec") or manifest.extra.get("spec"))
        if spec is None:
            raise StoreError(
                f"Block {block_id!r} has no spec in its manifest and none was "
                "supplied; cannot reconstruct it."
            )

        features_path = os.path.join(d, FEATURES_FILE)
        distances_path = os.path.join(d, DISTANCES_FILE)
        features = distances = None
        if os.path.exists(features_path):
            features = np.load(features_path)
        elif os.path.exists(distances_path):
            distances = np.load(distances_path)
        else:
            raise StoreError(
                f"Block {block_id!r} has neither {FEATURES_FILE} nor {DISTANCES_FILE}."
            )

        channels = {}
        for name in CHANNEL_SEMANTICS:
            channel_path = os.path.join(d, f"{CHANNEL_PREFIX}{name}.npy")
            if os.path.exists(channel_path):
                channels[name] = np.load(channel_path)
        # Blocks written before channels were named stored a single mask.npy,
        # which always meant censoring here.
        legacy_mask = os.path.join(d, MASK_FILE)
        if not channels and os.path.exists(legacy_mask):
            channels["censored"] = np.load(legacy_mask)

        return BlockResult(
            spec=spec,
            protids=protids,
            features=features,
            distances=distances,
            channels=channels,
            manifest=manifest.to_dict(),
        )

    # -- caching -----------------------------------------------------------

    def is_fresh(self, block_id: str, expected: Manifest) -> bool:
        """True when a stored block was built from exactly these inputs.

        Compares content hashes, never mtimes -- a rerun that produced identical
        inputs should hit the cache, and a touched file should not miss it.
        """
        if not self.has_block(block_id):
            return False
        try:
            stored = self.read_manifest(block_id)
        except StoreError:
            return False
        return stored.matches(expected)

    def invalidate(self, block_id: str) -> None:
        d = self.block_dir(block_id)
        if os.path.isdir(d):
            shutil.rmtree(d)

    def list_blocks(self) -> list:
        if not os.path.isdir(self.blocks_dir):
            return []
        return sorted(
            name
            for name in os.listdir(self.blocks_dir)
            if os.path.isdir(os.path.join(self.blocks_dir, name)) and not name.endswith(".tmp")
        )


def _spec_to_dict(spec: BlockSpec) -> dict:
    return {
        "id": spec.id,
        "kind": spec.kind,
        "fusable": spec.fusable,
        "metric": spec.metric,
        "normalization": spec.normalization,
        "provider": spec.provider,
        "params": spec.params,
        "not_fusable_reason": spec.not_fusable_reason,
        "version": spec.version,
        "symmetrization": spec.symmetrization,
        "distance_metric": spec.distance_metric,
    }


def _spec_from_dict(data) -> BlockSpec | None:
    if not data:
        return None
    return BlockSpec(**data)
