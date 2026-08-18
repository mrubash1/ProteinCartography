#!/usr/bin/env python
"""Provenance capture.

Every block and every space ships with enough information to recompute it
exactly: package versions, parameters, seeds, and content hashes of the inputs.
Without this a map is a picture; with it, it is a result.

Hashes are over *content*, never mtimes. A rerun with identical inputs should be
free, and a rerun with changed inputs must not be.

See ``docs/adr/0001-block-space-view.md``.
"""

from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache

__all__ = [
    "Manifest",
    "file_digest",
    "hash_params",
    "package_versions",
    "values_digest",
]

# Recorded on every manifest. Keep the list short and relevant -- these are the
# packages whose version changes can move a coordinate.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "umap-learn",
    "scanpy",
    "leidenalg",
    "ProteinCartography",
)

_CHUNK = 1 << 20


def file_digest(path, algorithm: str = "sha256") -> str:
    """Content hash of a file. Streams, so a 10 GB matrix does not blow up."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


def values_digest(array, algorithm: str = "sha256") -> str:
    """Content hash of a numpy array, including dtype and shape.

    dtype and shape are folded in because the same bytes under a different dtype
    are different data, and a silent dtype change is exactly the sort of thing
    that should invalidate a cache.
    """
    digest = hashlib.new(algorithm)
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np_ascontiguous(array).tobytes())
    return f"{algorithm}:{digest.hexdigest()}"


def np_ascontiguous(array):
    import numpy as np

    return np.ascontiguousarray(array)


def hash_params(params: Mapping, algorithm: str = "sha256") -> str:
    """Stable hash of a parameter mapping.

    ``sort_keys`` makes it independent of insertion order; ``default=str``
    keeps it from failing on a value that is meaningful but not JSON-native.
    """
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return f"{algorithm}:{hashlib.new(algorithm, payload.encode()).hexdigest()}"


# Bounded rather than unbounded: `packages` is caller-supplied, and in every
# call path in this repo it is exactly TRACKED_PACKAGES.
@lru_cache(maxsize=8)
def _package_versions(packages: tuple) -> tuple:
    """Version lookup for a fixed package tuple, memoised for the process.

    ``importlib.metadata.version`` walks ``sys.path`` and parses metadata on every
    call: 5.0-5.5 ms for :data:`TRACKED_PACKAGES`. :meth:`Manifest.build` calls it
    on every block written, so the unit suite paid it 257 times for one distinct
    answer -- 1.76 s measured in ``cartography_tidy``, out of a 13 s run.

    Installed versions cannot change inside a process without an interpreter-level
    reinstall, so the cache cannot go stale in any way a caller could observe.

    The cached value is a tuple of pairs rather than a dict *on purpose*:
    :attr:`Manifest.versions` is a plain mutable field, so handing out the cached
    object itself would give every manifest in the process the same dict, and one
    caller mutating its own manifest would silently rewrite the provenance of all
    the others. An immutable cached value makes that impossible rather than merely
    discouraged.
    """
    import importlib.metadata as importlib_metadata

    versions = []
    for name in packages:
        try:
            versions.append((name, importlib_metadata.version(name)))
        except importlib_metadata.PackageNotFoundError:
            versions.append((name, None))
    return tuple(versions)


def package_versions(packages: Iterable = TRACKED_PACKAGES) -> dict:
    """Installed versions of the packages whose changes can move a coordinate."""
    return dict(_package_versions(tuple(packages)))


@dataclass
class Manifest:
    """Everything needed to recompute an artifact, and to tell if it is stale.

    Attributes:
        kind: ``"block"`` or ``"space"``.
        id: the block or space id.
        provider: the registry name that produced it.
        params: the validated parameters.
        inputs: mapping of logical input name to content hash.
        n_proteins: size of the index this was computed over.
        protids_digest: hash of the protid list, so a cohort change invalidates.
        seed: the random seed, where one applies.
        versions: package versions.
        environment: interpreter and platform.
        extra: provider-specific detail, e.g. a model checkpoint hash or the
            symmetrization rule a pairwise block applied.
    """

    kind: str
    id: str
    provider: str = ""
    params: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    n_proteins: int = 0
    protids_digest: str = ""
    seed: int | None = None
    versions: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    #: Facts about the *output*, recorded when it is written -- content digests,
    #: observed censoring rate, the resolved spec. Deliberately excluded from
    #: :attr:`cache_key`, which is an identity of the *inputs*. Including them
    #: would make the key unreproducible by any caller who has not already
    #: computed the block, which is precisely the caller the cache exists for.
    derived: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        kind: str,
        id: str,
        *,
        provider: str = "",
        params: Mapping | None = None,
        inputs: Mapping | None = None,
        protids: Iterable | None = None,
        seed: int | None = None,
        extra: Mapping | None = None,
    ) -> Manifest:
        protids = list(protids or [])
        return cls(
            kind=kind,
            id=id,
            provider=provider,
            params=dict(params or {}),
            inputs=dict(inputs or {}),
            n_proteins=len(protids),
            protids_digest=protids_digest(protids) if protids else "",
            seed=seed,
            versions=package_versions(),
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            extra=dict(extra or {}),
        )

    @property
    def cache_key(self) -> str:
        """Identity of this artifact's *inputs*.

        Deliberately excludes ``environment``: rebuilding on a different machine
        with the same package versions should hit the cache. It includes
        ``versions``, because a numpy or scikit-learn change genuinely can move
        a coordinate.

        It also excludes :attr:`derived`. A cache key that folded in the output's
        content digest could only ever be computed by someone who had already
        produced the output, so it would never match on the one path that
        matters -- a fresh process asking "do I need to build this?".
        """
        payload = {
            "kind": self.kind,
            "id": self.id,
            "provider": self.provider,
            "params": self.params,
            "inputs": self.inputs,
            "protids_digest": self.protids_digest,
            "seed": self.seed,
            "versions": self.versions,
            "extra": self.extra,
        }
        return hash_params(payload)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cache_key"] = self.cache_key
        return data

    def write(self, path) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        """Rebuild from :meth:`to_dict` output, ignoring the derived cache key.

        Unknown keys are dropped rather than raising, so a manifest written by a
        newer version stays readable.
        """
        data = dict(data)
        data.pop("cache_key", None)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path) -> Manifest:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def matches(self, other: Manifest) -> bool:
        return self.cache_key == other.cache_key


def protids_digest(protids: Iterable, algorithm: str = "sha256") -> str:
    """Hash of an ordered protid list. Order matters -- it is the index."""
    digest = hashlib.new(algorithm)
    for p in protids:
        digest.update(str(p).encode())
        digest.update(b"\x00")
    return f"{algorithm}:{digest.hexdigest()}"
