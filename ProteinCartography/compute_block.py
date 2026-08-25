#!/usr/bin/env python
"""Compute one block and write it to the block store.

The snakemake entry point for a block. Everything interesting happens in the
provider; this is argument handling, provider lookup, and the cache check.

A block whose provider is unavailable is *skipped*, not failed. Optional
dependencies are expected to be missing (ADR 0006), and a run that cannot build
the PLM block should still build the structure map. The skip writes a manifest
recording why, so the absence is visible in the output rather than inferred from
a missing directory.
"""

from __future__ import annotations
import argparse
import json
import sys

from config_io import load_config
from config_schema import from_legacy
from spaces.registry import BLOCK_GROUP, ProviderNotFoundError, ProviderUnavailableError
from spaces.registry import get_provider as _get_provider
from spaces.store import BlockStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--configfile",
        required=True,
        help="the pipeline config, as JSON or YAML (see config_io.load_config)",
    )
    parser.add_argument("-b", "--block-id", required=True, help="which block to compute")
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when the stored block's inputs are unchanged",
    )
    parser.add_argument(
        "--provider-input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "a named input path for the provider, e.g. "
            "`--provider-input features_file=path/to/uniprot_features.tsv`. "
            "Repeatable. Used for inputs whose location depends on the run rather "
            "than on the config -- the features table lives in the output directory "
            "in search mode and in the user's input directory in cluster mode. "
            "These are paths, not parameters: they reach the provider through "
            "`ctx.extras` and never enter the manifest's params, so a block's cache "
            "key does not change when the same run is done from a different "
            "directory. What is recorded is the file's digest."
        ),
    )
    return parser.parse_args()


def parse_provider_inputs(pairs) -> dict:
    """Turn `NAME=PATH` strings into a dict for `PipelineContext.extras`."""
    inputs = {}
    for pair in pairs:
        name, separator, path = pair.partition("=")
        if not separator or not name.strip():
            raise SystemExit(
                f"--provider-input expects NAME=PATH, got {pair!r}. A path containing "
                "an '=' is fine; the split is on the first one."
            )
        inputs[name.strip()] = path
    return inputs


def _register_builtins() -> None:
    """Import the built-in providers so the registry can find them.

    Imports are inside the function and individually guarded: a provider whose
    optional dependency is missing must not stop the others from registering.
    """
    from blocks import biophys, domains, threedi, tmscore

    tmscore.register()
    threedi.register()
    biophys.register()
    domains.register()


def main() -> int:
    args = parse_args()
    config = from_legacy(load_config(args.configfile))

    if args.block_id not in config.blocks:
        raise SystemExit(
            f"block {args.block_id!r} is not defined in {args.configfile}. "
            f"Defined blocks: {config.block_ids() or '(none)'}"
        )
    block = config.blocks[args.block_id]

    _register_builtins()
    store = BlockStore(args.output_dir)

    try:
        provider = _get_provider(BLOCK_GROUP, block.provider, require_available=True)
    except ProviderUnavailableError as exc:
        # Expected for optional providers. Record the skip so it is visible.
        _write_skip(store, block, str(exc))
        print(f"[compute_block] skipping {args.block_id!r}: {exc}", file=sys.stderr)
        return 0
    except ProviderNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    from blocks.tmscore import PipelineContext

    extras = parse_provider_inputs(args.provider_input)
    # A `vocabulary_file` named in the CONFIG is honoured even when the caller
    # did not pass it as a `--provider-input`.
    #
    # The Snakefile passes it explicitly AND declares it as a rule input, which
    # is what makes snakemake rerun the block when the file changes. But a
    # config that names a vocabulary and a run that quietly ignores it is a
    # divergence between what the config says and what happened, and this entry
    # point is reachable by hand. An explicit `--provider-input` still wins, so
    # the Snakefile path is unchanged.
    if block.vocabulary_file and "vocabulary_file" not in extras:
        extras["vocabulary_file"] = block.vocabulary_file
    ctx = PipelineContext(output_dir=args.output_dir, extras=extras)
    params = dict(block.params)
    params.setdefault("block_id", block.id)
    if block.representation is not None:
        params.setdefault("representation", block.representation)
    # Only when the config actually asked for one. Filling in a default here
    # would reach the provider as an explicit choice and silence the provider's
    # own, which is what it used to do -- and it went unnoticed because the two
    # blocks that existed at the time both wanted the same value.
    if block.normalization is not None:
        params.setdefault("normalization", block.normalization)

    # ASK THE PROVIDER what it would write, instead of guessing.
    #
    # This used to be `Manifest.build(..., protids=[])` with no `inputs`, whose
    # key could not equal any written one -- so `is_fresh` was always False and
    # every block recomputed on every invocation snakemake allowed through.
    # That is FOLLOWUPS #27, and the fix is not a better guess: it is that only
    # the provider knows its own inputs, so only the provider can say.
    #
    # `getattr`, and `None` is a legitimate answer. A third-party provider
    # (ADR 0006) that never heard of `plan` recomputes exactly as it did before,
    # and so does one whose input is missing or unreadable -- `compute` then
    # raises the real error with its own message.
    plan = getattr(provider, "plan", None)
    expected = plan(ctx, params) if callable(plan) else None
    if not args.force and expected is not None and store.is_fresh(block.id, expected):
        print(f"[compute_block] {block.id!r} is up to date", file=sys.stderr)
        return 0

    result = provider.compute(ctx, params)
    path = store.write_block(result)
    censoring = result.censoring_rate
    print(
        f"[compute_block] wrote {block.id!r} to {path} "
        f"({result.n_proteins} proteins"
        + (f", {censoring:.1%} censored" if censoring is not None else "")
        + ")",
        file=sys.stderr,
    )
    return 0


def _write_skip(store: BlockStore, block, reason: str) -> None:
    import os

    directory = store.block_dir(block.id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "SKIPPED.json"), "w") as fh:
        json.dump(
            {"block_id": block.id, "provider": block.provider, "reason": reason},
            fh,
            indent=2,
        )
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
