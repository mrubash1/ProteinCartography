#!/usr/bin/env python
"""Load a pipeline config from YAML or JSON.

This exists because of a dependency, and the dependency is the interesting part.

``compute_block.py`` and ``reduce_space.py`` run in ``envs/analysis.yml``, which
pins numpy, scikit-learn, umap-learn and matplotlib against each other. It does
not contain PyYAML, so both scripts failed at import the first time they were
actually executed -- ``ModuleNotFoundError: No module named 'yaml'``.

The obvious fix is to add PyYAML to that environment. That is the wrong trade.
Changing ``envs/analysis.yml`` changes its hash, which forces a fresh solve of
the one environment whose package versions decide the pipeline's numeric output,
and this repository has already been bitten once by a fresh solve pulling in a
numpy-2-built matplotlib alongside a pinned numpy 1.23.5. Rebuilding that
environment to gain a config parser is a large risk for a small convenience.

So the parser goes away instead. The Snakefile writes the resolved config as
JSON, which the standard library reads, and PyYAML is imported lazily -- only on
the path where somebody runs one of these scripts by hand against a ``.yml``.
"""

from __future__ import annotations
import json
import os

__all__ = ["load_config"]

JSON_SUFFIXES = (".json",)
YAML_SUFFIXES = (".yml", ".yaml")


def load_config(path: str) -> dict:
    """Read a config file, choosing the parser by suffix.

    JSON is handled by the standard library. YAML needs PyYAML, which is
    imported here rather than at module scope so that an environment without it
    can still run the JSON path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file {path} does not exist.")

    suffix = os.path.splitext(path)[1].lower()
    if suffix in JSON_SUFFIXES:
        with open(path) as handle:
            return json.load(handle)

    if suffix not in YAML_SUFFIXES:
        raise ValueError(
            f"config file {path} has suffix {suffix!r}; expected one of "
            f"{', '.join(JSON_SUFFIXES + YAML_SUFFIXES)}."
        )

    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            f"reading {path} needs PyYAML, which is not installed in this "
            "environment. Inside the pipeline this should not happen: the "
            "Snakefile writes the config as JSON precisely so that the analysis "
            "environment does not need a YAML parser. Pass the JSON the "
            "`multispace_config` rule produces, or install PyYAML to read the "
            "YAML directly."
        ) from error

    with open(path) as handle:
        return yaml.safe_load(handle)
