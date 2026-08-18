#!/usr/bin/env python
"""Discovery of block, reducer, and fusion providers.

Two sources, in priority order:

1. **Built-ins**, registered in-process. These work when the repo is run from a
   source checkout with no installed distribution, which is how snakemake
   invokes the scripts and how the tests run.
2. **Entry points**, under the ``proteincartography.*`` groups. This is the
   extension path: a third party can add a representation by declaring an entry
   point in their own ``pyproject.toml``, without editing any file here.

Entry points win over built-ins of the same name, so a downstream package can
replace a default implementation.

The ``importlib.metadata`` entry-point API changed shape across the versions
this project supports -- a dict on Python 3.9, a selectable object from 3.10,
with the dict interface removed in 3.12 -- so the lookup is done behind a small
compatibility shim rather than assuming either form.

See ``docs/adr/0001-block-space-view.md`` and ``docs/adr/0006-optional-heavy-
dependencies.md``.
"""

from __future__ import annotations
import importlib.metadata as importlib_metadata
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "BLOCK_GROUP",
    "FUSION_GROUP",
    "REDUCER_GROUP",
    "ProviderNotFoundError",
    "ProviderUnavailableError",
    "available_providers",
    "clear_builtins",
    "get_provider",
    "list_providers",
    "register_builtin",
]

BLOCK_GROUP = "proteincartography.blocks"
REDUCER_GROUP = "proteincartography.reducers"
FUSION_GROUP = "proteincartography.fusion"

_BUILTINS: dict = {BLOCK_GROUP: {}, REDUCER_GROUP: {}, FUSION_GROUP: {}}


class ProviderNotFoundError(KeyError):
    """Raised when a config names a provider that is not installed."""

    def __str__(self) -> str:  # KeyError repr adds quotes we do not want
        return self.args[0] if self.args else ""


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider exists but its dependencies or weights are missing."""


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    group: str
    source: str  # "builtin" or "entry_point"
    available: bool
    reason: str


def register_builtin(group: str, name: str, factory) -> None:
    """Register an in-process provider. Called by the built-in modules."""
    if group not in _BUILTINS:
        raise ValueError(f"unknown provider group {group!r}")
    _BUILTINS[group][name] = factory


def clear_builtins(group: str | None = None) -> None:
    """Reset the built-in registry. For tests."""
    groups = list(_BUILTINS) if group is None else [group]
    for g in groups:
        _BUILTINS[g] = {}


@lru_cache(maxsize=1)
def _all_entry_points():
    """The raw ``importlib.metadata`` scan, memoised for the process.

    The scan walks every distribution on ``sys.path`` and parses its metadata --
    1.5-10 ms a call depending on machine load -- and :func:`_entry_point_factories`
    is reached from every provider lookup: 52 scans and 0.41 s across the unit
    suite for one unchanging answer. Nothing can install a distribution into a
    running interpreter in a way this would need to notice.

    The cache deliberately stops here, at the bare scan. Caching
    :func:`_iter_entry_points` or :func:`_entry_point_factories` instead would
    answer from the cache *before* the monkeypatch in
    ``test_spaces_registry_and_store.py`` -- two tests replace
    ``_iter_entry_points`` to inject a fake and a deliberately broken entry
    point, and both would silently stop testing anything.
    """
    return importlib_metadata.entry_points()


def _iter_entry_points(group: str):
    """Yield entry points for `group` across importlib.metadata API versions."""
    try:
        entry_points = _all_entry_points()
    except Exception:  # pragma: no cover - a broken installation, not our bug
        return
    select = getattr(entry_points, "select", None)
    if select is not None:  # Python 3.10+
        yield from select(group=group)
    else:  # Python 3.9: a plain dict of group -> list
        yield from entry_points.get(group, [])


def _entry_point_factories(group: str) -> dict:
    found = {}
    for ep in _iter_entry_points(group):
        found[ep.name] = ep
    return found


def list_providers(group: str) -> dict:
    """Every provider name in `group`, mapped to where it came from.

    Entry points shadow built-ins of the same name.
    """
    if group not in _BUILTINS:
        raise ValueError(f"unknown provider group {group!r}")
    names = {name: "builtin" for name in _BUILTINS[group]}
    for name in _entry_point_factories(group):
        names[name] = "entry_point"
    return dict(sorted(names.items()))


def _load(group: str, name: str):
    entry_points = _entry_point_factories(group)
    if name in entry_points:
        try:
            return entry_points[name].load(), "entry_point"
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Provider {name!r} in group {group!r} is declared as an entry point "
                f"but failed to import: {exc}\n"
                "This usually means the package that declares it is installed but its "
                "own dependencies are not."
            ) from exc
    if name in _BUILTINS[group]:
        return _BUILTINS[group][name], "builtin"

    known = list_providers(group)
    if known:
        options = "\n".join(f"  - {n} ({src})" for n, src in known.items())
        detail = f"Available in {group!r}:\n{options}"
    else:
        detail = (
            f"No providers are registered in {group!r} at all. If this project is "
            "installed from a source checkout, the built-ins register on import of "
            "the corresponding module; if you expected a third-party provider, check "
            "that its package is installed in this environment."
        )
    raise ProviderNotFoundError(f"Unknown provider {name!r}.\n{detail}")


def get_provider(group: str, name: str, *, require_available: bool = True):
    """Instantiate a provider by name.

    Args:
        require_available: when set, a provider whose ``is_available()`` reports
            False raises :class:`ProviderUnavailableError` with the provider's own
            explanation of what is missing. Callers that are surveying rather
            than computing -- the Snakefile deciding which spaces to skip --
            should pass False and consult ``is_available`` themselves.
    """
    factory, _source = _load(group, name)
    provider = factory() if callable(factory) else factory

    if require_available:
        checker = getattr(provider, "is_available", None)
        if checker is not None:
            available, reason = checker()
            if not available:
                raise ProviderUnavailableError(
                    f"Provider {name!r} is installed but not usable: {reason}\n"
                    "Optional providers are expected to be missing; if this space is "
                    "not essential, remove it from the config or let the pipeline "
                    "skip it."
                )
    return provider


def available_providers(group: str) -> list:
    """Survey `group`, reporting availability without raising.

    This is what the Snakefile uses to decide which spaces it can build.
    """
    infos = []
    for name, source in list_providers(group).items():
        try:
            provider = get_provider(group, name, require_available=False)
        except (ProviderNotFoundError, ProviderUnavailableError) as exc:
            infos.append(ProviderInfo(name, group, source, False, str(exc)))
            continue
        checker = getattr(provider, "is_available", None)
        if checker is None:
            infos.append(ProviderInfo(name, group, source, True, ""))
            continue
        try:
            available, reason = checker()
        except Exception as exc:  # a provider's own check should not kill the survey
            available, reason = False, f"is_available() raised: {exc}"
        infos.append(ProviderInfo(name, group, source, bool(available), reason))
    return infos
