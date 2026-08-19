#!/usr/bin/env python
"""Build the single-file interactive explorer for a finished run.

Phase 7, ADR 0005: one static self-contained HTML file, all data embedded, no
server, no CDN, no build step. It opens from `file://`, survives being emailed,
and can be committed beside a pub -- the properties that matter for how
ProteinCartography artifacts are actually distributed.

`plot_interactive.py` keeps working and keeps emitting its existing filenames.
This is additive, and like every rule in this work it stays out of the DAG
entirely unless a config asks for spaces.

**Plotly is inlined, not linked.** `plotly.offline.get_plotlyjs()` returns about
3.6 MB of JavaScript and it goes into the file. A CDN link would be smaller and
would break offline use and break the day the CDN version changes, which defeats
the archival property that motivated the decision in the first place.

**What this refuses to draw is the interesting part**, and it lives in
`explorer/payload.py` rather than here: a space the diagnostics call unreadable
is banded red and its points drawn hollow, an individual protein whose position
is not faithful is hollow in every panel, and a withheld cross-space number
renders as the word *withheld* with its reason rather than as a blank cell.
"""

from __future__ import annotations
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--configfile", required=True)
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument("--output", required=True, help="the HTML file to write")
    parser.add_argument(
        "--analysis-name",
        default=None,
        help="title for the page; defaults to the config's analysis_name",
    )
    parser.add_argument(
        "--also-cohort",
        action="append",
        default=[],
        metavar="NAME=CONFIG:OUTPUT_DIR",
        help=(
            "an additional cohort to embed, switchable from a dropdown. Repeatable. "
            "Cheap: of a single-cohort page's size, nearly all is the inlined Plotly "
            "bundle, which every cohort shares. Omit it and the page is exactly what "
            "it was before this option existed."
        ),
    )
    return parser.parse_args()


def parse_cohort(spec: str) -> tuple:
    """``NAME=CONFIG:OUTPUT_DIR`` -> ``(name, config_path, output_dir)``.

    Split from the right on ``:`` so a Windows-style or absolute path with a
    colon in it does not silently take the separator's place.
    """
    if "=" not in spec:
        raise SystemExit(
            f"[build_explorer] --also-cohort wants NAME=CONFIG:OUTPUT_DIR, got {spec!r}"
        )
    name, _, rest = spec.partition("=")
    config_path, sep, output_dir = rest.rpartition(":")
    if not sep or not name.strip() or not config_path or not output_dir:
        raise SystemExit(
            f"[build_explorer] --also-cohort wants NAME=CONFIG:OUTPUT_DIR, got {spec!r}"
        )
    return name.strip(), config_path, output_dir


def is_available() -> tuple:
    """``(available, explanation)``, in the shape ADR 0006 rule 2 specifies."""
    try:
        from plotly.offline import get_plotlyjs  # noqa: F401
    except ImportError as error:
        return False, (
            f"plotly is not importable ({error}). It is pinned in envs/plotting.yml, "
            "which is the environment `rule build_explorer` declares."
        )
    return True, "plotly is importable"


def main() -> int:
    args = parse_args()
    available, explanation = is_available()
    if not available:
        # A missing optional dependency is a reduced result, never an error --
        # but this rule's only product is the file, so there is nothing reduced
        # to hand back and the run should stop here rather than write a page
        # with no plotting library in it.
        raise SystemExit(f"[build_explorer] {explanation}")

    from config_io import load_config
    from config_schema import from_legacy
    from explorer.payload import build_payload
    from explorer.template import render
    from plotly.offline import get_plotlyjs

    raw = load_config(args.configfile)
    config = from_legacy(raw)
    if not config.spaces:
        raise SystemExit(
            "[build_explorer] this config defines no spaces, so there is nothing to "
            "explore. The legacy `plot_interactive` outputs are unaffected."
        )

    title = args.analysis_name or raw.get("analysis_name") or "ProteinCartography"
    payload = build_payload(config, args.output_dir, analysis_name=title)
    if not payload.spaces:
        raise SystemExit(
            "[build_explorer] no space produced an embedding, so there are no panels "
            "to draw. Run `reduce_space` first, or check whether every space was "
            "skipped for a missing provider."
        )

    # Additional cohorts, each a complete payload of its own. The page keeps the
    # first cohort's keys at the top level so a reader of the payload -- and every
    # test that reads `PAYLOAD["spaces"]` -- sees exactly what it saw before.
    document = payload.to_dict()
    cohorts = [dict(document, cohort_name=title)]
    for spec in args.also_cohort:
        name, config_path, output_dir = parse_cohort(spec)
        other_raw = load_config(config_path)
        other = from_legacy(other_raw)
        if not other.spaces:
            raise SystemExit(f"[build_explorer] cohort {name!r} defines no spaces")
        other_payload = build_payload(other, output_dir, analysis_name=name)
        if not other_payload.spaces:
            raise SystemExit(f"[build_explorer] cohort {name!r} produced no panels")
        cohorts.append(dict(other_payload.to_dict(), cohort_name=name))
    if len(cohorts) > 1:
        document["cohorts"] = cohorts

    html = render(document, get_plotlyjs(), title)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        handle.write(html)

    unreadable = [s.space_id for s in payload.spaces if s.verdict["level"] == "unreadable"]
    if unreadable:
        print(
            f"[build_explorer] {len(unreadable)} of {len(payload.spaces)} space(s) are "
            f"drawn but marked unreadable: {', '.join(unreadable)}. That is the "
            "diagnostics doing their job, not a failure.",
            file=sys.stderr,
        )
    size_mb = os.path.getsize(args.output) / 1e6
    print(
        f"[build_explorer] wrote {args.output} "
        f"({size_mb:.1f} MB, {len(payload.spaces)} panel(s), "
        f"{payload.provenance['n_proteins']} proteins)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
