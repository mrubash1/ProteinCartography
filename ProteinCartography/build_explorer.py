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


def _report_size(document, html, plotly_js):
    """Print what the page weighs, per part, against the recorded budget.

    To STDERR and never to the file: the page carries no generation timestamp on
    purpose, so that two runs of the same inputs produce the same bytes, and a
    size line inside it would be a second thing that varies with the machine.

    Nothing here fails a build. The budget is a number to be looked at, not a
    gate -- refusing to produce the artifact is the wrong response to it being
    large, since seeing how large is the reason someone asked.
    """
    from explorer.payload import HARD_BUDGET_BYTES, PREFERRED_BUDGET_BYTES, payload_bytes

    def mb(value):
        return f"{value / 1024**2:.2f} MB"

    accounting = payload_bytes(document)
    page = len(html.encode("utf-8"))
    print(
        f"[build_explorer] page {mb(page)} = plotly {mb(len(plotly_js))} "
        f"+ payload {mb(accounting['total'])} + template",
        file=sys.stderr,
    )
    for key, value in sorted(accounting["per_key"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"[build_explorer]   payload.{key}: {mb(value)}", file=sys.stderr)
    for cohort in accounting["cohorts"]:
        print(
            f"[build_explorer]   cohort {cohort['cohort_name']}: {mb(cohort['bytes'])}"
            f", of which tm_matrix {mb(cohort['tm_matrix_bytes'])}"
            f" (n={cohort['n_proteins']})",
            file=sys.stderr,
        )
    for name, limit in (("preferred", PREFERRED_BUDGET_BYTES), ("hard", HARD_BUDGET_BYTES)):
        side = "under" if page <= limit else "OVER"
        print(f"[build_explorer]   {side} the {name} budget of {mb(limit)}", file=sys.stderr)


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

    # Additional cohorts, each a complete payload of its own.
    #
    # THE FIRST COHORT USED TO BE SHIPPED TWICE. Its keys were kept at the top
    # level as well as inside `cohorts`, described as a compatibility guarantee
    # for readers of the payload. It was not one the renderer needed: the page
    # reads exactly `thresholds` off the top level (`const THRESHOLDS =
    # PAYLOAD.thresholds`), and `COHORTS = PAYLOAD.cohorts || [...]` takes the
    # array whenever it exists, so the copy was unreachable. Measured on the
    # two-cohort page it was 670,709 bytes -- 33.8% of the payload and 11.6% of
    # the whole file -- and it scales as n**2, because it carries its own
    # `tm_matrix`.
    #
    # THE SINGLE-COHORT PATH IS UNTOUCHED and must stay that way. With no
    # `--also-cohort` there is no `cohorts` key at all, and the renderer's
    # fallback rebuilds a one-cohort list FROM the top-level keys -- so a page
    # built without this flag depends on them being there. Only the multi-cohort
    # document is reduced.
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
        # `analysis_name` is the page's identity and costs nothing; `thresholds`
        # is the one key the renderer actually reads off the top level. Anything
        # else here is the copy.
        document = {
            key: value for key, value in document.items() if key in ("analysis_name", "thresholds")
        }
        document["cohorts"] = cohorts

    plotly_js = get_plotlyjs()
    html = render(document, plotly_js, title)
    _report_size(document, html, plotly_js)
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
