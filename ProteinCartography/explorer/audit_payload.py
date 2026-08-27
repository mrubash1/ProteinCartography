#!/usr/bin/env python
"""Read a built page and say whether its numbers are usable.

Every check here answers a question the browser cannot ask and the unit suite
does not: the payload is assembled in Python, serialized once, and then read by
JavaScript that assumes it is well formed. Between those two there is no gate.

**Why non-finite numbers are the sharpest of the three.** `json.dumps` emits
`NaN` and `Infinity` bare, which is not valid JSON -- but the page does not
parse JSON. `template.py:358` is ``const PAYLOAD = __PAYLOAD__;``, a JavaScript
*literal*, and in JavaScript `NaN` and `Infinity` are ordinary identifiers. So a
NaN in an overlay does not raise anywhere: it is written, it is read, it becomes
a real NaN, and the points it colours quietly stop being drawn. Nothing in this
repository would report that.

**What this does NOT do, stated so nobody assumes otherwise.** It reads the
DATA, not the picture. The defect that motivated the whole family of blank-panel
checks lived in `template.py`'s shape coordinates: the DOM reported the trace as
correct, every number in the payload was finite and in range, and only a
screenshot found it. An audit that passes here means the numbers are sound; it
does not mean the page drew them. The browser pass is a separate obligation and
this file does not discharge it.

Deliberately stdlib-only -- `base64`, `json`, `re` -- so it runs in the bare
environment, on a built page, with nothing installed.
"""

from __future__ import annotations
import base64
import json
import math
import re

__all__ = ["audit", "load_document", "payload_from_html", "HARD_BUDGET_BYTES"]

#: Kept in step with `payload.py` by importing when it is importable, so the
#: auditor and the builder cannot drift on what "too big" means. The literal is
#: the fallback for auditing a page with the package unavailable, which is the
#: case this file is meant to survive.
try:  # pragma: no cover - exercised by whichever branch the environment takes
    from explorer.payload import HARD_BUDGET_BYTES, PREFERRED_BUDGET_BYTES
except Exception:  # pragma: no cover
    HARD_BUDGET_BYTES = 20 * 1024**2
    PREFERRED_BUDGET_BYTES = 10 * 1024**2

_PAYLOAD_LINE = re.compile(r"^\s*const PAYLOAD = (.*);\s*$", re.M)


def payload_from_html(text: str) -> dict:
    """The payload out of a built page.

    Matched on `template.py`'s own assignment line rather than on a `<script>`
    boundary: the page carries plotly's bundle in a script tag too, and a
    boundary match would find that first.
    """
    found = _PAYLOAD_LINE.search(text)
    if not found:
        raise ValueError(
            "no `const PAYLOAD = ...;` line in this file. Either it is not a built "
            "explorer page, or template.py changed the assignment and this parser "
            "is now blind -- which is the failure mode worth checking for."
        )
    return json.loads(found.group(1))


def load_document(path: str) -> dict:
    """A payload from either artifact: a built page, or a payload JSON."""
    text = open(path, encoding="utf-8").read()
    if _PAYLOAD_LINE.search(text):
        return payload_from_html(text)
    return json.loads(text)


def _walk_numbers(value, trail: str = ""):
    """Every number in the document, with the path that reaches it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{trail}.{key}" if trail else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{trail}[{index}]")
    elif isinstance(value, float):
        yield trail, value


def check_finite(cohort: dict) -> list:
    """No NaN and no infinity anywhere in the cohort.

    Reported with the path, because "there is a NaN somewhere in 1.2 MB" is not
    something anyone can act on.
    """
    return [
        f"non-finite value at {path}: {value!r}"
        for path, value in _walk_numbers(cohort)
        if not math.isfinite(value)
    ]


def check_heatmap(cohort: dict) -> list:
    """The quantised matrix decodes to the range it declares, at the size it claims.

    Three failures, each silent in the browser: a short buffer draws a truncated
    map, a quantisation step above the declared `levels` decodes past the end of
    the colour scale, and an inverted range makes every decoded value wrong
    while every individual number still looks plausible.

    THERE IS DELIBERATELY NO PER-VALUE RANGE LOOP. Decoding is
    ``byte / levels * (high - low) + low``, so a decoded value can only leave
    [low, high] when `byte > levels` or when `high < low` -- both of which are
    checked directly above, in constant time. A loop over n**2 values could
    report nothing the two cheap checks miss, and at n=2530 it would walk 6.4
    million floats to say so. A check that cannot fail independently is not a
    check; it is cost.
    """
    matrix = cohort.get("tm_matrix") or {}
    if not matrix:
        return []
    n = matrix.get("n")
    levels = matrix.get("levels")
    low, high = matrix.get("low"), matrix.get("high")
    if None in (n, levels, low, high):
        return ["tm_matrix is present but does not declare all of n, levels, low, high"]
    problems = []
    raw = base64.b64decode(matrix.get("values") or "")
    if len(raw) != n * n:
        problems.append(f"tm_matrix decodes to {len(raw)} bytes, not n*n = {n * n}")
    if raw and max(raw) > levels:
        problems.append(
            f"tm_matrix holds a quantisation step {max(raw)} above its declared "
            f"{levels}, which decodes past `high`"
        )
    if high < low:
        problems.append(
            f"tm_matrix declares low={low} above high={high}; every decoded value "
            "is wrong while each one still looks plausible"
        )
    return problems


def check_budget(cohort: dict, name: str) -> list:
    """One cohort's serialized size against the budgets the builder enforces."""
    size = len(json.dumps(cohort, separators=(",", ":"), default=str))
    if size > HARD_BUDGET_BYTES:
        return [f"{name} is {size} bytes, over the hard budget of {HARD_BUDGET_BYTES}"]
    return []


def audit(document: dict) -> dict:
    """Every check over every cohort. Reports; decides nothing."""
    cohorts = document.get("cohorts") or [document]
    findings = []
    for index, cohort in enumerate(cohorts):
        name = cohort.get("cohort_name") or f"cohort[{index}]"
        for problem in check_finite(cohort) + check_heatmap(cohort) + check_budget(cohort, name):
            findings.append({"cohort": name, "problem": problem})
    return {"n_cohorts": len(cohorts), "findings": findings}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-i", "--input", required=True, help="a built explorer page, or a payload JSON"
    )
    return parser.parse_args()


def main():
    import sys

    args = parse_args()
    report = audit(load_document(args.input))
    print(f"[audit_payload] {args.input}: {report['n_cohorts']} cohort(s)")
    for finding in report["findings"]:
        print(f"  {finding['cohort']}: {finding['problem']}")
    if report["findings"]:
        sys.exit(f"[audit_payload] {len(report['findings'])} problem(s)")
    print("[audit_payload] finite, in range, and inside the byte budget")


if __name__ == "__main__":
    main()
