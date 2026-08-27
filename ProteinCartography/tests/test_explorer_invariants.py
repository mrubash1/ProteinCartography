"""Two ways a panel can be wrong without anything failing.

Both are the same shape and both have bitten this branch: a panel that is
declared drawable renders as NOTHING, and nothing notices, because an absent
renderer and an empty dataset look identical in a browser. Four blank-panel
defects passed the entire suite in one session here.

**(a) A drawable panel whose `panel_type` has no renderer.** `catalogue_for`
decides drawability in Python; `SHEET_PANELS` supplies the renderers in
JavaScript. Nothing joined the two, so a panel could be marked drawable, be
handed real data, and produce an empty box. The renderers are read from the
RENDERED PAGE rather than from `template.py`'s source, because the page is what
the reader receives -- the same reasoning that found `HTML-PLAN` shipping in
every built page when the metadata guard could not see it.

**(b) A `needs` key that no payload will ever supply.** A mistyped need is
worse than a missing panel: the panel awaits forever, and "awaiting" is a
legitimate state, so it reads as correct. Every need must therefore be either
producible today or on an explicit list of inputs the pipeline cannot build.

Both invariants HOLD at the commit that added them -- they are regression
guards, not bug reports -- so each carries a planted-failure test. A check that
has never been seen to fail is not a check.

Deliberately bare-env: `explorer.template`, `explorer.panels` and
`explorer.payload` import with nothing installed, which
`test_optional_dependencies.py` asserts, and this file must not be the reason
that stops being true.
"""

from __future__ import annotations
import re

from explorer.panels import CATALOGUE, catalogue_for
from explorer.payload import PRODUCIBLE_KEYS
from explorer.template import render

#: Needs no payload can supply today. Every one is an input the pipeline does
#: not build: a phylogeny (`gene_tree`, `species_tree`, `patristic`, `clades`,
#: `ancestors`, `tree_corpus`), the identity axis that would have to come with
#: one, the metric embedding, and the perturbation grid. They are listed so a
#: MISTYPED need is a failure rather than a panel that waits forever -- and so
#: that adding one is a decision someone makes on purpose.
AWAITING_INPUTS = frozenset(
    {
        "ancestors",
        "clades",
        "gene_tree",
        "identity",
        "metric_embedding",
        "patristic",
        "perturbation",
        "species_tree",
        "tree_corpus",
    }
)


def _declared_renderers() -> set:
    """Every panel_type the PAGE can actually draw.

    Parsed from the rendered output, not from `template.py`, because the page
    is the artifact. Two sources: `SHEET_PANELS.<type> = {...}` for panels the
    sheet renders, and `BUILT_ELSEWHERE` for the three the page assembles
    outside that table.
    """
    html = render({"spaces": []}, plotly_js="", title="t")
    renderers = set(re.findall(r"SHEET_PANELS\.([A-Za-z_][A-Za-z0-9_]*)\s*=", html))
    renderers |= set(re.findall(r"SHEET_PANELS\[[\"']([^\"']+)[\"']\]\s*=", html))
    elsewhere = re.search(r"BUILT_ELSEWHERE\s*=\s*new Set\(\[([^\]]*)\]\)", html)
    assert elsewhere, "BUILT_ELSEWHERE is gone from the page; this parser is now blind"
    renderers |= set(re.findall(r"[\"']([^\"']+)[\"']", elsewhere.group(1)))
    assert renderers, "parsed no renderers at all, which would make every check below vacuous"
    return renderers


def _drawable_without_a_renderer(entries, renderers) -> list:
    """The join nothing else performs: drawable in Python, absent in JavaScript."""
    return sorted(
        (entry["panel_id"], entry["panel_type"])
        for entry in entries
        if entry["drawable"] and entry["panel_type"] not in renderers
    )


def test_every_drawable_panel_has_something_that_can_draw_it():
    """Under everything the payload can produce, no panel is drawable and blank."""
    offences = _drawable_without_a_renderer(
        catalogue_for(set(PRODUCIBLE_KEYS)), _declared_renderers()
    )
    assert not offences, (
        f"these panels are marked drawable and the page has no renderer for them: {offences}. "
        "They would render as an empty box, which is indistinguishable from a panel "
        "whose data happened to be empty."
    )


def test_the_renderer_check_would_catch_one():
    """The detector against a planted panel, not a re-typed copy of its condition."""
    planted = [{"panel_id": "planted", "panel_type": "no_such_renderer", "drawable": True}]
    assert _drawable_without_a_renderer(planted, {"cards"}) == [("planted", "no_such_renderer")]
    # ...and an AWAITING panel with no renderer is fine, which is the whole
    # reason the check is on `drawable` rather than on every panel.
    waiting = [{"panel_id": "waiting", "panel_type": "no_such_renderer", "drawable": False}]
    assert _drawable_without_a_renderer(waiting, {"cards"}) == []

    # Stronger than the synthetic case: take a renderer away from the REAL page
    # and the REAL catalogue, and the real check must fire. A detector proven
    # only against hand-made input can still be blind to the shape it ships on.
    real = catalogue_for(set(PRODUCIBLE_KEYS))
    assert _drawable_without_a_renderer(real, _declared_renderers() - {"cards"}), (
        "removing the `cards` renderer left the check silent, so it is not "
        "actually reading the catalogue it claims to guard"
    )


def _unknown_needs(needs, producible, awaiting) -> list:
    """Needs that nothing produces and nobody declared missing."""
    return sorted(set(needs) - set(producible) - set(awaiting))


def test_every_need_is_either_producible_or_a_declared_missing_input():
    """A mistyped need makes a panel await forever, and awaiting looks correct."""
    unknown = _unknown_needs(
        {need for spec in CATALOGUE for need in spec.needs}, PRODUCIBLE_KEYS, AWAITING_INPUTS
    )
    assert not unknown, (
        f"these needs are neither produced nor declared missing: {unknown}. "
        "A need nothing supplies leaves its panel awaiting forever, which reads "
        "as a legitimate state. Add it to AWAITING_INPUTS deliberately, or fix the spelling."
    )


def test_the_awaiting_list_is_not_a_dumping_ground():
    """Every declared-missing input must still be claimed by a panel.

    Without this, `AWAITING_INPUTS` absorbs a typo forever: someone adds the
    misspelling to silence the test above and the panel never draws. An entry
    that no panel needs is either a fixed typo nobody cleaned up or an input
    that stopped being wanted.
    """
    claimed = {need for spec in CATALOGUE for need in spec.needs}
    orphans = sorted(AWAITING_INPUTS - claimed)
    assert not orphans, f"no panel needs these any more: {orphans}"
    assert not (AWAITING_INPUTS & set(PRODUCIBLE_KEYS)), (
        "an input cannot be both produced and awaited; if the pipeline learned to "
        "build one of these, take it off AWAITING_INPUTS so its panel goes drawable"
    )


def test_the_need_check_would_catch_a_typo():
    """The detector against a planted misspelling, and against the two spellings
    it must NOT flag: a produced key, and a declared-missing one."""
    assert _unknown_needs({"censoringg"}, {"censoring"}, {"patristic"}) == ["censoringg"]
    assert _unknown_needs({"censoring"}, {"censoring"}, {"patristic"}) == []
    assert _unknown_needs({"patristic"}, {"censoring"}, {"patristic"}) == []
