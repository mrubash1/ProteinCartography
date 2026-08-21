"""The follow-up file's entry numbers are load-bearing.

Twenty-two of them are cited by name from tracked source and documentation. A
renumber, a reorder, or an insertion above the last row silently invalidates
those citations in code that ships — and nothing else in the suite would notice,
because no other test reads this file at all.

Deliberately stdlib-only (`re`, `pathlib`) so it runs in the bare environment
alongside the rest of the explorer tests.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
FOLLOWUPS = REPO / "docs" / "FOLLOWUPS.md"

#: BOTH citation forms in use. `FOLLOWUPS #59` is the common one, but
#: `docs/ARCHITECTURE.md` and `docs/INTERPRETING.md` write ``\`docs/FOLLOWUPS.md\` #51``
#: with a backtick and a space between. A pattern matching only the first form
#: misses three citations across two documents that ship, which is exactly the
#: kind of hole that makes a guard read as passing while it checks less than it
#: claims.
CITATION = re.compile(r"FOLLOWUPS(?:\.md)?`?\s*#(\d+)")

#: A row is `| 59 |` or, once resolved, `| ~~59~~ |`. Struck rows still count:
#: a citation to a resolved entry must keep resolving.
ROW = re.compile(r"^\| ~?~?(\d+)~?~?\s*\|", re.M)

SEARCH_SUFFIXES = (".py", ".md", ".yml", ".yaml", ".toml", ".cfg")


def _tracked_text_files():
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".snakemake", "node_modules"} for part in path.parts):
            continue
        if path.suffix in SEARCH_SUFFIXES or path.name == "Snakefile":
            yield path


def test_every_cited_followup_number_still_resolves_to_a_row():
    """Renumbering is the one edit to this file that breaks something real."""
    rows = {int(n) for n in ROW.findall(FOLLOWUPS.read_text(encoding="utf-8"))}
    assert rows, "no numbered rows parsed out of docs/FOLLOWUPS.md"

    cited = {}
    for path in _tracked_text_files():
        if path == FOLLOWUPS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number in CITATION.findall(text):
            cited.setdefault(int(number), set()).add(str(path.relative_to(REPO)))

    assert cited, "the citation pattern matched nothing; it has probably drifted"

    dangling = {n: sorted(where) for n, where in cited.items() if n not in rows}
    assert not dangling, (
        "these FOLLOWUPS numbers are cited but no longer resolve to a row -- "
        f"an entry was renumbered, reordered or removed: {dangling}"
    )


def test_the_citation_pattern_catches_both_forms_actually_in_use():
    """The guard above is only as good as its pattern, so the pattern is pinned.

    A narrower `FOLLOWUPS #(\\d+)` would miss the backticked-path form and
    silently check less than it claims.
    """
    assert CITATION.findall("see FOLLOWUPS #59 for the transform") == ["59"]
    assert CITATION.findall("a threshold (`docs/FOLLOWUPS.md` #51).") == ["51"]
    assert CITATION.findall("FOLLOWUPS.md #7") == ["7"]
    assert CITATION.findall("no citation here") == []


def test_a_struck_row_still_counts_as_resolving():
    """Entries get struck when resolved; citations to them must keep working."""
    assert ROW.findall("| ~~16~~ | text |") == ["16"]
    assert ROW.findall("| 59 | text |") == ["59"]
