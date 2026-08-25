#!/usr/bin/env python
"""Invariants over `Snakefile_domain`, checked by parsing the text.

`Snakefile` declares `benchmark:` on 31 rules and `Snakefile_domain` declared it
on NONE, so the domain path — the half of this pipeline nobody has timed — was
the half with no timings. PC-020's cost estimate for an exhaustive domain run is
a measured floor precisely because no domain rule ever wrote a benchmark file.

Parsed rather than imported. Importing snakemake to ask this question would need
a config, a workflow object and a resolved DAG, and would then only answer it
for the rules that config happens to reach — which for the domain path is none
of them, because the gate is off by default. Reading the file answers it for
every rule unconditionally, which is what an invariant needs.

This is a check rather than a comment, and the difference is the point: a rule
added tomorrow without a benchmark fails here, rather than being noticed by
someone reading a docstring that says it should not happen.
"""

from __future__ import annotations
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SNAKEFILE_DOMAIN = REPO_ROOT / "Snakefile_domain"

BLOCK = re.compile(r"^(\s*)(rule|checkpoint)\s+(\w+):")
SECTION = re.compile(
    r"^(input|output|params|conda|log|threads|resources|benchmark|shell|run|"
    r"wildcard_constraints|priority|group|container|message|retries):"
)


def _blocks(text: str) -> list:
    """`[(name, kind, line_number, {section: [lines]})]` for one Snakefile."""
    lines = text.splitlines()
    found, i = [], 0
    while i < len(lines):
        match = BLOCK.match(lines[i])
        if not match:
            i += 1
            continue
        indent, kind, name = match.group(1), match.group(2), match.group(3)
        body_indent = indent + "    "
        sections: dict = {}
        current = None
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if (
                line.strip()
                and not line.startswith(body_indent)
                and not line.strip().startswith("#")
            ):
                break
            stripped = line.strip()
            section = SECTION.match(stripped)
            if section:
                current = section.group(1)
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
            j += 1
        found.append((name, kind, i + 1, sections))
        i = j
    return found


def test_every_domain_rule_is_timed():
    """A rule with no `benchmark:` is a rule whose cost is unknowable after the
    fact, and the domain path is the expensive half."""
    missing = [
        f"{kind} {name} at Snakefile_domain:{line}"
        for name, kind, line, sections in _blocks(SNAKEFILE_DOMAIN.read_text())
        if "benchmark" not in sections
    ]
    assert not missing, "these domain rules declare no benchmark:\n" + "\n".join(missing)


def test_there_are_domain_rules_to_check():
    """The guard above passes vacuously over an empty list, and a parser that
    silently stops matching is exactly how that would happen."""
    blocks = _blocks(SNAKEFILE_DOMAIN.read_text())
    assert len(blocks) >= 20, f"only {len(blocks)} rule/checkpoint blocks parsed"
    names = {name for name, _, _, _ in blocks}
    assert "domain_query_gate" in names
    assert "domain_plot_cluster_distributions" in names


def test_the_guard_would_catch_a_rule_without_one():
    """A check that has never been seen to fail is decoration.

    The offence is planted in a copy of the real text rather than in a synthetic
    fixture, so what is proved is that this parser catches it in THIS file's
    shape -- indented rules inside an `if`, multi-line `expand()` outputs and
    all.
    """
    text = SNAKEFILE_DOMAIN.read_text()
    stripped = re.sub(
        r"\n\s*benchmark:\n\s*BENCHMARKS_DIR / \"[^\"]*domain_assess_pdbs\.txt\"", "", text
    )
    assert stripped != text, "the planted edit changed nothing; the anchor moved"
    missing = [name for name, _, _, sections in _blocks(stripped) if "benchmark" not in sections]
    assert missing == ["domain_assess_pdbs"], missing


def test_a_benchmark_carries_the_same_wildcards_as_its_outputs():
    """snakemake refuses a rule whose benchmark and outputs disagree, and it
    refuses it at PARSE time -- so this is already enforced for anyone who runs
    a dry run.

    Asserted here anyway because the failure is easy to reintroduce and its
    message names the symptom rather than the cause. Two ways to get it wrong,
    both hit while writing this: `f"{ANALYSIS_NAME}_x.tsv"` is an f-string field
    interpolated at parse time and is NOT a wildcard, and inside `expand(...)`
    it is the DOUBLED braces that survive as the wildcard while single ones are
    consumed as expand's own parameters.
    """
    for name, _kind, line, sections in _blocks(SNAKEFILE_DOMAIN.read_text()):
        benchmark = " ".join(sections.get("benchmark", []))
        declared = set(re.findall(r"\{(\w+)\}", benchmark))
        outputs = " ".join(sections.get("output", []))
        expected = set(re.findall(r"\{\{(\w+)\}\}", outputs))
        if not expected:
            expected = {
                w
                for literal in re.finditer(r'(?<!f)"([^"]*)"', outputs)
                for w in re.findall(r"\{(\w+)\}", literal.group(1))
            }
            if "expand(" in outputs:
                continue  # settled by the doubled-brace branch above
        assert declared == expected, (
            f"{name} at Snakefile_domain:{line}: benchmark declares {sorted(declared)} "
            f"but its outputs carry {sorted(expected)}"
        )
