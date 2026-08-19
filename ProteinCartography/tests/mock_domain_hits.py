"""Domain-path mock hit lists. Kept free of ``requests`` so pandas-env scripts can import it."""

from pathlib import Path

# Distinct domain-path neighborhoods used by the two-domain Snakemake test.
# These accessions have AlphaFold artifacts. Protein-path mocks still return the
# full actin hit list, so a parent-PDB search that only relabels outputs cannot
# produce this partitioned union.
DOMAIN_SEARCH_HITS = {
    "d01": ["A0A286Q506"],
    "d02": ["Q6QAQ1"],
}


def maybe_write_per_domain_hits(output_file: str) -> bool:
    """If ``output_file`` is a domain-path hit list, write that domain's neighborhood.

    Returns True when the caller should skip normal extraction.
    """
    # Match the FILENAME's id segment, not the whole path. `token in name` also
    # fires when any parent directory happens to contain `__d01.` -- a workdir,
    # a tmpdir, a branch name -- and the caller then silently returns a mocked
    # hit list for a protein-path invocation. Under snakemake that surfaces as a
    # declared output never being written, which reads as a pipeline bug.
    stem = Path(output_file).name.split(".", 1)[0]
    for suffix, hits in DOMAIN_SEARCH_HITS.items():
        if stem.endswith(f"__{suffix}"):
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(hit + "\n" for hit in hits))
            return True
    return False
