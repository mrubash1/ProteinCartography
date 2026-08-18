"""The reader and the writer must agree on where a file is.

Four explorer defects have come from a reader and a writer that each spelled a
filename out for themselves and were never compared (GE.5 and GE.17 are the two
most recent). The tests written to prevent them could not: they built their
fixtures with the *reader's* spelling, so renaming the writer left every one of
them green.

Two kinds of test here, and the second is the one that matters.

1. `spaces.layout` is the single definition, and it stays import-light.
2. **End-to-end**: run the real writer, then read the result with the real
   reader. No filename literal appears on either side, so this fails if the two
   ever drift, whatever they drift to.
"""

from __future__ import annotations
import os
import subprocess
import sys
import tempfile

import pytest
from spaces import layout


def test_layout_does_not_import_the_heavy_stack():
    """The explorer payload imports this, and it runs in the bare environment.

    ADR 0006: the payload must not pull numpy, pandas or the reducer stack. If
    `layout` ever imports a sibling pipeline module, it drags that in with it.
    """
    code = (
        "import sys; import spaces.layout; "
        "bad = [m for m in ('numpy', 'pandas', 'sklearn', 'umap', 'plotly') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(os.path.abspath(layout.__file__)))]
        + [p for p in [env.get("PYTHONPATH")] if p]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing spaces.layout pulled in {result.stdout.strip()}; it must stay dependency-free "
        "so the explorer payload can import it in the bare environment"
    )


def test_the_pair_filename_round_trips_through_the_separator():
    assert layout.pair_filename("a", "b") == f"a{layout.PAIR_SEPARATOR}b.tsv"


def test_the_space_manifest_is_never_a_bare_manifest_json():
    """`manifest.json` is the *block* store's name. Reading it at the space
    level returned nothing and rendered an empty provenance footer (GE.5)."""
    for reducer in ("pca_umap", "pca_tsne", layout.DIAGNOSTICS_MANIFEST_KEY):
        assert layout.manifest_filename(reducer) != "manifest.json"
        assert reducer in layout.manifest_filename(reducer)


def test_coregister_writes_where_the_explorer_reads(monkeypatch, tmp_path):
    """The real guard: drive `coregister.main()`, then read it with the payload.

    Neither side is given a filename by this test. `coregister` builds its
    output paths from `spaces.layout` and `explorer.payload` rebuilds them from
    the same module, so if the two ever disagree this fails -- which is exactly
    what did not happen when disagreement mode shipped colouring every protein
    `null` (GE.17), and what a fixture written with the reader's spelling can
    never detect.
    """
    pytest.importorskip("pandas")
    test_coregister = pytest.importorskip("test_coregister")
    from explorer.payload import _read_comparisons

    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    features = output_dir / "protein_features"
    features.mkdir(parents=True)
    rows = ["protid\tSequence\tPfam\tInterPro"]
    for protid, sequence in test_coregister.SEQUENCES.items():
        rows.append(f"{protid}\t{sequence}\t{test_coregister.DOMAINS[protid]}\t")
    (features / "uniprot_features.tsv").write_text("\n".join(rows) + "\n")

    config_path = test_coregister.write_config(
        run_dir, {"compare": ["physicochemistry", "families"], "k": 2}
    )
    test_coregister.build_blocks(monkeypatch, config_path, str(output_dir))
    assert test_coregister.run(monkeypatch, config_path, str(output_dir)) == 0

    summary = layout.summary_path(str(output_dir))
    assert os.path.exists(summary), "coregister did not write the summary the explorer reads"

    rows = _read_comparisons(summary)
    assert rows, "the explorer read no comparison rows from a run that wrote them"
    for row in rows:
        assert row["per_protein"], (
            f"the explorer found no per-protein detail for "
            f"{row.get('space_a')} vs {row.get('space_b')}; the pair file it looked for is "
            f"{layout.pair_filename(row.get('space_a'), row.get('space_b'))}. This is the "
            "GE.17 failure: disagreement mode colours every protein null and nothing raises."
        )
