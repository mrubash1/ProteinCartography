"""TED gate, user TSV precedence, cache, and omit-on-miss behavior."""

from __future__ import annotations
from pathlib import Path

import assign_domains
import domain_utils as du
import pytest

P00698_PAYLOAD = {
    "data": [
        {
            "ted_id": "AF-P00698-F1-model_v4_TED01",
            "uniprot_acc": "P00698",
            "chopping": "22-141",
            "nres_domain": 120,
            "cath_label": "1.10.530.10",
        }
    ],
    "count": 1,
}

TWO_DOMAIN_PAYLOAD = {
    "data": [
        {
            "ted_id": "AF-P99999-F1-model_v4_TED01",
            "uniprot_acc": "P99999",
            "chopping": "1-80",
            "nres_domain": 80,
            "cath_label": "3.40.50.300",
        },
        {
            "ted_id": "AF-P99999-F1-model_v4_TED02",
            "uniprot_acc": "P99999",
            "chopping": "81-160",
            "nres_domain": 80,
            "cath_label": "1.10.10.10",
        },
    ],
    "count": 2,
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._payload = payload if payload is not None else {"data": [], "count": 0}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(
        self, responses: dict[str, FakeResponse], errors: dict[str, Exception] | None = None
    ):
        self.responses = responses
        self.errors = errors or {}
        self.urls: list[str] = []
        self._hits: dict[str, int] = {}

    def get(self, url, timeout=60):
        self.urls.append(url)
        for key, err in self.errors.items():
            if key in url:
                raise err
        for key, response in self.responses.items():
            if key in url:
                if isinstance(response, list):
                    idx = self._hits.get(key, 0)
                    self._hits[key] = idx + 1
                    return response[min(idx, len(response) - 1)]
                return response
        return FakeResponse(404)


def _write_query_files(input_dir: Path, protid: str, nres: int):
    seq = "A" * nres
    (input_dir / f"{protid}.fasta").write_text(f">{protid}\n{seq}\n")
    atoms = []
    for i in range(1, nres + 1):
        atoms.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  "
        )
    (input_dir / f"{protid}.pdb").write_text("\n".join(atoms) + "\nEND\n")


def test_p00698_gate_off(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P00698", 147)
    session = FakeSession({"P00698": FakeResponse(200, P00698_PAYLOAD)})
    gate = assign_domains.run_query_gate(
        ["P00698"], str(input_dir), str(tmp_path / "domain"), session=session
    )
    assert gate == "off"
    assert (tmp_path / "domain" / "query_domain_ids.txt").read_text() == ""


def test_two_domain_ted_gate_on(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    out = tmp_path / "domain"
    gate = assign_domains.run_query_gate(["P99999"], str(input_dir), str(out), session=session)
    assert gate == "on"
    ids = (out / "query_domain_ids.txt").read_text().splitlines()
    assert ids == ["P99999__d01", "P99999__d02"]
    assert (out / "query_structures" / "P99999__d01.fasta").is_file()
    assert (out / "query_structures" / "P99999__d02.pdb").is_file()
    fasta = (out / "query_structures" / "P99999__d01.fasta").read_text()
    assert fasta.startswith(">P99999__d01")
    assert "P99999.pdb" not in fasta


def test_user_tsv_wins_over_ted(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    tsv = tmp_path / "user.tsv"
    tsv.write_text("parent_protid\tchopping\nP99999\t1-50\nP99999\t51-160\n")
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    out = tmp_path / "domain"
    assign_domains.run_query_gate(
        ["P99999"],
        str(input_dir),
        str(out),
        user_domains_file=str(tsv),
        session=session,
    )
    text = (out / "query_domains.tsv").read_text()
    assert "user" in text
    assert "1-50" in text
    assert session.urls == []


def test_user_tsv_start_end_columns(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    tsv = tmp_path / "user.tsv"
    tsv.write_text("parent_protid\tstart\tend\nP99999\t1\t50\nP99999\t51\t160\n")
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    out = tmp_path / "domain"
    gate = assign_domains.run_query_gate(
        ["P99999"],
        str(input_dir),
        str(out),
        user_domains_file=str(tsv),
        session=session,
    )
    assert gate == "on"
    text = (out / "query_domains.tsv").read_text()
    assert "1-50" in text
    assert "51-160" in text


def test_mixed_queries_only_crop_multidomain(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P00698", 147)
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession(
        {
            "P00698": FakeResponse(200, P00698_PAYLOAD),
            "P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD),
        }
    )
    out = tmp_path / "domain"
    gate = assign_domains.run_query_gate(
        ["P00698", "P99999"], str(input_dir), str(out), session=session
    )
    assert gate == "on"
    ids = (out / "query_domain_ids.txt").read_text().splitlines()
    assert ids == ["P99999__d01", "P99999__d02"]
    assert not (out / "query_structures" / "P00698__d01.fasta").exists()
    assert (out / "query_structures" / "P99999__d01.fasta").is_file()


def test_query_ted_404_does_not_raise(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession({"P99999": FakeResponse(404)})
    gate = assign_domains.run_query_gate(
        ["P99999"], str(input_dir), str(tmp_path / "domain"), session=session
    )
    assert gate == "off"


def test_query_ted_500_does_not_raise(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession({"P99999": FakeResponse(500)})
    gate = assign_domains.run_query_gate(
        ["P99999"], str(input_dir), str(tmp_path / "domain"), session=session
    )
    assert gate == "off"


def test_ted_cache_skips_second_get(tmp_path: Path):
    cache = tmp_path / "cache"
    session = FakeSession({"P00698": FakeResponse(200, P00698_PAYLOAD)})
    first = assign_domains.fetch_ted_summary("P00698", session, cache)
    assert first["data"][0]["chopping"] == "22-141"
    assert len(session.urls) == 1
    session.errors = {"P00698": RuntimeError("network down")}
    second = assign_domains.fetch_ted_summary("P00698", session, cache)
    assert second["data"][0]["chopping"] == "22-141"
    assert len(session.urls) == 1


def test_assign_hits_omits_ted_404(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    atoms = [
        (
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  "
        )
        for i in range(1, 101)
    ]
    (pdb_dir / "P11111.pdb").write_text("\n".join(atoms) + "\nEND\n")
    hits = tmp_path / "hits.txt"
    hits.write_text("P11111\n")
    session = FakeSession({"P11111": FakeResponse(404)})
    rows = assign_domains.assign_and_crop_hits(
        accessions_file=str(hits),
        pdb_dir=str(pdb_dir),
        output_dir=str(tmp_path / "domain_structures"),
        output_tsv=str(tmp_path / "domain_features.tsv"),
        cache_dir=str(tmp_path / "cache"),
        session=session,
    )
    assert rows == []
    assert not (tmp_path / "domain_structures" / "P11111__d01.pdb").exists()
    tsv = (tmp_path / "domain_features.tsv").read_text()
    assert "full_chain" not in tsv


def test_assign_hits_crops_both_domains_and_matches_ca_count(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_query_files(pdb_dir, "P99999", 160)
    hits = tmp_path / "hits.txt"
    hits.write_text("P99999\n")
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    out_dir = tmp_path / "domain_structures"
    rows = assign_domains.assign_and_crop_hits(
        accessions_file=str(hits),
        pdb_dir=str(pdb_dir),
        output_dir=str(out_dir),
        output_tsv=str(tmp_path / "domain_features.tsv"),
        cache_dir=str(tmp_path / "cache"),
        session=session,
    )
    assert {row["protid"] for row in rows} == {"P99999__d01", "P99999__d02"}
    assert "full_chain" not in {row["assignment_source"] for row in rows}
    for row in rows:
        pdb_text = (out_dir / f"{row['protid']}.pdb").read_text()
        resseqs = [int(line[22:26]) for line in pdb_text.splitlines() if line.startswith("ATOM")]
        start, end = int(row["start"]), int(row["end"])
        assert resseqs
        assert min(resseqs) >= start
        assert max(resseqs) <= end
        assert int(row["nres_domain"]) == len(resseqs)


def test_domain_map_off_skips_ted(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    gate = assign_domains.run_query_gate(
        ["P99999"],
        str(input_dir),
        str(tmp_path / "domain"),
        domain_map="off",
        session=session,
    )
    assert gate == "off"
    assert session.urls == []


def test_assign_hits_skips_query_parents_already_on_the_map(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_query_files(pdb_dir, "P99999", 160)
    query_tsv = tmp_path / "query_domains.tsv"
    assign_domains._write_domains_tsv(
        query_tsv,
        [
            du.domain_row("P99999", 1, "1-80", "ted"),
            du.domain_row("P99999", 2, "81-160", "ted"),
        ],
    )
    query_structs = tmp_path / "query_structures"
    query_structs.mkdir()
    (query_structs / "P99999__d01.pdb").write_text((pdb_dir / "P99999.pdb").read_text())
    (query_structs / "P99999__d02.pdb").write_text((pdb_dir / "P99999.pdb").read_text())
    hits = tmp_path / "hits.txt"
    hits.write_text("P99999\n")
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    rows = assign_domains.assign_and_crop_hits(
        accessions_file=str(hits),
        pdb_dir=str(pdb_dir),
        output_dir=str(tmp_path / "domain_structures"),
        output_tsv=str(tmp_path / "domain_features.tsv"),
        cache_dir=str(tmp_path / "cache"),
        query_domains_tsv=str(query_tsv),
        query_structures_dir=str(query_structs),
        session=session,
    )
    protids = [row["protid"] for row in rows]
    assert protids == ["P99999__d01", "P99999__d02"]
    assert session.urls == []


def test_assign_hits_drops_empty_crop(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    atoms = [
        (
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  "
        )
        for i in range(200, 221)
    ]
    (pdb_dir / "P11111.pdb").write_text("\n".join(atoms) + "\nEND\n")
    hits = tmp_path / "hits.txt"
    hits.write_text("P11111\n")
    payload = {
        "data": [
            {
                "ted_id": "AF-P11111-F1-model_v4_TED01",
                "uniprot_acc": "P11111",
                "chopping": "1-80",
                "nres_domain": 80,
                "cath_label": "1.10.10.10",
            }
        ],
        "count": 1,
    }
    session = FakeSession({"P11111": FakeResponse(200, payload)})
    rows = assign_domains.assign_and_crop_hits(
        accessions_file=str(hits),
        pdb_dir=str(pdb_dir),
        output_dir=str(tmp_path / "domain_structures"),
        output_tsv=str(tmp_path / "domain_features.tsv"),
        cache_dir=str(tmp_path / "cache"),
        session=session,
    )
    assert rows == []
    assert not (tmp_path / "domain_structures" / "P11111__d01.pdb").exists()


def test_ted_429_retries_then_succeeds(tmp_path: Path):
    session = FakeSession(
        {
            "P00698": [
                FakeResponse(429),
                FakeResponse(200, P00698_PAYLOAD),
            ]
        }
    )
    payload = assign_domains.fetch_ted_summary("P00698", session, tmp_path / "cache")
    assert payload["data"][0]["chopping"] == "22-141"
    assert len(session.urls) == 2


def test_ted_paginates_until_short_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(du, "TED_PAGE_SIZE", 1)
    page1 = {"data": [TWO_DOMAIN_PAYLOAD["data"][0]], "count": 1}
    page2 = {"data": [TWO_DOMAIN_PAYLOAD["data"][1]], "count": 1}
    page3 = {"data": [], "count": 0}

    class PagingSession:
        def __init__(self):
            self.urls: list[str] = []

        def get(self, url, timeout=60):
            self.urls.append(url)
            if "skip=2" in url:
                return FakeResponse(200, page3)
            if "skip=1" in url:
                return FakeResponse(200, page2)
            return FakeResponse(200, page1)

    session = PagingSession()
    payload = assign_domains.fetch_ted_summary("P99999", session, tmp_path / "cache")
    assert len(payload["data"]) == 2
    assert payload["data"][0]["chopping"] == "1-80"
    assert payload["data"][1]["chopping"] == "81-160"
    assert len(session.urls) == 3


# ==========================================================================
# PC-021 phase 1 -- no TED payload can fail the run
# ==========================================================================


def _unusable_payload(accession: str) -> dict:
    """A TED payload built from `mocks.UNUSABLE_TED_PAYLOADS`, so the two failure
    shapes have one spelling between the unit tests here and the pipeline
    integration tests that go through `mocks.mock_ted_api_responses`."""
    from mocks import UNUSABLE_TED_PAYLOADS

    return {
        "data": [
            {
                "ted_id": f"AF-{accession}-F1-model_v4_TED{i:02d}",
                "uniprot_acc": accession,
                "chopping": chopping,
                "nres_domain": 80,
                "cath_label": "1.10.10.10",
            }
            for i, chopping in enumerate(UNUSABLE_TED_PAYLOADS[accession], start=1)
        ],
        "count": 2,
    }


@pytest.mark.parametrize(
    ("accession", "what"),
    [
        ("P90001", "a chopping string that is not a range"),
        # P90002 -- a chopping that overruns the staged FASTA -- USED to belong
        # here and moved to the phase-2 section below. It no longer needs
        # omitting: the crop clips it and keeps the domain. Left as a comment
        # rather than deleted, because "this case moved" is the fact a reader of
        # the phase-1 commit will be looking for.
    ],
)
def test_an_unusable_ted_payload_leaves_the_protein_path_green(
    tmp_path: Path, capsys, accession, what
):
    """`assign_parent`'s docstring said "Never raises" and it could.

    `parse_chopping` raises `DomainChoppingError` -- a `ValueError` -- from
    inside `domain_row`, while the row is being BUILT, so every caller of
    `rows_from_ted_payload` was a place a remote payload could fail the whole
    protein pipeline for a defect in an optional domain map.

    Both accessions parse as two domains, so each would pass the multi-domain
    gate if it parsed at all; that is the only way to reach the code under test.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, accession, 160)
    session = FakeSession({accession: FakeResponse(200, _unusable_payload(accession))})
    out = tmp_path / "domain"

    gate = assign_domains.run_query_gate([accession], str(input_dir), str(out), session=session)

    assert gate == "off", what
    # All three outputs exist, because the checkpoint's consumers read them
    # whatever the gate says.
    assert (out / "gate.txt").read_text().strip() == "off"
    assert (out / "query_domain_ids.txt").read_text() == ""
    assert (out / "query_domains.tsv").is_file()
    # And it says which accession and why, in the same shape as the existing
    # miss messages.
    stderr = capsys.readouterr().err
    assert accession in stderr
    assert "omitting" in stderr


def test_a_failure_that_is_not_bad_data_still_surfaces(tmp_path: Path):
    """The guard is a NAMED tuple, not `except Exception`.

    A payload that is a list where a dict was expected makes `payload.get`
    raise `AttributeError`. That is a bug in this file rather than bad data from
    TED, and swallowing it would turn every future mistake here into a silent
    "not multi-domain".
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P90003", 160)
    session = FakeSession({"P90003": FakeResponse(200, ["not", "a", "dict"])})
    with pytest.raises(AttributeError):
        assign_domains.run_query_gate(
            ["P90003"], str(input_dir), str(tmp_path / "domain"), session=session
        )


def test_one_bad_parent_does_not_take_the_others_with_it(tmp_path: Path, capsys):
    """The guard is inside the per-parent loop, so the gate goes on.

    Before it, one unparseable chopping ended the run; a cohort with one bad
    accession and one good one produced no domain map at all rather than a map
    of the good one.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P90001", 160)
    _write_query_files(input_dir, "P99999", 160)
    session = FakeSession(
        {
            "P90001": FakeResponse(200, _unusable_payload("P90001")),
            "P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD),
        }
    )
    out = tmp_path / "domain"

    gate = assign_domains.run_query_gate(
        ["P90001", "P99999"], str(input_dir), str(out), session=session
    )

    assert gate == "on"
    assert (out / "query_domain_ids.txt").read_text().splitlines() == [
        "P99999__d01",
        "P99999__d02",
    ]
    assert "P90001" in capsys.readouterr().err


def test_a_crop_that_cannot_read_its_files_omits_the_parent(tmp_path: Path, capsys, monkeypatch):
    """`OSError` is in the crop guard's set and not in the parse guard's.

    The crop branch touches the filesystem, and an unreadable PDB is the same
    kind of event as an unparseable chopping -- neither is a reason to fail the
    protein pipeline. Asserted separately because it reaches a different
    `except` from every test above.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)

    def explode(*args, **kwargs):
        raise OSError("Input/output error")

    monkeypatch.setattr(assign_domains, "_crop_query_files", explode)
    out = tmp_path / "domain"
    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})

    gate = assign_domains.run_query_gate(["P99999"], str(input_dir), str(out), session=session)

    assert gate == "off"
    stderr = capsys.readouterr().err
    assert "omitting P99999" in stderr
    assert "crop failed" in stderr


# ==========================================================================
# PC-021 phase 2 -- clip a FASTA crop the way a PDB crop is clipped
# ==========================================================================


def test_an_overrunning_chopping_clips_instead_of_failing(tmp_path: Path, capsys):
    """The behaviour change, stated as a test rather than left in a message.

    Before phase 2 this accession was OMITTED -- phase 1 caught the `ValueError`
    and the gate went off. Now the sequence follows the structure: the PDB crop
    has always kept the residues that exist, and the FASTA does too.

    `nres_domain` for the overrunning domain is 80, not the 99999 the chopping
    declared and not the 80 TED reported -- it is the CA count of the crop that
    actually happened.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P90002", 160)
    session = FakeSession({"P90002": FakeResponse(200, _unusable_payload("P90002"))})
    out = tmp_path / "domain"

    gate = assign_domains.run_query_gate(["P90002"], str(input_dir), str(out), session=session)

    assert gate == "on"
    structures = out / "query_structures"
    sequence = "".join((structures / "P90002__d02.fasta").read_text().splitlines()[1:])
    assert len(sequence) == 80, "the FASTA must carry the residues that exist, 81-160"
    # The two halves of the domain agree, which is the whole point of the reorder.
    atoms = (structures / "P90002__d02.pdb").read_text().splitlines()
    assert sum(1 for line in atoms if line.startswith("ATOM")) == len(sequence)

    stderr = capsys.readouterr().err
    assert "clipped P90002__d02" in stderr
    assert "declares residue 99999" in stderr
    assert "the sequence has 160" in stderr


def test_only_the_overrunning_segment_of_a_discontinuous_chopping_is_clipped(tmp_path: Path):
    """A domain built from two spans loses only what runs past the end.

    Clipping the whole chopping, or dropping it, would quietly change which
    residues the domain is made of rather than how many.
    """
    sequence = "".join(chr(65 + (i % 26)) for i in range(100))
    sliced, overrun = du._slice_with_loss(sequence, "1-20_90-150", on_overrun="clip")
    assert sliced == sequence[0:20] + sequence[89:100]
    assert overrun.declared_end == 150
    assert overrun.sequence_length == 100
    assert overrun.residues_lost == 50
    assert overrun.segments_dropped == 0


def test_a_segment_that_starts_past_the_end_is_dropped_not_clipped_to_nothing(tmp_path: Path):
    """A zero-length span and a dropped span are different facts about the
    chopping, and the report says which happened."""
    sequence = "A" * 50
    sliced, overrun = du._slice_with_loss(sequence, "1-20_80-120", on_overrun="clip")
    assert sliced == "A" * 20
    assert overrun.segments_dropped == 1
    assert overrun.residues_lost == 41


def test_a_chopping_entirely_past_the_end_writes_no_fasta(tmp_path: Path):
    """A FASTA carrying a header and no residues is a file every downstream
    reader would accept and none would flag."""
    source = tmp_path / "P.fasta"
    source.write_text(">P\n" + "A" * 50 + "\n")
    out = tmp_path / "P__d01.fasta"
    sliced, overrun = du.crop_fasta_file(source, "80-120", out, "P__d01", on_overrun="clip")
    assert sliced == ""
    assert overrun.segments_dropped == 1
    assert not out.exists()


def test_the_default_still_raises(tmp_path: Path):
    """`on_overrun` defaults to `raise`, so every existing caller and the
    pure-function test at the top of `test_domain_utils.py` are untouched."""
    with pytest.raises(ValueError, match="exceeds sequence length"):
        du.slice_fasta_sequence("A" * 50, "1-80")
    with pytest.raises(ValueError, match="exceeds sequence length"):
        du.slice_fasta_sequence("A" * 50, "1-80", on_overrun="raise")
    with pytest.raises(ValueError, match="on_overrun must be"):
        du.slice_fasta_sequence("A" * 50, "1-10", on_overrun="truncate")


def test_a_domain_whose_pdb_crops_to_nothing_leaves_no_fasta_behind(tmp_path: Path):
    """The reorder deletes the orphan-FASTA unlink branch rather than adding to
    it: a file never written needs no cleanup.

    The PDB here covers residues 1-80 only, so the second domain's crop returns
    no CA atoms at all.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_query_files(input_dir, "P99999", 160)
    # Truncate the PDB to the first domain's residues, leaving the FASTA whole.
    pdb = input_dir / "P99999.pdb"
    kept = [
        line
        for line in pdb.read_text().splitlines()
        if not line.startswith("ATOM") or int(line[22:26]) <= 80
    ]
    pdb.write_text("\n".join(kept) + "\n")

    session = FakeSession({"P99999": FakeResponse(200, TWO_DOMAIN_PAYLOAD)})
    out = tmp_path / "domain"
    gate = assign_domains.run_query_gate(["P99999"], str(input_dir), str(out), session=session)

    structures = out / "query_structures"
    assert (structures / "P99999__d01.fasta").is_file()
    assert not (structures / "P99999__d02.pdb").exists()
    assert not (
        structures / "P99999__d02.fasta"
    ).exists(), "the second domain has no structure, so it must have no sequence either"
    assert gate == "off", "one kept domain is not multi-domain"
