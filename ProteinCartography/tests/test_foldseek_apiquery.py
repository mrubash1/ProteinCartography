"""Unit tests for the Foldseek API query, focused on the polling loop.

These tests never touch the network: `session_with_retry` and `sleep` are both
replaced, so the whole poll runs in memory and in no time at all.
"""

from __future__ import annotations
import pathlib

import foldseek_apiquery
import pytest


class _FakeResponse:
    """The subset of `requests.Response` that `foldseek_apiquery` actually uses."""

    def __init__(self, *, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self.reason = "OK"
        self._json_data = json_data
        self._content = content

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=128):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]


class _FakeSession:
    """A Foldseek server that reports `status` forever, then serves a download."""

    def __init__(self, status, content=b"results"):
        self.status = status
        self.content = content
        self.polls = 0
        self.downloads = 0

    def post(self, url, data=None, auth=None):
        return _FakeResponse(json_data={"id": "TICKET-1"})

    def get(self, url, auth=None, stream=False):
        if "/api/result/download/" in url:
            self.downloads += 1
            return _FakeResponse(content=self.content)
        self.polls += 1
        return _FakeResponse(json_data={"status": self.status})


@pytest.fixture
def query_pdb(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "query.pdb"
    path.write_text("ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 80.00\n")
    return str(path)


@pytest.fixture
def fake_server(monkeypatch):
    """Install a fake session and neutralize the poll sleep. Returns a factory."""

    def install(status, content=b"results"):
        session = _FakeSession(status, content)
        monkeypatch.setattr(foldseek_apiquery, "session_with_retry", lambda: session)
        monkeypatch.setattr(foldseek_apiquery, "sleep", lambda seconds: None)
        return session

    return install


def test_a_ticket_that_never_completes_is_an_error_rather_than_a_download(
    query_pdb, tmp_path, fake_server
):
    """A ticket stuck in RUNNING must abort, not fall through to the download.

    This is a regression test for an off-by-one. The loop ran while
    `elapsed < FOLDSEEK_SERVER_TIMEOUT` and the guard below it fired only on
    `elapsed > FOLDSEEK_SERVER_TIMEOUT`. Both poll intervals divide the 1800 s
    timeout exactly, so `elapsed` landed on 1800, the guard was False, and the
    function downloaded the result archive for a search that had never
    finished -- with no error, and no indication in the output that anything
    had gone wrong.
    """
    session = fake_server("RUNNING")
    output_file = tmp_path / "results.tar.gz"

    with pytest.raises(SystemExit) as excinfo:
        foldseek_apiquery.foldseek_apiquery(
            input_file=query_pdb,
            output_file=str(output_file),
            mode="3diaa",
            database=["afdb50"],
            server=foldseek_apiquery.PUBLIC_FOLDSEEK_SERVER,
        )

    assert "failed to complete" in str(excinfo.value)
    assert session.downloads == 0, "a never-completing ticket must not be downloaded"
    assert not output_file.exists(), "no output file may be written for a failed search"


def test_the_poll_loop_is_bounded_by_the_timeout(query_pdb, tmp_path, fake_server):
    """The loop must stop after the timeout rather than polling forever."""
    session = fake_server("RUNNING")
    expected_polls = foldseek_apiquery.FOLDSEEK_SERVER_TIMEOUT // 30

    with pytest.raises(SystemExit):
        foldseek_apiquery.foldseek_apiquery(
            input_file=query_pdb,
            output_file=str(tmp_path / "results.tar.gz"),
            mode="3diaa",
            database=["afdb50"],
            server=foldseek_apiquery.PUBLIC_FOLDSEEK_SERVER,
        )

    assert session.polls == expected_polls


def test_a_completed_ticket_is_downloaded(query_pdb, tmp_path, fake_server):
    """The negative control: the guard must not reject a search that did finish.

    Without this, making the guard stricter would pass the test above by
    aborting every search, successful or not.
    """
    session = fake_server("COMPLETE", content=b"tar.gz payload")
    output_file = tmp_path / "results.tar.gz"

    foldseek_apiquery.foldseek_apiquery(
        input_file=query_pdb,
        output_file=str(output_file),
        mode="3diaa",
        database=["afdb50"],
        server=foldseek_apiquery.PUBLIC_FOLDSEEK_SERVER,
    )

    assert session.polls == 1
    assert session.downloads == 1
    assert output_file.read_bytes() == b"tar.gz payload"


def test_a_ticket_that_errors_aborts_before_any_download(query_pdb, tmp_path, fake_server):
    """An explicit ERROR status must abort too, and must not write output."""
    session = fake_server("ERROR")
    output_file = tmp_path / "results.tar.gz"

    with pytest.raises(SystemExit) as excinfo:
        foldseek_apiquery.foldseek_apiquery(
            input_file=query_pdb,
            output_file=str(output_file),
            mode="3diaa",
            database=["afdb50"],
            server=foldseek_apiquery.PUBLIC_FOLDSEEK_SERVER,
        )

    assert "status ERROR" in str(excinfo.value)
    assert session.downloads == 0
    assert not output_file.exists()
