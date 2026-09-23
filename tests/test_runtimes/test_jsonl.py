"""The append-only reader, tested before anything depends on it.

It is a small amount of code in the path of every telemetry event, and its
failure modes are the quiet kind: a record read twice, a record silently
dropped, a multi-byte character cut in half by a poll that happened to land
mid-write. Each of those looks like a telemetry bug much later and somewhere
else, so they are pinned here while the primitive is still simple enough to
test exhaustively.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader, CorruptRecordError


class Record(BaseModel):
    n: int
    text: str = ""


ADAPTER = TypeAdapter(Record)


def _reader(path: Path) -> AppendOnlyJsonlReader[Record]:
    return AppendOnlyJsonlReader(path, ADAPTER)


def _append(path: Path, *chunks: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(chunk)


def test_a_file_that_does_not_exist_yet_is_not_an_error(tmp_path: Path) -> None:
    """A reader is created before its writer, every time."""
    assert _reader(tmp_path / "absent.jsonl").read_new() == ()


def test_each_record_is_returned_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _append(path, json.dumps({"n": 1}) + "\n", json.dumps({"n": 2}) + "\n")
    reader = _reader(path)

    first = reader.read_new()
    second = reader.read_new()

    assert [r.n for r in first] == [1, 2]
    assert second == (), "a second call must not redeliver what it already gave"


def test_only_what_was_appended_comes_back(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _append(path, json.dumps({"n": 1}) + "\n")
    reader = _reader(path)
    reader.read_new()

    _append(path, json.dumps({"n": 2}) + "\n")

    assert [r.n for r in reader.read_new()] == [2]


def test_an_incomplete_record_waits_for_its_newline(tmp_path: Path) -> None:
    """A poll can land mid-write, and half a record is not a record."""
    path = tmp_path / "records.jsonl"
    _append(path, '{"n": 1, "te')
    reader = _reader(path)

    assert reader.read_new() == (), "a partial line is not yet anything"

    _append(path, 'xt": "done"}\n')

    delivered = reader.read_new()
    assert [(r.n, r.text) for r in delivered] == [(1, "done")]


def test_a_multibyte_character_split_across_reads_survives(tmp_path: Path) -> None:
    """Bytes are buffered, not decoded early.

    Decoding each chunk as it arrives would turn one character split across
    two appends into two decode errors, and the record would be lost or
    reported corrupt when nothing was wrong with it.
    """
    path = tmp_path / "records.jsonl"
    # ensure_ascii=False, or json would escape these to \u sequences and the
    # test would prove nothing about multi-byte handling.
    payload = (json.dumps({"n": 1, "text": "ünïcodé ✓"}, ensure_ascii=False) + "\n").encode("utf-8")
    cut = payload.index(b"\xc3") + 1  # mid-character

    with path.open("ab") as stream:
        stream.write(payload[:cut])
    reader = _reader(path)
    assert reader.read_new() == ()

    with path.open("ab") as stream:
        stream.write(payload[cut:])

    assert [r.text for r in reader.read_new()] == ["ünïcodé ✓"]


def test_a_complete_record_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    """Never skipped. A gap the consumer cannot see is worse than an error."""
    path = tmp_path / "records.jsonl"
    _append(path, json.dumps({"n": 1}) + "\n", "{not json at all}\n")
    reader = _reader(path)

    with pytest.raises(CorruptRecordError, match="line 2"):
        reader.read_new()


def test_a_complete_record_of_the_wrong_shape_is_an_error(tmp_path: Path) -> None:
    """Valid JSON is not the same as a valid record."""
    path = tmp_path / "records.jsonl"
    _append(path, json.dumps({"n": "not-an-int"}) + "\n")

    with pytest.raises(CorruptRecordError):
        _reader(path).read_new()


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """An explicit policy, not an accident.

    A trailing newline is ordinary; refusing it would make an innocuous write
    fatal while saying nothing about whether any record was lost.
    """
    path = tmp_path / "records.jsonl"
    _append(path, json.dumps({"n": 1}) + "\n", "\n", "   \n", json.dumps({"n": 2}) + "\n")

    assert [r.n for r in _reader(path).read_new()] == [1, 2]


def test_history_is_not_rescanned(tmp_path: Path) -> None:
    """The reason this exists at all.

    The reader must read only what was appended, so its work per poll depends
    on new bytes and not on how long the run has been going.
    """
    path = tmp_path / "records.jsonl"
    reader = _reader(path)

    for n in range(200):
        _append(path, json.dumps({"n": n}) + "\n")
        reader.read_new()

    before = reader.offset
    _append(path, json.dumps({"n": 200}) + "\n")
    reader.read_new()

    assert reader.offset - before == len(json.dumps({"n": 200}) + "\n")


def test_a_fresh_reader_replays_the_whole_file(tmp_path: Path) -> None:
    """Byte offset is process-local; durable replay is the StreamCursor.

    A restarted controller builds a new reader and filters against its own
    recorded position, so the file must still be readable from zero.
    """
    path = tmp_path / "records.jsonl"
    for n in range(5):
        _append(path, json.dumps({"n": n}) + "\n")

    used = _reader(path)
    used.read_new()
    assert used.read_new() == ()

    assert [r.n for r in _reader(path).read_new()] == [0, 1, 2, 3, 4]


def test_a_corrupt_line_does_not_take_its_neighbours_with_it(tmp_path: Path) -> None:
    """Raise, but hand over every valid record from the same read.

    The offset has already moved past the whole batch, so a record not
    delivered here is never delivered. Without this, one garbage line written
    by a worker would silently cost the good records on either side of it --
    a worse hole than the one being reported.
    """
    path = tmp_path / "records.jsonl"
    _append(
        path,
        json.dumps({"n": 1}) + "\n",
        "garbage\n",
        json.dumps({"n": 3}) + "\n",
        "{also bad}\n",
    )
    reader = _reader(path)

    with pytest.raises(CorruptRecordError) as caught:
        reader.read_new()

    assert [r.n for r in caught.value.records] == [1, 3]
    assert caught.value.line_numbers == (2, 4)
    assert "and 1 more" in str(caught.value)
    assert reader.read_new() == (), "reported once, not raised again on every poll"
