from pathlib import Path

from ai_sorter.database import Database
from ai_sorter.models import Destination
from ai_sorter.processor import DELETE_ALIAS, SortProcessor


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompt = ""
        self.images_were_sent = False

    def generate(self, model: str, prompt: str, images=None, json_response: bool = False):
        self.prompt = prompt
        self.images_were_sent = bool(images)
        return self.response


def test_sort_prompt_uses_aliases_and_delete_option(tmp_path):
    file_path = tmp_path / "invoice.pdf"
    file_path.write_text("invoice", encoding="utf-8")
    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    prompt = processor.build_sort_prompt(
        file_path,
        [Destination(1, "Work Docs", "/tmp/work", "work documents", "movies")],
    )

    assert "Work_Docs: work documents (AVOID: movies)" in prompt
    assert f"Allowed target aliases: Work_Docs, {DELETE_ALIAS}" in prompt
    assert '"action": either "move" or "delete"' in prompt


def test_classify_one_accepts_target_alias_json(tmp_path):
    file_path = tmp_path / "invoice.pdf"
    file_path.write_text("invoice", encoding="utf-8")
    destination = Destination(1, "Work Docs", "/tmp/work", "work documents", "movies")
    client = FakeClient('{"action":"move","target_alias":"Work_Docs","reason":"invoice","confidence":87}')

    decision = SortProcessor(Database(tmp_path / "db.sqlite3")).classify_one(client, "llama", file_path, [destination])

    assert decision.destination_id == 1
    assert decision.destination_name == "Work Docs"
    assert decision.confidence == 87
    assert decision.action == "move"
    assert "Allowed target aliases" in client.prompt


def test_classify_one_accepts_delete_json(tmp_path):
    file_path = tmp_path / "thumb.tmp"
    file_path.write_text("cache", encoding="utf-8")
    destination = Destination(1, "Docs", "/tmp/docs", "documents", "")
    client = FakeClient('{"action":"delete","target_alias":null,"reason":"temporary cache","confidence":92}')

    decision = SortProcessor(Database(tmp_path / "db.sqlite3")).classify_one(client, "llama", file_path, [destination])

    assert decision.action == "delete"
    assert decision.destination_id is None
    assert decision.destination_name == "Удалить"
    assert decision.confidence == 92


def test_prompt_update_prompt_is_english(tmp_path):
    file_path = tmp_path / "invoice.txt"
    file_path.write_text("invoice", encoding="utf-8")
    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))

    prompt = processor.build_prompt_update_prompt(
        file_path,
        Destination(1, "Invoices", "/tmp/invoices", "billing documents", ""),
        Destination(2, "Pictures", "/tmp/pictures", "photos", "documents"),
        "This is a billing file, not a photo.",
    )

    assert "The user corrected a file sorting decision." in prompt
    assert "Return only a JSON object" in prompt
    assert not any("а" <= char.lower() <= "я" or char == "ё" for char in prompt)


def test_archive_context_samples_text_and_lists_remaining_metadata(tmp_path):
    import zipfile

    archive_path = tmp_path / "documents.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index in range(12):
            archive.writestr(f"doc_{index:02d}.txt", f"important invoice text {index}")
        archive.writestr("data.bin", b"\x00\x01\x02")

    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    context = processor.build_file_context(archive_path, FakeClient('{"action":"move"}'), "vision")

    assert "Archive content analysis: 13 regular file(s) detected." in context
    assert "Sampled image/text file(s): 10 of 12 candidate(s)." in context
    assert "important invoice text" in context
    assert "Other archive files (metadata only):" in context
    assert "data.bin | extension: .bin | size: 3 bytes" in context


def test_archive_context_analyzes_sampled_images(tmp_path):
    import zipfile

    archive_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("photo.jpg", b"fake-image-bytes")

    client = FakeClient("a photographed receipt")
    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    context = processor.build_file_context(archive_path, client, "llava")

    assert "photo.jpg" in context
    assert "Image analysis: a photographed receipt" in context
    assert client.images_were_sent


def test_single_file_xz_archive_context_reads_text(tmp_path):
    import lzma

    archive_path = tmp_path / "report.txt.xz"
    archive_path.write_bytes(lzma.compress(b"quarterly archive report"))

    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    context = processor.build_file_context(archive_path, FakeClient(''), "")

    assert processor.is_supported_archive(archive_path)
    assert "report.txt" in context
    assert "quarterly archive report" in context


def test_single_file_bz2_archive_context_reads_text(tmp_path):
    import bz2

    archive_path = tmp_path / "notes.txt.bz2"
    archive_path.write_bytes(bz2.compress(b"compressed meeting notes"))

    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    context = processor.build_file_context(archive_path, FakeClient(''), "")

    assert processor.is_supported_archive(archive_path)
    assert "notes.txt" in context
    assert "compressed meeting notes" in context


def test_7z_and_rar_are_supported_archive_extensions(tmp_path):
    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))

    assert processor.is_supported_archive(tmp_path / "photos.7z")
    assert processor.is_supported_archive(tmp_path / "documents.rar")


def test_external_7z_reader_samples_only_ten_candidates(tmp_path, monkeypatch):
    import subprocess

    archive_path = tmp_path / "bundle.7z"
    processor = SortProcessor(Database(tmp_path / "db.sqlite3"))
    monkeypatch.setattr(processor, "_external_archive_tool", lambda: "7z")
    extract_calls = []

    def fake_run(args, check, capture_output, text=False):
        if args[1] == "l":
            records = []
            for index in range(12):
                records.append(f"Path = doc_{index:02d}.txt\nSize = 20\nFolder = -\n")
            records.append("Path = data.bin\nSize = 3\nFolder = -\n")
            return subprocess.CompletedProcess(args, 0, stdout="\n".join(records))
        if args[1] == "x":
            extract_calls.append(args[-1])
            return subprocess.CompletedProcess(args, 0, stdout=f"content from {args[-1]}".encode())
        raise AssertionError(args)

    monkeypatch.setattr("ai_sorter.processor.subprocess.run", fake_run)

    entries = processor._read_external_archive_entries(archive_path, 1024)

    assert len(entries) == 13
    assert len(extract_calls) == 10
    assert sum(1 for _name, _size, _data, has_content in entries if has_content) == 10
