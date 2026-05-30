from ai_sorter.file_utils import file_hash, iter_top_level_files, is_probably_text, move_file


def test_file_helpers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    text = source / "note.txt"
    text.write_text("hello", encoding="utf-8")
    binary = source / "bin.dat"
    binary.write_bytes(b"\x00\x01")
    (source / "nested").mkdir()
    (source / "nested" / "ignored.txt").write_text("ignored", encoding="utf-8")

    assert is_probably_text(text)
    assert not is_probably_text(binary)
    assert file_hash(text) == file_hash(text)
    assert iter_top_level_files([str(source)], {"bin.dat"}) == [text]

    destination = tmp_path / "dest"
    moved = move_file(text, destination)
    assert moved.exists()
    assert moved.parent == destination
