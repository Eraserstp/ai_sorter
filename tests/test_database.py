from ai_sorter.database import Database
from ai_sorter.models import AppSettings, Destination, MediaAnalysis


def test_database_roundtrip(tmp_path):
    db = Database(tmp_path / "app.sqlite3")
    db.save_settings(AppSettings("http://example:11434", "llama", "llava"))
    assert db.get_settings() == AppSettings("http://example:11434", "llama", "llava")

    db.replace_sources(["/tmp/downloads", ""])
    db.replace_exclusions(["skip.iso"])
    assert [item.path for item in db.list_sources()] == ["/tmp/downloads"]
    assert [item.path for item in db.list_exclusions()] == ["skip.iso"]

    dest_id = db.upsert_destination(Destination(None, "Docs", "/tmp/docs", "documents", "movies"))
    db.upsert_destination(Destination(dest_id, "Documents", "/tmp/docs", "text files", "video"))
    assert db.list_destinations() == [Destination(dest_id, "Documents", "/tmp/docs", "text files", "video")]

    analysis = MediaAnalysis("abc", "/tmp/a.png", "image", "a cat")
    db.save_media_analysis(analysis)
    assert db.get_media_analysis("abc") == analysis
