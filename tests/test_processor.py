from pathlib import Path

from ai_sorter.database import Database
from ai_sorter.models import Destination
from ai_sorter.processor import DELETE_ALIAS, SortProcessor


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompt = ""

    def generate(self, model: str, prompt: str, json_response: bool = False):
        self.prompt = prompt
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
