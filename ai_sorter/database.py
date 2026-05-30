from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .models import AppSettings, Destination, ExcludedFile, MediaAnalysis, SourceDirectory

APP_DIR = Path.home() / ".local" / "share" / "ai-sorter"
DB_PATH = APP_DIR / "ai_sorter.sqlite3"


class Database:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS source_directories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS excluded_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS destinations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    positive_prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS media_analysis (
                    file_hash TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            for key, value in {
                "ollama_url": "http://localhost:11434",
                "sorter_model": "",
                "vision_model": "",
            }.items():
                conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value))

    def get_settings(self) -> AppSettings:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        values = {row["key"]: row["value"] for row in rows}
        return AppSettings(
            ollama_url=values.get("ollama_url", "http://localhost:11434"),
            sorter_model=values.get("sorter_model", ""),
            vision_model=values.get("vision_model", ""),
        )

    def save_settings(self, settings: AppSettings) -> None:
        with self.connect() as conn:
            for key, value in settings.__dict__.items():
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )

    def list_sources(self) -> list[SourceDirectory]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id, path FROM source_directories ORDER BY path").fetchall()
        return [SourceDirectory(row["id"], row["path"]) for row in rows]

    def replace_sources(self, paths: Iterable[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM source_directories")
            conn.executemany(
                "INSERT OR IGNORE INTO source_directories(path) VALUES (?)",
                [(str(Path(path).expanduser()),) for path in paths if str(path).strip()],
            )

    def list_exclusions(self) -> list[ExcludedFile]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id, path FROM excluded_files ORDER BY path").fetchall()
        return [ExcludedFile(row["id"], row["path"]) for row in rows]

    def replace_exclusions(self, paths: Iterable[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM excluded_files")
            conn.executemany(
                "INSERT OR IGNORE INTO excluded_files(path) VALUES (?)",
                [(str(Path(path).expanduser()),) for path in paths if str(path).strip()],
            )

    def list_destinations(self) -> list[Destination]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id, name, path, positive_prompt, negative_prompt FROM destinations ORDER BY name").fetchall()
        return [Destination(row["id"], row["name"], row["path"], row["positive_prompt"], row["negative_prompt"]) for row in rows]

    def upsert_destination(self, destination: Destination) -> int:
        with self.connect() as conn:
            if destination.id is None:
                cur = conn.execute(
                    "INSERT INTO destinations(name, path, positive_prompt, negative_prompt) VALUES (?, ?, ?, ?)",
                    (destination.name, destination.path, destination.positive_prompt, destination.negative_prompt),
                )
                return int(cur.lastrowid)
            conn.execute(
                "UPDATE destinations SET name=?, path=?, positive_prompt=?, negative_prompt=? WHERE id=?",
                (destination.name, destination.path, destination.positive_prompt, destination.negative_prompt, destination.id),
            )
            return destination.id

    def delete_destination(self, destination_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM destinations WHERE id=?", (destination_id,))

    def get_media_analysis(self, file_hash: str) -> MediaAnalysis | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT file_hash, path, media_type, description FROM media_analysis WHERE file_hash=?",
                (file_hash,),
            ).fetchone()
        return None if row is None else MediaAnalysis(row["file_hash"], row["path"], row["media_type"], row["description"])

    def save_media_analysis(self, analysis: MediaAnalysis) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO media_analysis(file_hash, path, media_type, description) VALUES (?, ?, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET path=excluded.path, media_type=excluded.media_type,
                    description=excluded.description, created_at=CURRENT_TIMESTAMP
                """,
                (analysis.file_hash, analysis.path, analysis.media_type, analysis.description),
            )
