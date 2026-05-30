from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    ollama_url: str = "http://localhost:11434"
    sorter_model: str = ""
    vision_model: str = ""


@dataclass(frozen=True)
class SourceDirectory:
    id: int | None
    path: str


@dataclass(frozen=True)
class ExcludedFile:
    id: int | None
    path: str


@dataclass(frozen=True)
class Destination:
    id: int | None
    name: str
    path: str
    positive_prompt: str
    negative_prompt: str = ""


@dataclass(frozen=True)
class MediaAnalysis:
    file_hash: str
    path: str
    media_type: str
    description: str


@dataclass(frozen=True)
class SortDecision:
    file_path: Path
    destination_id: int | None
    destination_name: str
    destination_path: str
    reason: str
    confidence: int
    action: str = "move"
