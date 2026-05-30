from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Any

from .database import Database
from .file_utils import (
    describe_basic_file,
    extract_video_frames,
    file_hash,
    is_image,
    is_probably_text,
    is_video,
    read_text_preview,
)
from .models import Destination, MediaAnalysis, SortDecision
from .ollama import OllamaClient, OllamaError

ProgressCallback = Callable[[str], None]

DELETE_ALIAS = "DELETE"


class SortProcessor:
    def __init__(self, db: Database, progress: ProgressCallback | None = None) -> None:
        self.db = db
        self.progress = progress or (lambda _message: None)
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def _log(self, message: str) -> None:
        self.progress(message)

    def prepare_media_analysis(self, files: list[Path]) -> None:
        settings = self.db.get_settings()
        client = OllamaClient(settings.ollama_url)
        for path in files:
            if self.cancelled:
                return
            if not (is_image(path) or is_video(path)):
                continue
            digest = file_hash(path)
            if self.db.get_media_analysis(digest):
                self._log(f"Использован кэш анализа: {path.name}")
                continue
            if is_image(path):
                self._log(f"Анализ изображения: {path.name}")
                description = client.generate(
                    settings.vision_model,
                    "Describe the image content in concise but useful detail for later file sorting. Focus on visible subjects, text, document type, context, and any clues that identify the file category.",
                    images=[path],
                )
                self.db.save_media_analysis(MediaAnalysis(digest, str(path), "image", description))
            elif is_video(path):
                self._log(f"Извлечение кадров видео: {path.name}")
                cache = Path(tempfile.mkdtemp(prefix=f"ai-sorter-{path.stem}-"))
                try:
                    frames = extract_video_frames(path, cache, max_frames=10)
                    frame_notes: list[str] = []
                    for idx, frame in enumerate(frames, start=1):
                        if self.cancelled:
                            return
                        note = client.generate(
                            settings.vision_model,
                            f"Describe frame {idx} from this video for later overall video classification. Focus on visible subjects, scene, text, and category clues.",
                            images=[frame],
                        )
                        frame_notes.append(f"Frame {idx}: {note}")
                    description = client.generate(
                        settings.sorter_model,
                        "Summarize the likely video content from these frame descriptions for later file sorting. Mention subjects, scene type, document/screen/video category, and any strong classification clues:\n" + "\n".join(frame_notes),
                    )
                    self.db.save_media_analysis(MediaAnalysis(digest, str(path), "video", description))
                finally:
                    shutil.rmtree(cache, ignore_errors=True)

    def build_file_context(self, path: Path) -> str:
        context = describe_basic_file(path)
        if is_image(path) or is_video(path):
            analysis = self.db.get_media_analysis(file_hash(path))
            if analysis:
                context += f"\nCached media analysis: {analysis.description}"
        elif is_probably_text(path):
            context += "\nText preview:\n" + read_text_preview(path)
        else:
            context += "\nBinary/non-text file: classify by name, extension, MIME and size."
        return context

    def classify_files(self, files: list[Path]) -> list[SortDecision]:
        settings = self.db.get_settings()
        destinations = self.db.list_destinations()
        client = OllamaClient(settings.ollama_url)
        decisions: list[SortDecision] = []
        for path in files:
            if self.cancelled:
                break
            self._log(f"Классификация файла: {path.name}")
            decisions.append(self.classify_one(client, settings.sorter_model, path, destinations))
        return decisions

    def build_sort_prompt(self, path: Path, destinations: list[Destination]) -> str:
        folder_descriptions: list[str] = []
        folder_aliases: list[str] = []
        for dest in destinations:
            alias = self.destination_alias(dest)
            folder_aliases.append(alias)
            line = f"- {alias}: {dest.positive_prompt.strip()}"
            if dest.negative_prompt.strip():
                line += f" (AVOID: {dest.negative_prompt.strip()})"
            folder_descriptions.append(line)

        folders_block = "\n".join(folder_descriptions) or "- no configured folders"
        aliases_list = ", ".join(folder_aliases + [DELETE_ALIAS])
        return f"""You are a precise file organizer. Based on the file/directory information, choose exactly ONE destination.

Options:
- One of the folders described below.
- DELETE if the file is definitely junk (temporary internet files, duplicates, corrupted downloads, thumbnails, installers/cache leftovers, etc.).

Available folders:
{folders_block}

Allowed target aliases: {aliases_list}

Rules:
- Prefer the best semantic match between file context and folder description.
- Treat each AVOID clause as a strong exclusion for that folder.
- Use DELETE only for obvious junk; do not delete useful files merely because no folder is perfect.
- If uncertain, still choose the closest folder and lower the confidence.
- Do not invent aliases. The target_alias MUST be one of the allowed target aliases.

You MUST respond with a single JSON object containing:
- "action": either "move" or "delete"
- "target_alias": if "move", the exact alias from the list; if "delete", use null
- "reason": short explanation in English
- "confidence": integer 0-100

File/directory context:
{self.build_file_context(path)}
""".strip()

    def classify_one(self, client: OllamaClient, model: str, path: Path, destinations: list[Destination]) -> SortDecision:
        fallback = destinations[0] if destinations else None
        try:
            raw = client.generate(model, self.build_sort_prompt(path, destinations), json_response=True)
            data = self._parse_model_json(raw)
            return self._decision_from_model_data(path, data, destinations, fallback)
        except (OllamaError, json.JSONDecodeError, TypeError, ValueError):
            return SortDecision(
                file_path=path,
                destination_id=fallback.id if fallback else None,
                destination_name=fallback.name if fallback else "Нет назначения",
                destination_path=fallback.path if fallback else "",
                reason="Не удалось получить корректный JSON от модели; выбрано первое назначение как черновой вариант.",
                confidence=0,
                action="move" if fallback else "none",
            )

    def _decision_from_model_data(
        self,
        path: Path,
        data: dict[str, Any],
        destinations: list[Destination],
        fallback: Destination | None,
    ) -> SortDecision:
        action = str(data.get("action", "move")).lower()
        reason = str(data.get("reason", "Нет обоснования"))
        confidence = max(0, min(100, int(data.get("confidence", 0))))
        if action == "delete":
            return SortDecision(path, None, "Удалить", "", reason, confidence, "delete")

        target_alias = str(data.get("target_alias") or data.get("destination") or data.get("destination_id") or "")
        destination = self.destination_by_alias(destinations, target_alias) or fallback
        return SortDecision(
            file_path=path,
            destination_id=destination.id if destination else None,
            destination_name=destination.name if destination else "Нет назначения",
            destination_path=destination.path if destination else "",
            reason=reason,
            confidence=confidence,
            action="move" if destination else "none",
        )

    def _parse_model_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def destination_alias(self, destination: Destination) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", destination.name.strip()).strip("_") or f"dest_{destination.id}"

    def destination_by_alias(self, destinations: list[Destination], alias: str) -> Destination | None:
        alias_normalized = alias.strip().casefold()
        for dest in destinations:
            if alias_normalized in {self.destination_alias(dest).casefold(), dest.name.strip().casefold(), str(dest.id)}:
                return dest
        return None

    def build_prompt_update_prompt(
        self,
        file_path: Path,
        manual_destination: Destination,
        wrong_destination: Destination | None,
        reason: str,
    ) -> str:
        wrong_name = wrong_destination.name if wrong_destination else "none"
        wrong_negative = wrong_destination.negative_prompt if wrong_destination else ""
        return f"""
The user corrected a file sorting decision.

File context:
{self.build_file_context(file_path)}

Correct destination: {manual_destination.name}
Current positive prompt for the correct destination: {manual_destination.positive_prompt}
User's reason for the correction: {reason}
Incorrectly selected destination: {wrong_name}
Current negative prompt for the incorrect destination: {wrong_negative}

Return only a JSON object with these keys:
- "positive_prompt": an improved positive prompt for the correct destination
- "negative_prompt": an improved negative prompt for the incorrectly selected destination
""".strip()

    def suggest_prompt_update(
        self,
        file_path: Path,
        manual_destination: Destination,
        wrong_destination: Destination | None,
        reason: str,
    ) -> tuple[str, str]:
        settings = self.db.get_settings()
        client = OllamaClient(settings.ollama_url)
        prompt = self.build_prompt_update_prompt(file_path, manual_destination, wrong_destination, reason)
        raw = client.generate(settings.sorter_model, prompt, json_response=True)
        data = json.loads(raw)
        return str(data.get("positive_prompt", manual_destination.positive_prompt)), str(data.get("negative_prompt", ""))
