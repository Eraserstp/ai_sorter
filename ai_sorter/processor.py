from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

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
                    "Подробно, но кратко опиши содержимое изображения для последующей сортировки файла.",
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
                            f"Опиши содержимое кадра {idx} из видео для последующего общего анализа.",
                            images=[frame],
                        )
                        frame_notes.append(f"Кадр {idx}: {note}")
                    description = client.generate(
                        settings.sorter_model,
                        "Сделай краткий вывод о содержимом видео по описаниям кадров:\n" + "\n".join(frame_notes),
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

    def classify_one(self, client: OllamaClient, model: str, path: Path, destinations: list[Destination]) -> SortDecision:
        destination_block = "\n".join(
            f"ID {dest.id}: {dest.name}\nPath: {dest.path}\nPositive: {dest.positive_prompt}\nNegative: {dest.negative_prompt or '(none)'}"
            for dest in destinations
        )
        prompt = f"""
Ты сортируешь файлы по пользовательским назначениям. Выбери ровно одно назначение.
Учитывай positive prompt как признак попадания, negative prompt как запрет.
Верни только JSON: {{"destination_id": number|null, "reason": "short", "confidence": 0-100}}.

Назначения:
{destination_block}

Файл:
{self.build_file_context(path)}
""".strip()
        fallback = destinations[0] if destinations else None
        try:
            raw = client.generate(model, prompt, json_response=True)
            data = json.loads(raw)
            destination_id = data.get("destination_id")
            reason = str(data.get("reason", "Нет обоснования"))
            confidence = int(data.get("confidence", 0))
        except (OllamaError, json.JSONDecodeError, TypeError, ValueError):
            destination_id = fallback.id if fallback else None
            reason = "Не удалось получить корректный JSON от модели; выбрано первое назначение как черновой вариант."
            confidence = 0
        destination = next((dest for dest in destinations if dest.id == destination_id), fallback)
        return SortDecision(
            file_path=path,
            destination_id=destination.id if destination else None,
            destination_name=destination.name if destination else "Нет назначения",
            destination_path=destination.path if destination else "",
            reason=reason,
            confidence=max(0, min(100, confidence)),
        )

    def suggest_prompt_update(
        self,
        file_path: Path,
        manual_destination: Destination,
        wrong_destination: Destination | None,
        reason: str,
    ) -> tuple[str, str]:
        settings = self.db.get_settings()
        client = OllamaClient(settings.ollama_url)
        prompt = f"""
Пользователь исправил сортировку файла.
Контекст файла:
{self.build_file_context(file_path)}

Правильное назначение: {manual_destination.name}
Текущий positive prompt: {manual_destination.positive_prompt}
Причина пользователя: {reason}
Ошибочно выбранное назначение: {wrong_destination.name if wrong_destination else 'нет'}
Текущий negative prompt ошибочного назначения: {wrong_destination.negative_prompt if wrong_destination else ''}

Верни только JSON: {{"positive_prompt": "улучшенный positive prompt для правильного назначения", "negative_prompt": "улучшенный negative prompt для ошибочного назначения"}}.
""".strip()
        raw = client.generate(settings.sorter_model, prompt, json_response=True)
        data = json.loads(raw)
        return str(data.get("positive_prompt", manual_destination.positive_prompt)), str(data.get("negative_prompt", ""))
