from __future__ import annotations

import hashlib
import mimetypes
import json
import shutil
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".ini", ".toml", ".py", ".js", ".html", ".css"}


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_image(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    return path.suffix.lower() in IMAGE_EXTENSIONS or bool(mime and mime.startswith("image/"))


def is_video(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    return path.suffix.lower() in VIDEO_EXTENSIONS or bool(mime and mime.startswith("video/"))


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_text_preview(path: Path, max_chars: int = 6000) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="replace")[:max_chars]
        except OSError:
            raise
        except UnicodeError:
            continue
    return ""


def iter_top_level_files(directories: list[str], excluded: set[str]) -> list[Path]:
    result: list[Path] = []
    excluded_paths = {str(Path(item).expanduser().resolve()) for item in excluded}
    for directory in directories:
        root = Path(directory).expanduser()
        if not root.is_dir():
            continue
        for child in root.iterdir():
            try:
                resolved = str(child.resolve())
            except OSError:
                continue
            if child.is_file() and resolved not in excluded_paths and child.name not in excluded:
                result.append(child)
    return sorted(result, key=lambda item: item.name.lower())


def extract_video_frames(video_path: Path, cache_dir: Path, max_frames: int = 10) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = cache_dir / "frame_%03d.jpg"
    duration = _video_duration_seconds(video_path)
    if duration and duration > 0:
        # Sample at most max_frames frames uniformly across the known duration.
        vf = f"fps={max_frames / duration},scale=640:-1"
    else:
        # Fallback when ffprobe cannot determine duration: take scene-spaced frames.
        vf = "thumbnail=60,scale=640:-1"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(cache_dir.glob("frame_*.jpg"))


def _video_duration_seconds(video_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(completed.stdout)
        return float(data["format"]["duration"])
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def move_file(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    if target.exists():
        stem, suffix = source.stem, source.suffix
        counter = 1
        while target.exists():
            target = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    return Path(shutil.move(str(source), str(target)))


def trash_file(path: Path) -> None:
    try:
        subprocess.run(["gio", "trash", str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        trash_dir = Path.home() / ".local" / "share" / "Trash" / "files"
        trash_dir.mkdir(parents=True, exist_ok=True)
        move_file(path, trash_dir)


def describe_basic_file(path: Path) -> str:
    stat = path.stat()
    mime, _ = mimetypes.guess_type(path.name)
    return (
        f"Name: {path.name}\nExtension: {path.suffix or '(none)'}\n"
        f"MIME guess: {mime or 'unknown'}\nSize: {stat.st_size} bytes"
    )
