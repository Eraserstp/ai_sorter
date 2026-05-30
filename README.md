# AI Sorter

AI Sorter is a Linux GTK application for sorting download folders with a local or remote Ollama instance and a SQLite configuration/cache database.

## Features

- Configure one or more top-level source directories and excluded file names/paths.
- Configure Ollama URL, a general sorting model, and a multimodal model for image/video pre-analysis.
- Maintain destination folders with framed prompt editors and a filesystem folder chooser for destination paths.
- Pre-analyze images and videos, caching media descriptions by SHA-256 hash in SQLite.
- Extract up to 10 evenly sampled video frames with `ffmpeg`, analyze each frame, summarize the video, and remove frame cache files.
- Classify all top-level files with a strict alias-based JSON prompt that compares every destination, honors AVOID clauses, and allows DELETE only for obvious junk.
- Inspect ZIP and TAR archives by sampling up to 10 image/text entries for content analysis and listing remaining archive entries by name, extension, and size.
- Review model decisions, confidence, and rationale before moving files or sending files to the trash.
- Manually correct rejected decisions and ask Ollama to suggest prompt improvements.

## Requirements

- Python 3.11+
- GTK 4 and PyGObject (`python3-gi`, `gir1.2-gtk-4.0` on many distributions)
- Ollama reachable locally at `http://localhost:11434` or a configured remote URL
- `ffmpeg` for video frame extraction
- `gio` for trash integration (falls back to `~/.local/share/Trash/files`)

## Run from source

```bash
python -m ai_sorter.app
```

Or install the package and use:

```bash
ai-sorter
```

The application stores its SQLite database at `~/.local/share/ai-sorter/ai_sorter.sqlite3`.
