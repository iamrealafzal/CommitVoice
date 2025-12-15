# Git Commit Social Updates Agent

A tiny Python agent that reads recent git commits from any remote and produces shareable tweet- and LinkedIn-ready snippets. Optional Gemini support polishes tweets for a more human tone while keeping commit hashes intact.

## Features
- Shallow-clone from a remote URL (SSH/HTTPS), with branch and date filters.
- Summarize commits (hash, author, date, subject, touched files).
- Generate:  
  - Tweet-sized update (≤280 chars; ~270 target)  
  - LinkedIn blurb (concise + impact line)  
- Optional Gemini polish for tweets (`--gemini-model`, requires `GOOGLE_API_KEY`).

## Requirements
- Python 3.11+ (for `google-generativeai` wheels).  
- Git available on PATH.
- For Gemini polish: `google-generativeai` and `GOOGLE_API_KEY`.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need to pin a Python version, use pyenv (install Python 3.11+), then recreate the venv with that interpreter.

## Usage
Basic run:
```bash
python tools/social_updates.py \
  --remote git@bitbucket.org:afzal--lakdawala/shopycart.git \
  --branch main \
  --limit 5
```

Date filter:
```bash
python tools/social_updates.py --since "2024-01-01" --until "2024-12-31"
```

Use Gemini to humanize tweets (requires `GOOGLE_API_KEY` in env or `.env`):
```bash
python tools/social_updates.py \
  --remote git@bitbucket.org:afzal--lakdawala/shopycart.git \
  --gemini-model gemini-1.5-flash
```

Flags:
- `--remote` remote git URL (default: Bitbucket shopycart SSH).
- `--branch` branch name, falls back to `master` if `main` missing.
- `--limit` number of commits (default 5).
- `--since`/`--until` date filters (`YYYY-MM-DD`).
- `--depth` shallow clone depth (default 20).
- `--gemini-model` Gemini model name to polish tweets (optional).

## Output
For each commit, prints:
- Tweet text (≤280 chars, hash preserved).
- LinkedIn blurb with files touched and an impact line.

## Reusing as an agent
- Wrap `main()` from `tools/social_updates.py` or import and call `parse_commits` + `make_tweet`/`make_linkedin`.
- Swap `print_updates` with your own sink (e.g., post to APIs or save to files).
- Adjust prompts in `make_tweet`/`enhance_with_gemini` to fit tone/brand.

## Versioning
- Initial public release: `v0.1.0` (current).
- Plan: semantic versioning. Next milestones: file-output option, per-commit templating, configurable impact lines.

## License
MIT. Contributions welcome.

