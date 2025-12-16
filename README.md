# CommitVoice

A tiny Python agent that reads recent git commits from any remote and produces shareable tweet- and LinkedIn-ready snippets. Optional Gemini support polishes tweets for a more human tone while keeping commit hashes intact. A timesheet mode formats entries for quick logging.

## Features
- Shallow-clone from a remote URL (SSH/HTTPS), with branch and date filters.
- Summarize commits (hash, author, date, subject, touched files).
- Generate:  
  - Tweet-sized update (≤280 chars; ~270 target)  
  - LinkedIn blurb (concise + impact line)  
- Optional LLM polish for tweets:
  - Gemini (`--llm-provider gemini`, `GOOGLE_API_KEY`)
  - OpenAI (`--llm-provider openai`, `OPENAI_API_KEY`)
- Timesheet-friendly output (`--timesheet`) to summarize work logs.

## Requirements
- Python 3.11+ (for `google-generativeai` and modern OpenAI client).  
- Git available on PATH.
- For Gemini polish: `google-generativeai` and `GOOGLE_API_KEY`.
- For OpenAI polish: `openai` and `OPENAI_API_KEY`.

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
commitvoice \
  --remote git@bitbucket.org:afzal--lakdawala/shopycart.git \
  --branch main \
  --limit 5
```

Date filter:
```bash
commitvoice --since "2024-01-01" --until "2024-12-31"
```

Use Gemini to humanize tweets (requires `GOOGLE_API_KEY` in env or `.env`):
```bash
commitvoice \
  --remote git@bitbucket.org:afzal--lakdawala/shopycart.git \
  --llm-provider gemini \
  --llm-model gemini-1.5-flash
```

Use OpenAI to humanize tweets (requires `OPENAI_API_KEY` in env or `.env`):
```bash
commitvoice \
  --remote git@bitbucket.org:afzal--lakdawala/shopycart.git \
  --llm-provider openai \
  --llm-model gpt-4.1-mini
```

Timesheet format:
```bash
commitvoice --timesheet --timesheet-hours 0.5
```

Flags:
- `--remote` remote git URL (default: Bitbucket shopycart SSH).
- `--branch` branch name, falls back to `master` if `main` missing.
- `--limit` number of commits (default 5).
- `--since`/`--until` date filters (`YYYY-MM-DD`).
- `--depth` shallow clone depth (default 20).
- `--llm-provider` LLM provider for tweet polish (`gemini`, `openai`).
- `--llm-model` LLM model name (e.g. `gemini-1.5-flash`, `gpt-4.1-mini`).
- `--gemini-model` (deprecated) alias for `--llm-model` with Gemini provider.
- `--timesheet` switch to timesheet output.
- `--timesheet-hours` hours per commit in timesheet output (default 0.5h).

## Output
For each commit, prints:
- Tweet text (≤280 chars, hash preserved).
- LinkedIn blurb with files touched and an impact line.

## Reusing as an agent
- Wrap `main()` from `commitvoice.cli` or import and call `parse_commits` + `make_tweet`/`make_linkedin`.
- Swap `print_updates` with your own sink (e.g., post to APIs or save to files).
- Adjust prompts in `make_tweet`/`enhance_with_gemini` to fit tone/brand.

## Versioning
- Current: `v0.2.0` (CommitVoice rename + timesheet mode).
- Plan: semantic versioning. Next milestones: file-output option, per-commit templating, configurable impact lines.

## License
MIT. Contributions welcome.

