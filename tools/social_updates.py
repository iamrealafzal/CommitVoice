"""Shim to allow running via the legacy path; delegates to CommitVoice CLI."""

from commitvoice.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
