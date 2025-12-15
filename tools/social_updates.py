"""Shim to allow running via the legacy path; delegates to the package CLI."""

from social_updates.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
