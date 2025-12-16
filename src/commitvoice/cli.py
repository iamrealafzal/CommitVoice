"""
CommitVoice: generate tweet, LinkedIn, or timesheet snippets from recent git commits.

Usage examples:
    commitvoice --remote git@bitbucket.org:afzal--lakdawala/shopycart.git --branch main --limit 5
    commitvoice --since "2024-01-01" --gemini-model gemini-1.5-flash
    commitvoice --timesheet --timesheet-hours 0.5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dotenv import load_dotenv

DEFAULT_REMOTE = "git@bitbucket.org:afzal--lakdawala/shopycart.git"


def run_git(args: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    """Run a git command and return (code, stdout, stderr)."""
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def clone_repo(remote: str, branch: str, depth: int) -> Path:
    """Clone the repository into a temporary directory; fallback to master if needed."""
    temp_dir = Path(tempfile.mkdtemp(prefix="commitvoice-"))
    target_branch = branch
    code, _, err = run_git(
        ["clone", "--quiet", f"--depth={depth}", "--branch", target_branch, remote, str(temp_dir)],
        cwd=Path("."),
    )
    if code != 0 and branch != "master":
        target_branch = "master"
        code, _, err = run_git(
            ["clone", "--quiet", f"--depth={depth}", "--branch", target_branch, remote, str(temp_dir)],
            cwd=Path("."),
        )
    if code != 0:
        raise RuntimeError(f"git clone failed for {remote} ({err})")
    return temp_dir


def parse_commits(
    repo: Path,
    limit: int,
    since: Optional[str],
    until: Optional[str],
) -> List[dict]:
    """Return recent commits with metadata and changed files."""
    log_args = ["log", f"-n{limit}", "--date=iso", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s"]
    if since:
        log_args.append(f"--since={since}")
    if until:
        log_args.append(f"--until={until}")

    code, stdout, err = run_git(log_args, cwd=repo)
    if code != 0:
        raise RuntimeError(f"git log failed: {err}")

    commits: List[dict] = []
    for line in filter(None, stdout.splitlines()):
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        commit_hash, author, date_str, subject = parts
        files = list_changed_files(repo, commit_hash)
        commits.append(
            {
                "hash": commit_hash,
                "author": author,
                "date": date_str,
                "subject": subject,
                "files": files,
            }
        )
    return commits


def list_changed_files(repo: Path, commit_hash: str) -> List[str]:
    """Return files touched in a commit."""
    code, stdout, err = run_git(
        ["show", "--pretty=format:", "--name-only", commit_hash],
        cwd=repo,
    )
    if code != 0:
        raise RuntimeError(f"git show failed: {err}")
    return [line for line in stdout.splitlines() if line.strip()]


def shorten(text: str, width: int) -> str:
    """Shorten while keeping whole words."""
    return textwrap.shorten(text, width=width, placeholder="…")


def _enhance_with_gemini(prompt: str, model: str, fallback: str) -> str:
    """Send prompt to Gemini if available; otherwise return fallback."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return fallback

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return fallback

    genai.configure(api_key=api_key)
    try:
        response = genai.GenerativeModel(model=model).generate_content(prompt)
        text = getattr(response, "text", "") or fallback
        return shorten(text.strip(), 270)
    except Exception:
        return fallback


def _enhance_with_openai(prompt: str, model: str, fallback: str) -> str:
    """Send prompt to OpenAI if available; otherwise return fallback."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return fallback

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return fallback

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        text = getattr(choice.message, "content", "") or fallback
        return shorten(text.strip(), 270)
    except Exception:
        return fallback


def enhance_with_llm(
    prompt: str,
    provider: Optional[str],
    model: Optional[str],
    fallback: str,
) -> str:
    """Route to the configured LLM provider, or return fallback."""
    if not provider or not model:
        return fallback
    if provider == "gemini":
        return _enhance_with_gemini(prompt, model, fallback)
    if provider == "openai":
        return _enhance_with_openai(prompt, model, fallback)
    return fallback


def make_tweet(
    commit: dict,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> str:
    """Construct a concise tweet-sized update and optionally polish with an LLM."""
    files = commit.get("files", [])
    files_preview = ", ".join(files[:3]) + (" +" if len(files) > 3 else "")
    base = f"{commit['subject']} [{commit['hash'][:7]}]"
    detail = f"Files: {files_preview}" if files_preview else "Code update"
    draft = shorten(f"{base}\n{detail}", 270)
    return enhance_with_llm(
        prompt=(
            "Rewrite this commit note into a friendly, concise, developer-focused tweet. "
            "Keep it under 270 characters, avoid hashtags, and keep the hash id intact. "
            f"Draft: {draft}"
        ),
        provider=llm_provider,
        model=llm_model,
        fallback=draft,
    )


def make_linkedin(
    commit: dict,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> str:
    """Construct a summary-style LinkedIn blurb and optionally polish it with an LLM."""
    files = commit.get("files", [])
    files_preview = ", ".join(files[:4]) + (" +" if len(files) > 4 else "")
    date_part = commit["date"].split(" ")[0] if commit.get("date") else ""
    base = "\n".join(
        [
            f"Summary for {commit['hash'][:7]} - {commit['subject']}",
            "Timeline:",
            f"- {date_part}: Updated files {files_preview or 'See diff'}",
            "- Impact: Improves implementation and keeps the codebase moving.",
        ]
    )

    return enhance_with_llm(
        prompt=(
            "Rewrite this into a clear, professional LinkedIn-style summary with a title and a short timeline. "
            "Keep the first line as a concise title, then 1-3 bullet points forming a timeline (date first), "
            "avoid hashtags, and preserve the commit hash. "
            f"Draft:\n{base}"
        ),
        provider=llm_provider,
        model=llm_model,
        fallback=base,
    )


def print_updates(
    commits: List[dict],
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> None:
    """Print tweet + LinkedIn text for each commit."""
    if not commits:
        print("No commits found for the provided range.")
        return
    for commit in commits:
        print("=" * 60)
        print(f"Commit {commit['hash'][:7]} by {commit['author']} on {commit['date']}")
        print("\nTweet:\n" + make_tweet(commit, llm_provider, llm_model))
        print("\nLinkedIn:\n" + make_linkedin(commit, llm_provider, llm_model))
        print()


def print_timesheet(commits: List[dict], hours: float) -> None:
    """Print a lightweight timesheet-style summary."""
    if not commits:
        print("No commits found for the provided range.")
        return
    print("Timesheet")
    print("-" * 60)
    for commit in commits:
        date_part = commit["date"].split(" ")[0] if commit.get("date") else ""
        files = commit.get("files", [])
        files_preview = ", ".join(files[:3]) + (" +" if len(files) > 3 else "")
        print(f"{date_part} | {hours:.2f}h | {commit['subject']} [{commit['hash'][:7]}] | Files: {files_preview or 'n/a'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create social updates from git commits.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Remote git URL to read commits from.")
    parser.add_argument("--branch", default="main", help="Branch to pull commits from (falls back to master if missing).")
    parser.add_argument("--limit", type=int, default=5, help="Number of commits to include.")
    parser.add_argument("--since", help='Only include commits after this date (e.g., "2024-01-01").')
    parser.add_argument("--until", help='Only include commits up to this date (e.g., "2024-12-31").')
    parser.add_argument("--depth", type=int, default=20, help="Shallow clone depth; increase for older commits.")
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "openai"],
        help="LLM provider to polish tweets (e.g., gemini, openai).",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model name (e.g., gemini-1.5-flash, gpt-4.1-mini).",
    )
    parser.add_argument(
        "--gemini-model",
        help="Deprecated alias for --llm-model with --llm-provider=gemini.",
    )
    parser.add_argument(
        "--timesheet",
        action="store_true",
        help="Output in timesheet format instead of tweet/LinkedIn text.",
    )
    parser.add_argument(
        "--timesheet-hours",
        type=float,
        default=0.5,
        help="Hours to attribute per commit in timesheet output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    try:
        repo_path = clone_repo(args.remote, args.branch, args.depth)
        commits = parse_commits(repo_path, args.limit, args.since, args.until)

        # Derive LLM provider/model with backward compatibility for --gemini-model.
        llm_provider = args.llm_provider
        llm_model = args.llm_model
        if not llm_provider and args.gemini_model:
            llm_provider = "gemini"
            llm_model = args.gemini_model

        if args.timesheet:
            print_timesheet(commits, hours=args.timesheet_hours)
        else:
            if llm_provider and llm_model:
                if llm_provider == "gemini":
                    print("Using Gemini for tweet polish. Ensure GOOGLE_API_KEY is set.", file=sys.stderr)
                elif llm_provider == "openai":
                    print("Using OpenAI for tweet polish. Ensure OPENAI_API_KEY is set.", file=sys.stderr)
            print_updates(commits, llm_provider=llm_provider, llm_model=llm_model)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


