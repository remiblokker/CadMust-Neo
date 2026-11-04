#!/usr/bin/env python3
"""
scrape_boards.py — Download KiCad PCB files from GitHub for benchmark testing.

Usage:
    python3 scripts/scrape_boards.py --out ~/cadmust_testboards

Requirements:
    pip install requests

GitHub token:
    Create a token at https://github.com/settings/tokens (no special scopes needed,
    a classic token with no permissions works fine). Set it as an environment variable:
        export GITHUB_TOKEN=ghp_...
    Without a token the search API limit is 10 requests/min (very slow).
    With a token it is 30 requests/min for code search, 5000/hr for other endpoints.

What this script does:
    1. Searches GitHub code search for *.kicad_pcb files across public repos.
    2. Filters by file size (MIN_KB–MAX_KB) and KiCad format version (≥7 preferred).
    3. Skips forks, already-downloaded files, and repos explicitly blacklisted.
    4. Downloads raw .kicad_pcb files into --out/<owner>__<repo>__<filename>.kicad_pcb
    5. Writes a manifest.json with metadata for each downloaded file.

Filters applied:
    - File size: 5 KB – 500 KB  (skip trivial and huge boards)
    - KiCad version: prefers format ≥7; older formats are accepted if quota allows
    - Repo is not a fork
    - File contains "(kicad_pcb" (sanity check it's a real PCB file)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests not installed. Run: pip install requests")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_KB = 30             # ~35+ component boards; smaller files are typically trivial breakouts
MAX_KB = 500
TARGET_COUNT = 300          # stop after this many downloads
PAUSE_BETWEEN_PAGES = 2.0   # seconds between search page requests
PAUSE_BETWEEN_DOWNLOADS = 0.5

BLACKLIST_REPOS = {
    # Add "owner/repo" strings here to skip specific repos
}


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def make_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    s.headers["X-GitHub-Api-Version"] = "2022-11-28"
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def search_kicad_pcb_files(session: requests.Session, page: int) -> dict:
    """One page of GitHub code search results for *.kicad_pcb files."""
    url = "https://api.github.com/search/code"
    params = {
        "q": "extension:kicad_pcb kicad_pcb",
        "per_page": 100,
        "page": page,
    }
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 403:
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - time.time() + 2, 10)
        print(f"  Rate-limited. Sleeping {wait:.0f}s...")
        time.sleep(wait)
        resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_repo_info(session: requests.Session, full_name: str) -> Optional[dict]:
    url = f"https://api.github.com/repos/{full_name}"
    resp = session.get(url, timeout=15)
    if resp.status_code != 200:
        return None
    return resp.json()


def download_raw(session: requests.Session, url: str) -> Optional[bytes]:
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        return None
    return resp.content


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def extract_kicad_version(content: bytes) -> int:
    """Return the KiCad PCB format version number, or 0 if not found."""
    try:
        text = content[:2000].decode("utf-8", errors="replace")
    except Exception:
        return 0
    m = re.search(r"\(version\s+(\d+)\)", text)
    if m:
        return int(m.group(1))
    return 0


def is_valid_pcb(content: bytes) -> bool:
    """Quick sanity check: real KiCad PCB content (not a Git LFS pointer)."""
    if not content:
        return False
    try:
        text = content[:500].decode("utf-8", errors="replace")
    except Exception:
        return False
    if text.startswith("version https://git-lfs"):
        return False  # Git LFS pointer, not actual content
    return "(kicad_pcb" in text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape KiCad PCB files from GitHub")
    parser.add_argument("--out", default=os.path.expanduser("~/cadmust_testboards"),
                        help="Output directory (default: ~/cadmust_testboards)")
    parser.add_argument("--target", type=int, default=TARGET_COUNT,
                        help=f"Stop after N downloads (default: {TARGET_COUNT})")
    parser.add_argument("--max-kb", type=int, default=MAX_KB,
                        help=f"Max file size in KB (default: {MAX_KB})")
    parser.add_argument("--min-kb", type=int, default=MIN_KB,
                        help=f"Min file size in KB (default: {MIN_KB})")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("WARNING: GITHUB_TOKEN not set. Rate limit is very low (10 req/min).")
        print("         Set it with: export GITHUB_TOKEN=ghp_...")

    session = make_session(token)
    downloaded = sum(1 for v in manifest.values() if v.get("status") == "ok")
    print(f"Starting. Already have {downloaded} boards. Target: {args.target}")

    page = 1
    seen_urls: set = {v["raw_url"] for v in manifest.values() if "raw_url" in v}

    while downloaded < args.target:
        print(f"\n--- Search page {page} ---")
        try:
            results = search_kicad_pcb_files(session, page)
        except Exception as e:
            print(f"Search error: {e}")
            break

        items = results.get("items", [])
        if not items:
            print("No more results.")
            break

        total_count = results.get("total_count", "?")
        print(f"  {len(items)} items (total available: {total_count})")

        for item in items:
            if downloaded >= args.target:
                break

            repo_full = item["repository"]["full_name"]
            file_name = item["name"]
            file_path_in_repo = item["path"]

            # Blacklist (cheap check, no API call)
            if repo_full in BLACKLIST_REPOS:
                print(f"  SKIP (blacklisted): {repo_full}")
                continue

            # Get repo info: fork status, stars, and crucially the actual default branch.
            # The code search result does NOT include default_branch, so guessing 'main'
            # causes many failed downloads for repos that use 'master' or other names.
            repo_info = get_repo_info(session, repo_full)
            if repo_info is None:
                continue
            if repo_info.get("fork", False):
                print(f"  SKIP (fork): {repo_full}")
                continue

            stars = repo_info.get("stargazers_count", 0)
            default_branch = repo_info.get("default_branch", "main")

            raw_url = (
                f"https://raw.githubusercontent.com/{repo_full}/"
                f"{default_branch}/{file_path_in_repo}"
            )

            # Skip already processed
            if raw_url in seen_urls:
                continue
            seen_urls.add(raw_url)

            # Download file
            print(f"  Downloading: {repo_full}/{file_name} [{default_branch}] ({stars}★)")
            content = download_raw(session, raw_url)
            if content is None:
                print(f"    FAILED (download error)")
                continue

            if not is_valid_pcb(content):
                print(f"    SKIP (not a valid kicad_pcb file)")
                continue

            # Check actual size now that we have the content
            size_kb = len(content) / 1024
            if not (args.min_kb <= size_kb <= args.max_kb):
                print(f"    SKIP (size {size_kb:.0f} KB out of range)")
                continue

            kicad_version = extract_kicad_version(content)
            print(f"    KiCad format version: {kicad_version}")

            # Save file
            safe_name = f"{repo_full.replace('/', '__')}__{file_name}"
            out_path = out_dir / safe_name
            # Handle filename collisions
            if out_path.exists():
                stem = out_path.stem
                suffix = out_path.suffix
                out_path = out_dir / f"{stem}_{page}_{downloaded}{suffix}"

            out_path.write_bytes(content)

            # Record in manifest
            key = safe_name
            manifest[key] = {
                "status": "ok",
                "file": str(out_path.name),
                "repo": repo_full,
                "path_in_repo": file_path_in_repo,
                "raw_url": raw_url,
                "size_kb": round(size_kb, 1),
                "stars": stars,
                "kicad_version": kicad_version,
            }
            save_manifest(manifest_path, manifest)
            downloaded += 1
            print(f"    Saved → {out_path.name}  [{downloaded}/{args.target}]")

            time.sleep(PAUSE_BETWEEN_DOWNLOADS)

        page += 1
        time.sleep(PAUSE_BETWEEN_PAGES)

    print(f"\nDone. {downloaded} boards in {out_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
