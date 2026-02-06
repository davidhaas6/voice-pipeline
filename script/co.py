#!/usr/bin/env python3
"""
co.py

Generate a single Markdown "context pack" of a repo for feeding to an LLM.

Features:
- Repo tree summary
- Includes file contents for relevant text/code files (default: src/**/*.py, docs/**/*.md, plus other small text files)
- Skips common junk dirs (venv, .git, node_modules, dist, build, __pycache__, etc.)
- Best-effort .gitignore support if `pathspec` is installed
- Limits: per-file bytes, total bytes, max files
- Include/exclude globs

Usage examples:
  python tools/co.py --root . --out llm_context.md
  python tools/co.py --only src docs --out llm_context.md
  python tools/co.py --include "**/*.py" "**/*.md" --exclude "**/tests/**"
  python tools/co.py --max-total-bytes 300000 --max-file-bytes 20000
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    "out",
    ".next",
    ".cache",
    ".DS_Store",
}

DEFAULT_INCLUDED_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".bat",
    ".ps1",
}

BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp3",
    ".mp4",
    ".mov",
    ".wav",
}


def guess_lang_for_fence(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".rst": "rst",
        ".sh": "bash",
        ".ps1": "powershell",
        ".bat": "bat",
        ".txt": "",
    }.get(ext, "")


def is_probably_binary(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in BINARY_EXTS:
        return True
    # Heuristic: sample a small chunk and look for NUL bytes
    try:
        with path.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except Exception:
        return True


def normalize_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def load_gitignore_matcher(root: Path):
    """
    Best-effort .gitignore support. If `pathspec` is installed and .gitignore exists,
    return a function that returns True if a path should be ignored.
    """
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None

    try:
        import pathspec  # type: ignore
    except Exception:
        return None

    try:
        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        # GitWildMatchPattern handles standard gitignore patterns reasonably well
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)

        def is_ignored(rel_posix: str) -> bool:
            return spec.match_file(rel_posix)

        return is_ignored
    except Exception:
        return None


def matches_any_glob(rel_posix: str, globs: Sequence[str]) -> bool:
    for g in globs:
        if fnmatch.fnmatch(rel_posix, g):
            return True
    return False


@dataclass(frozen=True)
class FilePick:
    abs_path: Path
    rel_posix: str
    size_bytes: int


def build_tree_lines(paths: Sequence[str]) -> List[str]:
    """
    Build a compact tree view from a list of posix relative paths.
    """
    # Build nested dict structure
    root = {}
    for p in sorted(paths):
        parts = p.split("/")
        cur = root
        for part in parts:
            cur = cur.setdefault(part, {})

    lines: List[str] = []

    def walk(node: dict, prefix: str = ""):
        keys = list(node.keys())
        for i, k in enumerate(keys):
            is_last = i == len(keys) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + k)
            child = node[k]
            if child:
                extension = "    " if is_last else "│   "
                walk(child, prefix + extension)

    walk(root, "")
    return lines


def normalize_blank_lines(text: str, max_consecutive: int = 2) -> str:
    return re.sub(r"\n\s*\n", "\n\n", text)


def safe_read_text(path: Path, max_bytes: int) -> Tuple[str, bool]:
    data = b""
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes + 1)
    except Exception as e:
        return f"<<UNREADABLE: {e}>>", False

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("latin-1")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, truncated


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate an LLM-friendly Markdown context file for a repo."
    )
    ap.add_argument("--root", default=".", help="Repo root (default: .)")
    ap.add_argument("--out", default="llm_context.md", help="Output markdown file path")
    ap.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Restrict scanning to these top-level dirs/files (e.g. src docs). If empty, scans entire repo.",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Include globs (posix rel paths). If provided, only these are eligible (unless --only used to restrict scan). "
        'Example: "**/*.py" "docs/**/*.md"',
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help='Exclude globs (posix rel paths). Example: "**/tests/**" "**/*.min.*"',
    )
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=30_000,
        help="Max bytes per file content included (default 30000)",
    )
    ap.add_argument(
        "--max-total-bytes",
        type=int,
        default=500_000,
        help="Max total bytes of included contents (default 500000)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=250,
        help="Max number of files to include (default 250)",
    )
    ap.add_argument(
        "--no-gitignore", action="store_true", help="Ignore .gitignore even if present"
    )
    ap.add_argument(
        "--ext",
        nargs="*",
        default=[],
        help="Override included extensions. Example: --ext .py .md .txt (default includes a sensible set).",
    )
    ap.add_argument(
        "--header",
        default="",
        help="Optional header text to add at top of output markdown.",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out_path = (
        (root / args.out).resolve()
        if not Path(args.out).is_absolute()
        else Path(args.out).resolve()
    )

    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2

    included_exts = (
        set(e.lower() for e in args.ext) if args.ext else set(DEFAULT_INCLUDED_EXTS)
    )

    gitignore_match = None
    if not args.no_gitignore:
        gitignore_match = load_gitignore_matcher(root)

    # Determine scan roots
    scan_roots: List[Path] = []
    if args.only:
        for item in args.only:
            scan_roots.append((root / item).resolve())
    else:
        scan_roots = [root]

    picks: List[FilePick] = []

    def should_skip_dir(dir_name: str) -> bool:
        return dir_name in DEFAULT_EXCLUDED_DIRS

    # Walk
    for base in scan_roots:
        if not base.exists():
            continue
        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(base):
                # prune dirs in-place
                dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
                pdir = Path(dirpath)
                for fn in filenames:
                    candidates.append(pdir / fn)

        for p in candidates:
            if not p.is_file():
                continue
            rel = normalize_rel(p, root)

            # Gitignore (best effort)
            if gitignore_match and gitignore_match(rel):
                continue

            # Exclude dirs already handled; also exclude obvious junk files
            if any(part in DEFAULT_EXCLUDED_DIRS for part in Path(rel).parts):
                continue

            # Exclude globs
            if args.exclude and matches_any_glob(rel, args.exclude):
                continue

            # Must match include globs if provided
            if args.include and not matches_any_glob(rel, args.include):
                continue

            ext = p.suffix.lower()
            if ext and ext not in included_exts:
                # allow extensionless but common config files, only if explicitly included by glob
                continue

            # skip binary-ish
            if is_probably_binary(p):
                continue

            try:
                size = p.stat().st_size
            except Exception:
                continue

            picks.append(FilePick(abs_path=p, rel_posix=rel, size_bytes=size))

    # Sort: prioritize src/ and docs/, then smaller files first (helps pack more relevant context)
    def sort_key(fp: FilePick):
        rel = fp.rel_posix
        priority = 2
        if rel.startswith("src/"):
            priority = 0
        elif rel.startswith("docs/"):
            priority = 1
        return (priority, fp.size_bytes, rel)

    picks.sort(key=sort_key)

    # Enforce max-files
    picks = picks[: max(0, args.max_files)]

    # Build tree lines from selected files only
    tree_lines = build_tree_lines([fp.rel_posix for fp in picks])

    total_included = 0
    included_files: List[Tuple[FilePick, str, bool]] = []

    for fp in picks:
        remaining = args.max_total_bytes - total_included
        if remaining <= 0:
            break

        per_file_limit = min(args.max_file_bytes, remaining)
        text, truncated = safe_read_text(fp.abs_path, per_file_limit)
        total_included += min(fp.size_bytes, per_file_limit)
        included_files.append((fp, text, truncated))

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        if args.header.strip():
            out.write(args.header.strip() + "\n\n")

        out.write("# Repo context pack\n\n")
        out.write(f"- Root: `{root}`\n")
        out.write(
            f"- Included files: {len(included_files)} (scanned candidates: {len(picks)})\n"
        )
        out.write(f"- Max per file: {args.max_file_bytes} bytes\n")
        out.write(f"- Max total: {args.max_total_bytes} bytes\n\n")

        out.write("## File tree (included)\n\n")
        out.write("```text\n")
        for line in tree_lines:
            out.write(line + "\n")
        out.write("```\n\n")

        out.write("## Files\n\n")
        for fp, text, truncated in included_files:
            lang = guess_lang_for_fence(fp.abs_path)
            out.write(f"### `{fp.rel_posix}`\n\n")
            if truncated:
                out.write("> Note: truncated to fit limits.\n\n")
            out.write(f"```{lang}\n")
            # Avoid accidentally closing fence if file contains ```
            safe_text = normalize_blank_lines(text)
            safe_text = safe_text.replace("```", "``\\`")
            out.write(safe_text.rstrip() + "\n")
            out.write("```\n\n")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
