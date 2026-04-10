#!/usr/bin/env python3
"""Generate a folder-oriented view of git history.

The script prints the one-line summaries for every commit grouped by the
repository's top-level folders.  Merge commits are included and annotated.
In addition to the top-level breakdown, selected folders (like ``variations``
 or ``data``) are expanded into sub-folder sections to highlight more granular
activity.

Run from anywhere inside the repository:
    python report/list_commits_by_folder.py
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    summary: str
    is_merge: bool

    def format_line(self) -> str:
        merge_suffix = " (merge)" if self.is_merge else ""
        return f"- [{self.sha[:7]}] {self.summary}{merge_suffix}"


def iter_commits(revision_range: str) -> Iterator[Tuple[CommitInfo, List[str]]]:
    """Yield commits and their touched files for the revision range."""

    format_spec = "%x1e%H%x1f%P%x1f%s"
    result = subprocess.run(
        [
            "git",
            "log",
            revision_range,
            "--topo-order",
            "--name-only",
            f"--pretty=format:{format_spec}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    chunks = result.stdout.split("\x1e")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        header = lines[0]
        try:
            sha, parents_str, summary = header.split("\x1f")
        except ValueError:
            # Skip malformed entries.
            continue
        parents = parents_str.split()
        files = [line.strip() for line in lines[1:] if line.strip()]
        info = CommitInfo(sha=sha, summary=summary, is_merge=len(parents) > 1)
        yield info, files


def classify_files(files: Iterable[str]) -> Tuple[Dict[str, Set[str]], bool]:
    """Group file paths by top-level folder and capture sub-folders.

    Returns
    -------
    folder_map:
        Maps a top-level folder (or ``[root]`` for top-level files) to the set
        of immediate sub-folders (``"[files]"`` is used when the file lives
        directly under the folder).
    contains_files:
        Indicates if there were any files at the repository root.
    """
    folder_map: Dict[str, Set[str]] = defaultdict(set)
    contains_root_files = False

    for path in files:
        parts = Path(path).parts
        if not parts:
            continue
        if len(parts) == 1:
            folder_map["[root]"].add("[files]")
            contains_root_files = True
            continue
        top, rest = parts[0], parts[1:]
        if rest:
            folder_map[top].add(rest[0])
        else:
            folder_map[top].add("[files]")

    return folder_map, contains_root_files


def collect_history(revision_range: str) -> Tuple[
    Dict[str, List[CommitInfo]],
    Dict[str, Dict[str, List[CommitInfo]]],
]:
    """Collect commit information grouped by folders."""
    top_level_history: Dict[str, List[CommitInfo]] = defaultdict(list)
    detailed_history: Dict[str, Dict[str, List[CommitInfo]]] = defaultdict(
        lambda: defaultdict(list)
    )

    seen_top: Dict[str, Set[str]] = defaultdict(set)
    seen_detail: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for info, files in iter_commits(revision_range):
        if not files:
            # Skip commits without file changes (e.g., merge commits that only
            # updated metadata) because there is nothing to classify.
            continue
        folder_map, _ = classify_files(files)
        for folder, subfolders in folder_map.items():
            if info.sha not in seen_top[folder]:
                top_level_history[folder].append(info)
                seen_top[folder].add(info.sha)
            # Provide a detailed breakout for select folders.
            if folder in {"variations", "data", "train_variations"}:
                for sub in subfolders:
                    key = (folder, sub)
                    if info.sha in seen_detail[key]:
                        continue
                    detailed_history[folder][sub].append(info)
                    seen_detail[key].add(info.sha)

    return top_level_history, detailed_history


def print_history(
    top_level_history: Dict[str, List[CommitInfo]],
    detailed_history: Dict[str, Dict[str, List[CommitInfo]]],
) -> None:
    print("# Commits by Top-Level Folder\n")
    for folder in sorted(top_level_history.keys()):
        print(f"## {folder}")
        for info in top_level_history[folder]:
            print(info.format_line())
        print()

    if detailed_history:
        print("# Detailed Breakdowns\n")
        for folder in sorted(detailed_history.keys()):
            print(f"## {folder}")
            for subfolder in sorted(detailed_history[folder].keys()):
                label = "[files]" if subfolder == "[files]" else subfolder
                print(f"### {folder}/{label}")
                for info in detailed_history[folder][subfolder]:
                    print(info.format_line())
                print()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="List git commit one-liners grouped by folders."
    )
    parser.add_argument(
        "revision_range",
        nargs="?",
        default="HEAD",
        help="Revision range passed to git log (default: HEAD)",
    )
    args = parser.parse_args(argv)

    top_level_history, detailed_history = collect_history(args.revision_range)
    if not top_level_history:
        print("No file changes detected across the revision range.")
        return

    print_history(top_level_history, detailed_history)


if __name__ == "__main__":
    main()
