"""Shared bootstrap for notebooks in the AbstractGraph ecosystem.

This file is intended to be executed from notebooks via ``runpy.run_path``.
It resolves the current repo and workspace layout from a range of likely
starting directories, prepends available ``src`` directories to ``sys.path``,
and normalizes the process working directory to the current repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_NAME = "abstractgraph-generative"
SIBLING_REPOS = ("abstractgraph", "abstractgraph-ml", "abstractgraph-generative", "abstractgraph-graphicalizer")
EXTERNAL_SIBLING_REPOS = ("NSPPK",)


def _is_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "src").exists()


def _candidate_repo_roots(start: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (start, *start.parents):
        candidates.append(base)
        candidates.append(base / REPO_NAME)
        for sibling in SIBLING_REPOS:
            candidates.append(base / sibling)
    return candidates


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    seen: set[Path] = set()
    for candidate in _candidate_repo_roots(start):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == REPO_NAME and _is_repo_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root for {REPO_NAME!r} from starting path {start}"
    )


def find_workspace_root(repo_root: Path) -> Path:
    parent = repo_root.parent
    sibling_count = sum((parent / sibling).exists() for sibling in SIBLING_REPOS)
    return parent if sibling_count >= 2 else repo_root


def bootstrap(start: Path | None = None) -> dict[str, Path]:
    repo_root = find_repo_root(start)
    workspace_root = find_workspace_root(repo_root)

    external_search_roots = (
        workspace_root,
        workspace_root.parent,
        workspace_root.parent.parent,
    )
    candidate_src_dirs = [
        repo_root / "src",
        *(workspace_root / sibling / "src" for sibling in SIBLING_REPOS),
        *(
            external_root / sibling / "src"
            for external_root in external_search_roots
            for sibling in EXTERNAL_SIBLING_REPOS
        ),
    ]
    for src_dir in candidate_src_dirs:
        if src_dir.exists():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    os.chdir(repo_root)
    return {
        "repo_root": repo_root,
        "workspace_root": workspace_root,
    }


globals().update(bootstrap())
