"""Validate and synchronize the docs-as-code source for the GitHub Wiki."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

FORBIDDEN_FILENAME_CHARACTERS = frozenset('\\/:*?"<>|')
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
GITHUB_REPOSITORY_PATTERN = re.compile(
    r"The-Strategy-Unit/[A-Za-z0-9_.-]+", re.IGNORECASE
)
MARKDOWN_LINK_PATTERN = re.compile(
    r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)"
)
WIKI_HOST = "github.com"
WIKI_PATH = "/The-Strategy-Unit/nhp_products/wiki"
SOURCE_MANIFEST_NAME = "wiki-sources.json"
MIN_GITHUB_PATH_PARTS = 2


class WikiSyncError(ValueError):
    """Raised when synchronizing could lose data or publish invalid pages."""


def _markdown_pages(
    directory: Path, *, reject_unexpected_files: bool = False
) -> dict[str, Path]:
    """Return the top-level Markdown pages in a directory, keyed by filename."""
    if not directory.is_dir():
        raise WikiSyncError(f"Wiki directory does not exist: {directory}")

    pages: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name == ".git":
            continue
        if path.is_symlink():
            raise WikiSyncError(f"Wiki pages must not be symbolic links: {path.name}")
        if path.is_dir():
            raise WikiSyncError(f"Wiki pages must be top-level files: {path.name}")
        if path.suffix.lower() != ".md":
            if reject_unexpected_files:
                raise WikiSyncError(
                    "Wiki source contains an unsupported file; only .md pages are "
                    f"published: {path.name}"
                )
            continue
        pages[path.name] = path
    return pages


def _validate_internal_links(pages: dict[str, Path], contents: dict[str, str]) -> None:
    """Require canonical links to this Wiki to resolve to source pages."""
    for source_name, content in contents.items():
        for match in MARKDOWN_LINK_PATTERN.finditer(content):
            target = urlsplit(match.group("target"))
            if target.scheme != "https" or target.netloc != WIKI_HOST:
                continue
            if target.path == WIKI_PATH:
                target_slug = "Home"
            elif target.path.startswith(f"{WIKI_PATH}/"):
                target_slug = unquote(target.path.removeprefix(f"{WIKI_PATH}/"))
                target_slug = target_slug.rstrip("/") or "Home"
            else:
                continue

            target_name = f"{target_slug}.md"
            if "/" in target_slug or target_name not in pages:
                raise WikiSyncError(
                    f"Wiki page {source_name} links to a missing internal page: "
                    f"{match.group('target')}"
                )


def _linked_repositories(contents: dict[str, str]) -> dict[str, set[str]]:
    """Return Strategy Unit GitHub repositories linked by each Wiki page."""
    repositories: dict[str, set[str]] = {}
    for source_name, content in contents.items():
        for match in MARKDOWN_LINK_PATTERN.finditer(content):
            target = urlsplit(match.group("target"))
            if target.scheme != "https" or target.netloc.casefold() != WIKI_HOST:
                continue
            path_parts = target.path.strip("/").split("/")
            if (
                len(path_parts) < MIN_GITHUB_PATH_PARTS
                or path_parts[0].casefold() != "the-strategy-unit"
            ):
                continue
            repository = f"The-Strategy-Unit/{unquote(path_parts[1])}"
            repositories.setdefault(repository, set()).add(source_name)
    return repositories


def _validate_manifest_entry(
    repository: str,
    entry: object,
    pages: dict[str, Path],
    *,
    verified: bool,
) -> set[str]:
    """Validate one repository checkpoint and return its associated pages."""
    if not GITHUB_REPOSITORY_PATTERN.fullmatch(repository) or not isinstance(entry, dict):
        raise WikiSyncError(f"Invalid Wiki source manifest entry: {repository}")
    entry_data = cast(dict[str, object], entry)

    recorded_pages = entry_data.get("pages")
    if (
        not isinstance(recorded_pages, list)
        or not recorded_pages
        or not all(isinstance(page, str) for page in recorded_pages)
    ):
        raise WikiSyncError(f"Manifest entry has invalid pages: {repository}")
    missing_pages = sorted(set(recorded_pages) - pages.keys())
    if missing_pages:
        raise WikiSyncError(
            f"Manifest entry {repository} refers to missing pages: "
            f"{', '.join(missing_pages)}"
        )

    if verified:
        branch = entry_data.get("branch")
        head = entry_data.get("head")
        if not isinstance(branch, str) or not branch:
            raise WikiSyncError(f"Manifest entry has no branch: {repository}")
        if not isinstance(head, str) or not COMMIT_PATTERN.fullmatch(head):
            raise WikiSyncError(f"Manifest entry has an invalid head SHA: {repository}")
    else:
        reason = entry_data.get("reason")
        if not isinstance(reason, str) or not reason:
            raise WikiSyncError(f"Unverified manifest entry has no reason: {repository}")

    return set(cast(list[str], recorded_pages))


def _validate_source_manifest(
    source: Path, pages: dict[str, Path], contents: dict[str, str]
) -> None:
    """Validate the public-source checkpoints beside docs/wiki."""
    manifest_path = source.parent / SOURCE_MANIFEST_NAME
    if not manifest_path.exists():
        raise WikiSyncError(f"Wiki source manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WikiSyncError(f"Invalid Wiki source manifest: {manifest_path}") from error

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise WikiSyncError("Wiki source manifest has an unsupported schema")
    baseline = manifest.get("wiki_baseline")
    if not isinstance(baseline, dict) or not COMMIT_PATTERN.fullmatch(
        str(baseline.get("commit", ""))
    ):
        raise WikiSyncError("Wiki source manifest has an invalid baseline commit")

    verified = manifest.get("repositories")
    unverified = manifest.get("unverified_repositories")
    if not isinstance(verified, dict) or not isinstance(unverified, dict):
        raise WikiSyncError("Wiki source manifest repository groups must be objects")
    overlap = sorted(verified.keys() & unverified.keys())
    if overlap:
        raise WikiSyncError(
            "Manifest repositories are both verified and unverified: "
            f"{', '.join(overlap)}"
        )

    recorded: dict[str, set[str]] = {}
    for repository, entry in verified.items():
        recorded[repository.casefold()] = _validate_manifest_entry(
            repository, entry, pages, verified=True
        )
    for repository, entry in unverified.items():
        recorded[repository.casefold()] = _validate_manifest_entry(
            repository, entry, pages, verified=False
        )

    for repository, linked_pages in _linked_repositories(contents).items():
        recorded_pages = recorded.get(repository.casefold())
        if recorded_pages is None:
            raise WikiSyncError(
                f"Linked repository is absent from the source manifest: {repository}"
            )
        missing_mappings = sorted(linked_pages - recorded_pages)
        if missing_mappings:
            raise WikiSyncError(
                f"Manifest entry {repository} is missing linked pages: "
                f"{', '.join(missing_mappings)}"
            )


def validate_source(source: Path) -> dict[str, Path]:
    """Validate source pages and return them keyed by filename."""
    pages = _markdown_pages(source, reject_unexpected_files=True)
    if "Home.md" not in pages:
        raise WikiSyncError("The Wiki source must contain Home.md")

    contents: dict[str, str] = {}
    for name, path in pages.items():
        if any(character in FORBIDDEN_FILENAME_CHARACTERS for character in name):
            raise WikiSyncError(f"Wiki page has an unsafe filename: {name}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise WikiSyncError(f"Wiki page is not valid UTF-8: {name}") from error
        if not content.strip():
            raise WikiSyncError(f"Wiki page is empty: {name}")
        contents[name] = content

    _validate_internal_links(pages, contents)
    _validate_source_manifest(source, pages, contents)

    return pages


def find_drift(source: Path, destination: Path) -> tuple[list[str], list[str]]:
    """Return changed/missing pages and unmanaged destination pages."""
    source_pages = validate_source(source)
    destination_pages = _markdown_pages(destination)
    unmanaged = sorted(destination_pages.keys() - source_pages.keys())
    changed = sorted(
        name
        for name, source_path in source_pages.items()
        if name not in destination_pages
        or source_path.read_bytes() != destination_pages[name].read_bytes()
    )
    return changed, unmanaged


def synchronize(source: Path, destination: Path) -> list[str]:
    """Copy managed pages to a Wiki clone without deleting any Wiki pages."""
    if not (destination / ".git").exists():
        raise WikiSyncError(f"Destination is not a Git working tree: {destination}")

    changed, unmanaged = find_drift(source, destination)
    if unmanaged:
        page_list = ", ".join(unmanaged)
        raise WikiSyncError(
            "Refusing to delete unmanaged Wiki pages. Import or retire these pages "
            f"in docs/wiki first: {page_list}"
        )

    source_pages = validate_source(source)
    for name in changed:
        shutil.copyfile(source_pages[name], destination / name)
    return changed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate source pages")
    validate_parser.add_argument("source", type=Path)

    check_parser = subparsers.add_parser("check", help="compare source with a Wiki clone")
    check_parser.add_argument("source", type=Path)
    check_parser.add_argument("destination", type=Path)

    sync_parser = subparsers.add_parser("sync", help="copy source into a Wiki clone")
    sync_parser.add_argument("source", type=Path)
    sync_parser.add_argument("destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Wiki validation, comparison, or synchronization command."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            pages = validate_source(args.source)
            print(f"Validated {len(pages)} Wiki pages.")
            return 0

        if args.command == "check":
            changed, unmanaged = find_drift(args.source, args.destination)
            if unmanaged:
                print(f"Unmanaged Wiki pages: {', '.join(unmanaged)}", file=sys.stderr)
            if changed:
                print(f"Wiki pages with drift: {', '.join(changed)}", file=sys.stderr)
            if unmanaged or changed:
                return 1
            print("Wiki clone matches docs/wiki.")
            return 0

        changed = synchronize(args.source, args.destination)
        if changed:
            print(f"Synchronized Wiki pages: {', '.join(changed)}")
        else:
            print("Wiki is already up to date.")
        return 0
    except WikiSyncError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
