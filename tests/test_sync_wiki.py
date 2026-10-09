#!/usr/bin/env python

"""Offline tests for the docs-as-code Wiki synchronizer."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sync_wiki.py"


def _load_sync_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_wiki", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_wiki = _load_sync_module()


def _write_page(directory: Path, name: str, body: str = "# Page\n") -> None:
    (directory / name).write_text(body, encoding="utf-8")


def _write_source_manifest(
    source: Path,
    repositories: dict[str, object],
    unverified_repositories: dict[str, object] | None = None,
    excluded_repositories: dict[str, object] | None = None,
) -> None:
    manifest = {
        "schema_version": 1,
        "recorded_at": "2026-10-07T00:00:00Z",
        "wiki_baseline": {
            "repository": "The-Strategy-Unit/nhp_products.wiki",
            "commit": "a" * 40,
        },
        "repositories": repositories,
        "excluded_repositories": excluded_repositories or {},
        "unverified_repositories": unverified_repositories or {},
    }
    (source.parent / sync_wiki.SOURCE_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _run_git(
    directory: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _initialize_git_repository(directory: Path) -> None:
    _run_git(directory, "init", "--quiet")
    _run_git(directory, "config", "user.name", "Wiki test")
    _run_git(directory, "config", "user.email", "wiki-test@example.invalid")
    _run_git(directory, "add", "--all")
    _run_git(directory, "commit", "--quiet", "-m", "Initial Wiki")


def test_validate_source() -> None:
    """A source directory needs a non-empty Home page."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory)
        _write_page(source, "Home.md", "# Home\n")
        _write_page(source, "Data.md", "# Data\n")
        _write_source_manifest(source, {})

        pages = sync_wiki.validate_source(source)
        assert list(pages) == ["Data.md", "Home.md"]


def test_validate_source_requires_manifest() -> None:
    """Validation rejects source pages without their provenance manifest."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory) / "docs" / "wiki"
        source.mkdir(parents=True)
        _write_page(source, "Home.md")

        try:
            sync_wiki.validate_source(source)
        except sync_wiki.WikiSyncError as error:
            assert "manifest does not exist" in str(error)
        else:
            raise AssertionError("Expected validation to reject a missing manifest")


def test_validate_source_requires_home() -> None:
    """Validation rejects a source without the Wiki home page."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory)
        _write_page(source, "Data.md")

        try:
            sync_wiki.validate_source(source)
        except sync_wiki.WikiSyncError as error:
            assert "Home.md" in str(error)
        else:
            raise AssertionError("Expected validation to reject a missing Home.md")


def test_validate_source_rejects_unexpected_files() -> None:
    """A source file that cannot be published must not pass silently."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory)
        _write_page(source, "Home.md")
        (source / "Draft.mdx").write_text("# Draft\n", encoding="utf-8")

        try:
            sync_wiki.validate_source(source)
        except sync_wiki.WikiSyncError as error:
            assert "Draft.mdx" in str(error)
            assert "only .md" in str(error)
        else:
            raise AssertionError("Expected validation to reject a non-Markdown page")


def test_validate_source_rejects_missing_internal_link() -> None:
    """Canonical links to this Wiki must resolve to source pages."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory)
        _write_page(
            source,
            "Home.md",
            "[Missing](https://github.com/The-Strategy-Unit/nhp_products/wiki/Missing)\n",
        )

        try:
            sync_wiki.validate_source(source)
        except sync_wiki.WikiSyncError as error:
            assert "missing internal page" in str(error)
            assert "/wiki/Missing" in str(error)
        else:
            raise AssertionError("Expected validation to reject a broken Wiki link")


def test_validate_source_checks_adjacent_manifest() -> None:
    """A linked public repository needs a SHA checkpoint and page mapping."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory) / "docs" / "wiki"
        source.mkdir(parents=True)
        _write_page(
            source,
            "Home.md",
            "[Example](https://github.com/The-Strategy-Unit/example)\n",
        )
        _write_source_manifest(
            source,
            {
                "The-Strategy-Unit/example": {
                    "branch": "main",
                    "head": "b" * 40,
                    "pages": ["Home.md"],
                }
            },
        )

        pages = sync_wiki.validate_source(source)

        assert list(pages) == ["Home.md"]


def test_validate_source_rejects_unrecorded_repository_link() -> None:
    """A manifest must classify every linked Strategy Unit repository."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory) / "docs" / "wiki"
        source.mkdir(parents=True)
        _write_page(
            source,
            "Home.md",
            "[Missing](https://github.com/The-Strategy-Unit/missing)\n",
        )
        _write_source_manifest(source, {})

        try:
            sync_wiki.validate_source(source)
        except sync_wiki.WikiSyncError as error:
            assert "The-Strategy-Unit/missing" in str(error)
        else:
            raise AssertionError("Expected validation to reject an unrecorded repository")


def test_validate_source_accepts_owner_approved_repository_exclusion() -> None:
    """A linked repository may be classified by an explicit owner exclusion."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        source = Path(temporary_directory) / "docs" / "wiki"
        source.mkdir(parents=True)
        _write_page(
            source,
            "Home.md",
            "[Private](https://github.com/The-Strategy-Unit/private-source)\n",
        )
        _write_source_manifest(
            source,
            {},
            excluded_repositories={
                "The-Strategy-Unit/private-source": {
                    "reason": "Owner confirmed this source is intentionally private",
                    "pages": ["Home.md"],
                    "approved_by": "@wiki-owner",
                    "approved_at": "2026-10-09T12:00:00Z",
                }
            },
        )

        pages = sync_wiki.validate_source(source)

        assert list(pages) == ["Home.md"]


def test_synchronize_copies_changed_and_missing_pages() -> None:
    """Synchronization copies source bytes into an existing Wiki clone."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source"
        destination = root / "wiki"
        source.mkdir()
        destination.mkdir()
        (destination / ".git").mkdir()
        _write_page(source, "Home.md", "# New home\n")
        _write_page(source, "Data.md", "# Data\n")
        _write_source_manifest(source, {})
        _write_page(destination, "Home.md", "# Old home\n")

        changed = sync_wiki.synchronize(source, destination)

        assert changed == ["Data.md", "Home.md"]
        assert (destination / "Home.md").read_bytes() == (source / "Home.md").read_bytes()
        assert (destination / "Data.md").read_bytes() == (source / "Data.md").read_bytes()


def test_synchronize_refuses_unmanaged_pages_before_writing() -> None:
    """An unexpected destination page blocks the entire synchronization."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source"
        destination = root / "wiki"
        source.mkdir()
        destination.mkdir()
        (destination / ".git").mkdir()
        _write_page(source, "Home.md", "# New home\n")
        _write_source_manifest(source, {})
        _write_page(destination, "Home.md", "# Old home\n")
        _write_page(destination, "Unexpected.md")

        try:
            sync_wiki.synchronize(source, destination)
        except sync_wiki.WikiSyncError as error:
            assert "Unexpected.md" in str(error)
        else:
            raise AssertionError("Expected synchronization to reject an unmanaged page")

        assert "# Old home" in (destination / "Home.md").read_text(encoding="utf-8")


def test_git_publication_stages_and_commits_a_new_page() -> None:
    """The publication sequence includes pages that start out untracked."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source"
        destination = root / "wiki"
        source.mkdir()
        destination.mkdir()
        _write_page(
            source,
            "Home.md",
            "[Data](https://github.com/The-Strategy-Unit/nhp_products/wiki/Data)\n",
        )
        _write_page(source, "Data.md", "# Data\n")
        _write_source_manifest(
            source,
            {
                "The-Strategy-Unit/nhp_products": {
                    "branch": "main",
                    "head": "b" * 40,
                    "pages": ["Home.md"],
                }
            },
        )
        _write_page(destination, "Home.md", "# Old home\n")
        _initialize_git_repository(destination)

        changed = sync_wiki.synchronize(source, destination)
        _run_git(destination, "add", "--", "*.md")
        staged_diff = _run_git(destination, "diff", "--cached", "--quiet", check=False)

        assert changed == ["Data.md", "Home.md"]
        assert staged_diff.returncode == 1

        _run_git(destination, "commit", "--quiet", "-m", "Publish Wiki")
        published_data = _run_git(destination, "show", "HEAD:Data.md").stdout
        assert published_data == (source / "Data.md").read_text(encoding="utf-8")


def test_git_publication_does_not_commit_unchanged_pages() -> None:
    """An unchanged Wiki produces an empty staged diff."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source"
        destination = root / "wiki"
        source.mkdir()
        destination.mkdir()
        _write_page(source, "Home.md", "# Home\n")
        _write_source_manifest(source, {})
        _write_page(destination, "Home.md", "# Home\n")
        _initialize_git_repository(destination)

        changed = sync_wiki.synchronize(source, destination)
        _run_git(destination, "add", "--", "*.md")
        staged_diff = _run_git(destination, "diff", "--cached", "--quiet", check=False)

        assert changed == []
        assert staged_diff.returncode == 0


def test_git_publication_restores_prior_content_without_rewriting_history() -> None:
    """A rollback is a new commit whose tree restores the prior source."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source"
        destination = root / "wiki"
        source.mkdir()
        destination.mkdir()
        _write_page(source, "Home.md", "# Original home\n")
        _write_source_manifest(source, {})
        original_content = (source / "Home.md").read_bytes()
        (destination / "Home.md").write_bytes(original_content)
        _initialize_git_repository(destination)
        original_sha = _run_git(destination, "rev-parse", "HEAD").stdout.strip()

        _write_page(source, "Home.md", "# Mistaken home\n")
        assert sync_wiki.synchronize(source, destination) == ["Home.md"]
        _run_git(destination, "add", "--", "*.md")
        _run_git(destination, "commit", "--quiet", "-m", "Publish mistake")
        mistaken_sha = _run_git(destination, "rev-parse", "HEAD").stdout.strip()

        (source / "Home.md").write_bytes(original_content)
        assert sync_wiki.synchronize(source, destination) == ["Home.md"]
        _run_git(destination, "add", "--", "*.md")
        _run_git(destination, "commit", "--quiet", "-m", "Restore Wiki")
        restored_sha = _run_git(destination, "rev-parse", "HEAD").stdout.strip()

        assert (destination / "Home.md").read_bytes() == original_content
        assert _run_git(destination, "rev-parse", f"{mistaken_sha}^").stdout.strip() == (
            original_sha
        )
        assert _run_git(destination, "rev-parse", f"{restored_sha}^").stdout.strip() == (
            mistaken_sha
        )
        assert _run_git(destination, "rev-list", "--count", "HEAD").stdout.strip() == "3"


def main() -> int:
    """Run the offline smoke tests."""
    test_validate_source()
    test_validate_source_requires_manifest()
    test_validate_source_requires_home()
    test_validate_source_rejects_unexpected_files()
    test_validate_source_rejects_missing_internal_link()
    test_validate_source_checks_adjacent_manifest()
    test_validate_source_rejects_unrecorded_repository_link()
    test_synchronize_copies_changed_and_missing_pages()
    test_synchronize_refuses_unmanaged_pages_before_writing()
    test_git_publication_stages_and_commits_a_new_page()
    test_git_publication_does_not_commit_unchanged_pages()
    test_git_publication_restores_prior_content_without_rewriting_history()
    print("All Wiki synchronization tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
