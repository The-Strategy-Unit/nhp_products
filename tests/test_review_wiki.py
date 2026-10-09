#!/usr/bin/env python

"""Offline tests for the staged AI-assisted Wiki source review."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import urllib.error
from email.message import Message
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "review_wiki.py"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "wiki-review.yaml"
ATTESTATION_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "wiki-review-attestation.yaml"
)
OLD_HEAD = "a" * 40
NEW_HEAD = "b" * 40
EXPECTED_ERROR_EXIT_CODE = 2
EXPECTED_MODEL_CALLS = 2


def _load_review_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("review_wiki", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


review_wiki = _load_review_module()


def _run_git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(directory: Path, message: str) -> str:
    _run_git(directory, "add", "--all")
    _run_git(directory, "commit", "--quiet", "-m", message)
    return _run_git(directory, "rev-parse", "HEAD")


def _repository(root: Path) -> tuple[Path, str]:
    repository = root / "source"
    repository.mkdir()
    _run_git(repository, "init", "--quiet", "--initial-branch=main")
    _run_git(repository, "config", "user.name", "Wiki test")
    _run_git(repository, "config", "user.email", "wiki-test@example.invalid")
    (repository / "README.md").write_text("# Old heading\n", encoding="utf-8")
    return repository, _commit(repository, "Initial source")


def _wiki(root: Path, head: str, *, gaps: dict | None = None) -> tuple[Path, Path]:
    docs = root / "docs"
    wiki = docs / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "Home.md").write_text("# Old heading\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "recorded_at": "2026-10-07T00:00:00Z",
        "wiki_baseline": {
            "repository": "The-Strategy-Unit/nhp_products.wiki",
            "commit": "c" * 40,
        },
        "repositories": {
            "The-Strategy-Unit/example": {
                "branch": "main",
                "head": head,
                "pages": ["Home.md"],
                "review_paths": ["README.md"],
            }
        },
        "excluded_repositories": {},
        "unverified_repositories": gaps or {},
    }
    manifest_path = docs / "wiki-sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return wiki, manifest_path


def _source(*, complete: bool = True, blockers: list[str] | None = None) -> dict:
    return {
        "source_kind": "public_git",
        "source_id": "The-Strategy-Unit/example",
        "handling": {
            "classification": "public",
            "policy": "public-repository-v1",
            "public_report": True,
        },
        "previous_revision": OLD_HEAD,
        "current_revision": NEW_HEAD,
        "allowed_pages": ["Home.md"],
        "review_paths": ["README.md"],
        "review_mode": "incremental",
        "pages": {"Home.md": "# Old heading\n"},
        "compare_url": (
            f"https://github.com/The-Strategy-Unit/example/compare/{OLD_HEAD}...{NEW_HEAD}"
        ),
        "changed_paths": [{"status": "M", "old_path": "README.md", "path": "README.md"}],
        "excluded_paths": [],
        "fragments": [
            {
                "locator": "README.md#diff-1",
                "kind": "diff",
                "path": "README.md",
                "revision": NEW_HEAD,
                "line_start": 0,
                "citable": True,
                "content": "-# Old heading\n+# New heading\n",
            }
        ],
        "complete": complete,
        "blockers": blockers or [],
    }


def _evidence(source: dict | None = None) -> dict:
    return {
        "schema_version": review_wiki.SCHEMA_VERSION,
        "review_mode": "incremental",
        "inventory_gaps": {},
        "inventory_exclusions": {},
        "sources": [source or _source()],
    }


def _edit(**overrides: object) -> dict:
    edit: dict[str, object] = {
        "page": "Home.md",
        "old_text": "Old heading",
        "new_text": "New heading",
        "evidence": [{"locator": "README.md#diff-1", "quote": "+# New heading"}],
        "rationale": "The heading changed.",
    }
    edit.update(overrides)
    return edit


def _proposal(*, disposition: str = "no_change", edits: list[dict] | None = None) -> dict:
    return {
        "source_kind": "public_git",
        "source_id": "The-Strategy-Unit/example",
        "previous_revision": OLD_HEAD,
        "current_revision": NEW_HEAD,
        "compare_url": (
            f"https://github.com/The-Strategy-Unit/example/compare/{OLD_HEAD}...{NEW_HEAD}"
        ),
        "disposition": disposition,
        "edits": edits or [],
        "summary": "Reviewed.",
        "blockers": [],
        "verification": {"verdict": "supported", "summary": "Supported."},
        "audit": {
            "prompt_version": review_wiki.PROMPT_VERSION,
            "prompt_sha256": review_wiki.PROMPT_SHA256,
            "deployment": "deployment",
            "generator_model": "model-snapshot",
            "verifier_model": "model-snapshot",
        },
    }


def _review(proposal: dict | None = None) -> dict:
    return {
        "schema_version": review_wiki.SCHEMA_VERSION,
        "deployment": "deployment",
        "proposals": [proposal or _proposal()],
    }


def _evaluation() -> dict:
    return {
        "schema_version": 1,
        "prompt_version": review_wiki.PROMPT_VERSION,
        "prompt_sha256": review_wiki.PROMPT_SHA256,
        "deployment": "deployment",
        "expected_model": "model-snapshot",
        "suite_sha256": "d" * 64,
        "passed": 10,
        "total": 10,
    }


def _structured(output: dict) -> object:
    return review_wiki.StructuredResponse(output=output, model="model-snapshot")


def test_chunk_text_preserves_lines_and_rejects_oversized_lines() -> None:
    chunks = review_wiki.chunk_text("aa\nbb\ncc\n", max_bytes=5)
    assert chunks == ["aa\n", "bb\n", "cc\n"]
    try:
        review_wiki.chunk_text("long line\n", max_bytes=4)
    except review_wiki.WikiReviewError as error:
        assert "line exceeds" in str(error)
    else:
        raise AssertionError("Expected an oversized source line to be rejected")


def test_collect_git_source_collects_diff_and_complete_snapshot() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository, old_head = _repository(root)
        wiki, _ = _wiki(root, old_head)
        (repository / "README.md").write_text("# New heading\n", encoding="utf-8")
        new_head = _commit(repository, "Update docs")

        source = review_wiki.collect_git_source(
            "The-Strategy-Unit/example",
            {
                "branch": "main",
                "head": old_head,
                "pages": ["Home.md"],
                "review_paths": ["README.md"],
            },
            wiki,
            remote_url=str(repository),
        )

    assert source is not None
    assert source["complete"] is True
    assert source["current_revision"] == new_head
    snapshots = [item for item in source["fragments"] if item["kind"] == "snapshot"]
    assert "".join(item["content"] for item in snapshots) == "# New heading\n"


def test_collect_git_source_records_owner_scoped_paths_without_sending_them() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository, old_head = _repository(root)
        wiki, _ = _wiki(root, old_head)
        (repository / "generated.lock").write_text("generated\n", encoding="utf-8")
        _commit(repository, "Update generated content")

        source = review_wiki.collect_git_source(
            "The-Strategy-Unit/example",
            {
                "branch": "main",
                "head": old_head,
                "pages": ["Home.md"],
                "review_paths": ["README.md"],
            },
            wiki,
            remote_url=str(repository),
        )

    assert source is not None
    assert source["complete"] is True
    assert source["excluded_paths"] == ["generated.lock"]
    assert "generated\n" not in json.dumps(source["fragments"])


def test_collect_git_source_full_audit_runs_when_head_is_unchanged() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository, head = _repository(root)
        wiki, _ = _wiki(root, head)

        source = review_wiki.collect_git_source(
            "The-Strategy-Unit/example",
            {
                "branch": "main",
                "head": head,
                "pages": ["Home.md"],
                "review_paths": ["README.md"],
            },
            wiki,
            remote_url=str(repository),
            full_audit=True,
        )

    assert source is not None
    assert source["review_mode"] == "full"
    assert source["complete"] is True
    assert any(item["kind"] == "snapshot" for item in source["fragments"])


def test_collect_git_source_blocks_binary_in_scope() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository, old_head = _repository(root)
        wiki, _ = _wiki(root, old_head)
        (repository / "diagram.bin").write_bytes(b"\x00\x01\x02")
        _commit(repository, "Add binary")

        source = review_wiki.collect_git_source(
            "The-Strategy-Unit/example",
            {
                "branch": "main",
                "head": old_head,
                "pages": ["Home.md"],
                "review_paths": ["**"],
            },
            wiki,
            remote_url=str(repository),
        )

    assert source is not None
    assert source["complete"] is False
    assert "Binary change" in " ".join(source["blockers"])


def test_complete_inventory_is_required_for_automatic_runs() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        repository, head = _repository(root)
        wiki, manifest = _wiki(
            root,
            head,
            gaps={
                "The-Strategy-Unit/unresolved": {
                    "reason": "owner decision required",
                    "pages": ["Home.md"],
                }
            },
        )
        try:
            review_wiki.collect_repositories(
                manifest, wiki, require_complete_inventory=True
            )
        except review_wiki.WikiReviewError as error:
            assert "complete repository inventory" in str(error)
        else:
            raise AssertionError(
                "Expected unresolved inventory to block automatic review"
            )
        assert repository.exists()


def test_review_source_requires_independent_semantic_verification() -> None:
    generated = {
        "disposition": "edit",
        "edits": [_edit()],
        "summary": "Update heading.",
    }
    verified = {"verdict": "supported", "summary": "The evidence entails the edit."}
    with patch.object(
        review_wiki,
        "request_structured_response",
        side_effect=[_structured(generated), _structured(verified)],
    ) as request:
        proposal = review_wiki.review_source(
            _source(),
            "https://example.test",
            "deployment",
            expected_model="model-snapshot",
        )

    assert request.call_count == EXPECTED_MODEL_CALLS
    assert proposal["disposition"] == "edit"
    assert proposal["edits"] == [_edit()]
    assert proposal["audit"]["generator_model"] == "model-snapshot"
    assert proposal["verification"]["verdict"] == "supported"
    generator_prompt = json.loads(request.call_args_list[0].args[0].prompt)
    verifier_prompt = json.loads(request.call_args_list[1].args[0].prompt)
    assert generator_prompt["evidence_fragments"] == _source()["fragments"]
    assert verifier_prompt["candidate"]["edits"] == [_edit()]


def test_review_source_downgrades_an_unsupported_proposal_to_manual_review() -> None:
    generated = {
        "disposition": "edit",
        "edits": [_edit()],
        "summary": "Update heading.",
    }
    verified = {"verdict": "unsupported", "summary": "The quote is irrelevant."}
    with patch.object(
        review_wiki,
        "request_structured_response",
        side_effect=[_structured(generated), _structured(verified)],
    ):
        proposal = review_wiki.review_source(
            _source(), "https://example.test", "deployment"
        )

    assert proposal["disposition"] == "manual_review"
    assert proposal["edits"] == []
    assert "did not support" in " ".join(proposal["blockers"])


def test_review_source_never_calls_ai_for_incomplete_evidence() -> None:
    with patch.object(review_wiki, "request_structured_response") as request:
        proposal = review_wiki.review_source(
            _source(complete=False, blockers=["binary"]),
            "https://example.test",
            "deployment",
        )
    request.assert_not_called()
    assert proposal["disposition"] == "manual_review"


def test_review_source_rejects_a_quote_missing_from_evidence() -> None:
    generated = {
        "disposition": "edit",
        "edits": [_edit(evidence=[{"locator": "README.md#diff-1", "quote": "invented"}])],
        "summary": "Unsafe.",
    }
    with patch.object(
        review_wiki,
        "request_structured_response",
        return_value=_structured(generated),
    ):
        try:
            review_wiki.review_source(_source(), "https://example.test", "deployment")
        except review_wiki.WikiReviewError as error:
            assert "unverified evidence quote" in str(error)
        else:
            raise AssertionError("Expected fabricated evidence to be rejected")


def test_request_is_stateless_strict_and_records_the_resolved_model() -> None:
    response_body = {
        "model": "model-snapshot",
        "status": "completed",
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "disposition": "no_change",
                                "edits": [],
                                "summary": "No change.",
                            }
                        ),
                    }
                ]
            }
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(response_body).encode("utf-8")

    with patch.object(
        review_wiki.urllib.request, "urlopen", return_value=Response()
    ) as call:
        result = review_wiki.request_structured_response(
            review_wiki.StructuredRequest(
                endpoint="https://example.openai.azure.com",
                deployment="deployment",
                prompt="untrusted evidence",
                schema_name="test",
                schema=review_wiki.PROPOSAL_SCHEMA,
            ),
            token="token",
            expected_model="model-snapshot",
        )

    sent = json.loads(call.call_args.args[0].data)
    assert sent["store"] is False
    assert sent["text"]["format"]["strict"] is True
    assert result.model == "model-snapshot"
    assert result.output["disposition"] == "no_change"


def test_request_rejects_a_non_azure_endpoint_before_acquiring_a_token() -> None:
    with patch.object(review_wiki, "_azure_cli_token") as token:
        try:
            review_wiki.request_structured_response(
                review_wiki.StructuredRequest(
                    endpoint="https://attacker.example",
                    deployment="deployment",
                    prompt="public evidence",
                    schema_name="test",
                    schema=review_wiki.PROPOSAL_SCHEMA,
                )
            )
        except review_wiki.WikiReviewError as error:
            assert "approved HTTPS host" in str(error)
        else:
            raise AssertionError("Expected an untrusted endpoint to be rejected")
    token.assert_not_called()


def test_request_rejects_an_unexpected_model_without_exposing_content() -> None:
    response_body = {
        "model": "different-model",
        "status": "completed",
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {"disposition": "no_change", "edits": [], "summary": "Safe."}
                        ),
                    }
                ]
            }
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(response_body).encode("utf-8")

    with patch.object(review_wiki.urllib.request, "urlopen", return_value=Response()):
        try:
            review_wiki.request_structured_response(
                review_wiki.StructuredRequest(
                    endpoint="https://example.openai.azure.com",
                    deployment="deployment",
                    prompt="sensitive evidence",
                    schema_name="test",
                    schema=review_wiki.PROPOSAL_SCHEMA,
                ),
                token="token",
                expected_model="model-snapshot",
            )
        except review_wiki.WikiReviewError as error:
            assert "unexpected model" in str(error)
            assert "sensitive evidence" not in str(error)
        else:
            raise AssertionError("Expected model drift to be rejected")


def test_request_retries_throttling_without_exposing_body() -> None:
    error = urllib.error.HTTPError(
        "https://example.test",
        review_wiki.HTTP_TOO_MANY_REQUESTS,
        "throttled",
        Message(),
        None,
    )
    with (
        patch.object(review_wiki.urllib.request, "urlopen", side_effect=error),
        patch.object(review_wiki.time, "sleep"),
    ):
        try:
            review_wiki.request_structured_response(
                review_wiki.StructuredRequest(
                    endpoint="https://example.openai.azure.com",
                    deployment="deployment",
                    prompt="sensitive value",
                    schema_name="test",
                    schema=review_wiki.PROPOSAL_SCHEMA,
                ),
                token="token",
            )
        except review_wiki.WikiReviewError as caught:
            assert "sensitive value" not in str(caught)
            assert "4 attempts" in str(caught)
        else:
            raise AssertionError("Expected throttling exhaustion to fail")


def test_evaluation_requires_every_labelled_case_to_pass() -> None:
    cases = {
        "schema_version": 1,
        "cases": [
            {
                "name": "stale-heading",
                "source": _source(),
                "expected": {
                    "disposition": "edit",
                    "required_new_text": ["New heading"],
                },
            }
        ],
    }
    with patch.object(
        review_wiki,
        "review_source",
        return_value=_proposal(disposition="edit", edits=[_edit()]),
    ):
        result = review_wiki.evaluate_cases(
            cases, "https://example.test", "deployment", "model-snapshot"
        )
    assert result["passed"] == result["total"] == 1


def test_small_unapproved_evaluation_suite_cannot_enable_scheduling() -> None:
    cases = {"schema_version": 1, "approved_for_scheduling": False, "cases": [{}]}
    try:
        review_wiki.evaluate_cases(
            cases,
            "https://example.test",
            "deployment",
            "model-snapshot",
            require_scheduling_approval=True,
        )
    except review_wiki.WikiReviewError as error:
        assert "owner-approved evaluation suite" in str(error)
    else:
        raise AssertionError("Expected the seed evaluation suite to block scheduling")


def test_current_head_approval_requires_a_non_bot_codeowner() -> None:
    reviews = [
        {
            "state": "APPROVED",
            "commit_id": NEW_HEAD,
            "user": {"login": "yiwen-h", "type": "User"},
        }
    ]
    reviewer = review_wiki.verify_current_head_approval(
        reviews, NEW_HEAD, "* @yiwen-h @ai-mindset\n"
    )
    assert reviewer == "yiwen-h"

    reviews[0]["commit_id"] = OLD_HEAD
    try:
        review_wiki.verify_current_head_approval(
            reviews, NEW_HEAD, "* @yiwen-h @ai-mindset\n"
        )
    except review_wiki.WikiReviewError as error:
        assert "current AI Wiki proposal head" in str(error)
    else:
        raise AssertionError("Expected a stale approval to be rejected")


def test_report_is_findings_only_and_exposes_inventory_gaps() -> None:
    proposal = _proposal(disposition="edit", edits=[_edit()])
    evidence = _evidence()
    evidence["inventory_gaps"] = {
        "The-Strategy-Unit/unresolved": "owner decision required"
    }

    report = review_wiki.render_report(_review(proposal), evidence, _evaluation())

    assert "untrusted AI findings report" in report
    assert "never applied automatically" in report
    assert "The-Strategy-Unit/unresolved" in report
    assert NEW_HEAD in report
    assert "+# New heading" in report
    assert review_wiki.PROMPT_VERSION in report
    assert "10/10" in report
    assert "d" * 64 in report
    assert report.index("### Live evaluation gate") < report.index("| Source |")
    assert report.index("The-Strategy-Unit/unresolved") < report.index("| Source |")


def test_review_cli_requires_a_pinned_expected_model() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(json.dumps(_evidence()), encoding="utf-8")
        try:
            review_wiki._parser().parse_args(
                [
                    "review",
                    "--evidence",
                    str(evidence_path),
                    "--endpoint",
                    "https://example.test",
                    "--deployment",
                    "deployment",
                    "--output",
                    str(root / "review.json"),
                ]
            )
        except SystemExit as error:
            assert error.code == EXPECTED_ERROR_EXIT_CODE
        else:
            raise AssertionError("Expected --expected-model to be required")


def test_workflows_stage_findings_and_require_human_attestation() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    attestation = ATTESTATION_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "WIKI_AI_REVIEW_ENABLED" in workflow
    assert "--require-complete-inventory" in workflow
    assert "--expected-model" in workflow
    assert "evaluate" in workflow
    assert "apply" not in workflow
    assert "docs/wiki-review-proposal.md" in workflow
    assert "--draft" in workflow
    assert "contents: read" in workflow
    assert "contents: write" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "actions/download-artifact@" in workflow
    assert "pull_request_target:" in attestation
    assert "pull_request_review:" in attestation
    assert "verify-approval" in attestation
    assert "pull-requests: read" in attestation
    assert "pull-requests: write" not in attestation
    for path in (WORKFLOW_PATH, ATTESTATION_WORKFLOW_PATH):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "uses:" in line:
                reference = line.split("@", 1)[1].split()[0]
                assert len(reference) == review_wiki.COMMIT_SHA_LENGTH
