#!/usr/bin/env python

"""Offline tests for the signed SharePoint publication-projection boundary."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "collect_wiki_projection.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("collect_wiki_projection", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projection = _load_module()


class ProjectionRequest(Protocol):
    """Typed view of the dynamically imported projection request."""

    projection_path: Path
    attestation_path: Path
    signature_path: Path
    public_key_path: Path
    expected_key_sha256: str


def _fixture(root: Path) -> tuple[ProjectionRequest, Path, datetime]:
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    wiki = root / "wiki"
    wiki.mkdir()
    (wiki / "Home.md").write_text("# Existing public Wiki\n", encoding="utf-8")
    projection_path = root / "public-projection.md"
    projection_bytes = b"The approved public service name is OpenPlan.\n"
    projection_path.write_bytes(projection_bytes)
    attestation = {
        "schema_version": 1,
        "source_kind": "sharepoint_publication_projection",
        "source_id": "sharepoint-projection-1",
        "previous_revision": "version-1",
        "current_revision": "version-2",
        "pages": ["Home.md"],
        "projection_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        "classification": "approved-for-public-wiki",
        "approval_policy": "public-wiki-projection-v1",
        "approval_id": "approval-123",
        "dlp_status": "passed",
        "dlp_scan_id": "dlp-123",
        "approved_at": now.isoformat(),
        "expires_at": (now + timedelta(days=7)).isoformat(),
    }
    attestation_path = root / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    key_path = root / "approval-key.pem"
    key_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    raw_key = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_sha256 = hashlib.sha256(raw_key).hexdigest()
    signature_path = root / "attestation.sig"
    signature_path.write_text(
        base64.b64encode(
            private_key.sign(projection._canonical_json(attestation))
        ).decode("ascii"),
        encoding="ascii",
    )
    request = projection.ProjectionRequest(
        projection_path=projection_path,
        attestation_path=attestation_path,
        signature_path=signature_path,
        public_key_path=key_path,
        expected_key_sha256=key_sha256,
    )
    return request, wiki, now


def test_collect_projection_verifies_exact_public_content() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        request, wiki, now = _fixture(Path(temporary_directory))
        evidence = projection.collect_projection(request, wiki, now=now)

    source = evidence["sources"][0]
    assert source["source_kind"] == "approved_publication_projection"
    assert source["handling"]["classification"] == "approved-for-public-wiki"
    assert source["fragments"][0]["content"] == (
        "The approved public service name is OpenPlan.\n"
    )
    serialized = json.dumps(evidence)
    assert "public-projection.md" not in serialized
    assert "attestation.sig" not in serialized


def test_collect_projection_rejects_content_changed_after_approval() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        request, wiki, now = _fixture(Path(temporary_directory))
        request.projection_path.write_text("Unapproved replacement.\n", encoding="utf-8")
        try:
            projection.collect_projection(request, wiki, now=now)
        except projection.ProjectionError as error:
            assert "attestation is invalid" in str(error)
        else:
            raise AssertionError("Expected modified projection content to be rejected")


def test_collect_projection_rejects_untrusted_approval_key() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        request, wiki, now = _fixture(Path(temporary_directory))
        request = projection.ProjectionRequest(
            projection_path=request.projection_path,
            attestation_path=request.attestation_path,
            signature_path=request.signature_path,
            public_key_path=request.public_key_path,
            expected_key_sha256="0" * 64,
        )
        try:
            projection.collect_projection(request, wiki, now=now)
        except projection.ProjectionError as error:
            assert "trusted fingerprint" in str(error)
        else:
            raise AssertionError("Expected an untrusted approval key to be rejected")


def test_collect_projection_rejects_obvious_sensitive_content_before_model_use() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        request, wiki, now = _fixture(Path(temporary_directory))
        request.projection_path.write_text("password=not-public\n", encoding="utf-8")
        try:
            projection.collect_projection(request, wiki, now=now)
        except projection.ProjectionError as error:
            assert "defence-in-depth" in str(error)
        else:
            raise AssertionError("Expected obvious sensitive content to be rejected")
