#!/usr/bin/env python

"""Verify a signed, public-safe Wiki projection and emit review evidence.

This command accepts only a dedicated publication projection. It must never be
pointed at an original or otherwise confidential SharePoint document.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SCHEMA_VERSION = 2
ATTESTATION_SCHEMA_VERSION = 1
MAX_PROJECTION_BYTES = 80_000
MAX_APPROVAL_LIFETIME = timedelta(days=31)
OPAQUE_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:password|secret|token|api[_ -]?key)\s*[:=]", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b"),
)


class ProjectionError(ValueError):
    """Raised when a publication projection is not safe to consume."""


@dataclass(frozen=True)
class ProjectionRequest:
    """Files and pinned trust root for one publication projection."""

    projection_path: Path
    attestation_path: Path
    signature_path: Path
    public_key_path: Path
    expected_key_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionError("Could not read the publication attestation") from error
    if not isinstance(value, dict):
        raise ProjectionError("The publication attestation is not a JSON object")
    return cast(dict[str, Any], value)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ProjectionError(f"Approval attestation has an invalid {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProjectionError(f"Approval attestation has an invalid {field}") from error
    if parsed.tzinfo is None:
        raise ProjectionError(f"Approval attestation has an invalid {field}")
    return parsed.astimezone(UTC)


def _public_key(path: Path, expected_sha256: str) -> tuple[Ed25519PublicKey, str]:
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ProjectionError("The trusted approval-key fingerprint is invalid")
    try:
        loaded = serialization.load_pem_public_key(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ProjectionError("Could not load the trusted approval public key") from error
    if not isinstance(loaded, Ed25519PublicKey):
        raise ProjectionError("The approval key must be Ed25519")
    raw_key = loaded.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fingerprint = hashlib.sha256(raw_key).hexdigest()
    if fingerprint != expected_sha256:
        raise ProjectionError("The approval key does not match the trusted fingerprint")
    return loaded, fingerprint


def _validate_attestation(
    attestation: Mapping[str, Any], projection: bytes, now: datetime
) -> None:
    expected_fields = {
        "schema_version",
        "source_kind",
        "source_id",
        "previous_revision",
        "current_revision",
        "pages",
        "projection_sha256",
        "classification",
        "approval_policy",
        "approval_id",
        "dlp_status",
        "dlp_scan_id",
        "approved_at",
        "expires_at",
    }
    pages = attestation.get("pages")
    if (
        set(attestation) != expected_fields
        or attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or attestation.get("source_kind") != "sharepoint_publication_projection"
        or not isinstance(attestation.get("source_id"), str)
        or not OPAQUE_ID_PATTERN.fullmatch(cast(str, attestation["source_id"]))
        or not isinstance(attestation.get("previous_revision"), str)
        or not OPAQUE_ID_PATTERN.fullmatch(cast(str, attestation["previous_revision"]))
        or not isinstance(attestation.get("current_revision"), str)
        or not OPAQUE_ID_PATTERN.fullmatch(cast(str, attestation["current_revision"]))
        or not isinstance(pages, list)
        or not pages
        or not all(
            isinstance(page, str)
            and Path(page).name == page
            and Path(page).suffix.lower() == ".md"
            for page in pages
        )
        or len(set(cast(list[str], pages))) != len(pages)
        or attestation.get("projection_sha256") != hashlib.sha256(projection).hexdigest()
        or attestation.get("classification") != "approved-for-public-wiki"
        or attestation.get("approval_policy") != "public-wiki-projection-v1"
        or not isinstance(attestation.get("approval_id"), str)
        or not OPAQUE_ID_PATTERN.fullmatch(cast(str, attestation["approval_id"]))
        or attestation.get("dlp_status") != "passed"
        or not isinstance(attestation.get("dlp_scan_id"), str)
        or not OPAQUE_ID_PATTERN.fullmatch(cast(str, attestation["dlp_scan_id"]))
    ):
        raise ProjectionError("The signed publication attestation is invalid")
    approved_at = _timestamp(attestation.get("approved_at"), "approved_at")
    expires_at = _timestamp(attestation.get("expires_at"), "expires_at")
    if (
        approved_at > now
        or expires_at <= now
        or expires_at <= approved_at
        or expires_at - approved_at > MAX_APPROVAL_LIFETIME
    ):
        raise ProjectionError("The publication approval is not currently valid")


def collect_projection(
    request: ProjectionRequest,
    wiki_directory: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify an exact public projection and build an auditable evidence bundle."""
    try:
        projection = request.projection_path.read_bytes()
    except OSError as error:
        raise ProjectionError("Could not read the publication projection") from error
    if not projection or len(projection) > MAX_PROJECTION_BYTES:
        raise ProjectionError("The publication projection is empty or oversized")
    try:
        projection_text = projection.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProjectionError("The publication projection is not UTF-8 text") from error
    if any(pattern.search(projection_text) for pattern in SENSITIVE_PATTERNS):
        raise ProjectionError("The publication projection failed defence-in-depth checks")

    attestation = _load_json(request.attestation_path)
    _validate_attestation(attestation, projection, now or datetime.now(UTC))
    public_key, key_fingerprint = _public_key(
        request.public_key_path, request.expected_key_sha256
    )
    try:
        signature = base64.b64decode(
            request.signature_path.read_text(encoding="ascii").strip(), validate=True
        )
        public_key.verify(signature, _canonical_json(attestation))
    except (OSError, UnicodeDecodeError, binascii.Error, InvalidSignature) as error:
        raise ProjectionError(
            "Publication approval signature verification failed"
        ) from error

    pages = cast(list[str], attestation["pages"])
    wiki_pages: dict[str, str] = {}
    for page in pages:
        try:
            wiki_pages[page] = (wiki_directory / page).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ProjectionError(f"Could not read mapped Wiki page: {page}") from error
    source = {
        "source_kind": "approved_publication_projection",
        "source_id": attestation["source_id"],
        "handling": {
            "classification": "approved-for-public-wiki",
            "policy": "signed-publication-projection-v1",
            "public_report": True,
            "projection_sha256": attestation["projection_sha256"],
            "attestation_sha256": hashlib.sha256(
                _canonical_json(attestation)
            ).hexdigest(),
            "approval_key_sha256": key_fingerprint,
        },
        "previous_revision": attestation["previous_revision"],
        "current_revision": attestation["current_revision"],
        "allowed_pages": pages,
        "pages": wiki_pages,
        "changed_paths": [],
        "fragments": [
            {
                "locator": "publication-projection#1",
                "kind": "publication_projection",
                "path": "",
                "revision": attestation["current_revision"],
                "line_start": 1,
                "citable": True,
                "content": projection_text,
            }
        ],
        "complete": True,
        "blockers": [],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "review_mode": "publication_projection",
        "inventory_gaps": {},
        "inventory_exclusions": {},
        "sources": [source],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--approval-public-key", type=Path, required=True)
    parser.add_argument("--expected-key-sha256", required=True)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = collect_projection(
            ProjectionRequest(
                projection_path=args.projection,
                attestation_path=args.attestation,
                signature_path=args.signature,
                public_key_path=args.approval_public_key,
                expected_key_sha256=args.expected_key_sha256,
            ),
            args.wiki,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("Verified one signed public Wiki projection.")
        return 0
    except ProjectionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
