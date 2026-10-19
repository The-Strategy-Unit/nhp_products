"""Local regression checks for the pipeline's full-results copy selection."""

import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import nhpy.utils
from nhpy.config import ExitCodes

# Pipeline dependencies load .env during import; keep these checks independent of it.
with patch.object(nhpy.utils, "_load_dotenv_file", return_value=(False, None)):
    from nhpy import (
        pipeline,
        run_full_results,
        table_storage,
    )


SCENARIO_PATH = "aggregated-model-results/v6.0/RXX/scenario-a/20261013_100000/"
COPY_PATH = "aggregated-model-results/v6.0/RXX/scenario-a/20261013_110000"


def _configure_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            SCENARIO_PATH,
            "--account-url",
            "https://example.blob.core.windows.net/",
            "--results-container",
            "results",
            "--data-container",
            "data",
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(pipeline, "ensure_venv", lambda: True)
    detailed_paths = []
    monkeypatch.setattr(
        pipeline, "initialise_connections_and_params", lambda path, *_args: path
    )
    monkeypatch.setattr(
        pipeline,
        "run_detailed_results",
        lambda context, **_kwargs: detailed_paths.append(context),
    )
    return detailed_paths


def test_second_run_reuses_recorded_copy(monkeypatch, tmp_path):
    detailed_paths = _configure_pipeline(monkeypatch, tmp_path)
    original = {"PartitionKey": "RXX", "RowKey": "original-id", "full_results_copy": None}
    copy = {
        "PartitionKey": "RXX",
        "RowKey": "copy-id",
        "app_version": "v6.0",
        "scenario": "scenario-a",
        "create_datetime": datetime(2026, 10, 13, 11, 0, tzinfo=timezone.utc),
        "original_datetime": "20261013_100000",
    }
    model_runs = []
    full_results_paths = set()

    def find_original(path, **kwargs):
        assert path == SCENARIO_PATH
        assert kwargs == {}  # Table metadata uses STORAGE_ACCOUNT, not the Blob CLI URL.
        return original

    def find_copy(**kwargs):
        assert kwargs == {"partition_key": "RXX", "row_key": "copy-id"}
        return copy

    monkeypatch.setattr(pipeline, "find_entity_by_path", find_original)
    monkeypatch.setattr(pipeline, "get_entity", find_copy)
    monkeypatch.setattr(
        pipeline,
        "check_full_results",
        lambda path, **_kw: path.rstrip("/") in full_results_paths,
        raising=False,
    )

    def run_copy(**_kwargs):
        model_runs.append("copy-id")
        full_results_paths.add(COPY_PATH)
        return {"aggregated_results_path": COPY_PATH, "model_run_id": "copy-id"}

    def record_copy(**kwargs):
        assert kwargs["partition_key"] == original["PartitionKey"]
        assert kwargs["row_key"] == original["RowKey"]
        assert "account_url" not in kwargs
        original["full_results_copy"] = kwargs["copy_row_key"]

    monkeypatch.setattr(pipeline, "run_scenario_with_full_results", run_copy)
    monkeypatch.setattr(pipeline, "set_full_results_copy", record_copy)

    assert pipeline.main() == ExitCodes.SUCCESS_CODE
    assert original["full_results_copy"] == "copy-id"
    assert pipeline.main() == ExitCodes.SUCCESS_CODE
    assert model_runs == ["copy-id"]
    assert detailed_paths == [COPY_PATH, COPY_PATH]


def test_existing_original_full_results_are_used(monkeypatch, tmp_path):
    detailed_paths = _configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline, "check_full_results", lambda *_args, **_kw: True, raising=False
    )
    monkeypatch.setattr(
        pipeline,
        "find_entity_by_path",
        lambda *_args, **_kw: (_ for _ in ()).throw(
            AssertionError("unexpected Table Storage lookup")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_scenario_with_full_results",
        lambda **_kw: (_ for _ in ()).throw(AssertionError("unexpected model run")),
    )

    assert pipeline.main() == ExitCodes.SUCCESS_CODE
    assert detailed_paths == [SCENARIO_PATH]


def test_missing_original_metadata_prevents_new_model_run(monkeypatch, tmp_path):
    _configure_pipeline(monkeypatch, tmp_path)
    model_runs = []
    monkeypatch.setattr(
        pipeline, "check_full_results", lambda *_args, **_kw: False, raising=False
    )
    monkeypatch.setattr(pipeline, "find_entity_by_path", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        pipeline, "run_scenario_with_full_results", lambda **_kw: model_runs.append(True)
    )

    assert pipeline.main() == ExitCodes.EXCEPTION_CODE
    assert model_runs == []


def test_unusable_recorded_copy_does_not_create_another(monkeypatch, tmp_path):
    _configure_pipeline(monkeypatch, tmp_path)
    original = {
        "PartitionKey": "RXX",
        "RowKey": "original-id",
        "full_results_copy": "copy-id",
    }
    model_runs = []
    monkeypatch.setattr(
        pipeline, "check_full_results", lambda *_args, **_kw: False, raising=False
    )
    monkeypatch.setattr(pipeline, "find_entity_by_path", lambda *_args, **_kw: original)
    monkeypatch.setattr(pipeline, "get_entity", lambda *_args, **_kw: None)
    monkeypatch.setattr(pipeline, "set_full_results_copy", lambda **_kw: None)
    monkeypatch.setattr(
        pipeline, "run_scenario_with_full_results", lambda **_kw: model_runs.append(True)
    )

    assert pipeline.main() == ExitCodes.EXCEPTION_CODE
    assert model_runs == []


def test_copy_without_full_results_is_not_recorded(monkeypatch, tmp_path):
    _configure_pipeline(monkeypatch, tmp_path)
    original = {"PartitionKey": "RXX", "RowKey": "original-id", "full_results_copy": None}
    recorded = []
    monkeypatch.setattr(
        pipeline, "check_full_results", lambda *_args, **_kw: False, raising=False
    )
    monkeypatch.setattr(pipeline, "find_entity_by_path", lambda *_args, **_kw: original)
    monkeypatch.setattr(
        pipeline,
        "run_scenario_with_full_results",
        lambda **_kw: {"aggregated_results_path": COPY_PATH, "model_run_id": "copy-id"},
    )
    monkeypatch.setattr(
        pipeline, "set_full_results_copy", lambda **_kw: recorded.append(True)
    )

    assert pipeline.main() == ExitCodes.EXCEPTION_CODE
    assert recorded == []


def test_terminated_copy_run_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(run_full_results.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        run_full_results,
        "get_model_run_status",
        lambda *_args: {"state": "Terminated", "detail_status": "failed"},
    )

    with pytest.raises(RuntimeError, match="terminated"):
        run_full_results._track_container_status(
            {"dataset": "RXX", "model_run_id": "copy-id"}
        )


def test_copy_id_is_written_to_original_metadata_row(monkeypatch):
    updated = []

    class FakeTableClient:
        def get_entity(self, partition_key, row_key):
            assert (partition_key, row_key) == ("RXX", "original-id")
            return {"PartitionKey": partition_key, "RowKey": row_key}

        def update_entity(self, entity):
            updated.append(dict(entity))

    monkeypatch.setattr(
        table_storage, "get_table_client", lambda **_kwargs: FakeTableClient()
    )

    table_storage.set_full_results_copy(
        "RXX",
        "original-id",
        "copy-id",
    )

    assert updated == [
        {
            "PartitionKey": "RXX",
            "RowKey": "original-id",
            "full_results_copy": "copy-id",
        }
    ]
