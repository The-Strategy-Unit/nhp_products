"""Offline checks for Azure Table Storage lookup of model-run metadata."""

from datetime import datetime, timezone

import pytest
from azure.data.tables._serialize import _parameter_filter_substitution

from nhpy import table_storage


def test_table_client_uses_model_run_storage_account(monkeypatch):
    clients = []
    monkeypatch.setenv("STORAGE_ACCOUNT", "modelaccount")
    monkeypatch.setenv("AZ_STORAGE_EP", "https://resultsaccount.blob.core.windows.net")
    monkeypatch.setattr(table_storage, "DefaultAzureCredential", object)
    monkeypatch.setattr(
        table_storage,
        "TableClient",
        lambda **kwargs: clients.append((kwargs["endpoint"], kwargs["table_name"])),
    )

    table_storage.get_table_client()

    assert clients == [("https://modelaccount.table.core.windows.net", "modelruns")]


def test_table_client_requires_model_run_storage_account(monkeypatch):
    monkeypatch.delenv("STORAGE_ACCOUNT", raising=False)
    monkeypatch.setenv("AZ_STORAGE_EP", "https://resultsaccount.blob.core.windows.net")

    with pytest.raises(ValueError, match="STORAGE_ACCOUNT"):
        table_storage.get_table_client()


def test_find_entity_by_model_run_metadata(monkeypatch):
    path = "aggregated-model-results/v6.0/RXX/scenario-a/20261013_100000"
    entity = {
        "PartitionKey": "RXX",
        "RowKey": "original-id",
        "app_version": "v6.0",
        "scenario": "scenario-a",
        "create_datetime": datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc),
    }

    class FakeTableClient:
        def query_entities(self, query_filter, **kwargs):
            expanded = _parameter_filter_substitution(kwargs["parameters"], query_filter)
            assert "PartitionKey eq 'RXX'" in expanded
            assert "app_version eq 'v6.0'" in expanded
            assert "scenario eq 'scenario-a'" in expanded
            assert "create_datetime eq datetime'2026-10-13T10:00:00.000000Z'" in expanded
            return [entity]

    monkeypatch.setattr(
        table_storage, "get_table_client", lambda **_kwargs: FakeTableClient()
    )

    assert table_storage.find_entity_by_path(path) == entity
