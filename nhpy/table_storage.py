"""Azure Table Storage access for tracking full-results copies of model runs."""

import os
from datetime import datetime, timezone

from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableClient
from azure.identity import DefaultAzureCredential

from nhpy.run_full_results import _extract_scenario_components
from nhpy.utils import _construct_results_path, get_logger

__all__ = [
    "get_table_client",
    "get_entity",
    "find_entity_by_path",
    "aggregated_results_path_from_entity",
    "set_full_results_copy",
]

logger = get_logger()


def get_table_client() -> TableClient:
    """Connect to model-run metadata using the model client's storage account."""
    storage_account = os.environ.get("STORAGE_ACCOUNT")
    if not storage_account:
        raise ValueError("STORAGE_ACCOUNT is required for model-run Table Storage")

    return TableClient(
        endpoint=f"https://{storage_account}.table.core.windows.net",
        table_name="modelruns",
        credential=DefaultAzureCredential(),
    )


def get_entity(
    partition_key: str,
    row_key: str,
) -> dict | None:
    """Look up a model run by dataset and model-run ID."""
    table_client = get_table_client()
    try:
        return dict(table_client.get_entity(partition_key=partition_key, row_key=row_key))
    except ResourceNotFoundError:
        return None


def find_entity_by_path(
    aggregated_results_path: str,
) -> dict | None:
    """Find the model-run row for an aggregated results directory."""
    version, partition_key, scenario, created = _extract_scenario_components(
        aggregated_results_path
    )
    create_datetime = datetime.strptime(created, "%Y%m%d_%H%M%S").replace(
        tzinfo=timezone.utc
    )
    table_client = get_table_client()

    entities = list(
        table_client.query_entities(
            query_filter=(
                "PartitionKey eq @partition_key and "
                "app_version eq @version and scenario eq @scenario and "
                "create_datetime eq @created"
            ),
            parameters={
                "partition_key": partition_key,
                "version": version,
                "scenario": scenario,
                "created": create_datetime,
            },
        )
    )
    if not entities:
        return None
    if len(entities) > 1:
        logger.warning(
            "Multiple model-run rows found for %s; using the first",
            aggregated_results_path,
        )
    return dict(entities[0])


def aggregated_results_path_from_entity(entity: dict) -> str:
    """Build a results path from fields written to the model-run metadata row."""
    created = entity.get("create_datetime")
    if not isinstance(created, datetime):
        raise ValueError("Model-run metadata has no creation timestamp")
    params = {
        "app_version": entity.get("app_version"),
        "dataset": entity.get("PartitionKey"),
        "scenario": entity.get("scenario"),
        "create_datetime": created.strftime("%Y%m%d_%H%M%S"),
        "original_datetime": entity.get("original_datetime", ""),
    }
    if any(
        not isinstance(params[key], str) or not params[key]
        for key in ("app_version", "dataset", "scenario", "create_datetime")
    ):
        raise ValueError("Model-run metadata is missing results path components")
    return _construct_results_path(params)["aggregated_results_path"]


def set_full_results_copy(
    partition_key: str,
    row_key: str,
    copy_row_key: str,
) -> None:
    """Record a completed copy's model-run ID on the original scenario row."""
    table_client = get_table_client()
    try:
        entity = table_client.get_entity(partition_key=partition_key, row_key=row_key)
    except ResourceNotFoundError as exc:
        raise ValueError(
            f"Entity not found: PartitionKey={partition_key}, RowKey={row_key}. "
            "Cannot set full_results_copy on a non-existent entity."
        ) from exc

    entity["full_results_copy"] = copy_row_key
    table_client.update_entity(entity)
    logger.info("Recorded full-results copy %s for model run %s", copy_row_key, row_key)
