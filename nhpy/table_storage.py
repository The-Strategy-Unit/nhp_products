"""
Azure Table Storage utilities for NHP model run metadata.

This module provides functions to query and update entities in Azure Table Storage,
specifically for tracking full model results copies. When a full results copy is
created for a scenario, the copy's model_run_id (RowKey) is stored in the
original scenario's entity under the 'full_results_copy' column. This prevents
duplicate copy runs on subsequent pipeline executions.

The table is keyed by PartitionKey=dataset and RowKey=model_run_id. Entities also
contain an 'aggregated_results_path' column, which is the reliable way to match a
scenario file path to its table entity.

Usage:
    # Programmatic usage
    from nhpy.table_storage import (
        find_entity_by_path,
        get_full_results_copy,
        set_full_results_copy,
    )

    # Check if a copy exists for a scenario path
    copy_id = get_full_results_copy(
        "aggregated-model-results/dev/RXX/test/20260518_110000"
    )

    # Record a copy after it finishes
    set_full_results_copy("RXX", "c570a39a-...", "e9277ffb-...")

Configuration:
    Set environment variables: AZ_STORAGE_EP, AZ_TABLE_NAME
    Or provide credentials via function arguments.
"""

# %%
import os

from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableClient
from azure.identity import DefaultAzureCredential

from nhpy.config import DATASET_PATH_PARTS
from nhpy.utils import get_logger

# %%
# Define public API
__all__ = [
    "get_table_client",
    "get_entity",
    "find_entity_by_path",
    "get_full_results_copy",
    "set_full_results_copy",
]

# %%
# Get a logger for this module
logger = get_logger()


# %%
def _derive_table_endpoint(blob_endpoint: str) -> str:
    """
    Derive the Table Storage endpoint from a Blob Storage endpoint.

    Args:
        blob_endpoint: Blob Storage account URL
            (e.g. 'https://myaccount.blob.core.windows.net/')

    Returns:
        str: Table Storage account URL
            (e.g. 'https://myaccount.table.core.windows.net/')

    >>> _derive_table_endpoint("https://myaccount.blob.core.windows.net/")
    'https://myaccount.table.core.windows.net/'
    >>> _derive_table_endpoint("https://myaccount.blob.core.windows.net")
    'https://myaccount.table.core.windows.net'
    """
    return blob_endpoint.replace("blob", "table")


# %%
def get_table_client(
    table_name: str | None = None,
    account_url: str | None = None,
) -> TableClient:
    """
    Connect to an Azure Table Storage table.

    Uses DefaultAzureCredential for authentication, consistent with az.py.
    The table endpoint is derived from the blob endpoint (AZ_STORAGE_EP) by
    replacing 'blob' with 'table'.

    Args:
        table_name: Name of the Azure Table (default: AZ_TABLE_NAME env var)
        account_url: Blob endpoint URL to derive table endpoint from
            (default: AZ_STORAGE_EP env var)

    Returns:
        TableClient: Client for interacting with the Azure Table

    Raises:
        ValueError: If table_name or account_url cannot be determined

    >>> client = get_table_client(
    ...     table_name="modelruns",
    ...     account_url="https://myaccount.blob.core.windows.net/",
    ... )  # doctest: +SKIP
    >>> type(client).__name__  # doctest: +SKIP
    'TableClient'

    Raises ValueError when table_name is missing:

    >>> get_table_client(account_url="https://x.blob.core.windows.net/")
    Traceback (most recent call last):
        ...
    ValueError: Table name required. Provide as argument or set AZ_TABLE_NAME env var.

    Raises ValueError when account_url is missing:

    >>> get_table_client(table_name="modelruns")
    Traceback (most recent call last):
        ...
    ValueError: Account URL required. Provide as argument or set AZ_STORAGE_EP env var.
    """
    table_name = table_name or os.environ.get("AZ_TABLE_NAME")
    if not table_name:
        raise ValueError(
            "Table name required. Provide as argument or set AZ_TABLE_NAME env var."
        )

    blob_endpoint = account_url or os.environ.get("AZ_STORAGE_EP", "")
    if not blob_endpoint:
        raise ValueError(
            "Account URL required. Provide as argument or set AZ_STORAGE_EP env var."
        )

    table_endpoint = _derive_table_endpoint(blob_endpoint)

    credential = DefaultAzureCredential()
    table_client = TableClient(
        endpoint=table_endpoint,
        table_name=table_name,
        credential=credential,
    )

    return table_client


# %%
def get_entity(
    partition_key: str,
    row_key: str,
    table_name: str | None = None,
    account_url: str | None = None,
) -> dict | None:
    """
    Retrieve an entity from Azure Table Storage by partition and row key.

    Args:
        partition_key: PartitionKey of the entity (dataset)
        row_key: RowKey of the entity (model_run_id UUID)
        table_name: Name of the Azure Table (default: AZ_TABLE_NAME env var)
        account_url: Blob endpoint URL to derive table endpoint from
            (default: AZ_STORAGE_EP env var)

    Returns:
        dict | None: The entity as a dictionary, or None if not found

    >>> entity = get_entity("RXX", "c570a39a-...")  # doctest: +SKIP
    >>> entity["scenario"]  # doctest: +SKIP
    'test-v5-24-41'

    Returns None when entity is not found:

    >>> get_entity("RXX", "missing-id")  # doctest: +SKIP
    None
    """
    try:
        table_client = get_table_client(table_name=table_name, account_url=account_url)
        entity = table_client.get_entity(partition_key=partition_key, row_key=row_key)
        return dict(entity)
    except ResourceNotFoundError:
        logger.debug(f"Entity not found: PartitionKey={partition_key}, RowKey={row_key}")
        return None


# %%
def find_entity_by_path(
    aggregated_results_path: str,
    table_name: str | None = None,
    account_url: str | None = None,
) -> dict | None:
    """
    Find the table entity matching a given aggregated_results_path.

    Queries the table with a filter on PartitionKey (derived from the path)
    and the aggregated_results_path column.

    Args:
        aggregated_results_path: Path to the scenario results
            (e.g. 'aggregated-model-results/dev/RXX/test/20260518_110000')
        table_name: Name of the Azure Table (default: AZ_TABLE_NAME env var)
        account_url: Blob endpoint URL to derive table endpoint from
            (default: AZ_STORAGE_EP env var)

    Returns:
        dict | None: The matching entity, or None if not found

    >>> entity = find_entity_by_path(  # doctest: +SKIP
    ...     "aggregated-model-results/dev/RXX/test/20260518_110000")
    >>> entity["RowKey"]  # doctest: +SKIP
    'c570a39a-...'

    Returns None when no entity matches:

    >>> find_entity_by_path(  # doctest: +SKIP
    ...     "aggregated-model-results/dev/RXX/nonexistent/20260518_110000")
    """
    table_client = get_table_client(table_name=table_name, account_url=account_url)

    # Normalise path to match stored format (no trailing slash)
    path = aggregated_results_path.rstrip("/")

    # Extract dataset from path: aggregated-model-results/{version}/{dataset}/...
    parts = path.split("/")
    if len(parts) < DATASET_PATH_PARTS:
        raise ValueError(f"Cannot extract dataset from path: {aggregated_results_path}")

    partition_key = parts[2]

    filter_query = (
        f"PartitionKey eq '{partition_key}' and aggregated_results_path eq '{path}'"
    )

    entities = list(table_client.query_entities(filter_query))
    if not entities:
        logger.debug(f"No entity found for path: {path}")
        return None

    if len(entities) > 1:
        logger.warning(f"Multiple entities found for path {path}, using the first one.")

    return dict(entities[0])


# %%
def get_full_results_copy(
    aggregated_results_path: str,
    table_name: str | None = None,
    account_url: str | None = None,
) -> str | None:
    """
    Get the full_results_copy value for a scenario entity.

    This checks whether a full model results copy has already been created
    for the given scenario path.

    Args:
        aggregated_results_path: Path to the scenario results
            (e.g. 'aggregated-model-results/dev/RXX/test/20260518_110000')
        table_name: Name of the Azure Table (default: AZ_TABLE_NAME env var)
        account_url: Blob endpoint URL to derive table endpoint from
            (default: AZ_STORAGE_EP env var)

    Returns:
        str | None: The copy's model_run_id (RowKey), or None if not set

    >>> # Returns the copy's RowKey when a copy exists
    >>> get_full_results_copy(  # doctest: +SKIP
    ...     "aggregated-model-results/dev/RXX/test/20260518_110000")
    'e9277ffb-...'

    Returns None when no copy has been recorded:

    >>> get_full_results_copy(  # doctest: +SKIP
    ...     "aggregated-model-results/dev/RXX/test/20260518_110000")
    """
    entity = find_entity_by_path(
        aggregated_results_path=aggregated_results_path,
        table_name=table_name,
        account_url=account_url,
    )
    if entity is None:
        return None

    return entity.get("full_results_copy") or None


# %%
def set_full_results_copy(
    partition_key: str,
    row_key: str,
    copy_row_key: str,
    table_name: str | None = None,
    account_url: str | None = None,
) -> None:
    """
    Update a scenario entity with the full_results_copy value.

    After a full results copy scenario finishes running, this records the
    copy's model_run_id (RowKey) in the original scenario's entity.

    Args:
        partition_key: PartitionKey of the original entity (dataset)
        row_key: RowKey of the original entity (model_run_id UUID)
        copy_row_key: RowKey (model_run_id UUID) of the full results copy
        table_name: Name of the Azure Table (default: AZ_TABLE_NAME env var)
        account_url: Blob endpoint URL to derive table endpoint from
            (default: AZ_STORAGE_EP env var)

    Raises:
        ValueError: If the original entity doesn't exist

    >>> set_full_results_copy(  # doctest: +SKIP
    ...     "RXX", "c570a39a-...", "e9277ffb-...")

    Raises ValueError when the entity doesn't exist:

    >>> set_full_results_copy("RXX", "missing", "def-456")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Entity not found: PartitionKey=RXX, RowKey=missing. ...
    """
    table_client = get_table_client(table_name=table_name, account_url=account_url)

    try:
        entity = table_client.get_entity(partition_key=partition_key, row_key=row_key)
    except ResourceNotFoundError:
        raise ValueError(
            f"Entity not found: PartitionKey={partition_key}, "
            f"RowKey={row_key}. Cannot set full_results_copy "
            "on a non-existent entity."
        )

    entity["full_results_copy"] = copy_row_key
    table_client.update_entity(entity)

    logger.info(
        f"Updated full_results_copy for PartitionKey={partition_key}, "
        f"RowKey={row_key} -> {copy_row_key}"
    )
