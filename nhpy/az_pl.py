"""
Azure Blob Storage utilities for NHP model data - Polars implementation.

This module provides functions to connect to Azure Blob Storage containers and load
various types of data files related to NHP modelling, including JSON results, parquet
files, and aggregated model results. It handles version management and supports both
current and legacy data structures.

This module reuses non-Pandas functions from az.py and reimplements only the
Pandas-specific functions to use Polars instead.
"""

# Imports
import io
import logging
import time

import polars as pl
from azure.core.exceptions import (
    AzureError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.storage.blob import ContainerClient

from nhpy.types import ModelRunParams

# Configure module logger
logger = logging.getLogger(__name__)

# cache for Polars DataFrames
_model_results_cache: dict[str, dict[int, pl.DataFrame]] = {}


def load_parquet_file(
    container_client: ContainerClient,
    path_to_file: str,
    max_retries: int = 3,
    timeout: int = 120,
) -> pl.DataFrame:
    """Loads parquet file from Azure

    Args:
        container_client: Connection to the container
        path_to_file: Path to the file to be loaded
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Timeout in seconds for the download operation (default: 120)

    Returns:
        pl.DataFrame: DataFrame of the data

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file after all retries
    """
    retry_count = 0
    last_exception = None

    while retry_count < max_retries:
        try:
            blob_client = container_client.get_blob_client(path_to_file)
            # Set timeout for the download operation
            download_stream = blob_client.download_blob(timeout=timeout)
            stream_object = io.BytesIO(download_stream.readall())
            # Use Polars to read the parquet file directly
            data = pl.read_parquet(stream_object)

            return data
        except ResourceNotFoundError:
            # Don't retry for files that don't exist
            raise
        except ValueError as e:
            # Don't retry for invalid parquet files
            if "parquet" in str(e).lower():
                raise ValueError(f"Invalid parquet file: {e}") from e
            raise
        except (ServiceResponseError, ServiceRequestError, TimeoutError, AzureError) as e:
            last_exception = e
            retry_count += 1
            if retry_count < max_retries:
                # Exponential backoff: wait longer with each retry
                wait_time = 2**retry_count
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s. Retrying in %d seconds...",
                    retry_count,
                    max_retries,
                    path_to_file,
                    str(e),
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "Failed to download %s after %d attempts: %s",
                    path_to_file,
                    max_retries,
                    str(e),
                )

    # If we've exhausted all retries, raise the last exception
    if last_exception:
        raise AzureError(
            f"Failed to download blob after {max_retries} attempts: {str(last_exception)}"
        ) from last_exception

    # This should never happen, but just in case
    raise AzureError(
        f"Failed to download or process {path_to_file} with an unknown error"
    )


def load_data_file(
    container_client: ContainerClient,
    version: str,
    dataset: str,
    activity_type: str,
    year: int = 2023,
) -> pl.DataFrame:
    """Loads Parquet file from Azure containing NHP model data. Only works for >= v3.0.0

    Args:
        container_client: Connection to container with data files
        version: Version of the dataset to be used
        dataset: Name of the trust/dataset
        activity_type: Type of activity - options include ip, op, aae.
        year: Year for the data. Defaults to 2023.

    Returns:
        pl.DataFrame: DataFrame of the data

    Raises:
        FileNotFoundError: If no matching parquet file is found
        ResourceNotFoundError: If the found file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue accessing the container or downloading the file
    """
    try:
        blob_name = next(
            (
                b
                for b in container_client.list_blob_names(
                    name_starts_with=f"{version}/{activity_type}/fyear={year}/dataset={dataset}"
                )
                if b.endswith(".parquet")
            ),
            None,
        )

        if not blob_name:
            raise FileNotFoundError(
                f"No parquet files found for version={version}, dataset={dataset}, "
                f"activity_type={activity_type}, year={year}"
            )

        data = load_parquet_file(container_client, blob_name)

        return data
    except (FileNotFoundError, AzureError):
        raise


def load_data_file_old(
    container_client: ContainerClient,
    version: str,
    dataset: str,
    activity_type: str,
    year: int = 2019,
) -> pl.DataFrame:
    """Loads Parquet file from Azure containing NHP model data.
    Only works for model versions prior to v3.0

    Args:
        container_client: Connection to container with data files
        version: Version of the dataset to be used
        dataset: Name of the trust/dataset
        activity_type: Type of activity - options include ip, op, aae.
        year: Year for the data. Defaults to 2019.

    Returns:
        pl.DataFrame: DataFrame of the data

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
    blob_name = f"{version}/{year}/{dataset}/{activity_type}.parquet"
    data = load_parquet_file(container_client, blob_name)

    return data


def load_model_run_results_file(
    container_client: ContainerClient,
    params: ModelRunParams,
) -> pl.DataFrame:
    """Loads full model results from a specific run from Azure.
    Requires for the run to have had "--save-full-model-results" enabled.
    Can optionally load files in batches for efficiency.

    Args:
        container_client: Connection to container with data files
        params: Dictionary containing parameters:
            - version: Version of the dataset to be used
            - dataset: Name of the trust/dataset
            - scenario_name: Name of the scenario
            - run_id: ID of the specific model run of that scenario
            - activity_type: Type of activity - options include ip, op, aae.
            - run_number: Which of the Monte Carlo simulation runs it is
            - batch_size: If provided, loads files in batches of this size (internal
            caching)

    Returns:
        pl.DataFrame: DataFrame of the data for the requested run_number

    Raises:
        ResourceNotFoundError: If the requested file doesn't exist
        ValueError: If the requested file is not a valid parquet file
        AzureError: If there's an issue downloading the requested file
    """

    # Extract parameters
    version = params["version"]
    dataset = params["dataset"]
    scenario_name = params["scenario_name"]
    run_id = params["run_id"]
    activity_type = params["activity_type"]
    run_number = params["run_number"]
    batch_size = params.get("batch_size")

    path_components = [
        "full-model-results",
        version,
        dataset,
        scenario_name,
        run_id,
        activity_type,
        f"model_run={run_number}",
        "0.parquet",
    ]
    blob_name = "/".join(path_components)

    # If batch_size not provided, load single file directly
    if batch_size is None:
        return load_parquet_file(container_client, blob_name)

    # Use the module-level cache
    cache_key = f"{version}_{dataset}_{scenario_name}_{run_id}_{activity_type}"

    # Check if the requested run is in cache
    if (
        cache_key in _model_results_cache
        and run_number in _model_results_cache[cache_key]
    ):
        return _model_results_cache[cache_key][run_number]

    # Initialize cache for this key if needed
    if cache_key not in _model_results_cache:
        _model_results_cache[cache_key] = {}

    # Calculate batch range (centered on requested run_number if possible)
    half_batch = batch_size // 2 if batch_size is not None else 0
    batch_start = max(1, run_number - half_batch)
    batch_end = batch_start + (batch_size if batch_size is not None else 1)

    # Track if we need to separately load the requested run
    requested_run_loaded = False

    # Load batch of files
    for run in range(batch_start, batch_end):
        run_path_components = [
            "full-model-results",
            version,
            dataset,
            scenario_name,
            run_id,
            activity_type,
            f"model_run={run}",
            "0.parquet",
        ]
        run_blob_name = "/".join(run_path_components)

        try:
            _model_results_cache[cache_key][run] = load_parquet_file(
                container_client, run_blob_name
            )
            if run == run_number:
                requested_run_loaded = True
        except (ResourceNotFoundError, ValueError, AzureError) as e:
            if run == run_number:
                raise
            continue

    # If we haven't loaded the requested run yet, load it directly
    if not requested_run_loaded:
        _model_results_cache[cache_key][run_number] = load_parquet_file(
            container_client, blob_name
        )

    if run_number not in _model_results_cache[cache_key]:
        raise ResourceNotFoundError(
            f"Failed to load run {run_number} for {activity_type}"
        )

    return _model_results_cache[cache_key][run_number]


def load_agg_results(
    container_client: ContainerClient, path_to_agg_files: str, filename: str = "default"
) -> pl.DataFrame:
    """Loads aggregated model results from the folder containing aggregated results files,
    introduced in v3.1

    Args:
        container_client: Connection to container with data files
        path_to_agg_files: Path to "folder" containing the aggregated results files
        filename: Specific aggregation to load. Defaults to "default".

    Returns:
        pl.DataFrame: Dataframe of the results aggregation

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
    blob_name = f"{path_to_agg_files}/{filename}.parquet"
    file = load_parquet_file(container_client, blob_name)

    return file
