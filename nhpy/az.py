"""
Azure Blob Storage utilities for NHP model data.

This module provides functions to connect to Azure Blob Storage containers and load
various types of data files related to NHP modelling, including JSON results, parquet
files, and aggregated model results. It handles version management and supports both
current and legacy data structures.
"""

# %%
# Imports
import gzip
import io
import json
import os

import pandas as pd
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobPrefix, ContainerClient

from nhpy.config import EmptyContainerError


# %%
def get_azure_credentials(
    account_url: str | None = None,
    container_name: str | None = None,
) -> tuple[str, str]:
    """
    Get and validate Azure credentials from arguments or environment variables.

    Args:
        account_url: Azure Storage account URL
        container_name: Azure Storage container name

    Returns:
        tuple[str, str]: Validated (account_url, container_name)

    Raises:
        ValueError: If any required credential is missing
    """
    account_url = account_url or os.environ.get("AZ_STORAGE_EP")
    container_name = container_name or os.environ.get("AZ_STORAGE_RESULTS")

    if not all([account_url, container_name]):
        raise ValueError(
            "Missing Azure credentials. Provide as arguments or set environment vars: "
            "AZ_STORAGE_EP, AZ_STORAGE_RESULTS"
        )

    return account_url, container_name


# %%
def get_azure_blobs(
    path: str,
    account_url: str | None = None,
    container_name: str | None = None,
) -> list:
    """
    Connect to Azure and list blobs at the specified path using az.py utilities.

    Args:
        path: Path prefix within the container whose blobs we want to list
        account_url: Azure Storage account URL (default: from environment)
        container_name: Azure Storage container name (default: from environment)

    Returns:
        list: A list of blob objects representing blobs at the path

    Raises:
        ValueError: If any required credential is missing
        EmptyContainerError: If the container exists but contains no blobs at the path
        Various Azure exceptions: For authentication, network, or permission issues
    """
    # Get credentials from environment if not provided
    account_url, container_name = get_azure_credentials(
        account_url=account_url, container_name=container_name
    )

    # logger.info("Connecting to Azure...")
    # logger.debug(f"Connecting to container: {container_name}")

    container_client = connect_to_container(account_url, container_name)

    # List blobs at the specified path
    blobs = list(container_client.list_blobs(name_starts_with=path))

    if not blobs:
        raise EmptyContainerError(
            container_name=container_name,
            path=path,
            message=f"No blobs found in container '{container_name}' at path '{path}'",
        )

    return blobs


# %%
def find_latest_version(container_client: ContainerClient, version: str) -> str:
    """Finds latest version of data given a model version. Params and model results files
    do not include patch versions.
    For example, given version v3.0, if folders v3.0.1 and v3.0.0 exist,
    it will return v3.0.1

    Args:
        container_client: Connection to the container with the results files
        version: Version of the dataset to be used

    Returns:
        str: Latest patch version available, or None if no versions found

    Raises:
        AzureError: If there's an issue accessing the container
    """
    try:
        list_of_folders = []
        for blob in container_client.walk_blobs(name_starts_with=version, delimiter="/"):
            if isinstance(blob, BlobPrefix):
                list_of_folders.append(blob.name)

        if not list_of_folders:
            return None

        return sorted(list_of_folders)[-1].strip("/")
    except AzureError:
        raise


# %%
def connect_to_container(account_url: str, container_name: str) -> ContainerClient:
    """Connects to container on Azure

    Args:
        account_url: URL to the container
        container_name: Name of the container

    Returns:
        ContainerClient: Azure ContainerClient

    Raises:
        ClientAuthenticationError: If authentication fails
        ValueError: If URL or container name is invalid
    """
    try:
        default_credential = DefaultAzureCredential()
        container_client = ContainerClient(
            account_url, container_name, default_credential
        )

        return container_client
    except ClientAuthenticationError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid connection parameters: {e}") from e


# %%
def load_results_gzip_file(
    container_client: ContainerClient, path_to_results: str
) -> dict:
    """Loads a JSON NHP results file from Azure

    Args:
        container_client: Connection to the container with the results files
        path_to_results: Path to the results file on Azure Storage

    Returns:
        dict: JSON of the results file, in Python dict format

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If the file is not a valid gzip file
        AzureError: If there's an issue downloading the file
    """
    try:
        blob_client = container_client.get_blob_client(path_to_results)
        download_stream = blob_client.download_blob()
        stream_object = io.BytesIO(download_stream.readall())

        with gzip.open(stream_object, "rt") as f:
            results_json = json.load(f)

        return results_json
    except ResourceNotFoundError:
        raise
    except json.JSONDecodeError:
        raise
    except (gzip.BadGzipFile, OSError) as e:
        raise ValueError(f"Invalid gzip file: {e}") from e
    except AzureError:
        raise


# %%
def load_results_json_file(container_client: ContainerClient, path_to_json: str) -> dict:
    """Loads a plain JSON file from Azure

    Args:
        container_client: Connection to the container with the JSON files
        path_to_json: Path to the JSON file on Azure Storage

    Returns:
        dict: JSON file content, in Python dict format

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        ValueError: If the file has invalid encoding
        AzureError: If there's an issue downloading the file
    """
    try:
        blob_client = container_client.get_blob_client(path_to_json)
        download_stream = blob_client.download_blob()
        json_content = download_stream.readall().decode("utf-8")
        results_json = json.loads(json_content)

        return results_json
    except ResourceNotFoundError:
        raise
    except json.JSONDecodeError:
        raise
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid file encoding: {e}") from e
    except AzureError:
        raise


# %%
def load_parquet_file(
    container_client: ContainerClient, path_to_file: str
) -> pd.DataFrame:
    """Loads parquet file from Azure

    Args:
        container_client: Connection to the container
        path_to_file: Path to the file to be loaded

    Returns:
        pd.DataFrame: DataFrame of the data

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
    try:
        blob_client = container_client.get_blob_client(path_to_file)
        download_stream = blob_client.download_blob()
        stream_object = io.BytesIO(download_stream.readall())
        data = pd.read_parquet(stream_object)

        return data
    except ResourceNotFoundError:
        raise
    except (pd.errors.ParserError, ValueError) as e:
        if "parquet" in str(e).lower():
            raise ValueError(f"Invalid parquet file: {e}") from e
        raise
    except AzureError:
        raise


# %%
def load_data_file(
    container_client: ContainerClient,
    version: str,
    dataset: str,
    activity_type: str,
    year: int = 2019,
) -> pd.DataFrame:
    """Loads Parquet file from Azure containing NHP model data. Only works for >= v3.0.0

    Args:
        container_client: Connection to container with data files
        version: Version of the dataset to be used
        dataset: Name of the trust/dataset
        activity_type: Type of activity - options include ip, op, aae.
        year: Year for the data. Defaults to 2019.

    Returns:
        pd.DataFrame: DataFrame of the data

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


# %%
def load_data_file_old(
    container_client: ContainerClient,
    version: str,
    dataset: str,
    activity_type: str,
    year: int = 2019,
) -> pd.DataFrame:
    """Loads Parquet file from Azure containing NHP model data.
    Only works for model versions prior to v3.0

    Args:
        container_client: Connection to container with data files
        version: Version of the dataset to be used
        dataset: Name of the trust/dataset
        activity_type: Type of activity - options include ip, op, aae.
        year: Year for the data. Defaults to 2019.

    Returns:
        pd.DataFrame: DataFrame of the data

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
    blob_name = f"{version}/{year}/{dataset}/{activity_type}.parquet"
    data = load_parquet_file(container_client, blob_name)

    return data


# %%
# TODO: at a later stage, we could package input args into a dict
# to respect PLR0913 https://docs.astral.sh/ruff/rules/too-many-arguments/
def load_model_run_results_file(  # noqa
    container_client: ContainerClient,
    version: str,
    dataset: str,
    scenario_name: str,
    run_id: str,
    activity_type: str,
    run_number: int,
) -> pd.DataFrame:
    """Loads full model results from a specific run from Azure.
    Requires for the run to have had "--save-full-model-results" enabled

    Args:
        container_client: Connection to container with data files
        version: Version of the dataset to be used
        dataset: Name of the trust/dataset
        scenario_name: Name of the scenario
        run_id: ID of the specific model run of that scenario
        activity_type: Type of activity - options include ip, op, aae.
        run_number: Which of the Monte Carlo simulation runs it is

    Returns:
        pd.DataFrame: DataFrame of the data

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
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
    data = load_parquet_file(container_client, blob_name)

    return data


# %%
def load_agg_params(container_client: ContainerClient, path_to_agg_files: str) -> dict:
    """Loads params from the folder containing the aggregated results files,
    introduced in v3.1

    Args:
        container_client: Connection to container with data files
        path_to_agg_files: Path to "folder" containing the aggregated results files

    Returns:
        dict: Params for the model run

    Raises:
        ResourceNotFoundError: If the params.json file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        AzureError: If there's an issue downloading the file
    """
    blob_name = f"{path_to_agg_files}/params.json"
    try:
        blob_client = container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        stream_object = io.BytesIO(download_stream.readall())
        results_json = json.load(stream_object)

        return results_json
    except ResourceNotFoundError:
        raise
    except json.JSONDecodeError:
        raise
    except AzureError:
        raise


# %%
def load_agg_results(
    container_client: ContainerClient, path_to_agg_files: str, filename: str = "default"
) -> pd.DataFrame:
    """Loads aggregated model results from the folder containing aggregated results files,
    introduced in v3.1

    Args:
        container_client: Connection to container with data files
        path_to_agg_files: Path to "folder" containing the aggregated results files
        filename: Specific aggregation to load. Defaults to "default".

    Returns:
        pd.DataFrame: Dataframe of the results aggregation

    Raises:
        ResourceNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
        AzureError: If there's an issue downloading the file
    """
    blob_name = f"{path_to_agg_files}/{filename}.parquet"
    file = load_parquet_file(container_client, blob_name)

    return file
