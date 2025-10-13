"""
Type definitions module for scenario execution and environment configuration.

This module defines TypedDict classes that specify the structure for results and
configuration settings. It provides type safety and documentation for data structures
used throughout the application.
"""

# %%
# Imports
from typing import TypedDict

from azure.storage.blob import ContainerClient
from pandas import DataFrame


# %%
# Custom types
class ScenarioPaths(TypedDict):
    """Return type for scenario run results."""

    json_path: str
    aggregated_results_path: str
    full_results_path: str
    original_datetime: str


class EnvironmentConfig(TypedDict):
    """Type definition for environment configuration."""

    AZ_STORAGE_EP: str
    AZ_STORAGE_RESULTS: str
    AZ_STORAGE_DATA: str


class ProcessContext(TypedDict):
    """Context for detailed results processing."""

    results_connection: ContainerClient
    data_connection: ContainerClient
    params: dict[str, str]
    scenario_name: str
    trust: str
    model_version: str
    model_version_data: str
    baseline_year: int
    run_id: str
    actual_results_df: DataFrame


class ModelRunParams(TypedDict):
    """Parameters for loading model run results."""

    version: str
    dataset: str
    scenario_name: str
    run_id: str
    activity_type: str
    run_number: int
    batch_size: int | None
