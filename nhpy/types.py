"""
Type definitions module for scenario execution and environment configuration.

This module defines TypedDict classes that specify the structure for results and
configuration settings. It provides type safety and documentation for data structures
used throughout the application.
"""

# %%
# Imports
from typing import TypedDict


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
    API_KEY: str
    API_URL: str
