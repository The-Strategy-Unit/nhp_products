"""
Configurable utilities for Python applications.

This module provides utilities for:
- Logging configuration with appropriate formatters
- Environment variable loading and validation with error handling
- Module name detection that works in scripts, notebooks, and interactive shells
- Scenario path construction and parameter management
- Type definitions for environment configuration and scenario results
"""

# %%
# Imports
import logging
import re
from os import getenv
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from nhpy.az import connect_to_container, load_agg_params
from nhpy.config import EnvironmentVariableError
from nhpy.types import EnvironmentConfig, ScenarioPaths

# %% [markdown]
# Logging setup functions


# %%
def get_module_name():
    """Get current module name that works in both scripts and notebooks."""
    try:
        return __name__ if __name__ != "__main__" else Path(__file__).stem
    except (NameError, AttributeError):
        # We're in IPython/Jupyter
        try:
            from IPython import get_ipython  # noqa

            return get_ipython().user_ns.get("__name__", "notebook")
        except (ImportError, AttributeError):
            return "interactive"


# %%
def get_logger(module=None):
    """Get logger with appropriate name."""
    logger_name = module or get_module_name()

    return logging.getLogger(logger_name)


# %%
def configure_logging(level: int = logging.INFO):
    """Configure logging with appropriate formatting."""
    # Configure root logger - affects all loggers
    root_logger = logging.getLogger()

    # Suppress Azure SDK logs
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()

    if level <= logging.DEBUG:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


# %% [markdown]
# Init logger

# %%
logger = get_logger()


# %% [markdown]
# Load env variables


# %%
def _load_dotenv_file() -> None:
    """Load .env file with proper error handling."""
    # Get repository name from current directory
    repo_name = Path.cwd().name

    # Check ~/.config/repo_name/.env first
    config_path = Path.home() / ".config" / repo_name / ".env"
    if config_path.exists():
        try:
            load_dotenv(dotenv_path=config_path)
            return
        except (PermissionError, IOError) as e:
            logger.error(f"Error loading {config_path}: {e}")
            # Continue to try local .env

    # Fall back to local .env
    local_path = Path(".env")
    if local_path.exists():
        try:
            load_dotenv()
            return
        except PermissionError as e:
            logger.error(f"Permission denied when loading .env file: {e}")
            raise
        except IOError as e:
            logger.error(f"IO error when loading .env file: {e}")
            raise

    error_msg = f"No .env file found in {config_path} or {local_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def _validate_environment_variables(required_vars: list[str]) -> EnvironmentConfig:
    """Validate and return environment variables."""
    env_vars = {}
    missing_vars = []

    for var in required_vars:
        value = getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise EnvironmentVariableError(message=error_msg, missing_vars=missing_vars)

    return cast(EnvironmentConfig, env_vars)


def _load_environment_variables() -> EnvironmentConfig:
    """
    Load required environment variables for Azure Storage and API access.

    Returns:
        EnvironmentConfig: Dictionary containing environment variables

    Raises:
        EnvironmentVariableError: If any required environment variable is missing
        FileNotFoundError: If .env file cannot be found
        PermissionError: If permission is denied when accessing .env file
        IOError: If an I/O error occurs when reading .env file
    """
    _load_dotenv_file()

    required_vars = [
        "AZ_STORAGE_EP",
        "AZ_STORAGE_RESULTS",
        "AZ_STORAGE_DATA",
        "API_KEY",
        "API_URL",
        "AZ_VALID_PATH",
    ]

    return _validate_environment_variables(required_vars)


# %% [markdown]
# Scenario path and params


# %%
def _construct_results_path(params: dict[str, object]) -> ScenarioPaths:
    """
    Construct paths for the new scenario results.

    Args:
        params: Scenario parameters

    Returns:
        ScenarioPaths: Dictionary with result paths
    """
    json_path = (
        f"prod/{params['app_version']}/{params['dataset']}/"
        f"{params['scenario']}-{params['create_datetime']}.json.gz"
    )

    aggregated_results_path = (
        f"aggregated-model-results/{params['app_version']}/{params['dataset']}/"
        f"{params['scenario']}/{params['create_datetime']}"
    )

    full_results_path = (
        f"full-model-results/{params['app_version']}/{params['dataset']}/"
        f"{params['scenario']}/{params['create_datetime']}"
    )

    logger.info("📁 Result paths constructed:")
    logger.info(f"📄 JSON: {json_path}")
    logger.info(f"∑ Aggregated: {aggregated_results_path}")

    return ScenarioPaths(
        json_path=json_path,
        aggregated_results_path=aggregated_results_path,
        full_results_path=full_results_path,
        original_datetime=str(params["original_datetime"]),
    )


# %%
def _load_scenario_params(
    results_path: str, account_url: str, container_name: str
) -> dict[str, object]:
    """
    Load parameters from an existing scenario using load_agg_params.

    Args:
        results_path: Path to the aggregated results
        account_url: Azure Storage account URL
        container_name: Azure Storage container name

    Returns:
        dict: Scenario parameters

    Raises:
        Various Azure exceptions: For connection or file access issues
    """
    if not results_path:
        raise ValueError("Results path cannot be empty")

    if not isinstance(results_path, str):
        raise ValueError(f"Path must be a string, got {type(results_path).__name__}")

    logger.info("Loading scenario parameters...")
    logger.debug(f"Loading parameters from: {results_path}")

    container_client = connect_to_container(account_url, container_name)

    params = load_agg_params(container_client, results_path)

    # Extract the datetime from the path
    original_datetime = Path(results_path.strip()).name

    # Check if the name matches the expected datetime format
    if re.match(r"^\d{8}_\d{6}$", original_datetime):
        params["original_datetime"] = original_datetime

    logger.debug(f"Loaded parameters for scenario: {params.get('scenario', 'unknown')}")

    return params
