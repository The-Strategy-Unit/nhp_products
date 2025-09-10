"""
Run scenario with full model results enabled.

This module takes an existing scenario results path, extracts the parameters,
and runs a new scenario with save_full_model_results=True. It uses the existing
Azure utilities and follows the same patterns as other modules in the project.

Usage:
    # Programmatic usage
    from nhpy.run_full_results import run_scenario_with_full_results
    paths = run_scenario_with_full_results(
        "aggregated-model-results/v3.5/RXX/test/20250101_100000/"
    )

    # CLI usage
    uv run python nhpy/run_full_results.py \
        aggregated-model-results/v3.5/RXX/test/20250101_100000/

Configuration:
    Set environment variables based on your .env
    Or provide API credentials via function arguments.

Exit codes:
    0: Success
    2: Error occurred (authentication, network, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
import argparse
import json
import re
import sys
from logging import INFO
from pathlib import Path
from textwrap import dedent

import requests
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from jsonschema import ValidationError
from nhp.aci.run_model.helpers import validate_params
from traitlets import Bool

from nhpy.config import Colours, Constants, ExitCodes
from nhpy.types import ScenarioPaths
from nhpy.utils import (
    EnvironmentVariableError,
    _construct_results_path,
    _load_environment_variables,
    _load_scenario_params,
    configure_logging,
    get_logger,
)

# %%
# Define public API
__all__ = ["run_scenario_with_full_results"]


# %%
# Get a logger for this module
logger = get_logger()


# %%
def _extract_scenario_components(results_path: str) -> tuple[str, str, str, str]:
    """
    Extract version, dataset, scenario, and datetime from results path.

    Args:
        results_path: Path of the scenario results file you want full results for
        e.g."aggregated-model-results/v3.5/RXX/test/20250610_112654/"

    Returns:
        tuple[str, str, str, str]: (version, dataset, scenario, datetime)

    Raises:
        ValueError: If path format is invalid
    """
    path = results_path.strip("/")
    levels = path.split("/")

    if len(levels) < Constants.PATH_DEPTH or levels[0] != "aggregated-model-results":
        raise ValueError(
            f"Invalid results path: {results_path}. "
            "Expected format: aggregated-model-results/version/dataset/scenario/datetime/"
        )

    return levels[1], levels[2], levels[3], levels[4]


# %%
def _prepare_full_results_params(params: dict[str, object]) -> dict[str, object]:
    """
    Modify parameters for a full results run.

    Args:
        params: Original scenario parameters

    Returns:
        dict: Modified parameters for full results run
    """
    # Create a copy to avoid modifying the original
    new_params = params.copy()

    # Modify parameters for the new run
    new_params["scenario"] = params["scenario"]
    new_params["user"] = "ds-team"
    new_params["viewable"] = False
    new_params["original_datetime"] = params["original_datetime"]

    logger.debug(f"Prepared parameters for new scenario: {new_params['scenario']}")

    return new_params


# %%


def _validate_params(params: dict[str, object]) -> None:
    """Validates params against published schema for a specific model version

    Args:
        params (dict[str, object]): NHP model scenario parameters

    Raises:
        ValidationError: For schema validation errors

    """

    ver = str(params["app_version"])
    logger.info(f"Validating params against schema {ver}...")

    try:
        validate_params(params, ver)

    except ValidationError as e:
        logger.error(f"_validate_params(): Params validation failed: {e}")
        raise


# %%
def _submit_api_request(
    params: dict[str, object],
    api_url: str,
    api_key: str,
    timeout: int = 30,
) -> None:
    """
    Submit API request to run scenario with full model results.

    Args:
        params: Scenario parameters
        api_url: API endpoint URL
        api_key: API authentication key
        timeout: Request timeout in seconds

    Returns:
        str: The create_datetime from the API response

    Raises:
        requests.RequestException: For API request failures
        ValueError: For invalid API responses
    """
    logger.info("Submitting API request for full results run...")

    try:
        response = requests.post(
            url=api_url,
            params={
                "app_version": params["app_version"],
                "code": api_key,
                "save_full_model_results": "True",
            },
            json=params,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.RequestException as e:
        logger.error(f"_submit_api_request():API request failed: {e}")
        raise

    # Parse response
    try:
        response_data = response.json()
        server_datetime = response_data["create_datetime"]
        response_data["create_datetime"] = response_data["original_datetime"]
        logger.info(
            dedent(
                f"""
            ✅ API request successful 🥳
            {Colours.GREEN}Server datetime: {server_datetime}{Colours.RESET}
            Original datetime: {response_data["original_datetime"]}"""
            ).strip()
        )

        return server_datetime

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(
            f"_submit_api_request():API returned non-JSON response: {response.text[:200]}"
        )
        raise ValueError(f"Invalid API response format: {e}") from e


# %%
def run_scenario_with_full_results(
    results_path: str,
    api_url: str | None = None,
    api_key: str | None = None,
    account_url: str | None = None,
    container_name: str | None = None,
) -> ScenarioPaths:
    """
    Run a scenario with full model results enabled.

    Takes an existing scenario results path, loads the parameters, and submits
    a new run with save_full_model_results=True.

    Args:
        results_path: Path to existing aggregated results
        api_url: API endpoint URL (default: from environment)
        api_key: API authentication key (default: from environment)
        account_url: Azure Storage account URL (default: from environment)
        container_name: Azure Storage container name (default: from environment)

    Returns:
        ScenarioPaths: Dictionary containing paths to new scenario results

    Raises:
        ValueError: If path format is invalid or API response is malformed
        requests.RequestException: If API request fails
        Various Azure exceptions: For authentication, network, or permission issues
    """
    # Load environment variables if not provided
    if not all([api_url, api_key, account_url, container_name]):
        env_config = _load_environment_variables()
        api_url = api_url or env_config["API_URL"]
        api_key = api_key or env_config["API_KEY"]
        account_url = account_url or env_config["AZ_STORAGE_EP"]
        container_name = container_name or env_config["AZ_STORAGE_RESULTS"]

    # Validate the results path format
    try:
        _extract_scenario_components(results_path=results_path)
    except ValueError as e:
        logger.error(f"run_scenario_with_full_results():Invalid results path: {e}")
        raise

    if account_url and container_name:
        # Load scenario parameters
        params = _load_scenario_params(
            results_path=results_path,
            account_url=account_url,
            container_name=container_name,
        )

        # Prepare parameters for full results run
        mod_params = _prepare_full_results_params(params=params)

        # Submit API request
        if api_url and api_key:
            server_datetime = _submit_api_request(
                params=mod_params,
                api_url=api_url,
                api_key=api_key,
                timeout=Constants.TIMEOUT_SEC,
            )

        mod_params["create_datetime"] = server_datetime

        # Construct and return result paths
        full_results_params = _construct_results_path(params=mod_params)

        return full_results_params


# %%
def main() -> int:
    """
    CLI entry point when module is run directly.

    Returns:
        int: Exit code (0 for success, 2 for errors)
    """
    configure_logging(INFO)

    parser = argparse.ArgumentParser(
        description="Run scenario with full model results enabled"
    )
    parser.add_argument(
        "results_path",
        help="Path to existing aggregated results \
        (e.g. 'aggregated-model-results/v3.5/RXX/test/20250101_100000/')",
    )
    parser.add_argument("--api-url", help="API endpoint URL")
    parser.add_argument("--api-key", help="API authentication key")
    parser.add_argument("--account-url", help="Azure Storage account URL")
    parser.add_argument("--container", help="Azure Storage container name")

    args = parser.parse_args()

    try:
        result_paths = run_scenario_with_full_results(
            results_path=args.results_path,
            api_url=args.api_url,
            api_key=args.api_key,
            account_url=args.account_url,
            container_name=args.container,
        )

        logger.info("🎉 Scenario submitted successfully!")
        logger.info(f"Monitor results at: {result_paths['full_results_path']}")

        return ExitCodes.SUCCESS_CODE

    except (
        ValueError,
        requests.RequestException,
        EnvironmentVariableError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
    ) as e:
        logger.error(f"main():Error: {e}")
        return ExitCodes.EXCEPTION_CODE
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return ExitCodes.SIGINT_CODE


# %%
# Main guardrail
if __name__ == "__main__":
    sys.exit(main())
