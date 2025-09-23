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
import sys
import time
from logging import INFO
from textwrap import dedent

import requests
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.identity import CredentialUnavailableError
from jsonschema import ValidationError
from nhp.aci.run_model import create_model_run
from nhp.aci.run_model.helpers import validate_params
from nhp.aci.status.model_run_status import get_model_run_status

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

    version = str(params["app_version"])
    logger.info(f"Validating params against schema {version}...")

    try:
        validate_params(params, version)

    except ValidationError as e:
        logger.error(f"_validate_params(): Params validation failed: {e}")
        raise


# %%
# TODO: dict[str, object] is somewhat meaningless. Make this a TypedDict perhaps?
def _start_container(params: dict[str, object]) -> dict[str, str]:
    """Starts Azure container using submitted parameters, with save_full_model_results
    set to True

    Args:
        params (dict[str, object]): NHP model scenario parameters

    Returns:
        dict[str, str]: Metadata for container
    """

    logger.info("Starting container for full model results run...")
    try:
        metadata = create_model_run(
            params, str(params["app_version"]), save_full_model_results=True
        )
        logger.info(
            dedent(
                f"""
            ✅ Container successfully started 🥳
            {Colours.GREEN}Create datetime: {metadata["create_datetime"]}{Colours.RESET}
            Container ID: {metadata["id"]}"""
            ).strip()
        )
        return metadata

    except CredentialUnavailableError as e:
        logger.error(f"_start_container(): Unable to start container: {e}")
        raise


# %%
def _track_container_status(metadata: dict[str, str]):
    """Checks container status every 120 seconds to check on model run progress

    Args:
        metadata (dict[str, str]): Metadata for submitted model run

    """

    logger.info(
        f"Checking container status every 120 seconds... ⌚",
    )
    time.sleep(120)
    while True:
        try:
            status = get_model_run_status(metadata["id"])
        except Exception as e:
            logger.error(
                f"Error fetching container status: {e}. Retrying in 120 seconds..."
            )
            time.sleep(120)
            continue
        if status:
            state = status.get("state")
            detail_status = status.get("detail_status")
            if state == "Running":
                runs_completed = status.get("complete")
                model_runs = status.get("model_runs")
                logger.info(
                    f"Container state: {state}: {runs_completed} of {model_runs}",
                )
            # Check for completion and exit
            elif state == "Terminated":
                if detail_status == "Completed":
                    logger.info("✅ Container completed successfully.")
                else:
                    logger.error(f"❌ Container terminated with status {detail_status}")
                return
            else:
                logger.info(
                    f"Container state: {state}: {detail_status}",
                )
            # Wait and poll again
        time.sleep(120)


# %%
def run_scenario_with_full_results(
    results_path: str,
    account_url: str | None = None,
    container_name: str | None = None,
) -> ScenarioPaths:
    """
    Run a scenario with full model results enabled.

    Takes an existing scenario results path, loads the parameters, and submits
    a new run with save_full_model_results=True.

    Args:
        results_path: Path to existing aggregated results
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
    if not all([account_url, container_name]):
        env_config = _load_environment_variables()
        account_url = account_url or env_config["AZ_STORAGE_EP"]
        container_name = container_name or env_config["AZ_STORAGE_RESULTS"]

    # Check that all required parameters are now set. If not, raise an error.
    missing_params = [
        param_name
        for param_name, param_value in {
            "account_url": account_url,
            "container_name": container_name,
        }.items()
        if not param_value
    ]
    if missing_params:
        raise EnvironmentVariableError(
            missing_vars=missing_params,
            message=f"Missing required parameters: {', '.join(missing_params)}",
        )

    # Validate the results path format
    try:
        _extract_scenario_components(results_path=results_path)
    except ValueError as e:
        logger.error(f"run_scenario_with_full_results():Invalid results path: {e}")
        raise

    # Load scenario parameters
    # At this point account_url and container_name are guaranteed to be non-None
    # due to the checks above, but we need to help the type checker
    params = _load_scenario_params(
        results_path=results_path,
        account_url=account_url if account_url is not None else "",
        container_name=container_name if container_name is not None else "",
    )

    # Prepare parameters for full results run
    mod_params = _prepare_full_results_params(params=params)

    # Validate parameters against schema
    _validate_params(mod_params)

    container_metadata = _start_container(mod_params)

    # Track container status
    _track_container_status(container_metadata)

    # Use container creation datetime
    mod_params["create_datetime"] = (
        container_metadata["create_datetime"] if container_metadata else ""
    )

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
    parser.add_argument("--account-url", help="Azure Storage account URL")
    parser.add_argument("--container", help="Azure Storage container name")

    args = parser.parse_args()

    try:
        result_paths = run_scenario_with_full_results(
            results_path=args.results_path,
            account_url=args.account_url,
            container_name=args.container,
        )

        logger.info("🎉 Full results for scenario completed successfully!")
        logger.info(f"Results available at: {result_paths['full_results_path']}")

        return ExitCodes.SUCCESS_CODE

    except (
        ValueError,
        requests.RequestException,
        EnvironmentVariableError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
        CredentialUnavailableError,
        ValidationError,
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
