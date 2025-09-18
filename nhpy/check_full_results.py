"""
Check for full-model-results in Azure Blob Storage.

This module verifies whether complete model results exist for a given scenario in
Azure Blob Storage by analysing blob structure and paths. It reuses Azure connection
utilities from az.py for simpler, more maintainable code.

Usage:
    # Programmatic usage
    from nhpy.check_full_results import check_full_results
    exists = check_full_results("aggregated-model-results/v3.5/RXX/test/20250610_112654/")

    # CLI usage (quotes are optional)
    uv run python nhpy/check_full_results.py "agg/v1.3/ABC/test/20250110_111000/"
    uv run python nhpy/check_full_results.py "scenario/path/" --account-url URL \
        --container mycont

Configuration:
    Set environment variables: AZ_STORAGE_EP, AZ_STORAGE_RESULTS
    Or provide credentials via function arguments or CLI options.

Exit codes:
    0: Full results exist
    1: No results found
    2: Error occurred (authentication, network, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
import argparse
import sys
from logging import DEBUG, INFO, Logger

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from dotenv import load_dotenv

from nhpy.az import get_azure_blobs, get_azure_credentials
from nhpy.config import EmptyContainerError, ExitCodes
from nhpy.utils import (
    _construct_results_path,
    _load_scenario_params,
    configure_logging,
    get_logger,
)

# %%
load_dotenv()


# %%
# Define public API
__all__ = ["check_full_results"]

# %%
# Get a logger for this module
logger = get_logger()


# %%
def _analyse_blob_paths_for_full_results(paths: list[str]) -> tuple[bool, str]:
    """
    Analyse blob paths to determine if they represent full model results.

    Args:
        paths: List of blob paths to analyse

    Returns:
        tuple[bool, str]: (is_full_results, explanation)
    """
    has_directories = False
    has_model_run_dirs = False
    has_parquet_in_model_run = False

    for path in paths:
        # Check if this is a model_run directory containing a parquet file
        if "/model_run=" in path and path.endswith("/0.parquet"):
            has_directories = True
            has_model_run_dirs = True
            has_parquet_in_model_run = True
            break

        # Check for any directory structure
        if "/" in path:
            has_directories = True

            # Check for model_run directories
            if "/model_run=" in path:
                has_model_run_dirs = True

    # Determine if we have full results
    is_full_results = has_directories and has_model_run_dirs and has_parquet_in_model_run

    # Create explanation message
    if is_full_results:
        message = "✅"
    elif not has_directories:
        message = "📁❌ No directories found (flat structure only)"
    elif not has_model_run_dirs:
        message = "📂→❌ Directories found but no 'model_run=' subdirectories"
    elif not has_parquet_in_model_run:
        message = """📂→📂→📄❌ Found 'model_run=' subdirectories but no parquet file
        inside them"""
    else:
        message = "💥 Unexpected state in result analysis"

    return is_full_results, message


# %%
def check_full_results(
    scenario_path: str,
    account_url: str | None = None,
    container_name: str | None = None,
) -> bool:
    """
    Check if full-model-results exists for a given scenario.

    Args:
        scenario_path: Path to scenario (e.g.
            'aggregated-model-results/v3.5/RXX/test-new-server/20250610_112654/')
        account_url: Azure Storage account URL (default: AZ_STORAGE_EP)
        container_name: Azure Storage container name (default: AZ_STORAGE_RESULTS)

    Returns:
        bool: True if full-model-results exists, False otherwise

    Raises:
        ValueError: If any required credential is missing or path is invalid
        ClientAuthenticationError: If authentication fails
        ResourceNotFoundError: If container doesn't exist
        ServiceRequestError: If network connectivity issues occur
    """

    account_url_from_az, container_name_from_az = get_azure_credentials()
    account_url = account_url or account_url_from_az
    container_name = container_name or container_name_from_az

    params = _load_scenario_params(
        results_path=scenario_path,
        account_url=account_url,
        container_name=container_name,
    )

    try:
        results_path_dict = _construct_results_path(params=params)
        full_results_path = results_path_dict["full_results_path"]
    except ValueError as e:
        logger.error(f"Error constructing full results path: {e}")
        raise

    try:
        blobs = get_azure_blobs(
            path=full_results_path,
            account_url=account_url,
            container_name=container_name,
        )

        # Extract paths for analysis
        paths = [blob.name for blob in blobs]

        # Analyse paths to determine if we have full results
        is_full_results, explanation = _analyse_blob_paths_for_full_results(paths=paths)

        # Log the appropriate message
        logger.info(f"{explanation}")
        logger.debug(f"{explanation} at {full_results_path}")

        return is_full_results

    except EmptyContainerError:
        # No blobs found is a valid result = no results
        logger.info("🚫")
        logger.debug(f"No blobs found at {full_results_path}")
        return False
    except (
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
    ) as e:
        logger.error(f"Error accessing Azure Storage: {e}")
        raise


# %%
def main(level: Logger = INFO) -> int:
    """
    CLI entry point when module is run directly.

    Args:
        level: The logging level to set for terminal output

    Returns:
        int: Exit code (0 for success, 1 if no results found, 2 for errors)
    """
    configure_logging(level)

    parser = argparse.ArgumentParser(
        description="Check if full-model-results exists \
            for a scenario in Azure Blob Storage"
    )
    parser.add_argument(
        "scenario_path",
        help="Path to scenario (e.g. 'agg/v1.3/ABC/test-new-server/20250110_111000/')",
    )
    parser.add_argument("--container", help="Azure Storage container name")
    parser.add_argument("--account-url", help="Azure Storage account URL")

    args = parser.parse_args()

    try:
        result = check_full_results(
            args.scenario_path,
            account_url=args.account_url,
            container_name=args.container,
        )

        logger.info(f"Full model results {'exist' if result else 'do not exist'}")
        logger.debug(
            f"Full model results {'exist' if result else 'do not exist'} "
            f"for {args.scenario_path}"
        )

        return ExitCodes.SUCCESS_CODE if result else ExitCodes.ERROR_CODE

    except (
        ValueError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
    ) as e:
        logger.error(f"Error: {e}")
        return ExitCodes.EXCEPTION_CODE
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return ExitCodes.SIGINT_CODE


# %%
# Main guardrail
if __name__ == "__main__":
    sys.exit(main(level=INFO))
