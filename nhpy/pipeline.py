#!/usr/bin/env python
"""
NHP Products Pipeline

A cross-platform pipeline that integrates various modules to:
1. Update .venv with 'uv sync'
2. Check Azure Table Storage for an existing full results copy
3. Run scenario with full-model-results enabled if needed, and record in ATS
4. Run detailed results processing

Usage:
    # CLI usage
    uv run python nhpy/pipeline.py \
        aggregated-model-results/v4.x/RXX/test-new-server/20250101_100000/

    # With custom output directory
    uv run python nhpy/pipeline.py \
        aggregated-model-results/v4.x/RXX/test/20250101_100000/ -o ./results

Configuration:
    Set environment variables in .env file:
    - AZ_STORAGE_EP: Azure Storage account URL
    - AZ_STORAGE_RESULTS: Azure Storage container for results
    - AZ_STORAGE_DATA: Azure Storage container for data
    - AZ_TABLE_NAME: Azure Table Storage table name for model run metadata

Exit codes:
    0: Success
    1: No full results exist and failed to create them
    2: Error occurred (authentication, network, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# For cross-platform coloured terminal output
from colorama import Fore, Style, init

from nhpy.add_and_suppress_baseline import add_baseline_to_detailed_results
from nhpy.config import ExitCodes
from nhpy.custom_baseline_standard import produce_custom_suppressed_baseline
from nhpy.run_detailed_results import run_detailed_results
from nhpy.run_full_results import run_scenario_with_full_results
from nhpy.table_storage import (
    find_entity_by_path,
    get_entity,
    get_full_results_copy,
    set_full_results_copy,
)
from nhpy.utils import initialise_connections_and_params

# Initialise colorama with autoreset to avoid colour bleeding
init(autoreset=True)

# High-contrast accessible colours that work well on both light and dark backgrounds
INFO_COLOR = Fore.BLUE
SUCCESS_COLOR = Fore.GREEN
WARNING_COLOR = Fore.YELLOW
ERROR_COLOR = Fore.RED
RESET = Style.RESET_ALL

# %% [markdown]
# Logging configuration


# %%
def configure_logging():
    """Configure logging to display messages from all nhpy modules."""
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("nhpy").setLevel(logging.INFO)


configure_logging()

logger = logging.getLogger(__name__)


# %% [markdown]
# Virtual environment management functions


# %%
def is_venv_active() -> bool:
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


# %%
def ensure_uv_installed() -> bool:
    try:
        subprocess.run(
            "uv --version",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info(f"{INFO_COLOR}uv is already installed{RESET}")
        return True
    except subprocess.CalledProcessError:
        logger.info("uv not found, installing...")
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    "powershell -ExecutionPolicy ByPass -c '"
                    + "irm https://astral.sh/uv/install.ps1 | iex'",
                    shell=True,
                    check=True,
                    timeout=120,
                )
            else:
                subprocess.run(
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    shell=True,
                    check=True,
                    timeout=120,
                )
            logger.info("uv installed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Installation timed out. Check your network connection.")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"{ERROR_COLOR}Failed to install uv: {e}{RESET}")
            return False


# %%
def create_venv() -> bool:
    logger.info(".venv directory not found, creating a new virtual environment...")
    try:
        subprocess.run(
            "uv venv", shell=True, check=True, capture_output=True, text=True, timeout=60
        )
        logger.info(f"{SUCCESS_COLOR}Virtual environment created successfully{RESET}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{ERROR_COLOR}Virtual environment creation timed out{RESET}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{ERROR_COLOR}Failed to create virtual environment: {e}{RESET}")
        return False
    except Exception as e:
        logger.error(f"{ERROR_COLOR}Unexpected error: {e}{RESET}")
        return False


# %%
def get_activation_command(venv_path: Path) -> str:
    if platform.system() == "Windows":
        return f"call {venv_path / 'Scripts' / 'activate'}"
    else:
        return f"source {venv_path / 'bin' / 'activate'}"


# %%
def update_venv_dependencies() -> bool:
    logger.info("Updating virtual environment dependencies...")
    try:
        result = subprocess.run(
            "uv sync",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.stdout:
            logger.info(f"\n{result.stdout}")
        logger.info(f"{SUCCESS_COLOR}Virtual environment updated successfully{RESET}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{ERROR_COLOR}Dependency update timed out after 3 minutes{RESET}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{ERROR_COLOR}Failed to update: {e}{RESET}")
        return False
    except Exception as e:
        logger.error(f"{ERROR_COLOR}Unexpected error: {e}{RESET}")
        return False


# %%
def ensure_venv() -> bool:
    logger.info("Ensuring virtual environment is set up and activated...")
    current_dir = Path.cwd()
    venv_path = current_dir / ".venv"

    if not venv_path.exists():
        if not ensure_uv_installed():
            return False
        if not create_venv():
            return False

    if not is_venv_active():
        activate_cmd = get_activation_command(venv_path)
        logger.warning(
            f"{WARNING_COLOR}Venv exists but is not activated. Please run:{RESET}"
        )
        logger.warning(f"  {activate_cmd}")
        logger.warning(f"{WARNING_COLOR}Then run this script again.{RESET}")
        return False

    return update_venv_dependencies()


# %% [markdown]
# Main pipeline


# %%
def _parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="NHP Products Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"{INFO_COLOR}Example usage:{RESET}\n"
            f"  uv run python nhpy/pipeline.py \\\n"
            f"     aggregated-model-results/v4.x/RXX/test/20250101_100000/\n"
            f"  uv run python nhpy/pipeline.py \\\n"
            f"     aggregated-model-results/v4.x/RXX/test/20250101_100000/ -o ./results\n"
            f"{INFO_COLOR}For more information, visit:{RESET}\n"
            f"  https://github.com/The-Strategy-Unit/nhp_products"
        ),
    )
    parser.add_argument(
        "scenario_path",
        help="Path to scenario directory "
        "(e.g. 'aggregated-model-results/v4.x/RXX/test-new-server/20250101_100000/')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save detailed results (default: 'nhpy/data')",
        default="nhpy/data",
    )
    parser.add_argument("-a", "--account-url", help="Azure Storage account URL")
    parser.add_argument(
        "-r", "--results-container", help="Azure Storage container for results"
    )
    parser.add_argument("-d", "--data-container", help="Azure Storage container for data")
    parser.add_argument(
        "--agg-type",
        help="Which aggregation type to produce for detailed results",
        default="standard",
        choices=["standard", "hrg"],
    )
    parser.add_argument(
        "--include-baseline", help="Whether to include baseline", action="store_true"
    )
    return parser.parse_args()


def _resolve_existing_copy(scenario_path: str) -> str | None:
    """
    Check ATS for an existing full results copy.

    Returns the copy's aggregated_results_path if found, else None.
    """
    logger.info(f"{INFO_COLOR}Checking Azure Table Storage for full results copy:{RESET}")
    logger.info(f"  {scenario_path}")

    copy_id = get_full_results_copy(scenario_path)
    if not copy_id:
        return None

    logger.info(f"{INFO_COLOR}Full results copy found in ATS (RowKey: {copy_id}){RESET}")
    partition_key = scenario_path.split("/")[2]
    copy_entity = get_entity(partition_key=partition_key, row_key=copy_id)

    if copy_entity and copy_entity.get("aggregated_results_path"):
        copy_path = copy_entity["aggregated_results_path"]
        logger.info(f"{SUCCESS_COLOR}Using existing copy results at:{RESET}")
        logger.info(f"  {copy_path}")
        return copy_path

    logger.warning(
        f"{WARNING_COLOR}Copy entry found in ATS but copy entity or path missing.{RESET}"
    )
    logger.warning("Proceeding to create a new copy.")
    return None


def _create_copy_and_record(
    scenario_path: str,
    account_url: str | None,
    results_container: str | None,
) -> str:
    """
    Run a full results copy scenario and record it in ATS.

    Returns the copy's aggregated_results_path.
    """
    logger.info(f"{INFO_COLOR}No full results copy found.{RESET}")
    logger.info(f"{INFO_COLOR}Running scenario with full_model_results enabled...{RESET}")
    logger.info("This operation may take several minutes to complete.")

    result_paths = run_scenario_with_full_results(
        results_path=scenario_path,
        account_url=account_url,
        container_name=results_container,
    )

    copy_path = result_paths["aggregated_results_path"]
    logger.info(f"{SUCCESS_COLOR}Full model results generated at:{RESET}")
    logger.info(f"  {copy_path}")

    original_entity = find_entity_by_path(scenario_path)
    if original_entity:
        set_full_results_copy(
            partition_key=original_entity["PartitionKey"],
            row_key=original_entity["RowKey"],
            copy_row_key=result_paths["model_run_id"],
        )
        logger.info(f"{SUCCESS_COLOR}Recorded full results copy in ATS.{RESET}")
    else:
        logger.warning(
            f"{WARNING_COLOR}Could not find original entity in ATS to update.{RESET}"
        )

    return copy_path


def _run_detailed_results(args: argparse.Namespace, scenario_path: str) -> None:
    """Run detailed results processing and optional baseline."""
    logger.info(f"{INFO_COLOR}Running detailed results for:{RESET}")
    logger.info(f"  {scenario_path}")
    logger.info(f"{INFO_COLOR}This may take several minutes.{RESET}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    account_url = args.account_url or os.getenv("AZ_STORAGE_EP", "")
    results_container = args.results_container or os.getenv("AZ_STORAGE_RESULTS", "")
    data_container = args.data_container or os.getenv("AZ_STORAGE_DATA", "")

    context = initialise_connections_and_params(
        scenario_path,
        account_url,
        results_container,
        data_container,
    )

    results_paths = run_detailed_results(
        context,
        output_dir=str(output_dir),
        agg_type=args.agg_type,
    )

    if not args.include_baseline:
        return

    if args.agg_type == "hrg":
        add_baseline_to_detailed_results(
            results_paths, context, args.agg_type, str(output_dir)
        )
    if args.agg_type == "standard":
        produce_custom_suppressed_baseline(context, args.agg_type, str(output_dir))

    logger.info(f"{SUCCESS_COLOR}Pipeline completed successfully!{RESET}")
    logger.info(f"{SUCCESS_COLOR}Detailed results saved to:{RESET} {output_dir}")


def main() -> int:
    """
    Main pipeline implementing the flowchart logic.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    args = _parse_args()

    try:
        if not ensure_venv():
            logger.warning("Virtual environment setup or update failed")
            return ExitCodes.EXCEPTION_CODE

        # Step 2: Check ATS for existing copy
        copy_path = _resolve_existing_copy(args.scenario_path)

        # Step 3: Create copy if none exists
        if not copy_path:
            copy_path = _create_copy_and_record(
                args.scenario_path,
                args.account_url,
                args.results_container,
            )

        # Step 4: Run detailed results
        _run_detailed_results(args, copy_path)

        return ExitCodes.SUCCESS_CODE

    except KeyboardInterrupt:
        logger.info(f"{WARNING_COLOR}Operation cancelled by user{RESET}")
        return ExitCodes.SIGINT_CODE

    except Exception as e:
        logger.error(f"{ERROR_COLOR}Pipeline error: {e}{RESET}")
        return ExitCodes.EXCEPTION_CODE


# %%
if __name__ == "__main__":
    import inspect

    frame = inspect.currentframe()
    if frame is None:
        current_file = Path(__file__)
    else:
        current_file = Path(inspect.getfile(frame))
    module_name = f"nhpy.{current_file.stem}"
    logger = logging.getLogger(module_name)

    sys.exit(main())
