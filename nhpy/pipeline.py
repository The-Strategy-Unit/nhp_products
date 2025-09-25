#!/usr/bin/env python
"""
NHP Products Pipeline

A cross-platform pipeline that integrates various modules to:
1. Update .venv with 'uv sync'
2. Check if full-model-results exist for a scenario
3. Run scenario with full-model-results enabled if needed
4. Run detailed results processing

Usage:
    # CLI usage
    uv run python nhpy/pipeline.py \
        aggregated-model-results/v4.x/RXX/test-new-server/20250101_100000/

    # With custom output directory
    uv run python nhpy/pipeline.py \
        aggregated-model-results/v4.x/RXX/test/20250101_100000/ --output-dir ./results

Configuration:
    Set environment variables in .env file:
    - AZ_STORAGE_EP: Azure Storage account URL
    - AZ_STORAGE_RESULTS: Azure Storage container for results
    - AZ_STORAGE_DATA: Azure Storage container for data

Exit codes:
    0: Success
    1: No full results exist and failed to create them
    2: Error occurred (authentication, network, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
import argparse
import logging
import platform
import subprocess
import sys
from pathlib import Path

# For cross-platform coloured terminal output
from colorama import Fore, Style, init

from nhpy.check_full_results import check_full_results
from nhpy.config import ExitCodes
from nhpy.run_detailed_results import run_detailed_results
from nhpy.run_full_results import run_scenario_with_full_results

# Initialise colorama with autoreset to avoid colour bleeding
init(autoreset=True)

# High-contrast accessible colours that work well on both light and dark backgrounds
# Blue is bright enough for dark backgrounds but not too light for white backgrounds
INFO_COLOR = Fore.BLUE
# Green universally indicates success with good visibility
SUCCESS_COLOR = Fore.GREEN
# Yellow for warnings has good contrast on most backgrounds
WARNING_COLOR = Fore.YELLOW
# Red for errors stands out clearly on all backgrounds
ERROR_COLOR = Fore.RED
# Reset to return to default terminal colours
RESET = Style.RESET_ALL

# %% [markdown]
# Logging configuration


# %%
def configure_logging():
    """Configure logging to display messages from all nhpy modules."""
    root_logger = logging.getLogger()

    # Only add handler if it doesn't already have one
    if not root_logger.handlers:
        # Add console handler to root logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

        # Reduce noise from external libraries
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Enable all nhpy module logs
        logging.getLogger("nhpy").setLevel(logging.INFO)


# Configure logging
configure_logging()

# Set up logger for this module
# Use regular name when imported, special handling for __main__ is in the entry point
logger = logging.getLogger(__name__)


# %% [markdown]
# Virtual environment management functions


# %%
def is_venv_active() -> bool:
    """
    Check if a virtual environment is currently active.

    Returns:
        bool: True if a venv is active, False otherwise
    """
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


# %%
def ensure_uv_installed() -> bool:
    """
    Ensure that Astral's uv tool is installed.

    Returns:
        bool: True if uv is installed (or was successfully installed), False otherwise
    """
    try:
        # Check if uv is available
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
                logger.info("Installing uv on Windows (this may take a moment)...")
                subprocess.run(
                    "powershell -ExecutionPolicy ByPass -c '"
                    + "irm https://astral.sh/uv/install.ps1 | iex'",
                    shell=True,
                    check=True,
                    timeout=120,  # Network operation needs longer timeout
                )
            else:  # Unix-based (Linux/macOS)
                logger.info(
                    "Installing uv on Unix-based system (this may take a moment)..."
                )
                subprocess.run(
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    shell=True,
                    check=True,
                    timeout=120,
                )
            logger.info("uv installed successfully")

            return True
        except subprocess.TimeoutExpired:
            logger.error(
                "Installation timed out. Check your network connection and try again."
            )
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"{ERROR_COLOR}Failed to install uv: {e}{RESET}")
            return False


# %%
def create_venv() -> bool:
    """
    Create a new virtual environment in the current directory using uv.

    Creates a .venv directory in the current working directory using
    uv's built-in venv functionality.

    Returns:
        bool: True if virtual environment was created successfully, False otherwise
    """
    logger.info(".venv directory not found, creating a new virtual environment...")

    try:
        # Create new venv with uv
        logger.info("Creating virtual environment (this may take a moment)...")
        subprocess.run(
            "uv venv",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info(f"{SUCCESS_COLOR}Virtual environment created successfully{RESET}")

        return True
    except subprocess.TimeoutExpired:
        logger.error(
            f"{ERROR_COLOR}Virtual environment creation timed out after 60 seconds{RESET}"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{ERROR_COLOR}Failed to create virtual environment: {e}{RESET}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")

        return False
    except Exception as e:
        logger.error(
            f"{ERROR_COLOR}Unexpected error creating virtual environment: {e}{RESET}"
        )
        return False


# %%
def get_activation_command(venv_path: Path) -> str:
    """
    Get the appropriate command to activate the virtual environment.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        str: Command to activate the virtual environment
    """
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate"

        return f"call {activate_script}"
    else:  # Unix-based (Linux/macOS)
        activate_script = venv_path / "bin" / "activate"

        return f"source {activate_script}"


# %%
def update_venv_dependencies() -> bool:
    """
    Update virtual environment dependencies using uv sync.

    Returns:
        bool: True if dependencies were updated successfully, False otherwise
    """
    logger.info("Updating virtual environment dependencies...")
    try:
        logger.info(
            "Running uv sync (this may take a moment depending on package count)..."
        )
        result = subprocess.run(
            "uv sync",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,  # Allow up to 3 minutes for dependency resolution
        )
        if result.stdout:
            logger.info(f"\n{result.stdout}")

        logger.info(f"{SUCCESS_COLOR}Virtual environment updated successfully{RESET}")

        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{ERROR_COLOR}Dependency update timed out after 3 minutes{RESET}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{ERROR_COLOR}Failed to update virtual environment: {e}{RESET}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(
            f"{ERROR_COLOR}Unexpected error updating virtual environment: {e}{RESET}"
        )
        return False


# %%
def ensure_venv() -> bool:
    """
    Ensure a virtual environment exists, is activated, and up to date.

    1. Check if .venv exists in current directory
    2. If not, create it (after ensuring uv is installed)
    3. Check if venv is activated
    4. If not, provide activation instructions
    5. Update dependencies with uv sync

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Ensuring virtual environment is set up and activated...")

    # Get current directory using pathlib
    current_dir = Path.cwd()
    venv_path = current_dir / ".venv"

    # Check if .venv exists, if not create it
    if not venv_path.exists():
        if not ensure_uv_installed():
            return False

        if not create_venv():
            return False

    # Check if venv is activated
    if not is_venv_active():
        activate_cmd = get_activation_command(venv_path)

        # We can't directly activate the venv in the current process from Python
        logger.warning(
            f"{WARNING_COLOR}Venv exists but is not activated. Please run:{RESET}"
        )
        logger.warning(f"  {activate_cmd}")
        logger.warning(f"{WARNING_COLOR}Then run this script again.{RESET}")
        return False

    # Update dependencies with uv sync
    return update_venv_dependencies()


# %% [markdown]
# Main pipeline


# %%
def main() -> int:
    """
    Main pipeline implementing the flowchart
    https://github.com/The-Strategy-Unit/nhp_products/issues/7#issuecomment-3323738420
    logic.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
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
        help="""Path to scenario directory
        (e.g. 'aggregated-model-results/v4.x/RXX/test-new-server/20250101_100000/')""",
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

    args = parser.parse_args()

    try:
        # Step 1: Update .venv with 'uv sync' (ensuring it exists and is activated)
        if not ensure_venv():
            logger.warning("Virtual environment setup or update failed")
            return ExitCodes.EXCEPTION_CODE

        # Step 2: Check if full model results exist
        logger.info(f"{INFO_COLOR}Checking if full model results exist for:{RESET}")
        logger.info(f"  {args.scenario_path}")
        full_results_exist = check_full_results(
            args.scenario_path,
            account_url=args.account_url,
            container_name=args.results_container,
        )

        # Step 3: If full model results don't exist, run scenario with full results
        scenario_path_for_detailed = args.scenario_path

        if not full_results_exist:
            logger.info(f"{INFO_COLOR}Full model results don't exist.{RESET}")
            logger.info(
                f"{INFO_COLOR}Running scenario with full_model_results enabled...{RESET}"
            )
            logger.info("This operation may take several minutes to complete.")

            try:
                # Set expectations for long-running operation
                logger.info(f"{INFO_COLOR}Starting model run with full results{RESET}")
                logger.info(f"{INFO_COLOR}(please be patient)...{RESET}")

                result_paths = run_scenario_with_full_results(
                    results_path=args.scenario_path,
                    account_url=args.account_url,
                    container_name=args.results_container,
                )

                # Update scenario path to the new results
                scenario_path_for_detailed = result_paths["aggregated_results_path"]
                logger.info(f"{SUCCESS_COLOR}Full model results generated at:{RESET}")
                logger.info(f"  {scenario_path_for_detailed}")

            except Exception as e:
                logger.error(
                    f"{ERROR_COLOR}Failed to run scenario with full results: {e}{RESET}"
                )
                return ExitCodes.EXCEPTION_CODE
        else:
            logger.info(f"{INFO_COLOR}Full model results already exist{RESET}")

        # Step 4: Run detailed results
        logger.info(f"{INFO_COLOR}Running detailed results for:{RESET}")
        logger.info(f"  {scenario_path_for_detailed}")
        logger.info(f"{INFO_COLOR}This may take several minutes.{RESET}")

        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{INFO_COLOR}Starting detailed results processing{RESET}")
        logger.info(f"{INFO_COLOR}(IP, OP, and AAE data)...{RESET}")

        run_detailed_results(
            results_path=scenario_path_for_detailed,
            output_dir=str(output_dir),
            account_url=args.account_url,
            results_container=args.results_container,
            data_container=args.data_container,
        )

        logger.info(f"{SUCCESS_COLOR}Pipeline completed successfully!{RESET}")
        logger.info(f"{SUCCESS_COLOR}Detailed results saved to:{RESET} {output_dir}")

        return ExitCodes.SUCCESS_CODE

    except KeyboardInterrupt:
        logger.info(f"{WARNING_COLOR}Operation cancelled by user{RESET}")
        return ExitCodes.SIGINT_CODE

    except Exception as e:
        logger.error(f"{ERROR_COLOR}Pipeline error: {e}{RESET}")
        return ExitCodes.EXCEPTION_CODE


# %% [markdown]
# Entry point

# %%
if __name__ == "__main__":
    # Configure module-specific logger when run directly
    import inspect

    frame = inspect.currentframe()
    if frame is None:
        current_file = Path(__file__)  # Fallback to __file__ if frame is None
    else:
        current_file = Path(inspect.getfile(frame))
    module_name = f"nhpy.{current_file.stem}"
    logger = logging.getLogger(module_name)

    # Pass main's return code to the OS for programs to know if module succeeded/failed
    sys.exit(main())
