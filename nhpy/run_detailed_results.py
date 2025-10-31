"""
Generate detailed results for a model scenario.

This module produces detailed aggregations of IP, OP, and AAE model results
in CSV and Parquet formats. It assumes the scenario has already been run with
`full_model_results = True`. Outputs are stored in a `data/` folder.

Usage:
    # Programmatic usage
    from nhpy.run_detailed_results import run_detailed_results
    run_detailed_results("aggregated-model-results/v4.0/RXX/test/20250101_100000/")

    # CLI usage
    uv run python nhpy/run_detailed_results.py \
        aggregated-model-results/v4.0/RXX/test/20250101_100000/

Prerequisites:
    - Authentication via Azure CLI is required
    - Scenario must have full_model_results enabled
    - You can check using `nhpy.check_full_results`
    - You can enable full results using `nhpy.run_full_results`

Configuration:
    Set environment variables: AZ_STORAGE_EP, AZ_STORAGE_RESULTS, AZ_STORAGE_DATA

Exit codes:
    0: Success
    2: Error occurred (authentication, network, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
# Imports

import argparse
import gc
import os
import sys
import time
from logging import INFO
from pathlib import Path

import psutil
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from tqdm import tqdm

from nhpy import az, process_data, process_results
from nhpy.config import ExitCodes
from nhpy.types import ProcessContext
from nhpy.utils import (
    EnvironmentVariableError,
    _load_dotenv_file,
    configure_logging,
    get_logger,
)

# %%
# Define public API
__all__ = ["run_detailed_results"]

# %%
# Get a logger for this module
logger = get_logger()


def get_memory_usage():
    """Get current memory usage in MB.

    Uses psutil to get the Resident Set Size (RSS), which is the non-swapped physical
    memory a process has used. This is a cross-platform alternative to resource.getrusage
    that works on Windows, Mac, and Linux.
    """
    # Get the process memory info
    process = psutil.Process()
    memory_info = process.memory_info()

    # Return RSS (Resident Set Size) in MB
    return memory_info.rss / (1024 * 1024)  # Convert bytes to MB


# Load environment variables
_load_dotenv_file(interpolate=False)


def _initialise_connections_and_params(
    results_path: str,
    account_url: str,
    results_container: str,
    data_container: str,
) -> ProcessContext:
    """
    Initialize connections and load parameters from the aggregated results.

    Args:
        results_path: Path to the aggregated model results
        account_url: Azure Storage account URL
        results_container: Azure Storage container for results
        data_container: Azure Storage container for data

    Returns:
        dict containing all necessary objects and parameters for processing

    Raises:
        FileNotFoundError: If results folder or data version not found
    """
    # Connections and params
    results_connection = az.connect_to_container(account_url, results_container)
    data_connection = az.connect_to_container(account_url, data_container)
    params = az.load_agg_params(results_connection, results_path)

    # Get info from the results file
    scenario_name = params["scenario"]
    trust = params["dataset"]
    model_version = params["app_version"]
    baseline_year = int(params["start_year"])
    run_id = params["create_datetime"]

    # Patch model version for loading the data. Results folder name truncated,
    # e.g. v3.0 does not show the patch version. But data stores in format v3.0.1
    model_version_data = az.find_latest_version(data_connection, params["app_version"])
    logger.info(f"Using data: {model_version_data}")
    if model_version_data == "N/A":
        raise FileNotFoundError("Results folder not found")

    # Add principal to the "vanilla" model results
    actual_results_df = az.load_agg_results(results_connection, results_path)
    actual_results_df = process_results.convert_results_format(actual_results_df)

    return {
        "results_connection": results_connection,
        "data_connection": data_connection,
        "params": params,
        "scenario_name": scenario_name,
        "trust": trust,
        "model_version": model_version,
        "model_version_data": model_version_data,
        "baseline_year": baseline_year,
        "run_id": run_id,
        "actual_results_df": actual_results_df,
    }


def _check_results_exist(output_dir: str, scenario_name: str, activity_type: str) -> bool:
    """
    Check if results files for a specific activity type already exist.

    Args:
        output_dir: Directory where output files are stored
        scenario_name: Name of the scenario
        activity_type: Type of activity (ip, op, ae)

    Returns:
        bool: True if both CSV and Parquet files exist, False otherwise
    """
    results_base = f"{output_dir}/{scenario_name}_detailed_{activity_type}_results"
    csv_path = Path(f"{results_base}.csv")
    parquet_path = Path(f"{results_base}.parquet")

    if csv_path.exists() and parquet_path.exists():
        logger.info(f"Found existing {activity_type.upper()} results files:")
        logger.info(f"  - {csv_path}")
        logger.info(f"  - {parquet_path}")
        return True
    return False


def _process_inpatient_results(
    ctx: ProcessContext,
    output_dir: str,
) -> None:
    """
    Process inpatient detailed results.

    Args:
        ctx: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Extract scenario name first for file existence check
    scenario_name = ctx["scenario_name"]

    # Check if output files already exist
    if _check_results_exist(output_dir, scenario_name, "ip"):
        logger.info("Skipping IP processing as results already exist")
        return

    # Report memory usage at start
    logger.info(f"Memory usage before IP processing: {get_memory_usage():.2f} MB")

    # Extract needed variables from ctx
    results_connection = ctx["results_connection"]
    data_connection = ctx["data_connection"]
    model_version = ctx["model_version"]
    model_version_data = ctx["model_version_data"]
    trust = ctx["trust"]
    baseline_year = ctx["baseline_year"]
    run_id = ctx["run_id"]
    actual_results_df = ctx["actual_results_df"]

    # Load original data
    original_df = az.load_data_file(
        container_client=data_connection,
        version=model_version_data,
        dataset=trust,
        activity_type="ip",
        year=baseline_year,
    )

    # Pre-allocate dictionary
    model_runs = {}

    # Pre-create the reference dataframe copy once
    reference_df = original_df.copy().drop(columns=["speldur", "classpat"])

    # Choose a larger batch size for optimal I/O performance
    batch_size = 30  # Balance between memory usage and I/O performance

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting IP processing with {get_memory_usage():.2f} MB memory usage")
    for run in tqdm(range(1, 257), desc="IP"):
        # Load with batch functionality - this will cache surrounding runs
        df = az.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "ip",
                "run_number": run,
                "batch_size": batch_size,  # This enables batch loading
            },
        )

        # Use the pre-created reference dataframe
        merged = reference_df.merge(df, on="rn", how="inner")
        results = process_data.process_ip_detailed_results(merged)

        # More efficient dictionary update
        results_dict = results.to_dict()
        for k, v in results_dict["value"].items():
            if k not in model_runs:  # Avoid unnecessary .keys() call
                model_runs[k] = []
            model_runs[k].append(v)
    end = time.perf_counter()
    logger.info(
        f"All IP model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process model runs dictionary after the loop completes
    model_runs_df = process_data.process_model_runs_dict(
        model_runs,
        columns=[
            "sitetret",
            "age_group",
            "sex",
            "pod",
            "tretspef",
            "los_group",
            "maternity_delivery_in_spell",
            "measure",
        ],
    )
    logger.info(
        f"IP data processed into dataframe, memory usage: {get_memory_usage():.2f} MB"
    )

    # Validate results
    default_beddays_principal = (
        actual_results_df[actual_results_df["measure"] == "beddays"]["mean"]
        .sum()
        .astype(int)
    )
    detailed_beddays_principal = (
        model_runs_df.loc[
            (
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                "beddays",
            ),
            :,
        ]
        .sum()
        .loc["mean"]
        .astype(int)
    )

    try:
        assert abs(default_beddays_principal - detailed_beddays_principal) <= 1
    except AssertionError:
        logger.warning(
            f"""Validation mismatch: default={default_beddays_principal},
            detailed={detailed_beddays_principal}"""
        )

    # Save results
    model_runs_df.to_csv(f"{output_dir}/{scenario_name}_detailed_ip_results.csv")
    model_runs_df.to_parquet(f"{output_dir}/{scenario_name}_detailed_ip_results.parquet")

    # Clean up memory
    del model_runs_df, model_runs, original_df, reference_df
    if "az" in sys.modules and hasattr(sys.modules["az"], "_model_results_cache"):
        # Clear the cache after processing
        sys.modules["az"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after IP processing, current usage: {get_memory_usage():.2f} MB"
    )


def _process_outpatient_results(
    context: ProcessContext,
    output_dir: str,
) -> None:
    """
    Process outpatient detailed results.

    Args:
        context: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Extract scenario name first for file existence check
    scenario_name = context["scenario_name"]

    # Check if output files already exist
    if _check_results_exist(output_dir, scenario_name, "op"):
        logger.info("Skipping OP processing as results already exist")
        return

    # Report memory usage at start
    logger.info(f"Memory usage before OP processing: {get_memory_usage():.2f} MB")

    # Extract needed variables from context
    results_connection = context["results_connection"]
    data_connection = context["data_connection"]
    model_version = context["model_version"]
    model_version_data = context["model_version_data"]
    trust = context["trust"]
    baseline_year = context["baseline_year"]
    run_id = context["run_id"]
    actual_results_df = context["actual_results_df"]

    # Load original data
    original_df = az.load_data_file(
        data_connection, model_version_data, trust, "op", baseline_year
    ).fillna("unknown")
    original_df = original_df.rename(columns={"index": "rn"})

    # Pre-allocate dictionary
    op_model_runs = {}

    # Pre-create the reference dataframe copy once
    reference_df = original_df.copy().drop(columns=["attendances", "tele_attendances"])

    # Choose a larger batch size for optimal I/O performance
    batch_size = 30  # Balance between memory usage and I/O performance

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting OP processing with {get_memory_usage():.2f} MB memory usage")
    for run in tqdm(range(1, 257), desc="OP"):
        # Load with batch functionality - this will cache surrounding runs
        df = az.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "op",
                "run_number": run,
                "batch_size": batch_size,  # This enables batch loading
            },
        )

        assert df.shape[0] == original_df.shape[0]

        # Use the pre-created reference dataframe
        merged = reference_df.merge(df, on="rn", how="inner")
        results = process_data.process_op_detailed_results(merged)

        # Load conversion data with batch functionality
        df_conv = az.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "op_conversion",
                "run_number": run,
                "batch_size": batch_size,  # This enables batch loading
            },
        )

        df_conv = process_data.process_op_converted_from_ip(df_conv)
        results = process_data.combine_converted_with_main_results(df_conv, results)

        # More efficient dictionary update
        results_dict = results.to_dict()
        for k, v in results_dict["value"].items():
            if k not in op_model_runs:  # Avoid unnecessary .keys() call
                op_model_runs[k] = []
            op_model_runs[k].append(v)
    end = time.perf_counter()
    logger.info(
        f"All OP model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process results
    op_model_runs_df = process_data.process_model_runs_dict(
        op_model_runs, columns=["sitetret", "pod", "age_group", "tretspef", "measure"]
    )

    # Validate results
    detailed_attendances_principal = (
        op_model_runs_df.round(1)
        .loc[(slice(None), slice(None), slice(None), slice(None), "attendances"), :]
        .sum()
        .astype(int)
        .loc["mean"]
    )
    default_attendances_principal = (
        actual_results_df[actual_results_df["measure"] == "attendances"]["mean"]
        .sum()
        .astype(int)
    )

    # They're not always exactly the same because of rounding
    try:
        assert abs(default_attendances_principal - detailed_attendances_principal) <= 1
    except AssertionError:
        logger.warning(
            f"""Validation mismatch: default={default_attendances_principal},
            detailed={detailed_attendances_principal}"""
        )

    # Save results
    op_model_runs_df.to_csv(f"{output_dir}/{scenario_name}_detailed_op_results.csv")
    op_model_runs_df.to_parquet(
        f"{output_dir}/{scenario_name}_detailed_op_results.parquet"
    )

    # Clean up memory
    del op_model_runs_df, op_model_runs, original_df, reference_df
    if "az" in sys.modules and hasattr(sys.modules["az"], "_model_results_cache"):
        # Clear the cache after processing
        sys.modules["az"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after OP processing, current usage: {get_memory_usage():.2f} MB"
    )


def _validate_aae_metric(
    ae_model_runs_df, actual_results_df, measure_name: str, metric_label: str
):
    """
    Helper function to validate A&E metrics.

    Args:
        ae_model_runs_df: DataFrame with detailed results
        actual_results_df: DataFrame with aggregated results
        measure_name: Name of measure to validate (e.g., "ambulance", "walk-in")
        metric_label: Label for the measure in logs (e.g., "Ambulance", "Walk-in")
    """
    detailed_value = (
        ae_model_runs_df.loc[
            (
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                slice(None),
                measure_name,
            ),
            :,
        ]
        .sum()
        .loc["mean"]
        .round(0)
    )

    default_value = (
        actual_results_df[actual_results_df["measure"] == measure_name]["mean"]
        .sum()
        .round(0)
    )

    # They're not always exactly the same because of rounding
    try:
        assert abs(default_value - detailed_value) <= 1
    except AssertionError:
        logger.warning(
            f"""{metric_label} validation mismatch: default={default_value},
            detailed={detailed_value}"""
        )


def _process_aae_results(
    context: ProcessContext,
    output_dir: str,
) -> None:
    """
    Process A&E detailed results.

    Args:
        context: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Extract scenario name first for file existence check
    scenario_name = context["scenario_name"]

    # Check if output files already exist
    if _check_results_exist(output_dir, scenario_name, "ae"):
        logger.info("Skipping A&E processing as results already exist")
        return

    # Report memory usage at start
    logger.info(f"Memory usage before A&E processing: {get_memory_usage():.2f} MB")

    # Extract needed variables from context
    results_connection = context["results_connection"]
    data_connection = context["data_connection"]
    model_version = context["model_version"]
    model_version_data = context["model_version_data"]
    trust = context["trust"]
    baseline_year = context["baseline_year"]
    run_id = context["run_id"]
    actual_results_df = context["actual_results_df"]

    # Load original data
    original_df = az.load_data_file(
        data_connection, model_version_data, trust, "aae", baseline_year
    ).fillna("unknown")
    original_df = original_df.rename(columns={"index": "rn"})

    # Pre-allocate dictionary
    ae_model_runs = {}

    # Pre-create the reference dataframe copy once
    reference_df = original_df.drop(columns=["arrivals"])

    # Choose a larger batch size for optimal I/O performance
    batch_size = 30  # Balance between memory usage and I/O performance

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting A&E processing with {get_memory_usage():.2f} MB memory usage")
    for run in tqdm(range(1, 257), desc="A&E"):
        # Load with batch functionality - this will cache surrounding runs
        df = az.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "aae",
                "run_number": run,
                "batch_size": batch_size,  # This enables batch loading
            },
        )

        assert len(df) == len(original_df)

        # Use the pre-created reference dataframe
        merged = reference_df.merge(df, on="rn", how="inner")
        results = process_data.process_aae_results(merged)

        # Load conversion data with batch functionality
        df_conv = az.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "sdec_conversion",
                "run_number": run,
                "batch_size": batch_size,  # This enables batch loading
            },
        )

        df_conv = process_data.process_aae_converted_from_ip(df_conv)
        results = process_data.combine_converted_with_main_results(df_conv, results)

        # More efficient dictionary update
        results_dict = results.to_dict()
        for k, v in results_dict["arrivals"].items():
            if k not in ae_model_runs:  # Avoid unnecessary .keys() call
                ae_model_runs[k] = []
            ae_model_runs[k].append(v)
    end = time.perf_counter()
    logger.info(
        f"All AAE model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process results
    ae_model_runs_df = process_data.process_model_runs_dict(
        ae_model_runs,
        columns=[
            "sitetret",
            "pod",
            "age_group",
            "attendance_category",
            "aedepttype",
            "acuity",
            "measure",
        ],
    )

    # Validate results
    _validate_aae_metric(ae_model_runs_df, actual_results_df, "ambulance", "Ambulance")
    _validate_aae_metric(ae_model_runs_df, actual_results_df, "walk-in", "Walk-in")

    # Save results
    ae_model_runs_df.to_csv(f"{output_dir}/{scenario_name}_detailed_ae_results.csv")
    ae_model_runs_df.to_parquet(
        f"{output_dir}/{scenario_name}_detailed_ae_results.parquet"
    )

    # Clean up memory
    del ae_model_runs_df, ae_model_runs, original_df, reference_df
    if "az" in sys.modules and hasattr(sys.modules["az"], "_model_results_cache"):
        # Clear the cache after processing
        sys.modules["az"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after A&E processing, current usage: {get_memory_usage():.2f} MB"
    )


def run_detailed_results(
    results_path: str,
    output_dir: str | None = None,
    account_url: str | None = None,
    results_container: str | None = None,
    data_container: str | None = None,
) -> dict[str, str]:
    """
    Generate detailed results for a model scenario.

    Takes an existing scenario results path and produces detailed aggregations
    of IP, OP, and AAE model results in CSV and Parquet formats. If results already
    exist for any activity type, those processing steps are skipped.

    Args:
        results_path: Path to existing aggregated results
        output_dir: Directory to save output files (default: 'nhpy/data')
        account_url: Azure Storage account URL (default: from environment)
        results_container: Azure Storage container for results (default: from environment)
        data_container: Azure Storage container for data (default: from environment)

    Returns:
        dict[str, str]: dictionary containing paths to output files

    Raises:
        EnvironmentVariableError: If required environment variables are missing
        ValueError: If path format is invalid
        FileNotFoundError: If results folder or data version not found
        Various Azure exceptions: For authentication, network, or permission issues
    """
    # Start the total timing
    total_start_time = time.perf_counter()

    # Load environment variables if not provided
    account_url = account_url or os.getenv("AZ_STORAGE_EP", "")
    results_container = results_container or os.getenv("AZ_STORAGE_RESULTS", "")
    data_container = data_container or os.getenv("AZ_STORAGE_DATA", "")

    if not all([account_url, results_container, data_container]):
        missing = []
        if not account_url:
            missing.append("AZ_STORAGE_EP")
        if not results_container:
            missing.append("AZ_STORAGE_RESULTS")
        if not data_container:
            missing.append("AZ_STORAGE_DATA")
        raise EnvironmentVariableError(
            missing_vars=missing,
            message=f"Missing environment variables: {', '.join(missing)}",
        )

    # Set up output directory
    if output_dir is None:
        output_dir = "nhpy/data"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialise connections and load parameters
    context = _initialise_connections_and_params(
        results_path, account_url, results_container, data_container
    )

    scenario_name = context["scenario_name"]

    # Check for existing result files
    ip_exists = _check_results_exist(output_dir, scenario_name, "ip")
    op_exists = _check_results_exist(output_dir, scenario_name, "op")
    ae_exists = _check_results_exist(output_dir, scenario_name, "ae")

    # Track if we processed any new results
    processed_new_results = False

    # Process each type of results if they don't already exist
    if not ip_exists:
        _process_inpatient_results(context, output_dir)
        processed_new_results = True

    if not op_exists:
        _process_outpatient_results(context, output_dir)
        processed_new_results = True

    if not ae_exists:
        _process_aae_results(context, output_dir)
        processed_new_results = True

    if processed_new_results:
        # Calculate and report the total time only if we did some work
        total_end_time = time.perf_counter()
        total_duration = total_end_time - total_start_time
        minutes, seconds = divmod(total_duration, 60)
        time_str = f"{int(minutes)}m {seconds:.2f}s"
        logger.info(f"Total processing time for Pandas implementation: {time_str}")
    else:
        logger.info("All results already exist. No processing needed.")

    # Return paths to output files
    return {
        "ip_csv": f"{output_dir}/{scenario_name}_detailed_ip_results.csv",
        "ip_parquet": f"{output_dir}/{scenario_name}_detailed_ip_results.parquet",
        "op_csv": f"{output_dir}/{scenario_name}_detailed_op_results.csv",
        "op_parquet": f"{output_dir}/{scenario_name}_detailed_op_results.parquet",
        "ae_csv": f"{output_dir}/{scenario_name}_detailed_ae_results.csv",
        "ae_parquet": f"{output_dir}/{scenario_name}_detailed_ae_results.parquet",
    }


def main() -> int:
    """
    CLI entry point when module is run directly.

    Returns:
        int: Exit code (0 for success, 2 for errors)
    """
    configure_logging(INFO)

    parser = argparse.ArgumentParser(
        description="Generate detailed results for a model scenario"
    )
    parser.add_argument(
        "results_path",
        help="Path to existing aggregated results \
        (e.g. 'aggregated-model-results/v4.0/RXX/test/20250101_100000/')",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (default: 'nhpy/data')",
        default="nhpy/data",
    )
    parser.add_argument("--account-url", help="Azure Storage account URL")
    parser.add_argument("--results-container", help="Azure Storage container for results")
    parser.add_argument("--data-container", help="Azure Storage container for data")

    # Add mutually exclusive processing flags to match Polars implementation
    processing_group = parser.add_mutually_exclusive_group()
    processing_group.add_argument(
        "--ip", action="store_true", help="Check/process only inpatient results"
    )
    processing_group.add_argument(
        "--op", action="store_true", help="Check/process only outpatient results"
    )
    processing_group.add_argument(
        "--ae", action="store_true", help="Check/process only A&E results"
    )

    args = parser.parse_args()

    try:
        # First, check if specific result type was requested
        if args.ip or args.op or args.ae:
            # Get output directory and scenario name first
            output_dir = args.output_dir

            # We need to set up the output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Initialize connections to get scenario name
            account_url = args.account_url or os.getenv("AZ_STORAGE_EP", "")
            results_container = args.results_container or os.getenv(
                "AZ_STORAGE_RESULTS", ""
            )
            data_container = args.data_container or os.getenv("AZ_STORAGE_DATA", "")

            # Initialise context just to get scenario name
            context = _initialise_connections_and_params(
                args.results_path, account_url, results_container, data_container
            )
            scenario_name = context["scenario_name"]

            # Check for existing files for the requested type
            activity_type = "ip" if args.ip else "op" if args.op else "ae"
            if _check_results_exist(output_dir, scenario_name, activity_type):
                file_prefix = f"{scenario_name}_detailed_{activity_type}_results"
                results_base = f"{output_dir}/{file_prefix}"
                logger.info(f"Results already available for {activity_type.upper()}")
                logger.info(f"Paths:")
                logger.info(f"  - {results_base}.csv")
                logger.info(f"  - {results_base}.parquet")
                return ExitCodes.SUCCESS_CODE

        # If we're here, either no specific type was requested or files don't exist yet
        # Run detailed results generation
        result_files = run_detailed_results(
            results_path=args.results_path,
            output_dir=args.output_dir,
            account_url=args.account_url,
            results_container=args.results_container,
            data_container=args.data_container,
        )

        logger.info("🎉 Detailed results generated successfully!")
        logger.info(f"Results saved to: {args.output_dir}/")

        # List specific files based on what was requested
        if args.ip:
            logger.info(f"IP results: {result_files['ip_csv']}")
        elif args.op:
            logger.info(f"OP results: {result_files['op_csv']}")
        elif args.ae:
            logger.info(f"A&E results: {result_files['ae_csv']}")
        else:
            logger.info(f"Files generated: {len(result_files)}")

        return ExitCodes.SUCCESS_CODE

    except (
        ValueError,
        EnvironmentVariableError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
        FileNotFoundError,
    ) as e:
        logger.error(f"main():Error: {e}")
        return ExitCodes.EXCEPTION_CODE
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return ExitCodes.SIGINT_CODE


# Main guardrail
if __name__ == "__main__":
    sys.exit(main())
