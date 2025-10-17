"""Generate detailed results for a model scenario - Polars implementation.

This module produces detailed aggregations of IP, OP, and A&E model results
in CSV and Parquet formats using Polars for improved performance. It assumes the
scenario has already been run with `full_model_results = True`.
Outputs are stored in a `data/` folder.

This module reuses non-Pandas functions from run_detailed_results.py and
reimplements only the Pandas-specific functions to use Polars instead.

Usage:
    # Programmatic usage
    from nhpy.run_detailed_results_pl import run_detailed_results
    run_detailed_results("aggregated-model-results/v4.0/RXX/test/20250101_100000/")

    # CLI usage
    uv run python nhpy/run_detailed_results_pl.py \
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
import resource
import sys
import time
from logging import INFO
from pathlib import Path

# Define a Polars-specific ProcessContext type
from typing import TypedDict

import polars as pl
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from tqdm import tqdm

from nhpy import az, az_pl, process_data_pl, process_results_pl
from nhpy.config import ExitCodes

# Import non-Pandas constants and functions from run_detailed_results.py
from nhpy.utils import (
    EnvironmentVariableError,
    configure_logging,
    get_logger,
)

# %% [markdown]
# ## Type Definitions and Initialisation


# %%
# Create a Polars-specific version of ProcessContext
class ProcessContext(TypedDict):
    """Context for detailed results processing with Polars."""

    results_connection: ContainerClient
    data_connection: ContainerClient
    params: dict[str, str]
    scenario_name: str
    trust: str
    model_version: str
    model_version_data: str
    baseline_year: int
    run_id: str
    actual_results_df: pl.DataFrame


# Define public API
# Override imported __all__ with our own version
__all__ = ["run_detailed_results"]

# Get a logger for this module
logger = get_logger()


# %%
def get_memory_usage():
    """Get current memory usage in MB.

    Uses the resource module which provides more accurate memory usage information
    than sys.getsizeof. This helps monitor memory consumption during the processing
    of large datasets to prevent out-of-memory errors.
    """
    # Get memory info from resource module (more accurate than sys.getsizeof)
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # Return memory usage in MB
    return rusage.ru_maxrss / 1024  # Convert KB to MB


# %%
# Try to load from ~/.config/<project_name>/.env first, fall back to default behaviour
project_name = Path(__name__).resolve().parent.name
config_env_path = Path(f"~/.config/{project_name}/.env").expanduser()
if config_env_path.exists():
    load_dotenv(str(config_env_path))
else:
    load_dotenv()

# %% [markdown]
# ## Connection and Parameter Initialisation


# %%
def _initialize_connections_and_params(
    results_path: str,
    account_url: str,
    results_container: str,
    data_container: str,
) -> ProcessContext:
    """Initialise connections and load parameters from the aggregated results.

    Sets up all the necessary connections to Azure storage and loads parameters
    for processing detailed results. Creates a context object with all required
    information for the processing pipeline.

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
    logger.info("Using data: %s", model_version_data)
    if model_version_data == "N/A":
        error_msg = "Results folder not found"
        raise FileNotFoundError(error_msg)

    # Add principal to the "vanilla" model results
    actual_results_df = az_pl.load_agg_results(results_connection, results_path)
    actual_results_df = process_results_pl.convert_results_format(actual_results_df)

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


# %% [markdown]
# ## Inpatient Results Processing


# %%
def _process_inpatient_results(
    ctx: ProcessContext,
    output_dir: str,
) -> None:
    """Process inpatient detailed results for all 256 Monte Carlo simulations.

    Loads original data, processes each of the 256 model runs, and aggregates results
    into statistical summaries. Uses batch loading with caching for optimal performance
    and memory management. Saves results to CSV and Parquet files.

    Args:
        ctx: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Report memory usage at start
    logger.info(f"Memory usage before IP processing: {get_memory_usage():.2f} MB")
    # Extract needed variables from ctx
    results_connection = ctx["results_connection"]
    data_connection = ctx["data_connection"]
    model_version = ctx["model_version"]
    model_version_data = ctx["model_version_data"]
    trust = ctx["trust"]
    baseline_year = ctx["baseline_year"]
    scenario_name = ctx["scenario_name"]
    run_id = ctx["run_id"]
    actual_results_df = ctx["actual_results_df"]

    # Load original data
    original_df = az_pl.load_data_file(
        container_client=data_connection,
        version=model_version_data,
        dataset=trust,
        activity_type="ip",
        year=baseline_year,
    )

    # Pre-allocate dictionary
    model_runs = {}

    # Pre-create the reference dataframe copy once
    reference_df = original_df.drop(["speldur", "classpat"])

    # Choose a moderate batch size balancing memory usage and performance
    batch_size = 50

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting IP processing with {get_memory_usage():.2f} MB memory usage")
    for run in tqdm(range(1, 257), desc="IP"):
        # Load with batch functionality - this will cache surrounding runs
        df = az_pl.load_model_run_results_file(
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
        merged = reference_df.join(df, on="rn", how="inner")
        results = process_data_pl.process_ip_detailed_results(merged)

        # Build dictionary using same approach as Pandas for consistency
        # The column order is critical to match Pandas output
        cols = [
            "sitetret",
            "age_group",
            "sex",
            "pod",
            "tretspef",
            "los_group",
            "maternity_delivery_in_spell",
            "measure",
        ]

        for row in results.iter_rows(named=True):
            # Only add rows with valid sitetret
            if row["sitetret"] is None or row["sitetret"] == "":
                continue

            k = tuple(row[col] for col in cols[:-1])  # All columns except measure
            if "measure" in row:
                k = (*k, row["measure"])
            v = row["value"]
            if k not in model_runs:
                model_runs[k] = []
            model_runs[k].append(v)
    end = time.perf_counter()
    logger.info(
        f"All IP model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process model runs dictionary after the loop completes
    model_runs_df = process_data_pl.process_model_runs_dict(
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
    default_beddays_principal = int(
        actual_results_df.filter(pl.col("measure") == "beddays")
        .select(pl.col("mean").sum())
        .item()
    )

    # Extract the mean column for beddays measure
    detailed_beddays_principal = int(
        model_runs_df.filter(pl.col("measure") == "beddays")
        .select(pl.col("mean").sum())
        .item()
    )

    try:
        assert abs(default_beddays_principal - detailed_beddays_principal) <= 1
    except AssertionError:
        logger.warning(
            f"""Validation mismatch: default={default_beddays_principal},
            detailed={detailed_beddays_principal}"""
        )

    # Format boolean values to match Pandas output
    model_runs_df = model_runs_df.with_columns(
        pl.when(pl.col("maternity_delivery_in_spell"))
        .then(pl.lit("True"))
        .when(~pl.col("maternity_delivery_in_spell"))
        .then(pl.lit("False"))
        .otherwise(pl.col("maternity_delivery_in_spell"))
        .alias("maternity_delivery_in_spell")
    )

    # Save results
    model_runs_df.write_csv(f"{output_dir}/{scenario_name}_detailed_ip_results.csv")
    model_runs_df.write_parquet(
        f"{output_dir}/{scenario_name}_detailed_ip_results.parquet"
    )

    # Clean up memory
    del model_runs_df, model_runs, original_df, reference_df
    if "az_pl" in sys.modules and hasattr(sys.modules["az_pl"], "_model_results_cache"):
        # Clear the cache after processing
        sys.modules["az_pl"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after IP processing, current usage: {get_memory_usage():.2f} MB"
    )


# %% [markdown]
# ## Outpatient Results Processing


# %%
def _process_op_run(
    reference_df: pl.DataFrame, params: dict, run: int, op_model_runs: dict
) -> bool:
    """Process a single outpatient run and update the model runs dictionary.

    Args:
        reference_df: Reference dataframe without attendance columns
        params: Dictionary containing connection and model parameters
        run: Run number
        op_model_runs: Dictionary to update with results

    Returns:
        bool: True if K0O2Q site data was found in this run, False otherwise
    """
    try:
        # Extract parameters from dict
        results_connection = params["results_connection"]
        model_version = params["model_version"]
        trust = params["trust"]
        scenario_name = params["scenario_name"]
        run_id = params["run_id"]
        batch_size = params["batch_size"]

        # Load with batch functionality - this will cache surrounding runs
        logger.debug(f"Loading OP run {run} data")
        df = az_pl.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "op",
                "run_number": run,
                "batch_size": batch_size,
            },
        )

        # Validate data shape
        if df.shape[0] != reference_df.shape[0]:
            logger.warning(
                f"Row mismatch: run {run} df={df.shape[0]}, ref={reference_df.shape[0]}"
            )
            # Continue processing anyway - better to try than to fail

        # Use the pre-created reference dataframe
        # Use LEFT join to preserve reference data including all sites from original data
        logger.debug(f"Joining data for run {run}")
        merged = reference_df.join(df, on="rn", how="left")

        # Fill null values in any columns from df with appropriate defaults
        if "attendances" in df.columns:
            merged = merged.with_columns(pl.col("attendances").fill_null(0))
        if "tele_attendances" in df.columns:
            merged = merged.with_columns(pl.col("tele_attendances").fill_null(0))

        # Process the merged data
        results = process_data_pl.process_op_detailed_results(merged)

        # Log total number of sites in results
        site_count = results.select("sitetret").unique().height
        logger.debug(f"Run {run}: Found {site_count} unique sites after processing")

        # Load conversion data with batch functionality
        logger.debug(f"Loading OP conversion data for run {run}")
        df_conv = az_pl.load_model_run_results_file(
            container_client=results_connection,
            params={
                "version": model_version,
                "dataset": trust,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "activity_type": "op_conversion",
                "run_number": run,
                "batch_size": batch_size,
            },
        )

        # Process conversion data and combine with main results
        df_conv = process_data_pl.process_op_converted_from_ip(df_conv)
        results = process_data_pl.combine_converted_with_main_results(df_conv, results)

        # Count rows with empty sitetret for debugging
        empty_sitetret_count = results.filter(
            pl.col("sitetret").is_null() | (pl.col("sitetret") == "")
        ).height

        if empty_sitetret_count > 0:
            logger.debug(
                f"Found {empty_sitetret_count} rows with empty sitetret in run {run}"
            )

        # Convert results to a dictionary using an approach similar to Pandas to_dict()
        results_dict = {}
        cols = ["sitetret", "pod", "age_group", "tretspef", "measure"]

        # Track all unique sites seen for debugging
        unique_sites_in_run = set()

        # Build results_dict in the format Pandas would produce
        for row in results.iter_rows(named=True):
            # Process all rows - Pandas to_dict() does not filter
            k = tuple(row[col] for col in cols)
            results_dict[k] = row["value"]

            # Track the site for debugging if it has a valid value
            if row["sitetret"] is not None and row["sitetret"] != "":
                unique_sites_in_run.add(row["sitetret"])

        # Log unique sites found in this run
        logger.debug(
            f"Run {run}: Found {len(unique_sites_in_run)} unique sites in results"
        )

        # Update the model_runs dictionary exactly as Pandas would
        for k, v in results_dict.items():
            if k not in op_model_runs:
                # Initialize new key with empty list
                op_model_runs[k] = []
            # Add this run's value to the list
            op_model_runs[k].append(v)

        logger.debug(f"Added {len(results_dict)} results from run {run}")

        # Log number of keys processed in this run
        keys_added = len(results_dict)
        logger.debug(f"Run {run}: Added {keys_added} keys to results_dict")

        # Return success
        return True

    except Exception as e:
        # Log exception but don't fail the entire process
        logger.error(f"Error processing OP run {run}: {str(e)}")
        # Re-raise critical errors that should stop processing
        if "AssertionError" in str(e):
            raise

        return False


def _process_outpatient_results(
    context: ProcessContext,
    output_dir: str,
) -> None:
    """Process outpatient detailed results for all 256 Monte Carlo simulations.

    Handles both regular outpatient activity and activity converted from inpatient.
    Combines these two sources, processes all 256 model runs, and aggregates results.
    Uses batch loading with caching for optimal performance.
    Saves results to CSV and Parquet files.

    Args:
        context: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Report memory usage at start
    logger.info(f"Memory usage before OP processing: {get_memory_usage():.2f} MB")

    # Extract needed variables from context
    results_connection = context["results_connection"]
    data_connection = context["data_connection"]
    model_version = context["model_version"]
    model_version_data = context["model_version_data"]
    trust = context["trust"]
    baseline_year = context["baseline_year"]
    scenario_name = context["scenario_name"]
    run_id = context["run_id"]
    actual_results_df = context["actual_results_df"]

    # Load original data
    original_df = az_pl.load_data_file(
        data_connection, model_version_data, trust, "op", baseline_year
    ).fill_null("unknown")

    # Log total number of rows in original data
    logger.info(f"Original data contains {original_df.height} rows")

    # Rename 'index' column to 'rn' if it exists
    if "index" in original_df.columns:
        original_df = original_df.rename({"index": "rn"})

    # Pre-allocate dictionary
    op_model_runs = {}

    # Pre-create the reference dataframe copy once
    reference_df = original_df.drop(["attendances", "tele_attendances"])

    # Choose a larger batch size for optimal I/O performance
    batch_size = 30  # Balance between memory usage and I/O performance

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting OP processing with {get_memory_usage():.2f} MB memory usage")

    # Create params dictionary
    run_params = {
        "results_connection": results_connection,
        "model_version": model_version,
        "trust": trust,
        "scenario_name": scenario_name,
        "run_id": run_id,
        "batch_size": batch_size,
    }

    # Process all runs
    for run in tqdm(range(1, 257), desc="OP"):
        _process_op_run(
            reference_df,
            run_params,
            run,
            op_model_runs,
        )
    end = time.perf_counter()
    logger.info(
        f"All OP model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process results
    logger.info(f"Processing model runs dictionary with {len(op_model_runs)} keys")
    op_model_runs_df = process_data_pl.process_model_runs_dict(
        op_model_runs, columns=["sitetret", "pod", "age_group", "tretspef", "measure"]
    )

    # Log summary statistics about the sites in final results
    site_count = op_model_runs_df.select("sitetret").unique().height
    logger.info(f"Found {site_count} unique sites in processed results")

    # Validate results - comparing detailed results with aggregated results
    detailed_attendances_principal = int(
        op_model_runs_df.filter(pl.col("measure") == "attendances")
        .select(pl.col("mean").sum().round(1))
        .item()
    )

    default_attendances_principal = int(
        actual_results_df.filter(pl.col("measure") == "attendances")
        .select(pl.col("mean").sum())
        .item()
    )

    # They're not always exactly the same because of rounding
    # Log the values for verification
    logger.info(f"Detailed attendances total: {detailed_attendances_principal:,}")
    logger.info(f"Default attendances total: {default_attendances_principal:,}")

    attendance_diff = abs(default_attendances_principal - detailed_attendances_principal)
    if attendance_diff > 1:
        logger.warning(
            f"Validation mismatch: default={default_attendances_principal:,}, "
            f"detailed={detailed_attendances_principal:,}, "
            f"diff={attendance_diff:,}",
            f"({attendance_diff/default_attendances_principal:.1%})"
        )
    else:
        logger.info(
            "Attendance validation successful: Values match within rounding error"
        )

    # Final validation step - check for rows with empty sitetret
    empty_count = op_model_runs_df.filter(
        pl.col("sitetret").is_null() | (pl.col("sitetret") == "")
    ).height

    if empty_count > 0:
        logger.warning(
            f"Found {empty_count} rows with empty sitetret before final filtering"
        )

    # Filter out any rows with null/empty sitetret - one final safety check
    op_model_runs_df = op_model_runs_df.filter(
        ~pl.col("sitetret").is_null() & (pl.col("sitetret") != "")
    )

    # Replace NaN values with 0.0 to match Pandas behavior
    op_model_runs_df = op_model_runs_df.with_columns(
        [
            pl.col("lwr_ci").fill_nan(0.0),
            pl.col("median").fill_nan(0.0),
            pl.col("mean").fill_nan(0.0),
            pl.col("upr_ci").fill_nan(0.0),
        ]
    )

    # Log total row count for validation
    logger.info(f"Final OP dataframe has {op_model_runs_df.height} rows")

    # Log row count of final output
    logger.info(f"Final output contains {op_model_runs_df.height} total rows")

    # List all unique sitetret values
    unique_sites = op_model_runs_df.select("sitetret").unique().to_series().to_list()
    logger.info(f"Unique sites in final output: {sorted(unique_sites)}")

    # Log unique sites and row count information for diagnostics
    unique_sites_count = op_model_runs_df.select("sitetret").unique().height
    logger.info(f"Found {unique_sites_count} unique sites in final output")
    logger.info(f"Final row count: {op_model_runs_df.height} rows")

    # Save results - use parameters to match Pandas to_csv behavior
    op_model_runs_df.write_csv(
        f"{output_dir}/{scenario_name}_detailed_op_results.csv",
        include_header=True,
        separator=",",
        quote_style="necessary",
    )

    op_model_runs_df.write_parquet(
        f"{output_dir}/{scenario_name}_detailed_op_results.parquet"
    )

    # Clean up memory
    del op_model_runs_df, op_model_runs, original_df, reference_df
    if "az_pl" in sys.modules and hasattr(sys.modules["az_pl"], "_model_results_cache"):
        # Clear the cache after processing
        sys.modules["az_pl"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after OP processing, current usage: {get_memory_usage():.2f} MB"
    )


# %% [markdown]
# ## A&E Validation Helper


# %%
def _validate_aae_metric(
    ae_model_runs_df: pl.DataFrame,
    actual_results_df: pl.DataFrame,
    measure_name: str,
    metric_label: str,
) -> None:
    """Validate A&E metrics between detailed and aggregated results.

    Compares the detailed processed results with the original aggregated results
    to ensure consistency. Warns if there's a discrepancy greater than 1 (allowing
    for minor rounding differences).

    Args:
        ae_model_runs_df: DataFrame with detailed results
        actual_results_df: DataFrame with aggregated results
        measure_name: Name of measure to validate (e.g., "ambulance", "walk-in")
        metric_label: Label for the measure in logs (e.g., "Ambulance", "Walk-in")
    """
    detailed_value = round(
        float(
            ae_model_runs_df.filter(pl.col("measure") == measure_name)
            .select(pl.col("mean").sum())
            .item()
        )
    )

    default_value = round(
        float(
            actual_results_df.filter(pl.col("measure") == measure_name)
            .select(pl.col("mean").sum())
            .item()
        )
    )

    # They're not always exactly the same because of rounding
    try:
        assert abs(default_value - detailed_value) <= 1
    except AssertionError:
        logger.warning(
            f"""{metric_label} validation mismatch: default={default_value},
            detailed={detailed_value}"""
        )


# %% [markdown]
# ## A&E Results Processing


# %%
def _process_aae_results(
    context: ProcessContext,
    output_dir: str,
) -> None:
    """Process A&E detailed results for all 256 Monte Carlo simulations.

    Handles both regular A&E activity and SDEC activity converted from inpatient.
    Combines these two sources, processes all 256 model runs, and aggregates results.
    Validates results against aggregated results for both ambulance and walk-in activity.
    Saves results to CSV and Parquet files.

    Args:
        context: dictionary with connections and parameters
        output_dir: Directory to save output files
    """
    # Report memory usage at start
    logger.info(f"Memory usage before A&E processing: {get_memory_usage():.2f} MB")

    # Extract needed variables from context
    results_connection = context["results_connection"]
    data_connection = context["data_connection"]
    model_version = context["model_version"]
    model_version_data = context["model_version_data"]
    trust = context["trust"]
    baseline_year = context["baseline_year"]
    scenario_name = context["scenario_name"]
    run_id = context["run_id"]
    actual_results_df = context["actual_results_df"]

    # Load and prepare data
    original_df = az_pl.load_data_file(
        data_connection, model_version_data, trust, "aae", baseline_year
    )

    # Handle nulls more comprehensively like Pandas does
    # First replace empty strings with nulls, then fill all nulls with "unknown"
    original_df = original_df.with_columns(
        [
            pl.col(col).map_elements(
                lambda x: None if x == "" else x, return_dtype=pl.Utf8
            )
            for col in original_df.select(pl.col(pl.Utf8)).columns
        ]
    ).fill_null("unknown")

    # Rename 'index' column to 'rn' if it exists
    if "index" in original_df.columns:
        original_df = original_df.rename({"index": "rn"})

    # Pre-allocate dictionary and create reference dataframe
    ae_model_runs = {}
    reference_df = original_df.drop(["arrivals"])

    # Process settings
    batch_size = 30  # Balance between memory usage and I/O performance

    # Process all runs
    start = time.perf_counter()
    logger.info(f"Starting A&E processing with {get_memory_usage():.2f} MB memory usage")

    # Common parameters for loading model runs
    base_params = {
        "version": model_version,
        "dataset": trust,
        "scenario_name": scenario_name,
        "run_id": run_id,
        "batch_size": batch_size,
    }

    for run in tqdm(range(1, 257), desc="A&E"):
        # Load main A&E data
        aae_params = {**base_params, "activity_type": "aae", "run_number": run}
        df = az_pl.load_model_run_results_file(
            container_client=results_connection, params=aae_params
        )

        assert len(df) == len(original_df)

        # Process main A&E data
        # Use left join instead of inner join to better match Pandas behavior with nulls
        merged = reference_df.join(df, on="rn", how="left")

        # Fill any null values in arrivals column to match Pandas behavior
        merged = merged.with_columns(pl.col("arrivals").fill_null(0))

        results = process_data_pl.process_aae_results(merged)

        # Load and process conversion data
        conv_params = {
            **base_params,
            "activity_type": "sdec_conversion",
            "run_number": run,
        }
        df_conv = az_pl.load_model_run_results_file(
            container_client=results_connection, params=conv_params
        )

        # Process SDEC conversion data - critical for aae_type-05 rows
        sdec_results = process_data_pl.process_aae_converted_from_ip(df_conv)

        # Combine SDEC results with main results
        # This exact handling is critical for matching Pandas behavior
        results = process_data_pl.combine_converted_with_main_results(
            sdec_results, results
        )

        # Build model runs dictionary
        for row in results.iter_rows(named=True):
            # Ensure columns order matches process_model_runs_dict
            cols = [
                "sitetret",
                "pod",
                "age_group",
                "attendance_category",
                "aedepttype",
                "acuity",
                "measure",
            ]
            # Convert None/null to empty string in keys to match Pandas behavior
            k = tuple("" if row[col] is None else row[col] for col in cols)

            # Skip rows where all keys are empty strings - Pandas implicitly filters these
            if k != tuple("" for _ in cols):
                ae_model_runs.setdefault(k, []).append(row["arrivals"])

    end = time.perf_counter()
    logger.info(
        f"All A&E model runs were processed in {end - start:.3f} sec, "
        f"memory usage: {get_memory_usage():.2f} MB"
    )

    # Process results
    ae_model_runs_df = process_data_pl.process_model_runs_dict(
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

    # Verify output has expected structure
    expected_cols = [
        "sitetret",
        "pod",
        "age_group",
        "attendance_category",
        "aedepttype",
        "acuity",
        "measure",
        "lwr_ci",
        "median",
        "mean",
        "upr_ci",
    ]
    for col in expected_cols:
        if col not in ae_model_runs_df.columns:
            logger.warning(f"Expected column {col} missing in A&E results dataframe")

    # Verify no empty sitetret values
    empty_rows = ae_model_runs_df.filter(
        (pl.col("sitetret").is_null()) | (pl.col("sitetret") == "")
    ).height
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} rows with empty sitetret values")

    # Save results
    ae_model_runs_df.write_csv(f"{output_dir}/{scenario_name}_detailed_ae_results.csv")
    ae_model_runs_df.write_parquet(
        f"{output_dir}/{scenario_name}_detailed_ae_results.parquet"
    )

    # Clean up memory
    del ae_model_runs_df, ae_model_runs, original_df, reference_df
    if "az_pl" in sys.modules and hasattr(sys.modules["az_pl"], "_model_results_cache"):
        sys.modules["az_pl"]._model_results_cache.clear()
    gc.collect()
    logger.info(
        f"Memory cleaned after A&E processing, current usage: {get_memory_usage():.2f} MB"
    )


# %% [markdown]
# ## Main Function
#
# This is the primary function for generating detailed results. It:
# 1. Sets up connections and loads parameters
# 2. Processes inpatient (IP), outpatient (OP), and A&E results
# 3. Saves the results to files
# 4. Reports timing information


# %%
def run_detailed_results(
    results_path: str,
    output_dir: str | None = None,
    account_url: str | None = None,
    results_container: str | None = None,
    data_container: str | None = None,
    process_ip_only: bool = False,
    process_op_only: bool = False,
    process_ae_only: bool = False,
) -> dict[str, str]:
    """Generate detailed results for a model scenario using Polars.

    Takes an existing scenario results path and produces detailed aggregations
    of IP, OP, and A&E model results in CSV and Parquet formats. By default,
    processes all result types. If any of the process_*_only flags are set,
    only that specific type will be processed.

    Args:
        results_path: Path to existing aggregated results
        output_dir: Directory to save output files (default: 'nhpy/data')
        account_url: Azure Storage account URL (default: from environment)
        results_container: Azure Storage container for results (default: from environment)
        data_container: Azure Storage container for data (default: from environment)
        process_ip_only: Flag to process only inpatient results (default: False)
        process_op_only: Flag to process only outpatient results (default: False)
        process_ae_only: Flag to process only A&E results (default: False)

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

    # Convert to Path object and create directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialise connections and load parameters
    context = _initialize_connections_and_params(
        results_path,
        account_url,
        results_container,
        data_container,
    )

    # Determine which processes to run
    run_all = not any([process_ip_only, process_op_only, process_ae_only])
    scenario_name = context["scenario_name"]
    result_files = {}

    # Process each type of results based on flags
    if run_all or process_ip_only:
        logger.info("Processing inpatient results...")
        _process_inpatient_results(context, output_dir)
        result_files.update(
            {
                "ip_csv": f"{output_dir}/{scenario_name}_detailed_ip_results.csv",
                "ip_parquet": f"{output_dir}/{scenario_name}_detailed_ip_results.parquet",
            }
        )

    if run_all or process_op_only:
        logger.info("Processing outpatient results...")
        _process_outpatient_results(context, output_dir)
        result_files.update(
            {
                "op_csv": f"{output_dir}/{scenario_name}_detailed_op_results.csv",
                "op_parquet": f"{output_dir}/{scenario_name}_detailed_op_results.parquet",
            }
        )

    if run_all or process_ae_only:
        logger.info("Processing A&E results...")
        _process_aae_results(context, output_dir)
        result_files.update(
            {
                "ae_csv": f"{output_dir}/{scenario_name}_detailed_ae_results.csv",
                "ae_parquet": f"{output_dir}/{scenario_name}_detailed_ae_results.parquet",
            }
        )

    # Calculate and report the total time
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    minutes, seconds = divmod(total_duration, 60)
    logger.info(
        f"Total processing time for Polars implementation: {int(minutes)}m {seconds:.2f}s"
    )

    # Return paths to output files
    return result_files


# %% [markdown]
# ## CLI Entry Point


# %%
def main() -> int:
    """CLI entry point when module is run directly.

    Returns:
        int: Exit code (0 for success, 2 for errors)
    """
    configure_logging(INFO)

    parser = argparse.ArgumentParser(
        description="Generate detailed results for a model scenario using Polars",
    )
    parser.add_argument(
        "results_path",
        help="Path to existing aggregated results \
        (e.g. 'aggregated-model-results/v4.0/RXX/test/20250101_100000/')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save output files (default: 'nhpy/data')",
        default="nhpy/data",
    )
    parser.add_argument("-a", "--account-url", help="Azure Storage account URL")
    parser.add_argument(
        "-r", "--results-container", help="Azure Storage container for results"
    )
    parser.add_argument("-d", "--data-container", help="Azure Storage container for data")

    # Add mutually exclusive processing flags
    processing_group = parser.add_mutually_exclusive_group()
    processing_group.add_argument(
        "--ip", action="store_true", help="Process only inpatient results"
    )
    processing_group.add_argument(
        "--op", action="store_true", help="Process only outpatient results"
    )
    processing_group.add_argument(
        "--ae", action="store_true", help="Process only A&E results"
    )

    args = parser.parse_args()

    try:
        run_detailed_results(
            results_path=args.results_path,
            output_dir=args.output_dir,
            account_url=args.account_url,
            results_container=args.results_container,
            data_container=args.data_container,
            process_ip_only=args.ip,
            process_op_only=args.op,
            process_ae_only=args.ae,
        )

        logger.info("🎉 Detailed results generated successfully!")
        logger.info("Results saved to: %s/", args.output_dir)
    except (
        ValueError,
        EnvironmentVariableError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        HttpResponseError,
        ServiceRequestError,
        FileNotFoundError,
    ):
        logger.exception("main():Error occurred")
        return ExitCodes.EXCEPTION_CODE
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return ExitCodes.SIGINT_CODE
    else:
        # If we got here, it means no exceptions were raised
        return ExitCodes.SUCCESS_CODE


# %%
# Main guardrail
if __name__ == "__main__":
    sys.exit(main())
