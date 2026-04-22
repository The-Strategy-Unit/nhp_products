import argparse
from typing import List

import pandas as pd

from nhpy.az import load_data_file
from nhpy.config import DetailedResultsConfig, DetailedResultsHRG, DetailedResultsStandard
from nhpy.process_data import (
    process_aae_results,
    process_ip_detailed_results,
    process_op_detailed_results,
)
from nhpy.types import ProcessContext
from nhpy.utils import (
    get_logger,
)

logger = get_logger()


def suppress_small_counts(
    df: pd.DataFrame,
    suppress_cols: List[str],
    count_col: str = "baseline",
    threshold: int = 5,
) -> pd.DataFrame:
    """Suppression of small counts in detailed results. Filters dataframe to only rows with
    small numbers, then groups together values in specified columns and re-aggregates.

    Args:
        df (pd.DataFrame): Processed detailed results
        suppress_cols (List[str]): List of columns to use in suppressing small numbers,
        in order of preference.
        count_col (str, optional): The name of the column with the values to be suppressed
        and aggregated.
        Defaults to "value".
        threshold (int, optional): Maximum count allowed in the count_col

    Returns:
        pd.DataFrame: Processed detailed results with small counts aggregated together
    """
    logger.info("Beginning suppression...")
    full_index_cols = df.index.names.copy()
    for col in suppress_cols:
        df = df.reset_index()
        df[col] = df[col].astype(str)
        keep = df[df[count_col] >= threshold].copy()
        small = df[df[count_col] < threshold].copy()
        # avoid suppression if we don't need to
        if small.empty:
            df = df.groupby(full_index_cols).sum()
            break
        small[col] = "grouped"
        # reaggregate
        grouped = small.groupby(by=full_index_cols, dropna=False).sum().reset_index()
        # bind suppressed rows with the ones that didn't need suppression
        df = pd.concat([keep, grouped], ignore_index=True).groupby(full_index_cols).sum()
    return df.sort_index().sort_index(axis=1)


def add_baseline_to_detailed_results(
    results_paths: dict[str, str], context: ProcessContext, agg_type: str, output_dir: str
):
    """Adds baseline to detailed results and suppresses rows where counts are <5 in baseline,
    grouping together by the suppress_cols

    Args:
        results_paths (dict[str, str]): Results paths, returned from
        nhpy.run_detailed_results.run_detailed_results
        context (ProcessContext): Processing context with connection information and metadata
        agg_type (str): Type of aggregation: hrg or standard
        output_dir (str): Output directory to save results in
    """
    suppress_cols = {
        "ip": [
            "sushrg",
            "tretspef",
            "maternity_delivery_in_spell",
            "age_group",
            "sitetret",
        ],
        "op": ["tretspef", "age_group", "sitetret"],
        "aae": ["age_group", "attendance_category", "aedepttype", "sitetret"],
    }  # TODO: This is currently hardcoded to only work for the hrg agg_type
    for activity_type in ["ip", "op", "aae"]:
        if agg_type == "standard":
            config = DetailedResultsStandard()
        if agg_type == "hrg":
            config = DetailedResultsHRG()
        detailed_results = pd.read_parquet(results_paths[f"{activity_type}_parquet"])
        baseline = load_data_file(
            container_client=context["data_connection"],
            version=context["model_version_data"],
            dataset=context["trust"],
            activity_type=activity_type,
            year=context["baseline_year"],
        )
        if config.custom_age_groups:
            baseline["age_group"] = config.age_groups(baseline["age"])
        if activity_type == "ip":
            baseline_processed = process_ip_detailed_results(baseline, config.ip_agg_cols)
        if activity_type == "op":
            baseline_processed = process_op_detailed_results(
                baseline.rename(columns={"index": "rn"}), config.op_agg_cols
            )
        if activity_type == "aae":
            baseline_processed = process_aae_results(
                baseline, config.aae_agg_cols
            ).rename(columns={"arrivals": "value"})
        baseline_processed = baseline_processed.rename(columns={"value": "baseline"})
        unsuppressed_with_baseline = (
            detailed_results.merge(
                baseline_processed, left_index=True, right_index=True, how="outer"
            )
            .fillna(0)
            .drop(columns=["lwr_ci", "median", "upr_ci"])
            .copy()
        )
        suppressed_with_baseline = suppress_small_counts(
            unsuppressed_with_baseline,
            suppress_cols=suppress_cols[activity_type],
        )
        # TODO: ADD IN SITES FILTER
        suppressed_with_baseline.to_csv(
            f"{output_dir}/{context['scenario_name']}_detailed_{activity_type}_with_baseline.csv"
        )
        suppressed_with_baseline.to_parquet(
            f"{output_dir}/{context['scenario_name']}_detailed_{activity_type}_with_baseline.parquet"
        )
    logger.info("🎉 Baseline added successfully!")
    logger.info(f"Results saved to: {output_dir}/")
