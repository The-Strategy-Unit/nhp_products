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


def add_baseline_to_detailed_results(
    results_paths: dict[str, str], context: ProcessContext, agg_type: str, output_dir: str
):
    """Adds baseline to detailed results (HRG type) and does not suppress them

    Args:
        results_paths (dict[str, str]): Results paths, returned from
        nhpy.run_detailed_results.run_detailed_results
        context (ProcessContext): Processing context with connection information and metadata
        agg_type (str): Type of aggregation: hrg or standard
        output_dir (str): Output directory to save results in
    """
    for activity_type in ["ip", "op", "aae"]:
        if agg_type == "standard":
            raise NotImplementedError(
                "Including baseline not implemented for standard \
                agg type. Use custom_baseline_standard module instead"
            )
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
            .copy()
        )
        unsuppressed_with_baseline.to_csv(
            f"{output_dir}/{context['scenario_name']}_detailed_{activity_type}_with_baseline.csv"
        )
        unsuppressed_with_baseline.to_parquet(
            f"{output_dir}/{context['scenario_name']}_detailed_{activity_type}_with_baseline.parquet"
        )
    logger.info("🎉 Baseline added successfully!")
    logger.info(f"Results saved to: {output_dir}/")
