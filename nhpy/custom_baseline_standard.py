import argparse
import os
import sys
from collections import defaultdict
from logging import INFO
from typing import List

import pandas as pd
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from tqdm import tqdm

from nhpy import az, process_data, process_results
from nhpy.az import load_data_file
from nhpy.config import (
    DetailedResultsConfig,
    DetailedResultsHRG,
    DetailedResultsStandard,
    ExitCodes,
)
from nhpy.process_data import (
    add_pod_to_data_op,
    filter_sites,
    process_aae_results,
    process_ip_detailed_results,
    process_ip_results,
    process_op_detailed_results,
)
from nhpy.types import ProcessContext
from nhpy.utils import (
    EnvironmentVariableError,
    _load_dotenv_file,
    configure_logging,
    get_logger,
    initialise_connections_and_params,
)

logger = get_logger()


def suppress_small_counts(
    df: pd.DataFrame,
    full_index_cols: list[str],  # to help return/reorder cols we need in output
    suppress_group_cols: list[
        str
    ],  # the columns to aggregate by (a subset of full_index_cols)
    grouped_value="grouped",  # value to put in columns not part of suppress_group_cols
    threshold=5,  # consider for suppression if less than this
) -> pd.DataFrame:
    """Function to suppress small counts where values are < 5

    Args:
        df (pd.DataFrame): DataFrame with data to be suppressed
        full_index_cols (list[str]): To help return/reorder cols we need in output

    Returns:
        pd.DataFrame: Suppressed dataframe
    """

    df = df.reset_index()

    keep = df[df["value"] >= threshold].copy()
    small = df[df["value"] < threshold].copy()

    # avoid suppression if we don't need to
    if small.empty:
        return df.set_index(full_index_cols).sort_index()

    # reaggregate by subset of columns
    grouped = (
        small.groupby(suppress_group_cols, dropna=False)["value"].sum().reset_index()
    )

    # identify columns to reintroduce, fill with string, order as per original
    collapsed_cols = set(full_index_cols) - set(suppress_group_cols)
    for col in collapsed_cols:
        grouped[col] = grouped_value
    grouped = grouped[full_index_cols + ["value"]]

    # bind suppressed rows with the ones that didn't need suppression
    result = pd.concat([keep, grouped], ignore_index=True)

    return result.sort_values(by="value").reset_index(drop=True)


def add_custom_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom age groups for custom detailed baseline (standard agg type)

    Args:
        df (pd.DataFrame): DataFrame with "age" column

    Returns:
        pd.DataFrame: DataFrame with new "custom_age_groups" column
    """
    age_groups = defaultdict(
        lambda: "Adults >= 65",
        {
            **{i: "Paeds < 10" for i in range(0, 10)},
            **{i: "Paeds 10-17" for i in range(10, 18)},
            **{i: "Adults 18-64" for i in range(18, 65)},
        },
    )
    df["custom_age_groups"] = df["age"].map(age_groups)
    return df


def process_custom_ip_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Process IP data to produce custom baseline (standard agg type)

    Args:
        df (pd.DataFrame): IP data

    Returns:
        pd.DataFrame: IP data with custom_age_groups and custom_los_groups for custom
        baseline (standard agg type)
    """
    df = add_custom_age_groups(df)
    custom_los_groups = defaultdict(
        lambda: "15+ days",
        {
            0: "0 days",
            **{i: "1-6 days" for i in range(1, 7)},
            **{i: "7-14 days" for i in range(7, 15)},
        },
    )
    df = process_ip_results(df, custom_los_groups).rename(
        columns={"los_group": "custom_los_groups"}
    )
    grouped_data = df.groupby(
        [
            "sitetret",
            "custom_age_groups",
            "custom_los_groups",
            "tretspef",
            "pod",
            "maternity_delivery_in_spell",
        ],
        dropna=False,
    )[["admissions", "beddays", "procedures"]].sum()
    data = pd.DataFrame(grouped_data)
    data = data.melt(ignore_index=False, var_name="measure")
    data = data[data["value"] > 0].reset_index()
    data = data.groupby(
        [
            "sitetret",
            "custom_age_groups",
            "custom_los_groups",
            "tretspef",
            "pod",
            "maternity_delivery_in_spell",
            "measure",
        ]
    ).sum()
    return data.sort_values(by="value")


def process_custom_op_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Process OP data to produce custom baseline (standard agg type)

    Args:
        df (pd.DataFrame): OP data

    Returns:
        pd.DataFrame: OP data with custom_age_groups and custom_los_groups for custom
        baseline (standard agg type)
    """
    data = add_pod_to_data_op(data)
    data = add_custom_age_groups(data)
    measures = data.melt(["index"], ["attendances", "tele_attendances"], "measure")
    grouped_data = (
        data.merge(measures, on="index")
        .groupby(
            ["sitetret", "custom_age_groups", "tretspef", "pod", "measure"],
        )[["value"]]
        .sum()
        .sort_index()
    )
    return grouped_data


def process_custom_aae_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Process AAE data to produce custom baseline (standard agg type)

    Args:
        df (pd.DataFrame): AAE data

    Returns:
        pd.DataFrame: AAE data with custom_age_groups and custom_los_groups for custom
        baseline (standard agg type)
    """
    data["measure"] = "walk-in"
    data.loc[data["is_ambulance"], "measure"] = "ambulance"
    data = add_custom_age_groups(data)
    grouped_data = (
        data.groupby(
            [
                "sitetret",
                "pod",
                "custom_age_groups",
                "aedepttype",
                "acuity",
                "measure",
            ]
        )[["arrivals"]]
        .sum()
        .rename(columns={"arrivals": "value"})
    )
    return grouped_data


def produce_custom_suppressed_baseline(
    context: ProcessContext,
    agg_type: str,
    output_dir: str,
    sites_filter_dict: dict[str, list] = {"ip": ["ALL"], "op": ["ALL"], "aae": ["ALL"]},
):
    """Adds baseline to detailed results and suppresses rows where counts are <5 in
    baseline, grouping together by the suppress_cols

    Args:
        context (ProcessContext): Processing context with connection information and
        metadata
        agg_type (str): Type of aggregation: hrg or standard. ONLY WORKS FOR STANDARD
        output_dir (str): Output directory to save results in
        sites_filter_dict (dict[str, list]): sites to filter each activity_type by.
        Example:
        {"ip": ["ALL"], "op": ["ALL"], "aae": ["SITEA", "SITEB"]}
    """
    for activity_type in ["ip", "op", "aae"]:
        if agg_type == "standard":
            config = DetailedResultsStandard()
        else:
            raise NotImplementedError("Not implemented for HRG aggregation")
        baseline = load_data_file(
            container_client=context["data_connection"],
            version=context["model_version_data"],
            dataset=context["trust"],
            activity_type=activity_type,
            year=context["baseline_year"],
        )
        baseline_filtered = filter_sites(baseline, sites_filter_dict[activity_type])
        if activity_type == "ip":
            baseline_processed = process_custom_ip_baseline(baseline_filtered)
            full_index_cols = [
                "sitetret",
                "custom_age_groups",
                "custom_los_groups",
                "tretspef",
                "pod",
                "maternity_delivery_in_spell",
                "measure",
            ]
            suppress_group_cols = ["pod", "measure"]
        if activity_type == "op":
            baseline_processed = process_custom_op_baseline(baseline_filtered)
            full_index_cols = [
                "sitetret",
                "custom_age_groups",
                "tretspef",
                "pod",
                "measure",
            ]
            suppress_group_cols = ["pod"]
        if activity_type == "aae":
            baseline_processed = process_custom_aae_baseline(baseline_filtered)
            full_index_cols = [
                "sitetret",
                "pod",
                "custom_age_groups",
                "aedepttype",
                "acuity",
                "measure",
            ]
            suppress_group_cols = ["pod"]
        baseline_suppressed = suppress_small_counts(
            baseline_processed, full_index_cols, suppress_group_cols
        )

        baseline_suppressed.to_csv(
            f"{output_dir}/{context['scenario_name']}_custom_{activity_type}_baseline.csv",
            index=False,
        )
    logger.info("🎉 Custom detailed baseline (agg type standard) created!")
    logger.info(f"Results saved to: {output_dir}/")


def compile_sites_filter_dict(
    ip_sites: str, op_sites: str, aae_sites: str
) -> dict[str, list]:
    """Compiles sites filter dictionary from supplied command line arguments

    Args:
        ip_sites (str): Inpatients sites to filter baseline to
        op_sites (str): Outpatients sites to filter baseline to
        aae_sites (str): A&E sites to filter baseline to

    Returns:
        dict[str, list]: compiled sites filter dict for use in
        produce_custom_suppressed_baseline
    """
    return {
        "ip": ip_sites.upper().split(","),
        "op": op_sites.upper().split(","),
        "aae": aae_sites.upper().split(","),
    }


def main() -> int:
    """
    CLI entry point when module is run directly.

    Returns:
        int: Exit code (0 for success, 2 for errors)
    """
    configure_logging(INFO)

    parser = argparse.ArgumentParser(
        description="Generate custom detailed baseline (standard agg type)"
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
    parser.add_argument(
        "--ip_sites",
        help="Sites to filter IP data to",
        default="ALL",
    )
    parser.add_argument(
        "--op_sites",
        help="Sites to filter OP data to",
        default="ALL",
    )
    parser.add_argument(
        "--aae_sites",
        help="Sites to filter AAE data to",
        default="ALL",
    )
    parser.add_argument("--account-url", help="Azure Storage account URL")
    parser.add_argument("--results-container", help="Azure Storage container for results")
    parser.add_argument("--data-container", help="Azure Storage container for data")
    args = parser.parse_args()

    # Initialise context and connections
    account_url = args.account_url or os.getenv("AZ_STORAGE_EP", "")
    results_container = args.results_container or os.getenv("AZ_STORAGE_RESULTS", "")
    data_container = args.data_container or os.getenv("AZ_STORAGE_DATA", "")

    # Initialise context just to get scenario name
    context = initialise_connections_and_params(
        args.results_path, account_url, results_container, data_container
    )

    sites_filter_dict = compile_sites_filter_dict(
        args.ip_sites, args.op_sites, args.aae_sites
    )

    try:
        produce_custom_suppressed_baseline(
            context, "standard", args.output_dir, sites_filter_dict
        )
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


if __name__ == "__main__":
    sys.exit(main())
