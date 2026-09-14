"""
QA data checks for NHP model data (>= v3.0).

Compares the `dev` version of NHP model data against the latest released
version in Azure Blob Storage, across inpatients, outpatients, A&E activity,
and mitigator strategies (activity avoidance / efficiencies). Writes a
comparison CSV per activity type / mitigator set to a results directory, and
logs a warning for any activity type where dev and the latest release don't
match exactly.

This replaces the Databricks notebook at
notebooks/QA_prerelease-data-checks/QA_data_checks.ipynb, which relied on
`/Volumes/nhp/model_data/files/...` (a Databricks Unity Catalog volume) and
`dbutils`. This module reads the same data from Azure Blob Storage directly,
using az.py, so it can run locally.

Usage:
    # CLI usage
    uv run python -m nhpy.qa.data
    uv run python -m nhpy.qa.data --year 2024 --output-dir results
    uv run python -m nhpy.qa.data --compare-version v3.4.0

    # Programmatic usage
    from nhpy.qa.data import run_qa_checks
    run_qa_checks(year=2024, output_dir="results")

Configuration:
    Set environment variables: AZ_STORAGE_EP, AZ_STORAGE_DATA
    Or provide credentials via function arguments or CLI options.
    Authentication uses DefaultAzureCredential (e.g. `az login`).

Exit codes:
    0: Ran successfully, dev and latest release matched everywhere
    1: Ran successfully, but one or more mismatches were found (see logs/CSVs)
    2: Error occurred (authentication, network, missing data, etc.)
    130: Operation cancelled (Ctrl+C)
"""

# %%
import argparse
import os
import sys
from datetime import datetime
from logging import INFO
from pathlib import Path

import pandas as pd
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceNotFoundError,
)

from nhpy.az import (
    connect_to_container,
    find_latest_version,
    get_azure_credentials,
    list_all_versions,
    load_parquet_file,
)
from nhpy.config import EmptyContainerError, ExitCodes
from nhpy.process_data import get_all_ae_mitigators, get_all_op_mitigators
from nhpy.utils import _load_dotenv_file, configure_logging, get_logger

# %%
# Load environment variables
_load_dotenv_file()

# %%
__all__ = ["run_qa_checks"]

# %%
logger = get_logger()

DEFAULT_YEAR = 2024


# %%
# --- Azure helpers (folder-level read/list on top of az.py) --------------
#
# az.py's `load_parquet_file` loads a single blob, and `load_data_file` only
# grabs the first matching file for a dataset partition. The data here is
# written as one-or-more parquet part-files per partition, so we need to
# list and concatenate all files under a prefix - equivalent to what
# `pd.read_parquet()` does automatically for a directory on Databricks.


def _read_partition(container_client, prefix: str) -> pd.DataFrame:
    """Read and concatenate all parquet files found under a blob prefix.

    Args:
        container_client: Azure ContainerClient
        prefix: Blob name prefix, e.g. "dev/ip/fyear=2024/dataset=RXX/"

    Returns:
        pd.DataFrame: Concatenation of every parquet file under the prefix

    Raises:
        EmptyContainerError: If no parquet files are found under the prefix
    """
    names = [
        name
        for name in container_client.list_blob_names(name_starts_with=prefix)
        if name.endswith(".parquet")
    ]
    if not names:
        raise EmptyContainerError(
            container_name=container_client.container_name, path=prefix
        )
    frames = [load_parquet_file(container_client, name) for name in names]
    return pd.concat(frames, ignore_index=True)


def _list_datasets(
    container_client, version: str, activity_type: str, year: int
) -> list[str]:
    """List distinct trust/dataset names for a version + activity_type + year.

    Parses `dataset=<name>` segments out of blob names rather than downloading
    data, e.g. from "dev/ip/fyear=2024/dataset=RXX/0.parquet" -> "RXX".
    """
    prefix = f"{version}/{activity_type}/fyear={year}/dataset="
    datasets = set()
    for name in container_client.list_blob_names(name_starts_with=prefix):
        rest = name[len(prefix) :]
        dataset = rest.split("/", 1)[0]
        if dataset:
            datasets.add(dataset)
    if not datasets:
        raise EmptyContainerError(
            container_name=container_client.container_name, path=prefix
        )
    return sorted(datasets)


def _latest_version(container_client) -> str:
    """Find the latest `vX.Y.Z`-style version folder at the container root."""
    versions = [v for v in list_all_versions(container_client) if v.startswith("v")]
    if not versions:
        raise EmptyContainerError(
            container_name=container_client.container_name, path="/"
        )
    # sorts correctly for simple vMAJOR.MINOR.PATCH strings of equal width;
    # falls back to lexicographic order otherwise
    try:
        from packaging.version import Version

        return max(versions, key=lambda v: Version(v.lstrip("v")))
    except Exception:
        return sorted(versions)[-1]


# %%
# --- Aggregation functions (unchanged logic from the original notebook) --


def _aggregate_data_ip(data: pd.DataFrame, version: str) -> pd.DataFrame:
    data = data.copy()
    data["beddays"] = data["speldur"] + 1
    return (
        data.groupby("pod")
        .agg({"rn": "count", "beddays": "sum"})
        .rename(columns={"rn": f"{version}_admissions", "beddays": f"{version}_beddays"})
    )


def _aggregate_data_op(data: pd.DataFrame, version: str) -> pd.DataFrame:
    return (
        data.groupby("pod")
        .agg({"attendances": "sum", "tele_attendances": "sum"})
        .rename(
            columns={
                "attendances": f"{version}_attendances",
                "tele_attendances": f"{version}_tele_attendances",
            }
        )
    )


def _aggregate_data_aae(data: pd.DataFrame, version: str) -> pd.DataFrame:
    return (
        data.groupby(["pod", "group"])
        .agg({"arrivals": "sum"})
        .rename(columns={"arrivals": f"{version}_arrivals"})
    )


# %%
# --- Comparison builders ---------------------------------------------------


def _compare_activity(
    container_client,
    versions: list[str],
    year: int,
    activity_type: str,
    aggregate_fn,
) -> pd.DataFrame:
    """Build a per-trust, per-pod comparison table for ip/op/aae activity."""
    trusts = _list_datasets(container_client, versions[0], activity_type, year)
    data_dict = {}
    for trust in trusts:
        df = None
        for version in versions:
            prefix = f"{version}/{activity_type}/fyear={year}/dataset={trust}/"
            data = _read_partition(container_client, prefix)
            agg = aggregate_fn(data, version)
            if df is None:
                df = agg.copy()
            else:
                df = df.merge(agg, left_index=True, right_index=True, how="outer").fillna(
                    0
                )
                numeric_cols = df.select_dtypes(include="number").columns
                df[numeric_cols] = df[numeric_cols].astype(int)
        df["trust"] = trust
        data_dict[trust] = df
    index_cols = ["trust", "pod"] if activity_type != "aae" else ["trust", "pod", "group"]
    full_df = pd.concat(data_dict.values()).reset_index().set_index(index_cols)
    return full_df


def _check_columns_match(
    df: pd.DataFrame, dev_col: str, other_col: str, label: str
) -> bool:
    matches = df[dev_col].equals(df[other_col])
    if not matches:
        logger.warning(f"Mismatch found in {label}: '{dev_col}' vs '{other_col}'")
    return matches


def _compare_ip_mitigators(
    container_client, versions: list[str], year: int
) -> pd.DataFrame:
    mitigator_dict = {}
    for version in versions:
        aa = (
            _read_partition(
                container_client,
                f"{version}/ip_activity_avoidance_strategies/fyear={year}/",
            )
            .groupby("strategy")
            .agg({"rn": "count", "sample_rate": "sum"})
            .rename(columns={"rn": "count"})
        )
        ef = (
            _read_partition(
                container_client, f"{version}/ip_efficiencies_strategies/fyear={year}/"
            )
            .groupby("strategy")
            .agg({"rn": "count", "sample_rate": "sum"})
            .rename(columns={"rn": "count"})
        )
        mitigator_dict[version] = pd.concat([aa, ef], axis=0)
    dev, latest = versions[0], versions[-1]
    return mitigator_dict[dev].merge(
        mitigator_dict[latest],
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=(f"_{dev}", f"_{latest}"),
    )


def _compare_op_mitigators(
    container_client, versions: list[str], year: int
) -> tuple[pd.DataFrame, pd.Index]:
    mitigator_dict = {}
    mitigator_names = None
    for version in versions:
        op = _read_partition(container_client, f"{version}/op/fyear={year}/")
        op_mitigators = get_all_op_mitigators(op)
        mitigator_dict[version] = op_mitigators
        mitigator_names = op_mitigators.columns
    dev, latest = versions[0], versions[-1]
    combined = mitigator_dict[dev].merge(
        mitigator_dict[latest],
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=(f"_{dev}", f"_{latest}"),
    )
    return combined, mitigator_names


def _compare_aae_mitigators(
    container_client, versions: list[str], year: int
) -> tuple[pd.DataFrame, pd.Index]:
    mitigator_dict = {}
    mitigator_names = None
    for version in versions:
        aae = _read_partition(container_client, f"{version}/aae/fyear={year}/")
        aae_mitigators = get_all_ae_mitigators(aae)
        mitigator_dict[version] = aae_mitigators
        mitigator_names = aae_mitigators.columns
    dev, latest = versions[0], versions[-1]
    combined = mitigator_dict[dev].merge(
        mitigator_dict[latest],
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=(f"_{dev}", f"_{latest}"),
    )
    return combined, mitigator_names


# %%
def run_qa_checks(
    year: int = DEFAULT_YEAR,
    compare_version: str | None = None,
    output_dir: str = "results",
    account_url: str | None = None,
    container_name: str | None = None,
) -> bool:
    """Run all QA comparisons and write comparison CSVs.

    Args:
        year: fyear to check (default 2024)
        compare_version: version to compare `dev` against. Defaults to the
            latest `vX.Y.Z` version folder found in the container.
        output_dir: directory to write CSVs to (created if missing)
        account_url: Azure Storage account URL (default: AZ_STORAGE_EP)
        container_name: Azure Storage container name (default: AZ_STORAGE_DATA)

    Returns:
        bool: True if dev and the compared version matched everywhere,
            False if any mismatch was found (CSVs are still written either way)
    """
    account_url, container_name = get_azure_credentials(
        account_url, container_name=os.getenv("AZ_STORAGE_DATA")
    )
    container_client = connect_to_container(account_url, container_name)

    if compare_version is None:
        compare_version = _latest_version(container_client)
        logger.info(f"Comparing 'dev' against latest version found: {compare_version}")

    versions = ["dev", compare_version]
    today_date = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_matched = True

    # Inpatients
    logger.info("Comparing inpatients data...")
    full_df_ip = _compare_activity(
        container_client, versions, year, "ip", _aggregate_data_ip
    )
    all_matched &= _check_columns_match(
        full_df_ip, "dev_admissions", f"{compare_version}_admissions", "IP admissions"
    )
    all_matched &= _check_columns_match(
        full_df_ip, "dev_beddays", f"{compare_version}_beddays", "IP beddays"
    )
    full_df_ip.to_csv(out_dir / f"{today_date}_QA_ip.csv")

    # Outpatients
    logger.info("Comparing outpatients data...")
    full_df_op = _compare_activity(
        container_client, versions, year, "op", _aggregate_data_op
    )
    all_matched &= _check_columns_match(
        full_df_op,
        "dev_attendances",
        f"{compare_version}_attendances",
        "OP attendances",
    )
    all_matched &= _check_columns_match(
        full_df_op,
        "dev_tele_attendances",
        f"{compare_version}_tele_attendances",
        "OP tele attendances",
    )
    full_df_op.to_csv(out_dir / f"{today_date}_QA_op.csv")

    # A&E
    logger.info("Comparing A&E data...")
    full_df_aae = _compare_activity(
        container_client, versions, year, "aae", _aggregate_data_aae
    )
    all_matched &= _check_columns_match(
        full_df_aae, "dev_arrivals", f"{compare_version}_arrivals", "A&E arrivals"
    )
    full_df_aae.to_csv(out_dir / f"{today_date}_QA_aae.csv")

    # IP mitigators
    logger.info("Comparing IP mitigators...")
    ip_mitigators = _compare_ip_mitigators(container_client, versions, year)
    all_matched &= _check_columns_match(
        ip_mitigators, "count_dev", f"count_{compare_version}", "IP mitigator counts"
    )
    all_matched &= _check_columns_match(
        ip_mitigators,
        "sample_rate_dev",
        f"sample_rate_{compare_version}",
        "IP mitigator sample rates",
    )
    ip_mitigators.to_csv(out_dir / f"{today_date}_QA_ip_mitigators.csv")

    # OP mitigators
    logger.info("Comparing OP mitigators...")
    combined_op_mitigators, op_mitigator_names = _compare_op_mitigators(
        container_client, versions, year
    )
    for col in op_mitigator_names:
        all_matched &= _check_columns_match(
            combined_op_mitigators,
            f"{col}_dev",
            f"{col}_{compare_version}",
            f"OP mitigator '{col}'",
        )
    combined_op_mitigators.sort_index(axis=1).to_csv(
        out_dir / f"{today_date}_QA_op_mitigators.csv"
    )

    # A&E mitigators
    logger.info("Comparing A&E mitigators...")
    combined_aae_mitigators, aae_mitigator_names = _compare_aae_mitigators(
        container_client, versions, year
    )
    for col in aae_mitigator_names:
        all_matched &= _check_columns_match(
            combined_aae_mitigators,
            f"{col}_dev",
            f"{col}_{compare_version}",
            f"A&E mitigator '{col}'",
        )
    combined_aae_mitigators.sort_index(axis=1).to_csv(
        out_dir / f"{today_date}_QA_aae_mitigators.csv"
    )

    if all_matched:
        logger.info("✅ dev matches %s across all checks", compare_version)
    else:
        logger.warning(
            "⚠️ One or more mismatches found between dev and %s - see CSVs in %s",
            compare_version,
            out_dir,
        )

    return all_matched


# %%
def main(level: int = INFO) -> int:
    """CLI entry point when module is run directly."""
    configure_logging(level)

    parser = argparse.ArgumentParser(
        description="Compare NHP model 'dev' data against the latest released "
        "version in Azure Blob Storage, writing comparison CSVs."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"fyear to check (default {DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--compare-version",
        help="Version to compare 'dev' against (default: latest version found)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write comparison CSVs to (default: results)",
    )
    parser.add_argument("--account-url", help="Azure Storage account URL")
    parser.add_argument("--container", help="Azure Storage container name")

    args = parser.parse_args()

    try:
        matched = run_qa_checks(
            year=args.year,
            compare_version=args.compare_version,
            output_dir=args.output_dir,
            account_url=args.account_url,
            container_name=args.container,
        )
        return ExitCodes.SUCCESS_CODE if matched else ExitCodes.ERROR_CODE
    except (
        ValueError,
        ClientAuthenticationError,
        ResourceNotFoundError,
        AzureError,
        EmptyContainerError,
    ) as e:
        logger.error(f"Error: {e}")
        return ExitCodes.EXCEPTION_CODE
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return ExitCodes.SIGINT_CODE


# %%
if __name__ == "__main__":
    sys.exit(main(level=INFO))
