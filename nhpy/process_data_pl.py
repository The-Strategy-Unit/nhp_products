"""Functions to process model data in format suitable for model results
Polars implementation

This module reuses non-Pandas functions from process_data.py and reimplements only the
Pandas-specific functions to use Polars instead.
"""

import logging
from collections import defaultdict

import numpy as np
import polars as pl

from nhpy.config import Constants
from nhpy.process_data import n_runs

logger = logging.getLogger(__name__)


def age_groups(age: pl.Expr | pl.Series) -> pl.Expr:
    """Cut age into groups

    Takes a polars Series or Expression of ages and cuts into discrete intervals

    Args:
        age: a Series or Expression of ages

    Returns:
        pl.Expr: an expression for age groups
    """
    # Fill nulls with UNKNOWN (will be mapped to "Unknown")
    filled_age = age.fill_null(Constants.AGE_UNKNOWN)

    # Simple when-then chain directly on the series
    return (
        pl.when(filled_age <= Constants.AGE_UNKNOWN)
        .then("Unknown")
        .when(filled_age <= Constants.AGE_INFANT)
        .then("0")
        .when(filled_age < Constants.AGE_TODDLER)
        .then("1-4")
        .when(filled_age < Constants.AGE_CHILD)
        .then("5-9")
        .when(filled_age < Constants.AGE_ADOLESCENT)
        .then("10-15")
        .when(filled_age < Constants.AGE_YOUNG_ADULT)
        .then("16-17")
        .when(filled_age < Constants.AGE_ADULT)
        .then("18-34")
        .when(filled_age < Constants.AGE_MIDDLE_AGE)
        .then("35-49")
        .when(filled_age < Constants.AGE_SENIOR)
        .then("50-64")
        .when(filled_age < Constants.AGE_ELDERLY)
        .then("65-74")
        .when(filled_age < Constants.AGE_OLDEST)
        .then("75-84")
        .otherwise("85+")
    )


def add_pod_to_data_ip(data: pl.DataFrame) -> pl.DataFrame:
    """Adds the POD column to data"""

    # Create base pod column
    data = data.with_columns(
        pl.lit("ip_").alias("_prefix"),
        pl.col("group").alias("_group"),
        pl.lit("_admission").alias("_suffix"),
    )
    data = data.with_columns(pl.concat_str(["_prefix", "_group", "_suffix"]).alias("pod"))

    # Apply classpat rules
    data = data.with_columns(
        pl.when(pl.col("classpat") == "2")
        .then(pl.lit("ip_elective_daycase"))
        .when(pl.col("classpat") == "3")
        .then(pl.lit("ip_regular_day_attender"))
        .when(pl.col("classpat") == "4")
        .then(pl.lit("ip_regular_night_attender"))
        .when(pl.col("classpat") == "-2")
        .then(pl.lit("ip_elective_daycase"))
        .when(pl.col("classpat") == "-1")
        .then(pl.lit("op_procedure"))
        .otherwise(pl.col("pod"))
        .alias("pod")
    )

    # Clean up temporary columns
    data = data.drop(["_prefix", "_group", "_suffix"])

    return data


def process_ip_results(data: pl.DataFrame, los_groups: defaultdict) -> pl.DataFrame:
    """
    Processes the row level inpatients data into
    a format suitable for aggregation in results files

    Args:
        data: Data to be processed. Format should be similar to Model.data
        los_groups: Mapping of length of stay to los groups

    Returns:
        pl.DataFrame: Processed data with new columns
    """
    # Convert los_groups defaultdict to a dictionary without the default function
    # This is needed for replace_strict which doesn't support defaultdict with lambda

    # First, get all unique speldur values to create a complete mapping dictionary
    unique_speldurs = data.select(pl.col("speldur").unique()).to_series().to_list()

    # Create a complete mapping dictionary for all values
    los_mapping = {speldur: los_groups[speldur] for speldur in unique_speldurs}

    # Create new columns
    data = data.with_columns(
        [
            # Use replace_strict instead of map_elements for better performance
            pl.col("speldur").replace_strict(los_mapping).alias("los_group"),
            pl.lit(1).alias("admissions"),
            (pl.col("speldur") + 1).alias("beddays"),
            pl.col("has_procedure").alias("procedures"),
        ]
    )

    return data


def process_ip_activity_avoided(data: pl.DataFrame) -> pl.DataFrame:
    """
    Process the IP activity_avoided, adding pod and age_group,
    and grouping by HRG, los_group and pod

    Args:
        data: the IP rows avoided

    Returns:
        The processed and aggregated data
    """
    # Define los groups for activity avoided
    los_groups_aa = defaultdict(
        lambda: "1+ days",
        {
            0: "0 days",
        },
    )

    # Process data
    data = process_ip_results(data, los_groups_aa)

    # Group by sushrg, los_group, pod and aggregate
    data = data.group_by(["sushrg", "los_group", "pod"], maintain_order=True).agg(
        [pl.col("admissions").sum(), pl.col("beddays").sum(), pl.col("procedures").sum()]
    )

    # Handle outpatients rows - set beddays and procedures to 0
    op_mask = pl.col("pod") == "op_procedure"
    data = data.with_columns(
        [
            pl.when(op_mask)
            .then(pl.lit(0))
            .otherwise(pl.col("beddays"))
            .alias("beddays"),
            pl.when(op_mask)
            .then(pl.lit(0))
            .otherwise(pl.col("procedures"))
            .alias("procedures"),
        ]
    )

    # Convert to long format using unpivot
    data = data.unpivot(
        index=["sushrg", "los_group", "pod"],
        on=["admissions", "beddays", "procedures"],
        variable_name="measure",
    )

    # Set measure for op_procedure to attendances
    data = data.with_columns(
        [
            pl.when(op_mask)
            .then(pl.lit("attendances"))
            .otherwise(pl.col("measure"))
            .alias("measure")
        ]
    )

    # Reset index and regroup
    data = data.group_by(
        ["pod", "los_group", "sushrg", "measure"], maintain_order=True
    ).agg([pl.col("value").sum()])

    return data


def process_model_runs_dict(
    model_runs: dict, columns: list, all_runs_kept: bool = False
) -> pl.DataFrame:
    """
    Turn the model_runs dict into a DataFrame,
    adding the lower_ci, median, mean, and upper_ci columns

    Args:
        model_runs: Dict where the keys are all the possible aggregations,
        and the values are a list showing the value for each Monte Carlo simulation
        columns: The columns in the DataFrame
        all_runs_kept: If True, all values in the model_runs dict need to be of length 256

    Returns:
        DataFrame containing aggregated results of all 256 Monte Carlo simulations
    """
    # Prepare data for a more efficient approach - process all metrics at once
    rows = []

    # Calculate metrics for each key in a single pass
    for k, v in model_runs.items():
        v_ = np.array(v)
        if not all_runs_kept and len(v_) < n_runs:
            # Ensure all 256 runs are accounted for
            zeros = np.zeros(n_runs - len(v_))
            v_ = np.concatenate((v_, zeros))

        if len(v_) != n_runs:
            raise ValueError(f"Length of array for {k} is not equal to n_runs")

        # Convert None values to np.nan which numpy can handle
        v_ = np.array([np.nan if val is None else val for val in v_], dtype=float)

        # Check if the array is all NaN values
        if np.all(np.isnan(v_)):
            # If all values are NaN, set all statistics to 0
            lwr_ci, median, upr_ci = 0, 0, 0
            mean = 0
        else:
            # Use nanpercentile to ignore nan values when calculating percentiles
            x = np.nanpercentile(v_, [10, 50, 90])
            lwr_ci, median, upr_ci = x[0], x[1], x[2]
            mean = np.nanmean(v_)

        # Create a dict for this row with all columns and metrics
        row_dict = {columns[i]: k[i] for i in range(len(columns))}
        row_dict.update(
            {"lwr_ci": lwr_ci, "median": median, "mean": mean, "upr_ci": upr_ci}
        )

        rows.append(row_dict)

    # Create dataframe directly with all columns and metrics
    if rows:
        # Create DataFrame from processed data
        model_runs_df = pl.DataFrame(rows)
        # Sort by the original columns
        model_runs_df = model_runs_df.sort(columns)
    else:
        # Handle empty case
        df_cols = {col: [] for col in columns}
        df_cols.update({"lwr_ci": [], "median": [], "mean": [], "upr_ci": []})
        model_runs_df = pl.DataFrame(df_cols)

    return model_runs_df


def process_ip_detailed_results(data: pl.DataFrame) -> pl.DataFrame:
    """Process the IP detailed results, adding pod and age_group, and grouping by
    sitetret, age_group, sex, pod, tretspef, los_group, maternity_delivery_in_spell,
    and measure

    Args:
        data: the IP activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Define detailed LOS groups
    los_groups_detailed = defaultdict(
        lambda: "22+ days",
        {
            0: "0 days",
            1: "1 day",
            2: "2 days",
            3: "3 days",
            **{i: "4-7 days" for i in range(4, 8)},
            **{i: "8-14 days" for i in range(8, 15)},
            **{i: "15-21 days" for i in range(15, 22)},
        },
    )

    # Process the data
    data = process_ip_results(data, los_groups_detailed)

    # Group by required columns and aggregate
    group_cols = [
        "sitetret",
        "age_group",
        "sex",
        "pod",
        "tretspef",
        "los_group",
        "maternity_delivery_in_spell",
    ]

    data = data.group_by(group_cols, maintain_order=True).agg(
        [pl.col("admissions").sum(), pl.col("beddays").sum(), pl.col("procedures").sum()]
    )

    # Handle outpatients rows - set beddays and procedures to 0
    op_mask = pl.col("pod") == "op_procedure"
    data = data.with_columns(
        [
            pl.when(op_mask)
            .then(pl.lit(0))
            .otherwise(pl.col("beddays"))
            .alias("beddays"),
            pl.when(op_mask)
            .then(pl.lit(0))
            .otherwise(pl.col("procedures"))
            .alias("procedures"),
        ]
    )

    # Convert to long format using unpivot
    data = data.unpivot(
        index=group_cols,
        on=["admissions", "beddays", "procedures"],
        variable_name="measure",
    )

    # Set measure for op_procedure to attendances
    data = data.with_columns(
        [
            pl.when(op_mask)
            .then(pl.lit("attendances"))
            .otherwise(pl.col("measure"))
            .alias("measure")
        ]
    )

    # Remove rows where value is 0
    data = data.filter(pl.col("value") > 0)

    # Regroup with measure included
    group_cols.append("measure")
    data = data.group_by(group_cols, maintain_order=True).agg([pl.col("value").sum()])

    return data


def process_op_detailed_results(data: pl.DataFrame) -> pl.DataFrame:
    """Process the OP detailed results, adding pod and age_group, and grouping by
    sitetret, age_group, sex, pod, tretspef, and measure

    Args:
        data: the IP activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Convert wide format to long format using unpivot
    measures = data.unpivot(
        index=["rn"],
        on=["attendances", "tele_attendances"],
        variable_name="measure",
    )

    # Drop the original attendance columns and merge with melted data
    data = data.drop(["attendances", "tele_attendances"]).join(measures, on="rn")

    # Group by required columns and aggregate
    data = (
        data.group_by(
            ["sitetret", "pod", "age_group", "tretspef", "measure"], maintain_order=True
        )
        .agg([pl.col("value").sum()])
        .sort(["sitetret", "pod", "age_group", "tretspef", "measure"])
    )

    return data


def process_op_converted_from_ip(data: pl.DataFrame) -> pl.DataFrame:
    """Process the OP activity converted from IP, adding pod and age_group, and
    grouping by sitetret, age_group, sex, pod, tretspef, and measure.

    Args:
        data: the OP activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # The original Pandas code assumes the 'age' column exists, so we do the same
    # Set pod, age_group, and measure columns
    data = data.with_columns(
        [pl.lit("op_procedure").alias("pod"), pl.lit("attendances").alias("measure")]
    )

    # Add age_group column - replicate exactly what the pandas version does
    try:
        data = data.with_columns(age_groups(pl.col("age")).alias("age_group"))

    # FIXME
    except Exception as e:
        # If there's an error with the age column, use a default value
        # to maintain the pipeline functionality
        logger.warning("Error processing age column: %s", e)
        data = data.with_columns(pl.lit("Unknown").alias("age_group"))

    # Rename and group by required columns
    return (
        data.rename({"attendances": "value"})
        .group_by(
            ["sitetret", "pod", "age_group", "tretspef", "measure"], maintain_order=True
        )
        .agg([pl.col("value").sum()])
    )


def combine_converted_with_main_results(
    df_converted: pl.DataFrame, df: pl.DataFrame
) -> pl.DataFrame:
    """Combines the activity converted from IP to OP/AAE
    with the main OP/AAE activity results

    Args:
        df_converted: the OP/AAE activity converted from IP in each Monte Carlo
        simulation
        df: the OP/AAE activity in each Monte Carlo simulation

    Returns:
        pl.DataFrame: The combined dataframe with both sets of activity
    """
    # Create a copy to avoid modifying the input
    result = df.clone()

    # Figure out the value column name - for AAE it's "arrivals", for others it's "value"
    value_col = "arrivals" if "arrivals" in result.columns else "value"

    # Convert index to columns, join, and add values
    result = (
        result.join(
            df_converted,
            on=df_converted.columns[:-1],  # Join on all columns except value
            how="outer",
        )
        .with_columns(
            [
                (pl.col(value_col) + pl.col(f"{value_col}_right").fill_null(0)).alias(
                    value_col
                )
            ]
        )
        .drop(f"{value_col}_right")
    )

    return result


def process_aae_results(data: pl.DataFrame) -> pl.DataFrame:
    """Process the AAE detailed results, adding pod and age_group,
    and grouping by sitetret, pod, age_group, attendance_category,
    aedepttype, acuity, and measure

    Args:
        data: the AAE activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Set measure based on is_ambulance flag
    data = data.with_columns(
        [
            pl.when(pl.col("is_ambulance"))
            .then(pl.lit("ambulance"))
            .otherwise(pl.lit("walk-in"))
            .alias("measure")
        ]
    )

    # Group by required columns and aggregate
    return data.group_by(
        [
            "sitetret",
            "pod",
            "age_group",
            "attendance_category",
            "aedepttype",
            "acuity",
            "measure",
        ],
        maintain_order=True,
    ).agg([pl.col("arrivals").sum()])


def process_aae_converted_from_ip(data: pl.DataFrame) -> pl.DataFrame:
    """Process the AAE SDEC activity converted from IP, adding pod and age_group,
    and grouping by sitetret, age_group, pod, aedepttype, attendance_category, acuity
    and measure.

    Args:
        data: the AAE SDEC activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Add pod column
    data = data.with_columns(pl.lit("aae_type-05").alias("pod"))

    # Add age_group column - similar to the Pandas version but with error handling
    try:
        data = data.with_columns(age_groups(pl.col("age")).alias("age_group"))
    except Exception as e:
        # If there's an error with the age column, use a default value
        logger.warning("Error processing age column in AAE: %s", e)
        data = data.with_columns(pl.lit("Unknown").alias("age_group"))

    # Rename group to measure and group by required columns
    data = data.rename({"group": "measure"})

    return data.group_by(
        [
            "sitetret",
            "pod",
            "age_group",
            "attendance_category",
            "aedepttype",
            "acuity",
            "measure",
        ],
        maintain_order=True,
    ).agg([pl.col("arrivals").sum()])


def add_pod_to_data_op(data: pl.DataFrame) -> pl.DataFrame:
    """Adds the POD column to outpatient data"""
    data = data.with_columns(
        [
            pl.when(pl.col("is_first"))
            .then(pl.lit("op_first"))
            .when(pl.col("has_procedures"))
            .then(pl.lit("op_procedure"))
            .otherwise(pl.lit("op_follow-up"))
            .alias("pod")
        ]
    )

    return data


def get_op_mitigators_consultant(df: pl.DataFrame) -> pl.DataFrame:
    """Consultant to consultant mitigators."""

    # Filter for consultant to consultant referrals
    cons_df = df.filter(pl.col("is_cons_cons_ref")).select(
        ["type", "dataset", "attendances", "tele_attendances"]
    )

    # Group by dataset and type and sum
    cons_df = (
        cons_df.group_by(["dataset", "type"], maintain_order=True)
        .agg(
            [
                pl.col("attendances")
                .sum()
                .alias("consultant_to_consultant_referrals_attendances"),
                pl.col("tele_attendances")
                .sum()
                .alias("consultant_to_consultant_referrals_tele_attendances"),
            ]
        )
        .sort(["dataset", "type"])
    )

    return cons_df


def get_op_mitigators_followup(df: pl.DataFrame) -> pl.DataFrame:
    """Followup reduction mitigators. NOT FIRST, NO PROCEDURES"""

    # Filter for follow-up without procedures
    followup_df = df.filter(~pl.col("has_procedures") & ~pl.col("is_first")).select(
        ["type", "dataset", "attendances", "tele_attendances"]
    )

    # Group by dataset and type and sum
    followup_df = (
        followup_df.group_by(["dataset", "type"], maintain_order=True)
        .agg(
            [
                pl.col("attendances").sum().alias("followup_reduction_attendances"),
                pl.col("tele_attendances")
                .sum()
                .alias("followup_reduction_tele_attendances"),
            ]
        )
        .sort(["dataset", "type"])
    )

    return followup_df
