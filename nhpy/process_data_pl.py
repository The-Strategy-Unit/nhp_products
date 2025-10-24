"""Functions to process model data in format suitable for model results
Polars implementation

This module reuses non-Pandas functions from process_data.py and reimplements only the
Pandas-specific functions to use Polars instead.
"""

# %%
# Imports
import logging
from collections import defaultdict

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nhpy.config import Constants
from nhpy.process_data import n_runs

# %%
# Logging
logger = logging.getLogger(__name__)


# %% [markdown]
# ## Age Group Classification


# %%
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
        .then(pl.lit("Unknown"))
        .when(filled_age <= Constants.AGE_INFANT)
        .then(pl.lit("0"))
        .when(filled_age < Constants.AGE_TODDLER)
        .then(pl.lit("1-4"))
        .when(filled_age < Constants.AGE_CHILD)
        .then(pl.lit("5-9"))
        .when(filled_age < Constants.AGE_ADOLESCENT)
        .then(pl.lit("10-15"))
        .when(filled_age < Constants.AGE_YOUNG_ADULT)
        .then(pl.lit("16-17"))
        .when(filled_age < Constants.AGE_ADULT)
        .then(pl.lit("18-34"))
        .when(filled_age < Constants.AGE_MIDDLE_AGE)
        .then(pl.lit("35-49"))
        .when(filled_age < Constants.AGE_SENIOR)
        .then(pl.lit("50-64"))
        .when(filled_age < Constants.AGE_ELDERLY)
        .then(pl.lit("65-74"))
        .when(filled_age < Constants.AGE_OLDEST)
        .then(pl.lit("75-84"))
        .otherwise(pl.lit("85+"))
    )


# %% [markdown]
# ## Inpatient Data Processing


# %%
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


# %%
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


# %%
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

    # The `unpivot` method transforms data from wide format to long format.
    # Here it converts separate columns for different measures into a single value column
    # with a corresponding measure column.
    #
    # Example:
    # ```
    # # Wide format (before):
    # sushrg | los_group | pod | admissions | beddays | procedures
    # ------|-----------|-----|------------|---------|------------
    # AA22A  | 0 days    | ip  | 10         | 10      | 5
    #
    # # Long format (after):
    # sushrg | los_group | pod | measure    | value
    # ------|-----------|-----|------------|-------
    # AA22A  | 0 days    | ip  | admissions | 10
    # AA22A  | 0 days    | ip  | beddays    | 10
    # AA22A  | 0 days    | ip  | procedures | 5
    # ```
    #
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


# %% [markdown]
# ## Statistical Processing


# %%
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
    rows = []
    skipped_keys = 0
    error_keys = 0
    short_keys_processed = 0

    # Process ALL keys without filtering to exactly match Pandas behavior
    # Pandas does not skip keys with None sitetret
    for k, v in model_runs.items():
        try:
            # Convert to numpy array for calculations and handle None values in one step
            v_ = np.array([0.0 if val is None else float(val) for val in v], dtype=float)

            # For keys with fewer than n_runs values
            # still process them by padding with zeros to match Pandas behavior
            if not all_runs_kept and len(v_) < n_runs:
                short_keys_processed += 1
                # Pad with zeros to reach n_runs - ensures statistical calculations work
                zeros = np.zeros(n_runs - len(v_))
                v_ = np.concatenate((v_, zeros))

            # Use standard percentile calculation for confidence intervals
            # Extract percentiles individually to avoid subscripting
            lwr_ci = float(np.percentile(v_, 10))
            median = float(np.percentile(v_, 50))
            upr_ci = float(np.percentile(v_, 90))
            mean = float(np.mean(v_))

            # Create a dict for this row with all columns and metrics
            # Convert None to empty string in column values to match Pandas behavior
            row_dict = {
                columns[i]: "" if k[i] is None else k[i] for i in range(len(columns))
            }
            row_dict.update(
                {"lwr_ci": lwr_ci, "median": median, "mean": mean, "upr_ci": upr_ci}
            )

            rows.append(row_dict)
        except Exception as e:
            # Log error but continue - helps identify issues without breaking processing
            logger.warning(f"Error processing key {k}: {str(e)}")
            error_keys += 1
            continue

    if skipped_keys > 0:
        logger.info(f"Skipped {skipped_keys} keys due to incorrect run count")
    if error_keys > 0:
        logger.info(f"Skipped {error_keys} keys due to processing errors")
    if short_keys_processed > 0:
        logger.info(
            f"Processed {short_keys_processed} keys with fewer than {n_runs} runs"
        )

    # Create dataframe with all columns and metrics
    if rows:
        model_runs_df = pl.DataFrame(rows)
        model_runs_df = model_runs_df.sort(columns)
    else:
        # Handle empty case
        df_cols = {col: [] for col in columns}
        df_cols.update({"lwr_ci": [], "median": [], "mean": [], "upr_ci": []})
        model_runs_df = pl.DataFrame(df_cols)

    # Don't filter out rows after creating the DataFrame
    # Keep all rows to match Pandas behavior

    return model_runs_df


# %% [markdown]
# ## Detailed Results Processing


# %%
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


# %%
def process_op_detailed_results(data: pl.DataFrame) -> pl.DataFrame:
    """Process the OP detailed results, adding pod and age_group, and grouping by
    sitetret, age_group, sex, pod, tretspef, and measure

    Args:
        data: the IP activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Transform outpatient data that comes in wide format (with separate columns
    # for different attendance types) to long format using the unpivot operation.
    measures = data.unpivot(
        index=["rn"],
        on=["attendances", "tele_attendances"],
        variable_name="measure",
    )

    # Drop the original attendance columns and merge with melted data
    # Use LEFT join to preserve all original data including sites that might be dropped
    data = data.drop(["attendances", "tele_attendances"]).join(
        measures, on="rn", how="left"
    )

    # Fill any null values in the value column with 0 (from the unpivoted columns)
    data = data.with_columns(pl.col("value").fill_null(0))

    # Group by required columns and aggregate
    # Deliberately NOT filtering before grouping to match Pandas behavior exactly
    data = (
        data.group_by(
            ["sitetret", "pod", "age_group", "tretspef", "measure"], maintain_order=True
        )
        .agg([pl.col("value").sum()])
        .sort(["sitetret", "pod", "age_group", "tretspef", "measure"])
    )

    # Pandas version does not filter out NULL sitetret values after grouping
    # We'll keep all rows to exactly match Pandas behavior

    # Ensure numeric columns use consistent types
    data = data.with_columns([pl.col("value").cast(pl.Float64)])

    return data


# %%
def process_op_converted_from_ip(data: pl.DataFrame) -> pl.DataFrame:
    """Process the OP activity converted from IP, adding pod and age_group, and
    grouping by sitetret, age_group, sex, pod, tretspef, and measure.

    Args:
        data: the OP activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Start with a copy of the input data
    data = data.clone()

    # Always set pod and measure
    data = data.with_columns(
        [pl.lit("op_procedure").alias("pod"), pl.lit("attendances").alias("measure")]
    )

    # Add age_group if it doesn't exist
    if "age_group" not in data.columns:
        # Default to "Unknown"
        if "age" in data.columns:
            # Try to create age groups if age column exists
            try:
                data = data.with_columns(age_groups(pl.col("age")).alias("age_group"))
            except Exception as e:
                logger.warning(f"Error processing age column: {e}")
                data = data.with_columns(pl.lit("Unknown").alias("age_group"))
        else:
            data = data.with_columns(pl.lit("Unknown").alias("age_group"))

    # Add sitetret if missing
    if "sitetret" not in data.columns:
        data = data.with_columns(pl.lit("unknown").alias("sitetret"))

    # Add tretspef if missing
    if "tretspef" not in data.columns:
        data = data.with_columns(pl.lit("unknown").alias("tretspef"))

    # Create value column from attendances if it exists, otherwise use 1 as default
    if "attendances" in data.columns:
        data = data.with_columns(pl.col("attendances").alias("value"))
    else:
        data = data.with_columns(pl.lit(1).alias("value"))

    # Group by the needed columns and sum the value
    return data.group_by(
        ["sitetret", "pod", "age_group", "tretspef", "measure"], maintain_order=True
    ).agg(pl.col("value").sum())


# %% [markdown]
# ### Combining Data Sets
#
# The `combine_converted_with_main_results` function joins two dataframes and combines
# their values. This uses a join operation to match rows based on common columns, followed
# by column operations to sum the values.
#
# Example of the joining pattern:
# ```python
# # Joining two dataframes with matching keys, then summing the values
# df1 = pl.DataFrame({"key": ["A", "B"], "value": [1, 2]})
# df2 = pl.DataFrame({"key": ["A", "C"], "value": [10, 30]})
#
# # Result: DataFrame with keys A, B, C and summed values
# # A: 11 (1+10), B: 2, C: 30
# combined = df1.join(df2, on="key", how="outer").with_columns(
#     [(pl.col("value") + pl.col("value_right").fill_null(0)).alias("value")]
# ).drop("value_right")
# ```


# %%
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

    Note:
        We use concat instead of join because SDEC conversion data introduces
        entirely new rows (e.g., aae_type-05) that don't exist in the main dataset.
        These rows would be lost in a join operation, but preserved with concatenation.
        Schema alignment is critical for proper concatenation.
    """
    # Skip processing if SDEC frame is empty
    if len(df_converted) == 0:
        return df.clone()

    # Create a copy to avoid modifying the input
    result = df.clone()

    # Figure out the value column name - for AAE it's "arrivals", for others it's "value"
    value_col = "arrivals" if "arrivals" in result.columns else "value"

    # Create a list of all columns from both dataframes
    all_cols = list(set(result.columns) | set(df_converted.columns))

    # Get schema from both dataframes
    result_schema = {
        col: dtype for col, dtype in zip(result.columns, result.dtypes) if col in all_cols
    }
    converted_schema = {
        col: dtype
        for col, dtype in zip(df_converted.columns, df_converted.dtypes)
        if col in all_cols
    }

    # Create new dataframes with consistent schema
    new_result_cols = []
    new_converted_cols = []

    # Handle each column individually
    for col in all_cols:
        # Get target dtype, preferring Float64 for value column
        if col == value_col:
            target_dtype = pl.Float64
        elif col in result_schema and col in converted_schema:
            # If column exists in both, prefer the type from result
            target_dtype = result_schema[col]
        elif col in result_schema:
            target_dtype = result_schema[col]
        else:
            target_dtype = converted_schema[col]

        # Add column to result with correct type
        if col in result.columns:
            new_result_cols.append(pl.col(col).cast(target_dtype))
        else:
            new_result_cols.append(pl.lit(None).cast(target_dtype).alias(col))

        # Add column to converted with correct type
        if col in df_converted.columns:
            new_converted_cols.append(pl.col(col).cast(target_dtype))
        else:
            new_converted_cols.append(pl.lit(None).cast(target_dtype).alias(col))

    # Create new dataframes with consistent schemas
    new_result = result.with_columns(new_result_cols).select(all_cols)
    new_converted = df_converted.with_columns(new_converted_cols).select(all_cols)

    # Now concatenate with matching schemas
    combined = pl.concat([new_result, new_converted])

    # Group by all columns except the value column to sum any duplicate rows
    groupby_cols = [col for col in all_cols if col != value_col]

    # Only group if there are any groupby columns
    if groupby_cols:
        result = combined.group_by(groupby_cols).agg(pl.col(value_col).sum())
    else:
        # Unlikely to happen but handle edge case
        result = combined.select([pl.col(value_col).sum().alias(value_col)])

    return result


# %% [markdown]
# ## A&E Data Processing


# %%
def process_aae_results(data: pl.DataFrame) -> pl.DataFrame:
    """Process the A&E detailed results, adding pod and age_group,
    and grouping by sitetret, pod, age_group, attendance_category,
    aedepttype, acuity, and measure

    Args:
        data: the A&E activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Set measure based on is_ambulance flag - handle nulls explicitly like Pandas
    data = data.with_columns(
        [
            pl.when(pl.col("is_ambulance").is_null())
            .then(pl.lit("walk-in"))
            .when(pl.col("is_ambulance"))
            .then(pl.lit("ambulance"))
            .otherwise(pl.lit("walk-in"))
            .alias("measure")
        ]
    )

    # Explicit null handling required in Polars to match Pandas behavior
    # Polars differs from Pandas in these key ways:
    # 1. Nulls form valid group keys in Polars but are dropped by default in Pandas
    # 2. Nulls never match in joins in Polars (even null==null) unlike Pandas
    # 3. Polars preserves original dtypes with nulls while Pandas often converts them
    groupby_cols = [
        "sitetret",
        "pod",
        "age_group",
        "attendance_category",
        "aedepttype",
        "acuity",
        "measure",
    ]

    # Fill null values with "unknown" in categorical columns to exactly match Pandas
    # This is critical because Pandas fillna("unknown") was used in data loading
    for col in [
        "sitetret",
        "pod",
        "age_group",
        "attendance_category",
        "aedepttype",
        "acuity",
    ]:
        data = data.with_columns(pl.col(col).fill_null("unknown"))

    # Ensure measure column is never null - this matches Pandas behavior precisely
    data = data.with_columns(pl.col("measure").fill_null("walk-in"))

    # Group by required columns and aggregate
    grouped = data.group_by(
        groupby_cols,
        maintain_order=True,
    ).agg([pl.col("arrivals").sum()])

    # Cast to ensure consistent types with Pandas output
    grouped = grouped.with_columns(pl.col("arrivals").cast(pl.Float64))

    return grouped


# %%
def _prepare_aae_columns(data: pl.DataFrame) -> pl.DataFrame:
    """Helper function to prepare columns for A&E SDEC data converted from IP.

    Adds required columns and handles null values consistently.

    Args:
        data: DataFrame to prepare

    Returns:
        DataFrame with required columns added
    """
    # Create required columns if they don't exist
    cols_to_add = []

    # Always add pod as aae_type-05
    cols_to_add.append(pl.lit("aae_type-05").alias("pod"))

    # Add age_group if it doesn't exist
    if "age_group" not in data.columns:
        if "age" in data.columns:
            # Try to create age groups if age column exists
            try:
                cols_to_add.append(age_groups(pl.col("age")).alias("age_group"))
            except Exception as e:
                cols_to_add.append(pl.lit("Unknown").alias("age_group"))
        else:
            cols_to_add.append(pl.lit("Unknown").alias("age_group"))

    # Add other required columns if missing
    required_cols = {
        "sitetret": "unknown",
        "attendance_category": "unknown",
        "aedepttype": "05",  # Must be '05' for SDEC
        "acuity": "standard",  # Must be 'standard' for SDEC
    }

    for col, default in required_cols.items():
        if col not in data.columns:
            cols_to_add.append(pl.lit(default).alias(col))

    # Add all columns at once
    if cols_to_add:
        data = data.with_columns(cols_to_add)

    # Set 'measure' to 'walk-in'
    data = data.with_columns(pl.lit("walk-in").alias("measure"))

    # Make sure arrivals column exists
    if "arrivals" not in data.columns:
        data = data.with_columns(pl.lit(1).alias("arrivals"))

    return data


def process_aae_converted_from_ip(data: pl.DataFrame) -> pl.DataFrame:
    """Process the A&E SDEC activity converted from IP, adding pod and age_group,
    and grouping by sitetret, age_group, pod, aedepttype, attendance_category, acuity
    and measure.

    Args:
        data: the A&E SDEC activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
    # Skip processing if the data is empty
    if len(data) == 0:
        # Return empty DataFrame with required structure
        return pl.DataFrame(
            {
                "sitetret": [],
                "pod": [],
                "age_group": [],
                "attendance_category": [],
                "aedepttype": [],
                "acuity": [],
                "measure": [],
                "arrivals": [],
            }
        )

    # Prepare columns using helper function
    data = _prepare_aae_columns(data)

    # Define column values for null replacements
    null_replacements = {
        "pod": "aae_type-05",
        "aedepttype": "05",
        "acuity": "standard",
        "measure": "walk-in",
    }

    # Fill null values with known values to match Pandas
    groupby_cols = [
        "sitetret",
        "pod",
        "age_group",
        "attendance_category",
        "aedepttype",
        "acuity",
        "measure",
    ]

    # Fill all nulls in a single operation by constructing expressions
    fill_exprs = [
        pl.col(col).fill_null(null_replacements.get(col, "unknown")).alias(col)
        for col in groupby_cols
    ]
    data = data.with_columns(fill_exprs)

    # Group by and aggregate
    result = data.group_by(groupby_cols, maintain_order=True).agg(
        pl.col("arrivals").sum()
    )

    # Ensure arrivals is Float64 to match the main dataset's type
    result = result.with_columns(pl.col("arrivals").cast(pl.Float64))

    return result


# %% [markdown]
# ## Outpatient Data Processing


# %%
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


# %%
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


# %%
def get_op_mitigators_followup(df: pl.DataFrame) -> pl.DataFrame:
    """Followup reduction mitigators. NOT first, NO PROCEDURES"""

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
