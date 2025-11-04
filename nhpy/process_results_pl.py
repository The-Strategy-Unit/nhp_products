"""Functions to process results files - Polars implementation"""

import polars as pl

# %% [markdown]
# ## Results Format Conversion


# %%
def convert_results_format(results: pl.DataFrame, include_baseline=True) -> pl.DataFrame:
    """Get principal and baseline from new results format parquet files.

    Optimised Polars implementation that processes results with columns "model_run"
    and "value", replacing these with columns "baseline", "model_runs", and "mean".
    The function handles both cases with and without baseline data.

    Args:
        results (pl.DataFrame): Input DataFrame with "model_run" and "value" columns
        include_baseline (bool, optional): Whether to include baseline data.
        Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with baseline, model_runs and mean columns
    """
    # Get all columns except 'value' and 'model_run' for grouping
    group_cols = [col for col in results.columns if col not in {"value", "model_run"}]

    # Convert to lazy for better optimisation
    lazy_results = results.lazy()

    if include_baseline:
        # Process in one pass using window functions and lazy evaluation
        # for best performance
        # This implementation:
        # 1. Uses partitioning by group columns to get both baseline and model runs
        #    in a single pass
        # 2. Takes advantage of Polars' lazy evaluation for query optimisation
        # 3. Minimises data movement and reshaping operations
        df_combined = (
            lazy_results.with_columns(
                [
                    # Identify baseline rows (model_run = 0)
                    (pl.col("model_run") == 0).alias("is_baseline"),
                    # Flag for non-baseline rows
                    (pl.col("model_run") != 0).alias("is_model_run"),
                ]
            )
            .select(
                [
                    # Select all group columns
                    *[pl.col(col) for col in group_cols],
                    # Get baseline value (where model_run = 0)
                    pl.when(pl.col("is_baseline"))
                    .then(pl.col("value"))
                    .otherwise(None)
                    .alias("_baseline_value"),
                    # Get model run values (where model_run != 0)
                    pl.when(pl.col("is_model_run"))
                    .then(pl.col("value"))
                    .otherwise(None)
                    .alias("_model_run_value"),
                ]
            )
            .group_by(group_cols)
            .agg(
                [
                    # Get the first (and only) baseline value for each group
                    pl.col("_baseline_value")
                    .filter(pl.col("_baseline_value").is_not_null())
                    .first()
                    .alias("baseline"),
                    # Collect all non-null model run values into a list
                    pl.col("_model_run_value")
                    .filter(pl.col("_model_run_value").is_not_null())
                    .alias("model_runs"),
                ]
            )
            # Calculate mean in the same query
            .with_columns([pl.col("model_runs").list.mean().alias("mean")])
            .collect()
        )
    else:
        # For the case without baseline, simpler grouping is sufficient
        df_combined = (
            lazy_results.group_by(group_cols)
            .agg([pl.col("value").alias("model_runs")])
            .with_columns([pl.col("model_runs").list.mean().alias("mean")])
            .collect()
        )

    return df_combined
