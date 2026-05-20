"""Functions to process results files"""

from typing import Callable

import numpy as np
import pandas as pd


def convert_results_format(results: pd.DataFrame, include_baseline=True) -> pd.DataFrame:
    """Get principal and baseline from new results format parquet files.
    From results.generate_results_json agg_to_dict

    Args:
        results (pd.DataFrame): Processes results with columns "model_run" and
        "value", replacing these with columns "baseline" and "principal"
    """
    df = results.set_index("model_run")
    if include_baseline:
        baseline_df = (
            df.loc[0]
            .set_index([i for i in df.columns if i != "value"])
            .rename(columns={"value": "baseline"})
        )
        # First get the grouped data
        grouped_data = (
            df.loc[df.index != 0]
            .groupby([i for i in df.columns if i != "value"])
            .agg(list)
        )
        # Create a new DataFrame with the renamed column to satisfy type checker
        model_runs_df = pd.DataFrame(grouped_data)
        model_runs_df = model_runs_df.rename(columns={"value": "model_runs"})
        df_with_model_runs_combined = pd.concat(
            [baseline_df, model_runs_df],
            axis=1,
        ).reset_index()
    else:
        df_with_model_runs_combined = (
            df.groupby([i for i in df.columns if i != "value"])
            .agg(list)
            .rename(columns={"value": "model_runs"})
            .reset_index()
        )
    for i in range(df_with_model_runs_combined.shape[0]):
        df_with_model_runs_combined.loc[i, "mean"] = np.mean(
            df_with_model_runs_combined.loc[i, "model_runs"]
        )
    return df_with_model_runs_combined


def compare_results(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Helper function for the qa_model_runs notebook. Compares default model results for
    the same provider/scenario run on two different model versions, and calculates
    differences between pods

    Args:
        df_old (pd.DataFrame): Dataframe of default results from previous model version
        df_new (pd.DataFrame): Dataframe of default results from dev model version

    Returns:
        pd.DataFrame: Dataframe with the differences between pods calculated
    """
    df_old = df_old.groupby(["dataset", "pod", "measure"])[["baseline", "mean"]].sum()
    df_new = df_new.groupby(["dataset", "pod", "measure"])[["baseline", "mean"]].sum()
    combined = df_old.merge(
        df_new, left_index=True, right_index=True, suffixes=("_old", "_new"), how="outer"
    )
    # Create a temporary DataFrame with just the columns we need
    temp_df = pd.DataFrame(combined[["mean_old", "mean_new"]])
    # Use pct_change without the axis parameter, then extract the column we want
    pct_change_df = temp_df.pct_change(axis="columns") * 100
    combined["%_diff"] = abs(pct_change_df["mean_new"].fillna(0))
    return combined


def agg_stepcounts(stepcounts: pd.DataFrame) -> pd.Series:
    """Aggregates step_counts in old format of results JSON, grouping by 'change_factor',
    'measure' and 'strategy' and summing the 'principal' column
    TODO: Deprecate function; we should be working with Parquet format of model results

    Args:
        stepcounts (pd.DataFrame): The step counts from the old format of the results JSON

    Returns:
        pd.DataFrame: Aggregated step counts
    """
    return (
        stepcounts.groupby(["change_factor", "measure", "strategy"])["principal"]
        .sum()
        .astype(int)
    )


def compare_default(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """Compares two Dataframes with format produced by convert_results_format function,
    working out the % diff between the baseline and principal (mean) for each dataframe
    and comparing them

    Args:
        df1 (pd.DataFrame): First dataframe to be compared
        df2 (pd.DataFrame): First dataframe to be compared

    Returns:
        pd.DataFrame: Combined dataframe with both model results compared
    """
    for df in df1, df2:
        # Create a DataFrame with the columns explicitly to fix the typing issue
        pct_df = pd.DataFrame(df[["mean", "baseline"]])
        # Use pct_change without the axis parameter
        pct_change_df = pct_df.pct_change(periods=1, fill_method=None)
        df["%_diff"] = abs(pct_change_df["baseline"].round(4).fillna(0))
    cols_to_drop = [
        "dataset",
        "scenario",
        "app_version",
        "create_datetime",
        "model_runs",
    ]
    merged = pd.merge(
        df1.drop(columns=cols_to_drop),
        df2.drop(columns=cols_to_drop),
        on=["pod", "sitetret", "measure"],
        suffixes=[f"_{df1['scenario'].iloc[0]}", f"_{df2['scenario'].iloc[0]}"],
    ).sort_index(axis=1)
    return merged


def process_stepcounts(sc: pd.DataFrame) -> pd.DataFrame:
    """Calculates principal (mean) for step counts from parquet results, across all model runs

    Args:
        sc (pd.DataFrame): Step counts dataframe

    Returns:
        pd.DataFrame: Step counts aggregated
    """
    return (
        sc.groupby(
            [
                "dataset",
                "activity_type",
                "pod",
                "change_factor",
                "strategy",
                "measure",
            ]
        )[["value"]]
        .mean()
        .rename(columns={"value": "mean"})
    )


def compare_stepcounts(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Helper function for the qa_model_runs notebook. Compares step_counts in
    model results for the same provider/scenario run on two different model versions,
    and calculates differences

    Args:
        df_old (pd.DataFrame): Dataframe of default results from previous model version
        df_new (pd.DataFrame): Dataframe of default results from dev model version

    Returns:
        pd.DataFrame: Dataframe with the differences between step counts
    """
    df_old = process_stepcounts(df_old)
    df_new = process_stepcounts(df_new)
    combined = df_old.merge(
        df_new,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_old", "_new"),
    ).fillna(0)
    # Use pct_change without the axis parameter
    pct_change_df = combined.pct_change(axis=1)
    combined["%_diff"] = abs(pct_change_df["mean_new"].fillna(0)) * 100
    return combined
