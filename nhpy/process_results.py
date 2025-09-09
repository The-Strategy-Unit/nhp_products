"""Functions to process results files"""

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
        model_runs_df = (
            df.loc[df.index != 0]
            .groupby([i for i in df.columns if i != "value"])
            .agg(list)
            .rename(columns={"value": "model_runs"})
        )
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


def add_principal(model_results: pd.DataFrame) -> pd.DataFrame:
    """Adds column 'principal' to old JSON format of model results
    TODO: Deprecate function; we should be working with Parquet format of model results

    Args:
        model_results (pd.DataFrame):

    Returns:
        pd.DataFrame: Model results with principal (mean) column added
    """
    res = pd.DataFrame(model_results).fillna("-")
    for i in res.index:
        p = np.mean(res.loc[i, "model_runs"])
        res.loc[i, "principal"] = p
    return res


def agg_default(df: pd.DataFrame) -> pd.DataFrame:
    """Groups model results dataframe by pod and measure, summing the baseline and
    the principal columns

    Args:
        df (pd.DataFrame): Model results containing the columns 'pod', 'measure',
        'baseline' and 'principal'

    Returns:
        _type_: _description_
    """
    return (
        df.groupby(["pod", "measure"])
        .agg({"baseline": "sum", "principal": "sum"})
        .astype(int)
    )


def process_default(full_results: dict, trust: str) -> pd.DataFrame:
    """Processes the 'default' results from the old JSON format of model results
    TODO: Deprecate function; we should be working with Parquet format of model results

    Args:
        full_results (dict): Full JSON dictionary of model results
        trust (str): Provider code, usually the "dataset" from the model params

    Returns:
        pd.DataFrame: Aggregated 'default' results
    """
    df = full_results["results"]["default"]
    df_with_principal = add_principal(df)
    df_aggregated = agg_default(df_with_principal)
    df_aggregated["trust"] = trust
    return df_aggregated


def compare_results(results_dict: dict, trust: str) -> pd.DataFrame:
    """Helper function for the qa_model_runs notebook. Compares model results for
    the same provider/scenario run on two different model versions, and calculates
    differences between pods

    Args:
        results_dict (dict): Dictionary containing the results for the two runs
                            to be compared
        trust (str): Provider code for the trust

    Returns:
        pd.DataFrame: Dataframe with the differences between pods calculated
    """
    df_old = process_default(results_dict[trust]["results_old"], trust)
    df_new = process_default(results_dict[trust]["results_new"], trust)
    combined = df_old.merge(
        df_new.drop(columns=["baseline", "trust"]),
        left_index=True,
        right_index=True,
        suffixes=("_old", "_new"),
    )
    combined["%_diff"] = abs(
        combined[["principal_old", "principal_new"]]
        .pct_change(axis=1)["principal_new"]
        .round(4)
        .fillna(0)
    )
    return combined


def agg_stepcounts(stepcounts: pd.DataFrame) -> pd.DataFrame:
    """Aggregates step_counts in old format of results JSON, grouping by 'change_factor',
    'measure' and 'strategy' and summing the 'principal' column
    TODO: Deprecate function; we should be working with Parquet format of model results

    Args:
        stepcounts (pd.DataFrame): The step counts from the old format of the results JSON

    Returns:
        pd.DataFrame: Aggregated step counts
    """
    return (
        stepcounts.groupby(["change_factor", "measure", "strategy"])
        .sum("principal")
        .astype(int)
    )


def process_stepcounts(full_results: dict) -> pd.DataFrame:
    """Processes the step counts in the full model results JSON
    TODO: Deprecate function; we should be working with Parquet format of model results

    Args:
        full_results (dict): Full JSON dictionary of model results

    Returns:
        pd.DataFrame: Aggregated 'step_counts' from results
    """
    df = pd.DataFrame(full_results["results"]["step_counts"]).fillna("-")
    df_with_principal = add_principal(df)
    return agg_stepcounts(df_with_principal)


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
        df["%_diff"] = abs(
            df[["mean", "baseline"]]
            .pct_change(fill_method=None, axis=1)["baseline"]
            .round(4)
            .fillna(0)
        )
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


def compare_stepcounts(results_dict: dict, trust: str) -> pd.DataFrame:
    """Helper function for the qa_model_runs notebook. Compares step_counts in
    model results for the same provider/scenario run on two different model versions,
    and calculates differences

    Args:
        results_dict (dict): Dictionary containing the results for the two runs to
        be compared
        trust (str): Provider code for the trust

    Returns:
        pd.DataFrame: Dataframe with the differences between step counts
    """
    df_old = process_stepcounts(results_dict[trust]["results_old"])
    df_new = process_stepcounts(results_dict[trust]["results_new"])
    combined = df_old.merge(
        df_new,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_old", "_new"),
    ).fillna(0)
    combined["trust"] = trust
    combined["%_diff"] = abs(
        combined[["principal_old", "principal_new"]]
        .pct_change(axis=1)["principal_new"]
        .round(4)
        .fillna(0)
    )
    return combined


def create_time_profiles(horizon_years: int, year: int) -> dict[str, callable]:
    """Create time profile functions. Creates time_profiles_dict which is taken by
    the get_time_profiles_factor function

    :param horizon_years: how many years the model is running over
    :type horizon_years: int
    :param year: the year (in `(0, horizon_years]`) that we want the factor for
    :return: the time profiles, in a dictionary
    :rtype: dict[str, callable]
    """
    return {
        "none": 1,
        "linear": year / horizon_years,
        "front_loaded": np.sqrt(horizon_years**2 - (horizon_years - year) ** 2)
        / horizon_years,
        "back_loaded": 1 - np.sqrt(horizon_years**2 - year**2) / horizon_years,
        "step": lambda y: int(year >= y),
    }


def get_time_profiles_factor(time_profile_type, time_profiles_dict, baseline_year):
    """
    Gets factor to use for each year, given the time_profile type selected by the user,
    the baseline year, and the horizon year

    Args:
        time_profile_type (str): The "time profile" type specified in the params. Options
        are "stepXXXX" where XXXX is the year the step change occurs, "linear",
        "front_loaded" or "back_loaded"
        time_profiles_dict (dict): The dict created by the create_time_profiles function
        baseline_year (int): The baseline year, from the results JSON file

    Returns:
        float: Factor that should be used for the specific parameter
    """
    if time_profile_type[:4] == "step":
        step_year = int(time_profile_type[4:]) - baseline_year
        factor = time_profiles_dict["step"](step_year)
    else:
        factor = time_profiles_dict[time_profile_type]
    return factor
