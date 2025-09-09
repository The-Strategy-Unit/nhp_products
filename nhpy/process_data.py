"""Functions to process model data in format suitable for model results"""

from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd

n_runs = 256


# this is from model.helpers.age_groups
def age_groups(age: pd.Series) -> pd.Series:
    """Cut age into groups

    Takes a pandas Series of age's and cut's into discrete intervals

    :param age: a Series of ages
    :type age: pandas.Series

    :returns: a Series of age groups
    :rtype: pandas.Series
    """

    return pd.cut(
        age.fillna(-1),
        [-1, 0, 1, 5, 10, 16, 18, 35, 50, 65, 75, 85, 1000],
        right=False,
        labels=[
            "Unknown",
            "0",
            "1-4",
            "5-9",
            "10-15",
            "16-17",
            "18-34",
            "35-49",
            "50-64",
            "65-74",
            "75-84",
            "85+",
        ],
    ).astype(str)


# this is from InpatientsModel._add_pod_to_data()


def add_pod_to_data_ip(data: pd.DataFrame) -> pd.DataFrame:
    """Adds the POD column to data"""

    data["pod"] = "ip_" + data["group"] + "_admission"
    classpat = data["classpat"]
    data.loc[classpat == "2", "pod"] = "ip_elective_daycase"
    data.loc[classpat == "3", "pod"] = "ip_regular_day_attender"
    data.loc[classpat == "4", "pod"] = "ip_regular_night_attender"
    # handle the type conversions: change the pods (from InpatientsModel.process_results)
    data.loc[data["classpat"] == "-2", "pod"] = "ip_elective_daycase"
    data.loc[data["classpat"] == "-1", "pod"] = "op_procedure"

    return data


# Adapted from InpatientsModel.process_results
def process_ip_results(data: pd.DataFrame, los_groups: defaultdict) -> pd.DataFrame:
    """
    Processes the row level inpatients data into
    a format suitable for aggregation in results files

    :param data: Data to be processed. Format should be similar to Model.data
    :type data: pd.DataFrame
    """

    data = data.assign(
        los_group=lambda x: x["speldur"].map(los_groups),
        admissions=1,
        beddays=lambda x: x["speldur"] + 1,
        procedures=lambda x: x["has_procedure"],
    )

    return data


def process_ip_activity_avoided(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the IP activity_avoided, adding pod and age_group,
    and grouping by HRG, los_group and pod

    Args:
        data: the IP rows avoided

    Returns:
        The processed and aggregated data
    """

    los_groups_aa = defaultdict(
        lambda: "1+ days",
        {
            0: "0 days",
        },
    )
    data = process_ip_results(data, los_groups_aa)
    data = data.groupby(["sushrg", "los_group", "pod"], dropna=False)[
        ["admissions", "beddays", "procedures"]
    ].sum()

    # handle the outpatients rows
    data.loc[
        data.index.get_level_values("pod") == "op_procedure",
        ["beddays", "procedures"],
    ] = 0
    data = data.melt(ignore_index=False, var_name="measure")
    data.loc[data.index.get_level_values("pod") == "op_procedure", "measure"] = (
        "attendances"
    )
    data = data.reset_index()
    data = data.groupby(["pod", "los_group", "sushrg", "measure"]).sum()

    return data


def process_model_runs_dict(
    model_runs: dict, columns: list, all_runs_kept: bool = False
) -> pd.DataFrame:
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

    model_runs_df = (
        pd.DataFrame(
            model_runs.keys(),
            columns=columns,
        )
        .set_index(columns)
        .sort_index()
    )
    for k, v in model_runs.items():
        v_ = np.array(v)
        if not all_runs_kept:
            # make sure all 256 runs are accounted for! Some groups don't show up
            # in the individual model runs.
            if len(v_) < n_runs:
                zeros = np.zeros(n_runs - len(v_))
                v_ = np.concatenate((v_, zeros))
        if len(v_) != n_runs:
            raise ValueError(f"Length of array for {k} is not equal to n_runs")
        x = cast(np.ndarray, np.percentile(v_, [10, 50, 90]))
        model_runs_df.loc[k, "lwr_ci"] = x[0]
        model_runs_df.loc[k, "median"] = x[1]
        model_runs_df.loc[k, "mean"] = v_.mean()
        model_runs_df.loc[k, "upr_ci"] = x[2]

    return model_runs_df


def process_ip_detailed_results(data: pd.DataFrame) -> pd.DataFrame:
    """Process the IP detailed results, adding pod and age_group, and grouping by
    sitetret, age_group, sex, pod, tretspef, los_group, maternity_delivery_in_spell,
    and measure

    Args:
        data: the IP activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """
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
    data = process_ip_results(data, los_groups_detailed)
    data = data.groupby(
        [
            "sitetret",
            "age_group",
            "sex",
            "pod",
            "tretspef",
            "los_group",
            "maternity_delivery_in_spell",
        ],
        dropna=False,
    )[["admissions", "beddays", "procedures"]].sum()

    # handle the outpatients rows
    data.loc[
        data.index.get_level_values("pod") == "op_procedure",
        ["beddays", "procedures"],
    ] = 0
    data = data.melt(ignore_index=False, var_name="measure")
    data.loc[data.index.get_level_values("pod") == "op_procedure", "measure"] = (
        "attendances"
    )
    # remove any row where the measure value is 0
    data = data[data["value"] > 0].reset_index()
    data = data.groupby(
        [
            "sitetret",
            "age_group",
            "sex",
            "pod",
            "tretspef",
            "los_group",
            "maternity_delivery_in_spell",
            "measure",
        ]
    ).sum()

    return data


def process_op_detailed_results(data: pd.DataFrame) -> pd.DataFrame:
    """Process the OP detailed results, adding pod and age_group, and grouping by
    sitetret, age_group, sex, pod, tretspef, and measure

    Args:
        data: the IP activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """

    # From aggregate
    measures = data.melt(["rn"], ["attendances", "tele_attendances"], "measure")
    data = (
        data.drop(["attendances", "tele_attendances"], axis="columns")
        .merge(measures, on="rn")
        .groupby(
            ["sitetret", "pod", "age_group", "tretspef", "measure"],
        )[["value"]]
        .sum()
        .sort_index()
    )
    return data


def process_op_converted_from_ip(data: pd.DataFrame) -> pd.Series:
    """Process the OP activity converted from IP, adding pod and age_group, and
    grouping by sitetret, age_group, sex, pod, tretspef, and measure.

    Args:
        data: the OP activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """

    # activity converted to OP should only be procedures
    data["pod"] = "op_procedure"
    data["age_group"] = age_groups(data["age"])
    # op conversion should only ever be attendances, not teleattendances
    data["measure"] = "attendances"
    return (
        data.rename(columns={"attendances": "value"})
        .groupby(["sitetret", "pod", "age_group", "tretspef", "measure"])["value"]
        .sum()
    )

    return data


def combine_converted_with_main_results(
    df_converted: pd.Series, df: pd.DataFrame
) -> pd.DataFrame:
    """Combines the activity converted from IP to OP/AAE
    with the main OP/AAE activity results

    Args:
        df_converted: the OP/AAE activity converted from IP in each Monte Carlo
        simulation
        df: the OP/AAE activity in each Monte Carlo simulation

    Returns:
        DataFrame: The combined dataframe with both sets of activity
    """
    # Create a copy to avoid modifying the input
    result = df.copy()

    # Convert Series to DataFrame with same columns as df
    df_converted_frame = pd.DataFrame(
        {col: df_converted for col in df.columns}, index=df_converted.index
    )

    # Use pandas' add method with fill_value=0 to handle missing indices
    result = result.add(df_converted_frame, fill_value=0)

    return result


def process_aae_results(data: pd.DataFrame) -> pd.DataFrame:
    """Process the AAE detailed results, adding pod and age_group,
    and grouping by sitetret, pod, age_group, attendance_category,
    aedepttype, acuity, and measure

    Args:
        data: the AAE activity in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """

    data["measure"] = "walk-in"
    data.loc[data["is_ambulance"], "measure"] = "ambulance"

    return data.groupby(
        [
            "sitetret",
            "pod",
            "age_group",
            "attendance_category",
            "aedepttype",
            "acuity",
            "measure",
        ]
    )[["arrivals"]].sum()


def process_aae_converted_from_ip(data: pd.DataFrame) -> pd.Series:
    """Process the AAE SDEC activity converted from IP, adding pod and age_group,
    and grouping by sitetret, age_group, pod, aedepttype, attendance_category, acuity
    and measure.

    Args:
        data: the AAE SDEC activity converted from IP in each Monte Carlo simulation

    Returns:
        The processed and aggregated data
    """

    # activity converted to AAE should only be aae_type-05
    data["pod"] = "aae_type-05"
    data["age_group"] = age_groups(data["age"])
    data = data.rename(columns={"group": "measure"})
    data = data.groupby(
        [
            "sitetret",
            "pod",
            "age_group",
            "attendance_category",
            "aedepttype",
            "acuity",
            "measure",
        ]
    )["arrivals"].sum()

    return data


def add_pod_to_data_op(data):
    data.loc[data["is_first"], "pod"] = "op_first"
    data.loc[~data["is_first"], "pod"] = "op_follow-up"
    data.loc[data["has_procedures"], "pod"] = "op_procedure"

    return data


def get_op_mitigators_consultant(df):
    """Consultant to consultant mitigators."""

    cons_df = df[df["is_cons_cons_ref"]][
        ["type", "dataset", "attendances", "tele_attendances"]
    ]
    cons_df = (
        cons_df.groupby(["dataset", "type"])
        .sum()
        .rename(
            columns={
                "attendances": "consultant_to_consultant_referrals_attendances",
                "tele_attendances": "consultant_to_consultant_referrals_tele_attendances",
            }
        )
    )

    return cons_df.sort_index()


def get_op_mitigators_followup(df):
    """Followup reduction mitigators. NOT FIRST, NO PROCEDURES"""

    followup_df = df[~df["has_procedures"]]
    followup_df = followup_df[~followup_df["is_first"]][
        ["dataset", "type", "attendances", "tele_attendances"]
    ]
    followup_df = (
        followup_df.groupby(["dataset", "type"])
        .sum()
        .rename(
            columns={
                "attendances": "followup_reduction_attendances",
                "tele_attendances": "followup_reduction_tele_attendances",
            }
        )
    )

    return followup_df.sort_index()


def get_op_mitigators_gp(df):
    """GP referred mitigators. IS FIRST AND IS GP REFERRED"""

    gp_df = df[df["is_gp_ref"]]
    gp_df = gp_df[gp_df["is_first"]][
        ["dataset", "type", "attendances", "tele_attendances"]
    ]
    gp_df = (
        gp_df.groupby(["dataset", "type"])
        .sum()
        .rename(
            columns={
                "attendances": "gp_referred_attendances",
                "tele_attendances": "gp_referred_tele_attendances",
            }
        )
    )

    return gp_df.sort_index()


def get_op_mitigators_tele(op_data):
    """Convert to teleattendance mitigators. NOT PROCEDURES, NOT TELE"""

    data = op_data.copy()
    tele_df = data[~data["has_procedures"]]
    tele_df = tele_df.groupby(["dataset", "type"])[["attendances"]].sum()
    tele_df = tele_df.rename(
        columns={
            "attendances": "convert_to_tele: attendances",
        }
    )

    return tele_df.sort_index()


def get_all_op_mitigators(op_data):
    """Get table of all OP mitigators, grouped by dataset and type"""
    tele_mitigators = get_op_mitigators_tele(op_data)
    gp_mitigators = get_op_mitigators_gp(op_data)
    cons_mitigators = get_op_mitigators_consultant(op_data)
    followup_mitigators = get_op_mitigators_followup(op_data)
    all_op_mitigators = pd.concat(
        [tele_mitigators, gp_mitigators, cons_mitigators, followup_mitigators], axis=1
    )

    return all_op_mitigators


def get_ae_aggregation(df, col_name):
    """Get counts of A&E arrivals by column"""

    df = df[df[col_name]].groupby(["dataset", "hsagrp"])[["arrivals"]].sum()
    df = df.rename(columns={"arrivals": col_name})

    return df


def get_all_ae_mitigators(ae_data):
    """Get table of all A&E mitigators, grouped by dataset and hsagrp"""

    ae_df = pd.DataFrame(index=ae_data.groupby(["dataset", "hsagrp"]).count().index)

    for col in [
        "is_discharged_no_treatment",
        "is_frequent_attender",
        "is_left_before_treatment",
        "is_low_cost_referred_or_discharged",
    ]:
        df = get_ae_aggregation(ae_data, col)
        ae_df = ae_df.merge(df, left_index=True, right_index=True, how="outer").fillna(0)

    return ae_df
