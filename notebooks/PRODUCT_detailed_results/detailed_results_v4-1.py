"""
This module is for producing detailed results for model v4.0.0.

Assumes you have already authenticated via Azure CLI -
[instructions here](https://github.com/The-Strategy-Unit/data_science/blob/fa37cbc01513127626364049124d71f06a35183a/blogs/posts/2024-05-22_storing-data-safely/azure_python.ipynb#L43-L47).
Outputs into a `data/` folder the detailed aggregations of IP, OP, and AAE model
results in CSV and Parquet formats.

Also assumes the scenario has already been run with `full_model_results = True`.

You can check if this has happened using `nhpy.check_full_results`, and if not,
produce full model results using `nhpy.run_full_results`
"""

# %%
# Imports

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from tqdm import tqdm

from nhpy import az, process_data, process_results

load_dotenv()

# %% [markdown]
# Aggregated model results path
# ⚠️ Set this to the path where the aggregated model results are saved
# NOTE: This should already have full_model_results available! If not, please run the
# PRODUCT_run-scenario-with-full-results notebook copy the new_json_path variable that
# gets output at the end and use that for the results_path variable here
# %%
agg_results_folder = "aggregated-model-results/vX.X/RXX/scenarioname/datetime/"

# %% [markdown]
# Setup

# %%
# Load env vars
account_url = os.getenv("AZ_STORAGE_EP", "")
results_container = os.getenv("AZ_STORAGE_RESULTS", "")
data_container = os.getenv("AZ_STORAGE_DATA", "")
api_key = os.getenv("API_KEY", "")

# %%
# Connections and params
results_connection = az.connect_to_container(account_url, results_container)
data_connection = az.connect_to_container(account_url, data_container)
params = az.load_agg_params(results_connection, agg_results_folder)

# %%
# Get lots of info from the results file
scenario_name = params["scenario"]
trust = params["dataset"]
model_version = params["app_version"]
baseline_year = params["start_year"]
run_id = params["create_datetime"]

# %%
# Patch model version for loading the data. Results folder name truncated,
# e.g. v3.0 does not show the patch version. But data stores in format v3.0.1
model_version_data = az.find_latest_version(data_connection, params["app_version"])
print(f"Using data: {model_version_data}")
if model_version_data == "N/A":
    raise FileNotFoundError("Results folder not found")

# %%
# Add `data/` folder if it doesn't exist
Path("notebooks/PRODUCT_detailed_results/data/").mkdir(parents=True, exist_ok=True)

# %%
# Add principal to the "vanilla" model results
actual_results_df = az.load_agg_results(results_connection, agg_results_folder)
actual_results_df = process_results.convert_results_format(actual_results_df)

# %% [markdown]
# ## Inpatients

# %%
# Load original data
original_df = az.load_data_file(
    data_connection, model_version_data, trust, "ip", baseline_year
)

# %%
# Load all model runs, using batching
# In [1]: %%timeit
# 7min 21s ± 1min 22s per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Pre-allocate dictionary
model_runs = {}

# Pre-create the reference dataframe copy once
reference_df = original_df.copy().drop(columns=["speldur", "classpat"])

# Choose an appropriate batch size for your file sizes and memory constraints
batch_size = 20

# Process all runs
start = time.perf_counter()
for run in tqdm(range(1, 257), desc="IP"):
    # Load with batch functionality - this will cache surrounding runs
    df = az.load_model_run_results_file(
        container_client=results_connection,
        version=model_version,
        dataset=trust,
        scenario_name=scenario_name,
        run_id=run_id,
        activity_type="ip",
        run_number=run,
        batch_size=batch_size,  # This enables batch loading
    )

    # Use the pre-created reference dataframe
    merged = reference_df.merge(df, on="rn", how="inner")
    results = process_data.process_ip_detailed_results(merged)

    # More efficient dictionary update
    results_dict = results.to_dict()
    for k, v in results_dict["value"].items():
        if k not in model_runs:  # Avoid unnecessary .keys() call
            model_runs[k] = []
        model_runs[k].append(v)
end = time.perf_counter()
print(f"All IP model runs were processed in {end - start:.3f} sec")

# %%
# Process model runs dictionary after the loop completes
model_runs_df = process_data.process_model_runs_dict(
    model_runs,
    columns=[
        "sitetret",
        "age_group",
        "sex",
        "pod",
        "tretspef",
        "los_group",
        "maternity_delivery_in_spell",
        "measure",
    ],
)

# %%
# Useful for checking if "main" model results from Azure line up with aggregated model
# results. Not always the same because of rounding

default_beddays_principal = (
    actual_results_df[actual_results_df["measure"] == "beddays"]["mean"].sum().astype(int)
)
detailed_beddays_principal = (
    model_runs_df.loc[
        (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            "beddays",
        ),
        :,
    ]
    .sum()
    .loc["mean"]
    .astype(int)
)

try:
    assert abs(default_beddays_principal - detailed_beddays_principal) <= 1
except AssertionError:
    print(default_beddays_principal)
    print(detailed_beddays_principal)

# %%
# Save

model_runs_df.to_csv(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_ip_results.csv"
)
model_runs_df.to_parquet(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_ip_results.parquet"
)

# %% [markdown]
# ## Outpatients

# %%
original_df = az.load_data_file(
    data_connection, model_version_data, trust, "op", baseline_year
).fillna("unknown")
original_df = original_df.rename(columns={"index": "rn"})


# %%
# In [2]: %% timeit
# 2min 3s ± 7.05 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Pre-allocate dictionary
op_model_runs = {}

# Pre-create the reference dataframe copy once
reference_df = original_df.copy().drop(columns=["attendances", "tele_attendances"])

# Choose an appropriate batch size for your file sizes and memory constraints
# A batch size of 20 means we'll load 20 files at a time
batch_size = 20

# Process all runs
start = time.perf_counter()
for run in tqdm(range(1, 257), desc="OP"):
    # Load with batch functionality - this will cache surrounding runs
    df = az.load_model_run_results_file(
        container_client=results_connection,
        version=model_version,
        dataset=trust,
        scenario_name=scenario_name,
        run_id=run_id,
        activity_type="op",
        run_number=run,
        batch_size=batch_size,  # This enables batch loading
    )

    assert df.shape[0] == original_df.shape[0]

    # Use the pre-created reference dataframe
    merged = reference_df.merge(df, on="rn", how="inner")
    results = process_data.process_op_detailed_results(merged)

    # Load conversion data with batch functionality
    df_conv = az.load_model_run_results_file(
        container_client=results_connection,
        version=model_version,
        dataset=trust,
        scenario_name=scenario_name,
        run_id=run_id,
        activity_type="op_conversion",
        run_number=run,
        batch_size=batch_size,  # This enables batch loading
    )

    df_conv = process_data.process_op_converted_from_ip(df_conv)
    results = process_data.combine_converted_with_main_results(df_conv, results)

    # More efficient dictionary update
    results_dict = results.to_dict()
    for k, v in results_dict["value"].items():
        if k not in op_model_runs:  # Avoid unnecessary .keys() call
            op_model_runs[k] = []
        op_model_runs[k].append(v)
end = time.perf_counter()
print(f"All OP model runs were processed in {end - start:.3f} sec")

# %%
op_model_runs_df = process_data.process_model_runs_dict(
    op_model_runs, columns=["sitetret", "pod", "age_group", "tretspef", "measure"]
)
op_model_runs_df.head()

# %%
# Useful for checking if "main" model results from Azure line up with aggregated model
# results using "full model results"
detailed_attendances_principal = (
    op_model_runs_df.round(1)
    .loc[(slice(None), slice(None), slice(None), slice(None), "attendances"), :]
    .sum()
    .astype(int)
    .loc["mean"]
)
default_attendances_principal = (
    actual_results_df[actual_results_df["measure"] == "attendances"]["mean"]
    .sum()
    .astype(int)
)
# They're not always exactly the same because of rounding
try:
    assert abs(default_attendances_principal - detailed_attendances_principal) <= 1
except AssertionError:
    print(default_attendances_principal)
    print(detailed_attendances_principal)

# %%
op_model_runs_df.to_csv(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_op_results.csv"
)
op_model_runs_df.to_parquet(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_op_results.parquet"
)

# %% [markdown]
# ## AAE

# %%
original_df = az.load_data_file(
    data_connection, model_version_data, trust, "aae", baseline_year
).fillna("unknown")
original_df = original_df.rename(columns={"index": "rn"})

# %%
# In [3]: %%timeit
# 25.3 s ± 2.49 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Pre-allocate dictionary
ae_model_runs = {}

# Pre-create the reference dataframe copy once
reference_df = original_df.drop(columns=["arrivals"])

# Choose an appropriate batch size for your file sizes and memory constraints
# A batch size of 20 means we'll load 20 files at a time
batch_size = 20

# Process all runs
start = time.perf_counter()
for run in tqdm(range(1, 257), desc="A&E"):
    # Load with batch functionality - this will cache surrounding runs
    df = az.load_model_run_results_file(
        container_client=results_connection,
        version=model_version,
        dataset=trust,
        scenario_name=scenario_name,
        run_id=run_id,
        activity_type="aae",
        run_number=run,
        batch_size=batch_size,  # This enables batch loading
    )

    assert len(df) == len(original_df)

    # Use the pre-created reference dataframe
    merged = reference_df.merge(df, on="rn", how="inner")
    results = process_data.process_aae_results(merged)

    # Load conversion data with batch functionality
    df_conv = az.load_model_run_results_file(
        container_client=results_connection,
        version=model_version,
        dataset=trust,
        scenario_name=scenario_name,
        run_id=run_id,
        activity_type="sdec_conversion",
        run_number=run,
        batch_size=batch_size,  # This enables batch loading
    )

    df_conv = process_data.process_aae_converted_from_ip(df_conv)
    results = process_data.combine_converted_with_main_results(df_conv, results)

    # More efficient dictionary update
    results_dict = results.to_dict()
    for k, v in results_dict["arrivals"].items():
        if k not in ae_model_runs:  # Avoid unnecessary .keys() call
            ae_model_runs[k] = []
        ae_model_runs[k].append(v)
end = time.perf_counter()
print(f"All AAE model runs were processed in {end - start:.3f} sec")

# %%
ae_model_runs_df = process_data.process_model_runs_dict(
    ae_model_runs,
    columns=[
        "sitetret",
        "pod",
        "age_group",
        "attendance_category",
        "aedepttype",
        "acuity",
        "measure",
    ],
)


# %%
# Useful for checking if "main" model results from Azure line up with
# aggregated model results using full model results
detailed_ambulance_principal = (
    ae_model_runs_df.loc[
        (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            "ambulance",
        ),
        :,
    ]
    .sum()
    .loc["mean"]
    .round(0)
)
default_ambulance_principal = (
    actual_results_df[actual_results_df["measure"] == "ambulance"]["mean"].sum().round(0)
)

# They're not always exactly the same because of rounding
try:
    assert abs(default_ambulance_principal - detailed_ambulance_principal) <= 1
except AssertionError:
    print("OH NO!!")
    print(default_ambulance_principal)
    print(detailed_ambulance_principal)

# %%
# Useful for checking if "main" model results from Azure line up with
# aggregated model results using full model results
detailed_walkins_principal = (
    ae_model_runs_df.loc[
        (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            "walk-in",
        ),
        :,
    ]
    .sum()
    .loc["mean"]
    .round(0)
)
default_walkins_principal = (
    actual_results_df[actual_results_df["measure"] == "walk-in"]["mean"].sum().round(0)
)

# They're not always exactly the same because of rounding
try:
    assert abs(default_walkins_principal - detailed_walkins_principal) <= 1
except AssertionError:
    print("OH NO!!")
    print(default_walkins_principal)
    print(detailed_walkins_principal)

# %%
# Save
ae_model_runs_df.to_csv(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_ae_results.csv"
)
ae_model_runs_df.to_parquet(
    f"notebooks/PRODUCT_detailed_results/data/{scenario_name}_detailed_ae_results.parquet"
)
