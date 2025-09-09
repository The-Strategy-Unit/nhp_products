# Databricks notebook source
# MAGIC %md
# MAGIC # Population based model
# MAGIC ## Link activity avoided back to original data
# MAGIC
# MAGIC This notebook uses Databricks to link the activity avoided saved from each
# Monte Carlo simulation back to the original data, aggregating the inpatients activity
# avoided by HRG, an indicator on whether the length of stay from admission to discharge
# is zero, and pod.
# MAGIC
# MAGIC For this notebook to work we first need to have run the national_run notebook
# from nhp_model, with save_full_model_results set to TRUE for inpatients.
# MAGIC
# MAGIC This notebook currently only works for IP, not OP or AAE.

# COMMAND ----------

# Load the path to the results to use
dbutils.widgets.text("results_folder", "", "Path to Full Model Results Folder")

# COMMAND ----------

import sys

sys.path.append(spark.conf.get("bundle.sourcePath", "."))
import json
import os
import pandas as pd
import pyspark.sql.functions as F
from nhpy import process_data

%load_ext autoreload
%autoreload 2

# COMMAND ----------

# Load and set variables

results_folder = dbutils.widgets.get("results_folder")
file_path = f"/Volumes/nhp/results/files/{results_folder}"
params_path = file_path.replace("full-model-results", "aggregated-model-results")
with open(f"{params_path}/params.json", "rb") as f:
    params = json.load(f)
scenario_name = params["scenario"]

# COMMAND ----------
# Check that there hasn't been a patch release of data - this is very rare
data_version = results_folder.split("/")[1] + ".0"
data_path = f"/Volumes/nhp/model_data/files/{data_version}"

# COMMAND ----------

apc = (
            spark.read.parquet(f"{data_path}/ip")
            .filter(F.col("fyear") == params["start_year"])
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .sample(fraction=0.01, seed=params["seed"])
            .persist()
            .drop("speldur", "classpat")
)

# COMMAND ----------

model_runs = {}
for run in range(1, 257):
    print(run)
    df = spark.read.parquet(f"{file_path}/ip_avoided/model_run={run}")
    df = df.join(apc, on="rn", how="inner")
    merged = df.toPandas()
    merged["sushrg"] = merged["sushrg"].fillna("NULL")
    results_dict = process_data.process_ip_activity_avoided(merged).to_dict()
    for k, v in results_dict["value"].items():
        # handle group not previously seen
        if k not in model_runs.keys():
            model_runs[k] = [0] * (run-1) + [v]
        # handle case where group already seen
        else:
            model_runs[k].append(v)
    for k2 in model_runs.keys():
        # handle if group not in this specific model run
        if k2 not in results_dict["value"].keys():
            model_runs[k2].append(0)



# COMMAND ----------

model_runs_df = process_data.process_model_runs_dict(
    model_runs, columns=["pod", "los_group", "sushrg", "measure"],
    all_runs_kept = True
)
notebook_admissions = model_runs_df.loc[
    (slice(None), slice(None), slice(None), "admissions")].sum().loc["mean"]
notebook_admissions

# COMMAND ----------

# Checking results of this notebook against "principal" from main model results
# avoided_activity

activity_avoided = pd.read_parquet(f"{params_path}/avoided_activity.parquet")
assert (notebook_admissions *100) == (
    activity_avoided[activity_avoided["measure"] == "admissions"]["value"].sum()/256)

# COMMAND ----------


notebook_beddays = model_runs_df.loc[
    (slice(None), slice(None), slice(None), "beddays")].sum().loc["mean"]
assert (notebook_beddays * 100) == (
    activity_avoided[activity_avoided["measure"] == "beddays"]["value"].sum()/256)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving outputs

# COMMAND ----------

if not os.path.exists("data/"):
    os.makedirs("data/")

# COMMAND ----------

# Save results
(model_runs_df * 100).to_csv(
    f"data/{scenario_name}_ip_activity_avoided_hrg.csv", float_format='%.11f'
)
(model_runs_df * 100).to_parquet(
    f"data/{scenario_name}_ip_activity_avoided_hrg.parquet"
)
