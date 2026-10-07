1. Run scenario with full model results. For a NHP model run to be converted to capacity, full model results must exist for the scenario. [Use nhp_products to run a scenario with full results](https://github.com/The-Strategy-Unit/nhp_products?tab=readme-ov-file#run-scenario-with-full-results).

1. Run `generate_token.ps1` PowerShell script in nhp_functional_area_mapping to generate temporary SAS tokens for Databricks

1. Log in to Databricks and run the `produce_functional_area_aggs.ipynb` notebook. Set the path to the full model results folder and the sites you wish to use for each activity type using the widgets at the top of the notebook. Sites can can be obtained from the tagged runs table.

1. The Databricks notebook should have added a new entry to the functional area mappings table on Azure Table Storage. Note the GUID from Azure Table Storage.

1. Run the module from nhp_capacity_conversion_logic using the GUID. e.g. `uv run -m nhp.capacity_conversion.aae GUID`