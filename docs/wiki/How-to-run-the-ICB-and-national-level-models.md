⚠️ This wiki is for internal use only ⚠️

# How do the ICB/national level models work?

nhp_model code is a package on GitHub. We tag specific versions of the nhp_model, e.g. v4.2.1.

nhp_model_databricks has as its package dependency the tagged version of nhp_model code, available from GitHub. So for example, you can specify v4.2.1 or v4.1.0 of nhp_model. This is currently specified in the pyproject.toml file.

nhp_model_databricks contains specific Data loading classes for ICB, national and provider level model runs. These tell the nhp_model code to load the data for these runs from Databricks.

nhp_model_databricks repository also has workflow yaml files that tell Databricks what to do. The workflow basically is: look on Azure for params files in a specific folder, then run a specific notebook using those params.

nhp_model_databricks basically uses the same extracted parquet data as the provider level model, just samples from it differently. For some of the reference data it reads tables directly from Databricks.

# Current limitations of the ICB/national level models

- We have to manually update all the references to nhp_model versions whenever we launch a new version. 
- Because nhp_model_databricks lives in a separate repo from nhp_model, it could be hard to develop in both simultaneously.
- Some functionality has not been sorted yet in the databricks model at ICB and national level, namely:
    - Inequalities
    - Full model results for OP and ECDS
- Note that Expat/repat is not relevant to ICB/national level models
- Note also that national model samples data at 0.01 by default and [does not scale the results back up](https://github.com/The-Strategy-Unit/nhp_model_databricks/issues/10), which can be confusing.
- There are no JSON model results for the ICB model run because the file size is too large. There are still JSON model results for the provider and national level models. [Full details](https://github.com/The-Strategy-Unit/nhp_model_databricks/issues/28)

# How do I run a local/national/ICB level run using nhp_model_databricks?

1. Prepare your params. Note [the handy new functions](https://connect.strategyunitwm.nhs.uk/nhp/model_documentation/#generating-sample-parameters) in nhp_model if using sample params
1. Save them in the right folder here: `/Volumes/nhp/model_data/queue` NOTE: I can’t find it on Azure Storage Explorer for some reason, only on Databricks?
1. Go to "Jobs and Pipelines" on Databricks and kick off the correct job. NOTE: Currently we only have dev workflows deployed, these do not auto-update when we do a new release but this is a [planned future development](https://github.com/The-Strategy-Unit/nhp_model_databricks/issues/8). Currently you have to pull from main, then deploy the asset bundles yourself. Currently only YiWen and Tom know how to do this.
1. Wait for workflows to complete. Results can be viewed on Azure as per usual, and on the Outputs app. Currently outputs both JSON and parquet results but we hope to deprecate JSON soon.


# How to develop ICB/national level models 

(this information is correct as of October 2025 - it's very flawed but we're working on improving this)

1. Develop in nhp_model. 
1. Once PR approved and pushed to main, when working in nhp_model_databricks, you need to change references to a specific nhp_model version e.g. v4.2.1 to main instead. 
1. Make your changes to the code in nhp_model_databricks. 
1. Deploy the nhp_model_databricks workflow to dev on Databricks. Then run it.
1. If nhp_model_databricks workflow runs successfully, make PR