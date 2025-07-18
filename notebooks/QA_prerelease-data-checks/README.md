# Modal data QA checks

This notebook automates the process of comparing the `dev` data for the NHP model
with the data from the latest two versions. It draws the relevant data for the NHP schemes from Azure, aggregates it, and collates it in one dataframe.

The purpose of this is to flag any significant changes in the model data from one version to another, so that:
- We can gauge the impact of changes to data on schemes
- Any changes can be flagged with MRMs if appropriate.
