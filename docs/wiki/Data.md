⚠️ This wiki is for internal use only ⚠️

# Data pipeline

1. All our data processing happens on Databricks, using the reproducible pipelines in [nhp_data](https://github.com/The-Strategy-Unit/nhp_data)
1. The data pipelines are deployed as Databricks asset bundles to prod on GitHub tagged release. The main pipelines are as follows, they need to be run in order: 
    1. Generate NHP data (processes the data, saves as tables on Databricks)
    1. Extract NHP for containers (actually extracts the data from Databricks tables into parquet format in Azure storage)

# Data storage

The NHP model data is stored in Azure, in versioned folders.

# Synthetic data

This is currently for dev purposes only. It is not truly synthetic and should be treated as sensitive data. It is created as part of the data extraction pipeline and is also on Azure.

# Data access through products

A person (Provider/ICB) assigned to groups on Posit Connect when their account is set up. Each Provider group corresponds to the Provider dataset and where a person requires access across several Providers they will be added to each group. In most cases, data set corresponds to a particular Trust code. 

There are particular special cases where the user is assigned to either `nhp_devs` or `nhp_power_users` and this will give them greater access to datasets in some products (depending on how they are coded).
