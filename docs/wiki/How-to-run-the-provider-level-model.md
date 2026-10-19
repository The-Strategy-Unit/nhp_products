⚠️ This wiki is for internal use only ⚠️

# Use inputs app

NB: This is the "normal" user journey

1. Login to [inputs app](https://connect.strategyunitwm.nhs.uk/nhp/inputs/)
1. You can either copy a scenario from another user (if in nhp_devs group), create a new scenario, or edit existing
1. Go to "Run Model" and submit model run

# Run the model on Azure Container Instances with nhp_aci (preferred)

1. Prepare your parameters file
1. Install [nhp_aci](https://github.com/The-Strategy-Unit/nhp_aci) - follow README instructions
1. Run `nhp_aci run path/to/params.json` in terminal.

# Run the model locally (for local testing only)

1. Clone [nhp_model](https://github.com/The-Strategy-Unit/nhp_model) Github repo and install 
1. Get data from Azure. See Data page on wiki for more info
1. Run model using `uv run -m nhp.model path/to/params.json -d path/to/data`

# What about the API?

Access to [the api](https://github.com/The-Strategy-Unit/nhp-api-v2) is limited by IP. It was a pain to manually enable IP addresses for team members, so as of August 2025 we bypass this by using nhp_aci instead