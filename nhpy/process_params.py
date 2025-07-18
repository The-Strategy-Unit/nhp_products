"""Processes parameters for NHP model"""


def upgrade_ndg(params: dict) -> dict:
    """Upgrades pre-v3.3 parameters with the old format for non-demographic growth so they can be used with model v3.3 onwards

    Args:
        params (dict): Full model parameters in JSON format

    Returns:
        dict: Full model parameters with the non-demographic adjustment section amended
    """
    ndg_new_format = {
        "variant": "ndg-test",
        "value-type": "year-on-year-growth",
        "values": params["non-demographic_adjustment"].copy(),
    }
    params["non-demographic_adjustment"] = ndg_new_format
    return params
