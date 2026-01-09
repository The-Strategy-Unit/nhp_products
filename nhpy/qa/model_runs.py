import argparse

from nhpy.run_full_results import _extract_scenario_components
from nhpy.utils import (
    EnvironmentVariableError,
    _load_environment_variables,
    _load_scenario_params,
    get_logger,
)

logger = get_logger()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario_filepaths",
        help="""Filepaths to aggregated results folders for the scenarios to rerun on dev.
        Use the most recent model version. Supply one filepath at minimum. No maximum.
        e.g. 'aggregated-model-results/vX.X/RXX/scenario-name/20250101_100000/'""",
        type=str,
        nargs="+",
    )
    return parser.parse_args()


def load_env_vars():
    env_config = _load_environment_variables()
    account_url = env_config["AZ_STORAGE_EP"]
    container_name = env_config["AZ_STORAGE_RESULTS"]
    # Check that all required parameters are now set. If not, raise an error.
    missing_params = [
        param_name
        for param_name, param_value in {
            "account_url": account_url,
            "container_name": container_name,
        }.items()
        if not param_value
    ]
    if missing_params:
        raise EnvironmentVariableError(
            missing_vars=missing_params,
            message=f"Missing required parameters: {', '.join(missing_params)}",
        )
    return account_url, container_name


def main(scenario_filepaths: list[str]):
    account_url, container_name = load_env_vars()

    existing_params = []
    for results_path in scenario_filepaths:
        # Validate the results path format
        try:
            _extract_scenario_components(results_path=results_path)
        except ValueError as e:
            logger.error(f"run_scenario_with_full_results():Invalid results path: {e}")
            raise

        existing_params.append(
            _load_scenario_params(
                results_path=results_path,
                account_url=account_url,
                container_name=container_name,
            )
        )
    print(existing_params)
    # check app_version consistent across all params to check
    # update params to run on dev instead, change name etc
    # start new model runs on dev using nhp_aci
    # compare default results
    # compare step counts
    # print headlines to terminal and save output files locally
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args.scenario_filepaths)
