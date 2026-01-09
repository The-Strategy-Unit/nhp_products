import argparse

from nhpy.utils import get_logger

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


def main(scenario_filepaths: list[str]):
    print(scenario_filepaths)
    # should allow as many scenarios to be added as we want
    # get params from azure
    # update params to run on dev instead, change name etc
    # start new model runs on dev using nhp_aci
    # compare default results
    # compare step counts
    # print headlines to terminal and save output files locally
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args.scenario_filepaths)
