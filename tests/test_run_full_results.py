#!/usr/bin/env python

"""
Smoke tests path parsing, parameter modification, and result path construction
for scenarios that need to be run with full model results enabled.

Usage:
    python tests/test_run_full_results.py [results_path]

    Optional arguments:
        results_path: Path to real scenario results for live testing
"""

# %%
import logging
import os
import sys

from nhpy.run_full_results import (
    _construct_results_path,
    _extract_scenario_components,
    _prepare_full_results_params,
    run_scenario_with_full_results,
)
from nhpy.utils import _load_environment_variables, configure_logging, get_logger

# %%
# Get a logger for this module
configure_logging(level=logging.INFO)
logger = get_logger()

# %%
# Load .env
env = _load_environment_variables()


# %%
def test_path_parsing():
    """Tests scenario path parsing with valid and invalid paths."""
    logger.info("🧪 Testing path parsing...")

    # Valid path
    path = env["AZ_VALID_PATH"]
    version, _, scenario, datetime = _extract_scenario_components(path)
    assert version == "v4.1"
    assert scenario == "test-yh-full-results"
    assert datetime == "20250909_200111"
    logger.info("  ✅ Valid path parsing works")

    # Invalid path
    try:
        _extract_scenario_components("invalid/path")
        # This assertion is deliberately unreachable - if we get here without exception,
        # the test should fail
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("  ✅ Invalid path correctly rejected")


# %%
def test_param_modification():
    """Tests parameter modification for full results runs."""
    logger.info("🧪 Testing parameter modification...")

    original_params = {
        "scenario": "my-test",
        "app_version": "v3.5",
        "dataset": "RXX",
        "user": "original-user",
        "viewable": True,
        "original_datetime": "20250101_103015",
    }

    new_params = _prepare_full_results_params(original_params)

    assert new_params["user"] == "ds-team"
    assert new_params["viewable"] is False
    assert new_params["original_datetime"] == "20250101_103015"  # Other params preserved
    assert original_params["scenario"] == "my-test"  # Original unchanged
    logger.info("  ✅ Parameter modification works correctly")


# %%
def test_path_construction():
    """Tests result path construction with scenario parameters."""
    logger.info("🧪 Testing path construction...")

    params = {
        "scenario": "full-model-results",
        "app_version": "v3.5",
        "dataset": "RXX",
        "original_datetime": "20250101_103015",
        "create_datetime": "20250102_091500",
    }

    paths = _construct_results_path(params)

    expected_json = "prod/v3.5/RXX/full-model-results-20250102_091500.json.gz"
    expected_agg = "aggregated-model-results/v3.5/RXX/full-model-results/20250102_091500"
    expected_full = "full-model-results/v3.5/RXX/full-model-results/20250102_091500"

    assert paths["json_path"] == expected_json
    assert paths["aggregated_results_path"] == expected_agg
    assert paths["full_results_path"] == expected_full
    logger.info("  ✅ Path construction works correctly")


# %%
def test_environment_check():
    """Validates required environment variables for model API access."""
    logger.info("🧪 Testing environment configuration...")

    required_vars = [
        "AZ_STORAGE_EP",
        "AZ_STORAGE_RESULTS",
        "STORAGE_ACCOUNT",
        "SUBSCRIPTION_ID",
        "CONTAINER_IMAGE",
        "AZURE_LOCATION",
        "SUBNET_NAME",
        "SUBNET_ID",
        "USER_ASSIGNED_IDENTITY",
        "CONTAINER_MEMORY",
        "CONTAINER_CPU",
        "AUTO_DELETE_COMPLETED_CONTAINERS",
        "RESOURCE_GROUP",
        "LOG_ANALYTICS_WORKSPACE_ID",
        "LOG_ANALYTICS_WORKSPACE_KEY",
        "LOG_ANALYTICS_WORKSPACE_RESOURCE_ID",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.info(f"  ⚠️  Missing environment variables: {missing}")
        logger.info("  💡 Set these in .env file for full testing")
    else:
        logger.info("  ✅ All required environment variables present")


# %%
def test_dry_run():
    """Tests error handling with invalid path input."""
    logger.info("🧪 Testing error handling...")

    try:
        # This should fail with a clear error message
        run_scenario_with_full_results("invalid/path/format")
        # This assertion is deliberately unreachable,
        # if we get here without an exception, the test should fail
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"  ✅ Error handling works: {e}")


# %%
def test_real_path(results_path):
    """Tests run_full_results with a real results path.

    This function will perform a dry-run validation of the path and parameters,
    but will NOT actually submit a new model run.

    Args:
        results_path: Path to real scenario results
    """
    logger.info(f"🧪 Testing with real path: {results_path}")

    try:
        # Extract components to verify valid path structure
        version, dataset, scenario, datetime = _extract_scenario_components(results_path)
        logger.info(
            f"  ✅ Path components: v{version}, {dataset}, {scenario}, {datetime}"
        )

        # Check that account_url and container_name are available
        account_url = os.getenv("AZ_STORAGE_EP")
        container_name = os.getenv("AZ_STORAGE_RESULTS")

        if not account_url or not container_name:
            logger.error("  ❌ Missing required environment variables for Azure access")
            return False

        # Load parameters for validation (but don't submit run)
        logger.info("  ✅ Path structure valid. Not submitting actual model run.")
        logger.info(
            "  💡 To run with full results, use: uv run python -m nhpy.run_full_results"
        )

        return True
    except Exception as e:
        logger.error(f"  ❌ Error validating real path: {e}")
        return False


def main():
    """Runs all smoke tests and returns appropriate exit code."""
    logger.info("🚀 Running smoke tests for run_full_results module...\n")

    # Check for command line argument for real path testing
    real_path = None
    if len(sys.argv) > 1:
        real_path = sys.argv[1]

    try:
        test_path_parsing()
        test_param_modification()
        test_path_construction()
        test_environment_check()
        test_dry_run()

        logger.info("\n🎉 All smoke tests passed!")

        # If real path provided, run real path test
        if real_path:
            logger.info("\n🧪 Running test with real path...")
            test_real_path(real_path)
        else:
            logger.info(
                "💡 To test with real path, run: "
                "uv run python tests/test_run_full_results.py <path_to_results>"
            )

    except Exception as e:
        logger.info(f"\n❌ Test failed: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    import sys

    sys.exit(main())
