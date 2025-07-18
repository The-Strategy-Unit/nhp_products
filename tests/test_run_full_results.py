#!/usr/bin/env python

"""
Simple smoke tests for run_full_results module.
Run this to verify basic functionality without complex test frameworks.
"""

# %%
import logging
import os

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
    """Test scenario path parsing."""
    logger.info("🧪 Testing path parsing...")

    # Valid path
    path = env["AZ_VALID_PATH"]
    version, _, scenario, datetime = _extract_scenario_components(path)
    assert version == "v3.6"
    assert scenario == "test-patch-3-6-1"
    assert datetime == "20250630_080446"
    logger.info("  ✅ Valid path parsing works")

    # Invalid path
    try:
        _extract_scenario_components("invalid/path")
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("  ✅ Invalid path correctly rejected")


# %%
def test_param_modification():
    """Test parameter modification for full results."""
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
    """Test result path construction."""
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
    """Check if environment is properly configured."""
    logger.info("🧪 Testing environment configuration...")

    required_vars = ["AZ_STORAGE_EP", "AZ_STORAGE_RESULTS", "API_KEY", "API_URL"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.info(f"  ⚠️  Missing environment variables: {missing}")
        logger.info("  💡 Set these in .env file for full testing")
    else:
        logger.info("  ✅ All required environment variables present")


# %%
def test_dry_run():
    """Test the main function with invalid path (should fail gracefully)."""
    logger.info("🧪 Testing error handling...")

    try:
        # This should fail with a clear error message
        run_scenario_with_full_results("invalid/path/format")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"  ✅ Error handling works: {e}")


# %%
def main():
    """Run all smoke tests."""
    logger.info("🚀 Running smoke tests for run_full_results module...\n")

    try:
        test_path_parsing()
        test_param_modification()
        test_path_construction()
        test_environment_check()
        test_dry_run()

        logger.info("\n🎉 All smoke tests passed!")
        logger.info(
            "💡 To test with real Azure/API, use a valid results path with the CLI"
        )

    except Exception as e:
        logger.info(f"\n❌ Test failed: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    import sys

    sys.exit(main())
