#!/usr/bin/env python

"""
Smoke tests the ability to detect if full model results exist for a scenario by
analysing blob paths and validating path formats.

Usage:
    python tests/test_check_full_results.py [results_path]

    Optional arguments:
        results_path: Path to real scenario results for live testing
"""

# %%
import logging
import os
import sys

from fastcore.test import test_eq as fc_eq
from fastcore.test import test_is as fc_is
from fastcore.test import test_fail as fc_fail
from fastcore.test import test_ne as fc_ne
from fastcore.test import test_close as fc_close
from fastcore.test import test_warns as fc_warns
from fastcore.test import test_stdout as fc_stdout
from fastcore.test import test_fig_exists as fc_fig_exists
from fastcore.test import test_eq_type as fc_eq_type
from nhpy.utils import configure_logging, get_logger

# %%
# Validate imports before running tests
try:
    from nhpy.check_full_results import (
        _analyse_blob_paths_for_full_results,
        check_full_results,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Private function may not be exported - check module structure")
    sys.exit(2)

# %%
# Get a logger for this module
configure_logging(level=logging.INFO)
logger = get_logger()

# %%
# Test data constants
FULL_RESULTS_PATHS = [
    "some/path/model_run=1/0.parquet",
    "some/path/model_run=2/0.parquet",
    "some/path/model_run=3/0.parquet",
]

FLAT_PATHS = [
    "file1.json",
    "file2.csv",
    "results.txt",
]

NO_MODEL_RUN_PATHS = [
    "some/path/data.csv",
    "another/path/results.json",
    "deep/nested/path/file.txt",
]

NO_PARQUET_PATHS = [
    "some/path/model_run=1/data.csv",
    "some/path/model_run=2/results.json",
    "some/path/model_run=3/info.txt",
]


# %%
def test_blob_path_analysis():
    """Tests blob path analysis for different result structure scenarios."""
    logger.info("🧪 Testing blob path analysis...")

    # Test 1: Full results (has model_run directories with parquet files)
    is_full, explanation = _analyse_blob_paths_for_full_results(FULL_RESULTS_PATHS)
    fc_eq(is_full, True)
    fc_eq(explanation, "✅")
    logger.info("  ✅ Full results detection works")

    # Test 2: No directories (flat structure)
    is_full, explanation = _analyse_blob_paths_for_full_results(FLAT_PATHS)
    fc_eq(is_full, False)
    fc_is(type(explanation), str)
    fc_eq("No directories found" in explanation, True)
    logger.info("  ✅ Flat structure detection works")

    # Test 3: Has directories but no model_run
    is_full, explanation = _analyse_blob_paths_for_full_results(NO_MODEL_RUN_PATHS)
    fc_eq(is_full, False)
    fc_eq("no 'model_run=' subdirectories" in explanation, True)
    logger.info("  ✅ No model_run detection works")

    # Test 4: Has model_run directories but no parquet files
    is_full, explanation = _analyse_blob_paths_for_full_results(NO_PARQUET_PATHS)
    fc_eq(is_full, False)
    fc_eq("no parquet file" in explanation, True)
    logger.info("  ✅ No parquet file detection works")

    # Test 5: Empty paths list
    is_full, explanation = _analyse_blob_paths_for_full_results([])
    fc_eq(is_full, False)
    fc_eq("No directories found" in explanation, True)
    logger.info("  ✅ Empty paths handling works")


# %%
def test_blob_analysis_edge_cases():
    """Tests edge cases for blob path analysis logic."""
    logger.info("🧪 Testing blob analysis edge cases...")

    # Test with mixed valid and invalid paths
    mixed_paths = [
        "some/regular/file.txt",
        "path/model_run=1/0.parquet",  # This should make it qualify as full results
        "another/regular/file.csv",
    ]
    is_full, explanation = _analyse_blob_paths_for_full_results(mixed_paths)
    fc_eq(is_full, True)
    fc_eq(explanation, "✅")
    logger.info("  ✅ Mixed paths analysis works")

    # Test with model_run but wrong parquet file name
    wrong_parquet_paths = [
        "some/path/model_run=1/1.parquet",  # Should be 0.parquet
        "some/path/model_run=2/data.parquet",
    ]
    is_full, explanation = _analyse_blob_paths_for_full_results(wrong_parquet_paths)
    fc_eq(is_full, False)
    fc_eq("no parquet file" in explanation, True)
    logger.info("  ✅ Wrong parquet file name detection works")


# %%
def test_environment_check():
    """Validates environment configuration for Azure access."""
    logger.info("🧪 Testing environment configuration...")

    required_vars = ["AZ_STORAGE_EP", "AZ_STORAGE_RESULTS"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.info(f"  ⚠️  Missing environment variables: {missing}")
        logger.info("  💡 Set these in .env file for full testing")
    else:
        logger.info("  ✅ All required environment variables present")


# %%
def test_error_handling():
    """Tests error handling with invalid scenario paths."""
    logger.info("🧪 Testing error handling...")

    # Use fastcore.test's fail to validate exception raising
    from azure.core.exceptions import ResourceNotFoundError
    fc_fail(lambda: check_full_results("invalid/path/format"),
              exc=ResourceNotFoundError)
    logger.info("  ✅ Invalid path ResourceNotFoundError correctly raised")


# %%
def test_path_validation():
    """Tests path validation with edge case inputs."""
    logger.info("🧪 Testing path validation...")

    # Test with empty string
    fc_fail(lambda: check_full_results(""),
              exc=(ValueError, TypeError),
              contains="path")
    logger.info("  ✅ Empty path correctly rejected")

    # Test with None (will fail at _load_scenario_params, not type checking)
    fc_fail(lambda: check_full_results(None),
              exc=(ValueError, TypeError, AttributeError))
    logger.info("  ✅ None path correctly rejected")


# %%
def test_public_api():
    """Verifies that public API functions are properly exported."""
    logger.info("🧪 Testing public API...")

    # Verify check_full_results is in __all__
    try:
        from nhpy.check_full_results import __all__  # noqa PLC0415
        assert "check_full_results" in __all__, "check_full_results should be in __all__"
        logger.info("  ✅ check_full_results properly exported")
    except ImportError:
        logger.info("  ⚠️  No __all__ defined in module")

    # Verify function signature
    import inspect  # noqa PLC0415

    sig = inspect.signature(check_full_results)
    params = list(sig.parameters.keys())
    expected_params = ["scenario_path", "account_url", "container_name"]

    fc_eq(params, expected_params)
    logger.info("  ✅ Function signature correct")


# %%
def check_full_results_with_real_path(results_path):
    """Tests check_full_results with a real results path.

    Args:
        results_path: Path to real scenario results
    """
    logger.info(f"🧪 Testing with real path: {results_path}")

    try:
        result = check_full_results(
            results_path,
            account_url=os.getenv("AZ_STORAGE_EP"),
            container_name=os.getenv("AZ_STORAGE_RESULTS"),
        )
        logger.info(f"  ✅ Real path check result: {result}")
        # Report if full results exist
        if result:
            logger.info("  ✅ Full model results exist for this scenario")
        else:
            logger.info("  ⚠️ Full model results do not exist for this scenario")
        return True
    except Exception as e:
        logger.error(f"  ❌ Error checking real path: {e}")
        return False


def main():
    """Runs all smoke tests and returns appropriate exit code."""
    logger.info("🚀 Running smoke tests for check_full_results module...\n")

    # Check for command line argument for real path testing
    real_path = None
    if len(sys.argv) > 1:
        real_path = sys.argv[1]

    try:
        test_blob_path_analysis()
        test_blob_analysis_edge_cases()
        test_environment_check()
        test_public_api()
        test_error_handling()
        test_path_validation()

        logger.info("\n🎉 All smoke tests passed!")

        # If real path provided, run real path test
        if real_path:
            logger.info("\n🧪 Running test with real path...")
            check_full_results_with_real_path(real_path)
        else:
            logger.info(
                "💡 To test with real Azure Storage, run: "
                "uv run python tests/test_check_full_results.py <path_to_results>"
            )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main())
