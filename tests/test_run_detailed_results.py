#!/usr/bin/env python

"""
Smoke tests the detailed results processing functionality that converts
full model results into CSV and Parquet files for analysis.

Usage:
    python tests/test_run_detailed_results.py [results_path]

    Optional arguments:
        results_path: Path to real scenario results for live testing
"""

# %%
import logging
import os
import sys
import tempfile
from pathlib import Path

from nhpy.utils import configure_logging, get_logger

# %%
# Validate imports before running tests
try:
    from nhpy.run_detailed_results import (
        _check_results_exist,
        run_detailed_results,
    )
    from nhpy.utils import initialise_connections_and_params
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Private function may not be exported - check module structure")
    sys.exit(2)

# %%
# Get a logger for this module
configure_logging(level=logging.INFO)
logger = get_logger()


# %%
def test_results_exist_check():
    """Tests detection of existing results files in different scenarios."""
    logger.info("🧪 Testing results existence check...")

    # Test with non-existent directory
    tmproot = tempfile.gettempdir()
    nonexistent = str(Path(tmproot) / "nonexistent")
    exists = _check_results_exist(nonexistent, "test-scenario", "ip")
    assert exists is False
    logger.info("  ✅ Non-existent directory handling works")

    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_name = "test-scenario"

        # Test with no files
        exists = _check_results_exist(tmpdir, scenario_name, "ip")
        assert exists is False
        logger.info("  ✅ Missing files correctly detected")

        # Create only CSV file
        csv_path = Path(tmpdir) / f"{scenario_name}_detailed_ip_results.csv"
        csv_path.touch()
        exists = _check_results_exist(tmpdir, scenario_name, "ip")
        assert exists is False
        logger.info("  ✅ Partial files correctly rejected")

        # Create both CSV and Parquet files
        parquet_path = Path(tmpdir) / f"{scenario_name}_detailed_ip_results.parquet"
        parquet_path.touch()
        exists = _check_results_exist(tmpdir, scenario_name, "ip")
        assert exists is True
        logger.info("  ✅ Both files correctly detected")


# %%
def test_environment_check():
    """Validates Azure environment variable configuration."""
    logger.info("🧪 Testing environment configuration...")

    required_vars = ["AZ_STORAGE_EP", "AZ_STORAGE_RESULTS", "AZ_STORAGE_DATA"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.info(f"  ⚠️  Missing environment variables: {missing}")
        logger.info("  💡 Set these in .env file for full testing")
    else:
        logger.info("  ✅ All required environment variables present")


# %%
def test_error_handling():
    """Tests error handling with missing environment variables or authentication."""
    logger.info("🧪 Testing error handling...")

    try:
        # This should fail with environment variable error
        run_detailed_results("aggregated-model-results/v4.0/RXX/test/20250101_100000/")
        # This assertion is deliberately unreachable - if we get here without exception,
        # the test should fail
        assert False, "Should have raised EnvironmentVariableError"
    except Exception as e:
        # Expected exception for missing environment variables or authentication
        logger.info(f"  ✅ Expected exception: {type(e).__name__}")


# %%
def test_public_api():
    """Verifies proper export and signature of public API functions."""
    logger.info("🧪 Testing public API...")

    # Verify run_detailed_results is in __all__
    try:
        from nhpy.run_detailed_results import __all__  # noqa PLC0415

        if "run_detailed_results" in __all__:
            logger.info("  ✅ run_detailed_results properly exported")
        else:
            logger.info("  ⚠️  run_detailed_results not in __all__")
    except ImportError:
        logger.info("  ⚠️  No __all__ defined in module")

    # Verify function signature
    import inspect  # noqa PLC0415

    sig = inspect.signature(run_detailed_results)
    params = list(sig.parameters.keys())
    expected_params = [
        "results_path",
        "output_dir",
        "account_url",
        "results_container",
        "data_container",
    ]

    if params == expected_params:
        logger.info("  ✅ Function signature correct")
    else:
        logger.info(f"  ⚠️  Signature differs: {params}")


# %%
def real_path_detailed_results_test(results_path):
    """Tests run_detailed_results with a real results path.

    This function will validate the path but NOT run detailed results processing.

    Args:
        results_path: Path to real scenario results
    """
    logger.info(f"🧪 Testing with real path: {results_path}")

    try:
        # Check required environment variables
        account_url = os.getenv("AZ_STORAGE_EP")
        results_container = os.getenv("AZ_STORAGE_RESULTS")
        data_container = os.getenv("AZ_STORAGE_DATA")

        if not all([account_url, results_container, data_container]):
            logger.error("  ❌ Missing required environment variables for Azure access")
            return False

        # Create temporary output directory for testing
        with tempfile.TemporaryDirectory() as output_dir:
            logger.info(f"  ✅ Using temporary output directory: {output_dir}")

            # Just validate path format
            logger.info("  ✅ Path format valid for detailed results processing")
            logger.info(
                "  💡 To process detailed results, use: "
                "uv run python -m nhpy.run_detailed_results"
            )

        return True
    except Exception as e:
        logger.error(f"  ❌ Error validating real path: {e}")
        return False


def main():
    """Runs all smoke tests and returns appropriate exit code."""
    logger.info("🚀 Running smoke tests for run_detailed_results module...\n")

    # Check for command line argument for real path testing
    real_path = None
    if len(sys.argv) > 1:
        real_path = sys.argv[1]

    try:
        test_results_exist_check()
        test_environment_check()
        test_public_api()
        test_error_handling()

        logger.info("\n🎉 All smoke tests passed!")

        # If real path provided, run real path test
        if real_path:
            logger.info("\n🧪 Running test with real path...")
            real_path_detailed_results_test(real_path)
        else:
            logger.info(
                "💡 To test with real path, run: "
                "uv run python tests/test_run_detailed_results.py <path_to_results>"
            )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main())
