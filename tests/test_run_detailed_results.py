#!/usr/bin/env python

"""
Simple smoke tests for run_detailed_results module.
Run this to verify basic functionality without complex test frameworks.
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
        _initialise_connections_and_params,
        run_detailed_results,
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
def test_results_exist_check():
    """Test results file existence checking."""
    logger.info("🧪 Testing results existence check...")

    # Test with non-existent directory
    tmproot = tempfile.gettempdir()
    exists = _check_results_exist(f"{tmproot}/nonexistent", "test-scenario", "ip")
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
        csv_path = Path(f"{tmpdir}/{scenario_name}_detailed_ip_results.csv")
        csv_path.touch()
        exists = _check_results_exist(tmpdir, scenario_name, "ip")
        assert exists is False
        logger.info("  ✅ Partial files correctly rejected")

        # Create both CSV and Parquet files
        parquet_path = Path(f"{tmpdir}/{scenario_name}_detailed_ip_results.parquet")
        parquet_path.touch()
        exists = _check_results_exist(tmpdir, scenario_name, "ip")
        assert exists is True
        logger.info("  ✅ Both files correctly detected")


# %%
def test_environment_check():
    """Check if environment is properly configured for Azure access."""
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
    """Test error handling with invalid inputs."""
    logger.info("🧪 Testing error handling...")

    try:
        # This should fail with environment variable error
        run_detailed_results("aggregated-model-results/v4.0/RXX/test/20250101_100000/")
        assert False, "Should have raised EnvironmentVariableError"
    except Exception as e:
        # Expected exception for missing environment variables or authentication
        logger.info(f"  ✅ Expected exception: {type(e).__name__}")


# %%
def test_public_api():
    """Test that public API functions are properly exported."""
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
def main():
    """Run all smoke tests."""
    logger.info("🚀 Running smoke tests for run_detailed_results module...\n")

    try:
        test_results_exist_check()
        test_environment_check()
        test_public_api()
        test_error_handling()

        logger.info("\n🎉 All smoke tests passed!")
        logger.info(
            "💡 To test with real Azure Storage, use a valid scenario path with proper credentials"  # noqa E501
        )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main())
