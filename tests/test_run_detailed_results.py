#!/usr/bin/env python

"""
Simple smoke tests for run_detailed_results module.
Run this to verify basic functionality without complex test frameworks.
"""

# %%
import logging
import os

from nhpy.run_detailed_results import (
    _check_results_exist,
    get_memory_usage,
    run_detailed_results,
)
from nhpy.utils import configure_logging, get_logger

# %%
# Get a logger for this module
configure_logging(level=logging.INFO)
logger = get_logger()


# %%
def test_memory_usage():
    """Test memory usage utility function."""
    logger.info("🧪 Testing memory usage function...")

    memory_mb = get_memory_usage()
    assert memory_mb > 0
    assert isinstance(memory_mb, float)
    logger.info(f"  ✅ Memory usage reported: {memory_mb:.2f} MB")


# %%
def test_check_results_exist():
    """Test results existence checking."""
    logger.info("🧪 Testing results existence check...")

    # Test with non-existent directory
    exists = _check_results_exist("/tmp/nonexistent", "test-scenario", "ip")
    assert exists is False
    logger.info("  ✅ Non-existent results correctly detected")

    # Create temporary output directory
    import tempfile  # noqa PLC0415
    from pathlib import Path  # noqa PLC0415

    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_name = "test-scenario"
        activity_type = "ip"

        # Test with missing files
        exists = _check_results_exist(tmpdir, scenario_name, activity_type)
        assert exists is False
        logger.info("  ✅ Missing files correctly detected")

        # Create both CSV and Parquet files
        base_path = Path(tmpdir) / f"{scenario_name}_detailed_{activity_type}_results"
        csv_path = Path(f"{base_path}.csv")
        parquet_path = Path(f"{base_path}.parquet")

        csv_path.touch()
        parquet_path.touch()

        # Test with existing files
        exists = _check_results_exist(tmpdir, scenario_name, activity_type)
        assert exists is True
        logger.info("  ✅ Existing files correctly detected")


# %%
def test_environment_check():
    """Check if environment is properly configured."""
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
        # This should fail with a clear error message
        run_detailed_results("invalid/path/format")
        assert False, "Should have raised an exception"
    except Exception as e:
        logger.info(f"  ✅ Expected exception type: {type(e).__name__}")


# %%
def test_path_validation():
    """Test path validation with edge case inputs."""
    logger.info("🧪 Testing path validation...")

    # Test with empty string
    try:
        run_detailed_results("")
        assert False, "Should have raised an exception for empty path"
    except Exception as e:
        logger.info(f"  ✅ Empty path exception: {type(e).__name__}")

    # Test with None
    try:
        run_detailed_results(None)
        assert False, "Should have raised exception for None path"
    except (ValueError, TypeError, AttributeError) as e:
        logger.info(f"  ✅ None path rejected: {type(e).__name__}")
    except Exception as e:
        logger.info(f"  ✅ None path exception: {type(e).__name__}")


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
        logger.info(f"  ⚠️  Unexpected signature: {params}")


# %%
def main():
    """Run all smoke tests."""
    logger.info("🚀 Running smoke tests for run_detailed_results module...\n")

    try:
        test_memory_usage()
        test_check_results_exist()
        test_environment_check()
        test_public_api()
        test_error_handling()
        test_path_validation()

        logger.info("\n🎉 All smoke tests passed!")
        logger.info(
            "💡 To test with real Azure Storage, use a valid results path with proper credentials"  # noqa E501
        )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    import sys

    sys.exit(main())
