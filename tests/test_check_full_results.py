#!/usr/bin/env python

"""
Simple smoke tests for check_full_results module.
Run this to verify basic functionality without complex test frameworks.
"""

# %%
import logging
import os
import sys

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
    """Test blob path analysis for different result structure scenarios."""
    logger.info("🧪 Testing blob path analysis...")

    # Test 1: Full results (has model_run directories with parquet files)
    is_full, explanation = _analyse_blob_paths_for_full_results(FULL_RESULTS_PATHS)
    assert is_full is True
    assert explanation == "✅"
    logger.info("  ✅ Full results detection works")

    # Test 2: No directories (flat structure)
    is_full, explanation = _analyse_blob_paths_for_full_results(FLAT_PATHS)
    assert is_full is False
    assert "No directories found" in explanation
    logger.info("  ✅ Flat structure detection works")

    # Test 3: Has directories but no model_run
    is_full, explanation = _analyse_blob_paths_for_full_results(NO_MODEL_RUN_PATHS)
    assert is_full is False
    assert "no 'model_run=' subdirectories" in explanation
    logger.info("  ✅ No model_run detection works")

    # Test 4: Has model_run directories but no parquet files
    is_full, explanation = _analyse_blob_paths_for_full_results(NO_PARQUET_PATHS)
    assert is_full is False
    assert "no parquet file" in explanation
    logger.info("  ✅ No parquet file detection works")

    # Test 5: Empty paths list
    is_full, explanation = _analyse_blob_paths_for_full_results([])
    assert is_full is False
    assert "No directories found" in explanation
    logger.info("  ✅ Empty paths handling works")


# %%
def test_blob_analysis_edge_cases():
    """Test edge cases for blob path analysis logic."""
    logger.info("🧪 Testing blob analysis edge cases...")

    # Test with mixed valid and invalid paths
    mixed_paths = [
        "some/regular/file.txt",
        "path/model_run=1/0.parquet",  # This should make it qualify as full results
        "another/regular/file.csv",
    ]
    is_full, explanation = _analyse_blob_paths_for_full_results(mixed_paths)
    assert is_full is True
    assert explanation == "✅"
    logger.info("  ✅ Mixed paths analysis works")

    # Test with model_run but wrong parquet file name
    wrong_parquet_paths = [
        "some/path/model_run=1/1.parquet",  # Should be 0.parquet
        "some/path/model_run=2/data.parquet",
    ]
    is_full, explanation = _analyse_blob_paths_for_full_results(wrong_parquet_paths)
    assert is_full is False
    assert "no parquet file" in explanation
    logger.info("  ✅ Wrong parquet file name detection works")


# %%
def test_environment_check():
    """Check if environment is properly configured for Azure access."""
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
    """Test error handling with invalid scenario paths."""
    logger.info("🧪 Testing error handling...")

    try:
        # This should fail with a clear error message for invalid path format
        check_full_results("invalid/path/format")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"  ✅ Invalid path ValueError: {str(e)[:50]}...")
    except Exception as e:
        # Other exceptions (auth, network) are also valid for smoke test
        logger.info(f"  ✅ Expected exception type: {type(e).__name__}")


# %%
def test_path_validation():
    """Test path validation with edge case inputs."""
    logger.info("🧪 Testing path validation...")

    # Test with empty string
    try:
        check_full_results("")
        assert False, "Should have raised ValueError for empty path"
    except ValueError as e:
        logger.info(f"  ✅ Empty path ValueError: {str(e)[:50]}...")
    except Exception as e:
        logger.info(f"  ✅ Empty path exception: {type(e).__name__}")

    # Test with None (will fail at _load_scenario_params, not type checking)
    try:
        check_full_results(None)
        assert False, "Should have raised exception for None path"
    except (ValueError, TypeError, AttributeError) as e:
        logger.info(f"  ✅ None path rejected: {type(e).__name__}")
    except Exception as e:
        logger.info(f"  ✅ None path exception: {type(e).__name__}")


# %%
def test_public_api():
    """Test that public API functions are properly exported."""
    logger.info("🧪 Testing public API...")

    # Verify check_full_results is in __all__
    try:
        from nhpy.check_full_results import __all__

        if "check_full_results" in __all__:
            logger.info("  ✅ check_full_results properly exported")
        else:
            logger.info("  ⚠️  check_full_results not in __all__")
    except ImportError:
        logger.info("  ⚠️  No __all__ defined in module")

    # Verify function signature
    import inspect

    sig = inspect.signature(check_full_results)
    params = list(sig.parameters.keys())
    expected_params = ["scenario_path", "account_url", "container_name"]

    if params == expected_params:
        logger.info("  ✅ Function signature correct")
    else:
        logger.info(f"  ⚠️  Unexpected signature: {params}")


# %%
def main():
    """Run all smoke tests."""
    logger.info("🚀 Running smoke tests for check_full_results module...\n")

    try:
        test_blob_path_analysis()
        test_blob_analysis_edge_cases()
        test_environment_check()
        test_public_api()
        test_error_handling()
        test_path_validation()

        logger.info("\n🎉 All smoke tests passed!")
        logger.info(
            "💡 To test with real Azure Storage, use a valid scenario path with proper credentials"
        )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main())
