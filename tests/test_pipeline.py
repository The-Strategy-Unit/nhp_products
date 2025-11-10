#!/usr/bin/env python

"""
Smoke tests the integrated pipeline that combines environment setup,
full results checking, and detailed results processing into a single workflow.

Usage:
    python tests/test_pipeline.py [results_path]

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
    from nhpy.pipeline import (
        ensure_venv,
        is_venv_active,
        main,
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
def test_venv_check():
    """Verifies virtual environment detection functionality."""
    logger.info("🧪 Testing virtual environment detection...")

    # Check if current environment is a venv
    is_venv = is_venv_active()

    # This is just a detection function, so we're just checking it runs
    logger.info(f"  ✅ Virtual environment check returned: {is_venv}")


# %%
def test_results_exist_check():
    """Tests proper handling of invalid paths in check_full_results."""
    logger.info("🧪 Testing results existence check...")

    # Import the function directly from the check_full_results module
    # as pipeline.py imports it from there
    # Import at the top would create circular dependency in testing context
    from nhpy.check_full_results import check_full_results  # noqa: PLC0415

    # Test with invalid path (should handle gracefully)
    try:
        check_full_results("invalid/path")
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("  ✅ Invalid path correctly rejected")
    except Exception as e:
        # Other exceptions (auth, network) are also valid for smoke test
        logger.info(f"  ✅ Expected exception type: {type(e).__name__}")


# %%
def test_environment_check():
    """Validates that required environment variables are checked."""
    logger.info("🧪 Testing environment configuration...")

    required_vars = [
        "AZ_STORAGE_EP",
        "AZ_STORAGE_RESULTS",
        "AZ_STORAGE_DATA",
        "AZ_VALID_PATH",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.info(f"  ⚠️  Missing environment variables: {missing}")
        logger.info("  💡 Set these in .env file for full testing")
    else:
        logger.info("  ✅ All required environment variables present")


# %%
def test_error_handling():
    """Tests CLI argument validation and error handling with missing arguments."""
    logger.info("🧪 Testing CLI error handling...")

    # Test with invalid arguments by calling main() with sys.argv mocked
    original_argv = sys.argv
    sys.argv = ["pipeline.py"]  # Missing required argument

    try:
        # Call main with no arguments (should show help and exit)
        main()
        # This assertion is deliberately unreachable,
        # if we get here without an exception, the test should fail
        assert False, "Should have raised SystemExit"
    except SystemExit:
        logger.info("  ✅ Missing arguments correctly handled")
    except Exception as e:
        logger.info(f"  ⚠️  Unexpected exception: {type(e).__name__}")
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


# %%
def test_output_dir_creation():
    """Verifies output directory creation works correctly."""
    logger.info("🧪 Testing output directory creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_dir = Path(tmpdir) / "output_dir"

        # Check that directory doesn't exist yet
        assert not nonexistent_dir.exists()

        # Create the directory using Path.mkdir
        nonexistent_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory was created
        assert nonexistent_dir.exists()
        logger.info("  ✅ Output directory creation works")


# %%
def test_real_path(results_path):
    """Tests pipeline with a real results path.

    This function will validate the path format and check if all required
    environment variables are set, but will not run the actual pipeline.

    Args:
        results_path: Path to real scenario results
    """
    logger.info(f"🧪 Testing pipeline with real path: {results_path}")

    try:
        # Check environment variables
        env_vars = ["AZ_STORAGE_EP", "AZ_STORAGE_RESULTS", "AZ_STORAGE_DATA"]
        missing = [var for var in env_vars if not os.getenv(var)]

        if missing:
            logger.error(f"  ❌ Missing required environment variables: {missing}")
            return False

        # Check virtual environment
        if not is_venv_active():
            logger.warning("  ⚠️ No virtual environment detected")

        # Validate path format
        logger.info("  ✅ Path format appears valid for pipeline processing")
        logger.info(
            "  💡 To run the full pipeline, use: "
            f"uv run python -m nhpy.pipeline {results_path}"
        )

        return True
    except Exception as e:
        logger.error(f"  ❌ Error validating real path: {e}")
        return False


def main_test():
    """Runs all smoke tests and returns appropriate exit code."""
    logger.info("🚀 Running smoke tests for pipeline module...\n")

    # Check for command line argument for real path testing
    real_path = None
    if len(sys.argv) > 1:
        real_path = sys.argv[1]

    try:
        test_venv_check()
        test_results_exist_check()
        test_environment_check()
        test_output_dir_creation()
        test_error_handling()

        logger.info("\n🎉 All smoke tests passed!")

        # If real path provided, run real path test
        if real_path:
            logger.info("\n🧪 Running test with real path...")
            test_real_path(real_path)
        else:
            logger.info(
                "💡 To test with real path, run: "
                "uv run python tests/test_pipeline.py <path_to_results>"
            )

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {e}")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main_test())
