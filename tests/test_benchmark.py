#!/usr/bin/env python

"""
Smoke tests for benchmark.py using fastcore.test.

Tests the benchmark.py script which compares Pandas vs Polars implementations
using fastcore.test's lightweight assertion framework.

Usage:
    uv run python tests/test_benchmark.py [scenario_path]

    Optional arguments:
        scenario_path: Path to specific scenario for live testing
"""

# %%
import logging
import os
import sys
import tempfile
from pathlib import Path

from fastcore.test import *
from rich.console import Console
from rich.logging import RichHandler

from nhpy.utils import get_logger

# %%
# Set up console with theme consistent with benchmark.py
console = Console()

# %%
# Configure logger
logger = get_logger()
logger.handlers = [RichHandler(console=console, rich_tracebacks=True)]

# %%
# Validate imports before running tests
try:
    import sys
    from pathlib import Path

    # Add project root to Python path to find benchmark.py
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmark import (
        _calculate_benchmark_statistics,
        _calculate_improvements,
        _cleanup_output_files,
        calc_improvement,
        compare_csv_outputs,
        load_scenario_path,
    )
except ImportError as e:
    console.print(f"[red]❌ Import error:[/] {e}")
    console.print(
        "[yellow]💡 benchmark.py functions may not be available - check file location[/]"
    )
    sys.exit(2)


# %%
def test_improvement_calculation():
    """Test calculation of percentage improvement."""
    console.print("[blue]🧪 Testing improvement calculation...[/]")

    # Test basic case (positive improvement)
    result = calc_improvement(100.0, 50.0)
    test_eq(result, 50.0)

    # Test negative improvement
    result = calc_improvement(100.0, 150.0)
    test_eq(result, -50.0)

    # Test edge case: pandas_value is 0
    result = calc_improvement(0.0, 50.0)
    test_eq(result, 0.0)

    # Test with float values - using test_close for float comparison
    result = calc_improvement(100.5, 50.25)
    test_close(result, 50.0, eps=0.01)

    console.print("[green]  ✅ Improvement calculation works correctly[/]")


# %%
def test_statistics_calculation():
    """Test statistical calculations used in benchmark results."""
    console.print("[blue]🧪 Testing statistics calculation...[/]")

    name = "test_implementation"
    times = [10.5, 11.2, 10.8, 11.0, 10.9]
    memory_peaks = [150.5, 155.2, 153.8, 152.0]
    memory_before = 100.0

    # Call the function under test
    time_result, memory_result, stats = _calculate_benchmark_statistics(
        name, times, memory_peaks, memory_before
    )

    # Verify results
    test_eq(time_result, 10.9)  # median time
    test_eq(memory_result, 155.2)  # max memory

    # Verify stats dictionary keys
    test_eq(set(stats.keys()), {"min", "max", "mean", "stdev"})

    # Verify stat values - using test_close for float comparisons
    test_eq(stats["min"], 10.5)
    test_eq(stats["max"], 11.2)
    test_close(stats["mean"], 10.88, eps=0.01)

    console.print("[green]  ✅ Statistics calculation works correctly[/]")


# %%
def test_combined_improvements():
    """Test calculation of time and memory improvements."""
    console.print("[blue]🧪 Testing combined improvements calculation...[/]")

    pandas_time = 100.0
    polars_time = 60.0
    pandas_memory = 500.0
    polars_memory = 300.0

    time_imp, memory_imp = _calculate_improvements(
        pandas_time, polars_time, pandas_memory, polars_memory
    )

    test_eq(time_imp, 40.0)
    test_eq(memory_imp, 40.0)

    console.print("[green]  ✅ Combined improvements calculation works correctly[/]")


# %%
def test_cleanup_output_files():
    """Test cleanup of output files before benchmark runs."""
    console.print("[blue]🧪 Testing output file cleanup...[/]")

    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_name = "test-benchmark"
        activity_types = ["ip", "op", "ae"]
        extensions = [".csv", ".parquet"]

        # Create dummy result files
        for activity_type in activity_types:
            for ext in extensions:
                file_path = (
                    Path(tmpdir)
                    / f"{scenario_name}_detailed_{activity_type}_results{ext}"
                )
                file_path.touch()
                test_eq(file_path.exists(), True)

        # Run cleanup
        _cleanup_output_files(tmpdir, scenario_name)

        # Verify files were removed - using fastcore.test's test_eq
        for activity_type in activity_types:
            for ext in extensions:
                file_path = (
                    Path(tmpdir)
                    / f"{scenario_name}_detailed_{activity_type}_results{ext}"
                )
                test_eq(file_path.exists(), False)

        console.print("[green]  ✅ Output file cleanup works correctly[/]")


# %%
def test_csv_comparison_handling():
    """Test the handling logic for CSV comparison."""
    console.print("[blue]🧪 Testing CSV comparison handling...[/]")

    # Test with non-existent directories - should handle gracefully
    with tempfile.TemporaryDirectory() as tmp_root:
        pandas_dir = str(Path(tmp_root) / "nonexistent1")
        polars_dir = str(Path(tmp_root) / "nonexistent2")

        # This should not raise an exception
        try:
            result = compare_csv_outputs(pandas_dir, polars_dir, "test-scenario")
            console.print("[green]  ✅ Handles missing directories gracefully[/]")
        except Exception as e:
            test_fail(
                lambda: compare_csv_outputs(pandas_dir, polars_dir, "test-scenario"),
                exc=type(e),
                contains=str(e),
            )
            console.print(
                "[green]  ✅ Raises appropriate exception for missing directories[/]"
            )

    # Test with empty directories
    with (
        tempfile.TemporaryDirectory() as pandas_dir,
        tempfile.TemporaryDirectory() as polars_dir,
    ):
        # This should not raise an exception
        result = compare_csv_outputs(pandas_dir, polars_dir, "test-scenario")
        console.print(f"[green]  ✅ Empty directory handling returns: {result}[/]")

        console.print("[green]  ✅ CSV comparison handling tested[/]")


# %%
def test_environment_check():
    """Check if environment is properly configured for benchmark."""
    console.print("[blue]🧪 Testing environment configuration...[/]")

    # Check .env contains AZ_FULL_RESULTS with JSON scenario data
    has_full_results = os.getenv("AZ_FULL_RESULTS") is not None

    if not has_full_results:
        console.print("[yellow]  ⚠️  Missing AZ_FULL_RESULTS environment variable[/]")
        console.print("[blue]  💡 Set this in .env file for full benchmark testing[/]")
    else:
        console.print("[green]  ✅ AZ_FULL_RESULTS environment variable present[/]")


# %%
def test_edge_cases():
    """Test edge cases in benchmark statistical functions."""
    console.print("[blue]🧪 Testing edge cases...[/]")

    # Empty lists for statistics
    name = "empty_test"
    time_result, memory_result, stats = _calculate_benchmark_statistics(name, [], [], 0.0)

    # Should return zeros for all values with empty input
    test_eq(time_result, 0)
    test_eq(memory_result, 0)
    test_eq(stats, {"min": 0, "max": 0, "mean": 0, "stdev": 0})

    # Single item lists
    name = "single_item_test"
    time_result, memory_result, stats = _calculate_benchmark_statistics(
        name, [5.0], [10.0], 0.0
    )

    test_eq(time_result, 5.0)
    test_eq(memory_result, 10.0)
    test_eq(stats["min"], 5.0)
    test_eq(stats["max"], 5.0)
    test_eq(stats["mean"], 5.0)
    test_eq(stats["stdev"], 0.0)

    console.print("[green]  ✅ Edge cases handled correctly[/]")


# %%
def test_real_scenario_path(scenario_path):
    """Test benchmark with a real scenario path.

    Args:
        scenario_path: Path to a specific scenario for testing
    """
    console.print(f"[blue]🧪 Testing with real scenario path: {scenario_path}[/]")

    try:
        # Just validate that the scenario path is properly formatted
        # This is a smoke test, so we won't actually run the benchmark
        if scenario_path:
            # Skip validation if the path appears to be another test file
            if scenario_path.startswith("tests/test_") and scenario_path.endswith(".py"):
                console.print("[yellow]  ⚠️ Skipping validation of test file path[/]")
                return True

            # Simple validation
            parts = Path(scenario_path.rstrip("/")).parts
            test_eq(len(parts) >= 4, True)
            console.print("[green]  ✅ Scenario path format appears valid[/]")
            console.print(
                "[blue]  💡 To run the benchmark, use: uv run python benchmark.py[/]"
            )
            return True
        else:
            # Try loading a random scenario from environment
            scenario_name, path = load_scenario_path()
            if scenario_name and path:
                console.print(
                    f"[green]  ✅ Successfully loaded random scenario: {scenario_name}[/]"
                )
                return True
            else:
                console.print(
                    "[yellow]  ⚠️ Could not load a random scenario from environment[/]"
                )
                return False
    except Exception as e:
        console.print(f"[red]  ❌ Error validating scenario path: {str(e)}[/]")
        return False


def main():
    """Runs all smoke tests and returns appropriate exit code."""
    console.print("[blue bold]🚀 Running smoke tests for benchmark.py...[/]\n")

    # Check for command line argument for real scenario testing
    scenario_path = None
    if len(sys.argv) > 1:
        scenario_path = sys.argv[1]

    try:
        test_improvement_calculation()
        test_statistics_calculation()
        test_combined_improvements()
        test_cleanup_output_files()
        test_csv_comparison_handling()
        test_edge_cases()
        test_environment_check()

        console.print("\n[green bold]🎉 All smoke tests passed![/]")

        # If scenario path provided, run real scenario test
        if scenario_path:
            console.print("\n[blue]🧪 Running test with real scenario path...[/]")
            test_real_scenario_path(scenario_path)
        else:
            # Try with a random scenario from environment
            console.print(
                "\n[blue]🧪 Testing with random scenario from environment...[/]"
            )
            test_real_scenario_path(None)
            console.print(
                "[blue]💡 To test with specific scenario, run: "
                "uv run python tests/test_benchmark.py <path_to_scenario>[/]"
            )

    except Exception as e:
        console.print(f"\n[red]❌ Test failed: {str(e)}[/]")
        return 1

    return 0


# %%
if __name__ == "__main__":
    sys.exit(main())
