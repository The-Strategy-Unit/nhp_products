#!/usr/bin/env python
"""
Cross-platform benchmarking script for comparing Pandas vs Polars implementations.

This script benchmarks the nhpy.run_detailed_results (Pandas) and
nhpy.run_detailed_results_pl (Polars) implementations, providing
median execution times and calculating the performance improvement.

Usage:
    uv run python benchmark.py [-r RUNS] [-p PANDAS_DIR] [-o POLARS_DIR] [-d]

Example:
    uv run python benchmark.py -r 5

Configuration:
    The script looks for a .env file in the following platform-specific locations:
    - Linux/Unix: ~/.config/nhp_products/.env
    - Windows: %USERPROFILE%\\AppData\\Local\\nhp_products\\.env
    - macOS: ~/Library/Application Support/nhp_products/.env

    The .env file should contain an AZ_FULL_RESULTS variable with JSON defining
    scenario paths.
"""

# Imports
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from logging import DEBUG, INFO
from pathlib import Path
from typing import Any, Literal, TypedDict

import psutil
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from nhpy.utils import _load_dotenv_file, get_logger


# TypedDict definitions for better typing
class BenchmarkData(TypedDict):
    """Dictionary containing benchmark result data."""

    pandas_completed: bool
    polars_completed: bool
    pandas_time: float
    polars_time: float
    pandas_memory: float
    polars_memory: float
    pandas_stats: dict[str, float]
    polars_stats: dict[str, float]
    scenario_name: str


class BenchmarkInit(TypedDict):
    """Container for benchmark initialization results."""

    result_data: BenchmarkData
    pandas_cmd: str
    polars_cmd: str
    pandas_dir: str
    polars_dir: str
    checkpoint_path: str
    initial_memory: float
    checkpoint_data: dict[str, Any] | None
    scenario_name: str


class ComparisonParams(TypedDict):
    """Parameters for benchmark comparison."""

    pandas_time: float
    polars_time: float
    pandas_memory: float
    polars_memory: float
    pandas_stats: dict[str, float]
    polars_stats: dict[str, float]
    pandas_dir: str
    polars_dir: str
    scenario_name: str
    checkpoint_path: str
    debug_mode: bool


class RunConfig(TypedDict):
    """Configuration for running a benchmark implementation."""

    implementation: str
    cmd: str
    runs: int
    result_data: BenchmarkData
    checkpoint_data: dict[str, Any] | None
    checkpoint_path: str
    scenario_name: str  # Added for file clean-up


class RunBenchmarksConfig(TypedDict):
    """Configuration for running both Pandas and Polars benchmarks."""

    args: Any
    result_data: BenchmarkData
    pandas_cmd: str
    polars_cmd: str
    checkpoint_path: str
    checkpoint_data: dict[str, Any] | None
    scenario_name: str  # Added for file clean-up


# Create a custom theme with accessible colours
custom_theme = Theme(
    {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "pandas": "yellow",
        "polars": "cyan",
        "heading": "bold blue",
        "metric": "magenta",
        "improvement_positive": "green",
        "improvement_negative": "red",
    }
)

# Set up console with custom theme
console = Console(theme=custom_theme, highlight=False)

# Configure logger to use Rich
logger = get_logger()
logger.handlers = [RichHandler(console=console, rich_tracebacks=True)]

# Constants
MIN_PATH_COMPONENTS = 2
DIFF_LINES = 10
SIGINT_EXIT_CODE = 130  # Standard exit code for SIGINT/KeyboardInterrupt
MIN_EXPECTED_TIME = 60.0
SUSPICIOUSLY_FAST_TIME = 30
# Coefficient of variation threshold for detecting high variance
HIGH_VARIANCE_THRESHOLD = 1.0
# Activity types used throughout the code
ACTIVITY_TYPES = ["ip", "op", "ae"]


def _get_checkpoint_path(scenario_name: str) -> str:
    """Get path to the checkpoint file for saving/resuming benchmark progress."""
    data_dir = Path("nhpy/data")  # This path is already in .gitignore
    data_dir.mkdir(exist_ok=True, parents=True)

    return str(data_dir / f"benchmark_{scenario_name}_checkpoint.json")


def _save_checkpoint(checkpoint_path: str, data: dict[str, Any] | BenchmarkData) -> None:
    """Save benchmark progress to a checkpoint file."""
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.debug(f"Error saving checkpoint: {str(e)}")


def _load_checkpoint(checkpoint_path: str) -> dict | None:
    """Load benchmark progress from a checkpoint file if it exists."""
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        console.print(f"[info]Found checkpoint at {checkpoint_path}[/]")
        return data
    except Exception as e:
        logger.debug(f"Error loading checkpoint: {str(e)}")
        return None


def _check_implementation_completed(output_dir: str, implementation: str) -> bool:
    """Check if an implementation has already completed by looking for output files.

    NOTE: This function now only detects file existence for logging purposes.
    It's no longer used to skip benchmark runs.
    """
    for activity_type in ACTIVITY_TYPES:
        pattern = f"*_detailed_{activity_type}_results.csv"
        if not list(Path(output_dir).glob(pattern)):
            logger.debug(f"No {activity_type} output files found for {implementation}")
            return False

    console.print(f"[info]Found existing output files for {implementation}[/]")

    return True


def _execute_command(command: str) -> None:
    """Execute a shell command and handle interruptions gracefully.

    Args:
        command: The command to execute

    Raises:
        KeyboardInterrupt: If the command was interrupted by the user
        subprocess.CalledProcessError: If the command failed
    """
    try:
        # Ensure any pending I/O finishes. This ensures we get accurate time measurements
        time.sleep(0)

        # Capture full execution time.
        # Set `capture_output=False` for displaying command output in real-time
        subprocess.run(command, shell=True, check=True, capture_output=False)

    except KeyboardInterrupt:  # Handle Ctrl+C gracefully
        console.print("\n[warning]Operation cancelled by user[/]")
        raise  # Re-raise to let caller handle it
    except subprocess.CalledProcessError as e:
        console.print(f"[error]Command failed with exit code {e.returncode}[/]")
        raise
    except Exception as e:
        console.print(f"[error]Command failed: {str(e)}[/]")
        raise


# %%
def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Uses psutil to get the Resident Set Size (RSS), which is the non-swapped physical
    memory a process has used. This is a cross-platform alternative to resource.getrusage
    that works on Windows, Mac, and Linux with minimal overhead.
    """
    # Get the process memory info
    process = psutil.Process()
    memory_info = process.memory_info()

    # Return RSS (Resident Set Size) in MB
    return memory_info.rss / (1024 * 1024)  # Convert bytes to MB


# %%
def run_benchmark(
    name: str, command: str, runs: int = 5
) -> tuple[float, float, dict[str, float]]:
    """Run a benchmark multiple times and report statistics.

    Args:
        name: Name of the implementation being benchmarked
        command: Command to run
        runs: Number of benchmark runs

    Returns:
        tuple: (median_execution_time, max_memory_usage, time_stats)
            - median_execution_time: Median execution time in seconds
            - max_memory_usage: Maximum memory usage observed during runs in MB
            - time_stats: Dictionary with min, max, mean, stdev time statistics
    """
    console.print(f"[heading]Running {name}[/heading]")
    times = []
    memory_peaks = []
    memory_usage_before = get_memory_usage()
    logger.debug(f"Memory usage before {name}: {memory_usage_before:.2f} MB")

    for i in range(runs):
        console.print(f"[metric]Run {i + 1}/{runs}[/]")
        start = time.time()
        start_memory = get_memory_usage()
        logger.debug(f"Memory usage before run {i + 1}: {start_memory:.2f} MB")

        # Log the command being executed
        console.print(f"[info]Executing: {command}[/]")
        console.print("[warning]Progress output (this may take a while):[/]")

        # Run command in a separate process
        _execute_command(command)

        end = time.time()
        elapsed = end - start
        times.append(elapsed)

        # Log time and memory usage
        end_memory = get_memory_usage()
        memory_diff = end_memory - start_memory
        memory_peaks.append(end_memory)  # Track peak memory usage
        console.print(f"[success]Time: {elapsed:.2f}s[/]")

        # Add early warning if run completed suspiciously quickly
        if elapsed < MIN_EXPECTED_TIME and "nhpy.run_detailed_results" in command:
            console.print(
                f"[warning]⚠️ Run {i + 1} completed too quickly."
                f" Expected > {MIN_EXPECTED_TIME}s for 256 Monte Carlo runs.[/]"
            )
            console.print(
                "[warning]⚠️ This likely means existing output files were detected "
                "and processing was skipped.[/]"
            )
            console.print(
                "[warning]⚠️ Check that files are being properly cleaned between runs "
                "for accurate timing.[/]"
            )

        logger.debug(
            f"Memory usage after run {i + 1}: {end_memory:.2f} MB"
            f"(delta: {memory_diff:+.2f} MB)"
        )

        # Reduce memory build up between runs
        gc.collect()
        after_gc_memory = get_memory_usage()
        logger.debug(f"Memory usage after garbage collection: {after_gc_memory:.2f} MB")

    return _calculate_benchmark_statistics(name, times, memory_peaks, memory_usage_before)


def _calculate_benchmark_statistics(
    name: str, times: list[float], memory_peaks: list[float], memory_usage_before: float
) -> tuple[float, float, dict[str, float]]:
    """Calculate statistics for benchmark runs.

    Args:
        name: Name of the implementation
        times: List of execution times
        memory_peaks: List of memory usage peaks
        memory_usage_before: Memory usage before benchmark started

    Returns:
        tuple: (median_execution_time, max_memory_usage, time_stats)
    """
    if not times:
        console.print(
            f"[warning]{name}: No successful runs to calculate statistics from[/]"
        )
        return 0, 0, {"min": 0, "max": 0, "mean": 0, "stdev": 0}

    # Time statistics
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0

    # Memory statistics
    max_memory = max(memory_peaks) if memory_peaks else 0

    # Time log
    console.print(f"[success]{name}: Median time: {median_time:.2f}s[/]")
    logger.debug(f"{name}: Min time: {min_time:.2f}s, Max time: {max_time:.2f}s")
    logger.debug(f"{name}: Mean time: {mean_time:.2f}s, StdDev: {stdev_time:.2f}s")

    if memory_peaks:
        logger.debug(f"{name}: Max memory usage: {max_memory:.2f} MB")

    # Final memory report
    final_memory = get_memory_usage()
    memory_diff = final_memory - memory_usage_before
    logger.debug(
        f"Final memory usage after {name}: {final_memory:.2f} MB"
        f" (total delta: {memory_diff:+.2f} MB)"
    )

    time_stats = {
        "min": min_time,
        "max": max_time,
        "mean": mean_time,
        "stdev": stdev_time,
    }

    return median_time, max_memory, time_stats


def _compare_single_file_pair(
    pandas_file: str, polars_file: str, activity_type: str
) -> bool:
    """Compare a single pair of CSV files and report differences.

    Returns:
        bool: True if files match, False otherwise
    """
    diff_cmd = f"diff {pandas_file} {polars_file}"
    logger.debug(f"Running: {diff_cmd}")

    try:
        process = subprocess.Popen(
            diff_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
    except Exception as e:
        console.print(f"[error]Failed to execute diff command: {str(e)}[/]")
        return False

    # Collect output for later analysis
    diff_lines = []

    if process is None or process.stdout is None:
        console.print("[error]Process or stdout is None, cannot collect diff output[/]")
        return False

    for line in process.stdout:
        diff_lines.append(line.strip())

    returncode = process.wait()

    if returncode == 0:
        console.print(f"[success]✅ {activity_type.upper()} outputs MATCH[/]")
        return True
    else:
        console.print(f"[warning]❌ {activity_type.upper()} outputs DIFFER[/]")
        console.print("[warning]First few differences:[/]")
        for line in diff_lines[:10]:  # Show first 10 lines of differences
            console.print(f"[warning]  {line}[/]")
        if len(diff_lines) > DIFF_LINES:
            console.print(
                f"[warning]  ... and {len(diff_lines) - 10} more differences[/]"
            )
        return False


def compare_csv_outputs(pandas_dir: str, polars_dir: str, scenario_name: str) -> bool:
    """Compare CSV outputs between Pandas and Polars implementations.

    Args:
        pandas_dir: Directory containing Pandas outputs
        polars_dir: Directory containing Polars outputs
        scenario_name: Name of the scenario for file pattern matching

    Returns:
        bool: True if all files match, False otherwise
    """
    console.print("[heading]COMPARING CSV OUTPUTS[/heading]")

    activity_types = ACTIVITY_TYPES
    all_match = True

    # Find matching CSV files
    for activity_type in activity_types:
        # Use scenario_name if provided, otherwise use a wildcard
        pattern_base = f"_detailed_{activity_type}_results.csv"
        if scenario_name:
            pandas_pattern = f"{scenario_name}{pattern_base}"
            polars_pattern = f"{scenario_name}{pattern_base}"
        else:
            pandas_pattern = f"*{pattern_base}"
            polars_pattern = f"*{pattern_base}"

        # List all matching files in each directory
        pandas_files = list(Path(pandas_dir).glob(pandas_pattern))
        polars_files = list(Path(polars_dir).glob(polars_pattern))

        if not pandas_files or not polars_files:
            console.print(
                f"[warning]Skipping {activity_type} comparison."
                f" No matching files found[/]"
            )
            continue

        # Use the first matching file from each directory
        pandas_file = str(pandas_files[0])
        polars_file = str(polars_files[0])

        logger.debug(f"Comparing {activity_type} files:")
        logger.debug(f"  Pandas: {pandas_file}")
        logger.debug(f"  Polars: {polars_file}")

        if not Path(pandas_file).exists() or not Path(polars_file).exists():
            console.print(
                f"[warning]Skipping {activity_type} comparison - files don't exist[/]"
            )
            continue

        console.print(f"[info]Comparing {activity_type.upper()} outputs:[/]")

        try:
            file_match = _compare_single_file_pair(
                pandas_file, polars_file, activity_type
            )
            if not file_match:
                all_match = False
        except Exception as e:
            console.print(f"[error]Error comparing {activity_type} files: {str(e)}[/]")
            all_match = False

    return all_match


# %%
def load_scenario_path() -> tuple[str | None, str | None]:
    """Load a random scenario path from AZ_FULL_RESULTS environment variable.
    Try to load from platform-specific config directory, fall back to default

    Returns:
        tuple: (scenario_name, scenario_path) or (None, None) if not found
    """
    project_name = Path(__file__).resolve().parent.name

    # Load environment variables with interpolate=False to handle complex JSON values
    _load_dotenv_file(interpolate=False)

    # If .env contains AZ_FULL_RESULTS, parse it as json
    az_full_results = os.getenv("AZ_FULL_RESULTS")
    if az_full_results:
        try:
            az_full_paths = json.loads(az_full_results)
        except json.JSONDecodeError as e:
            console.print(f"[error]Error parsing AZ_FULL_RESULTS JSON: {str(e)}[/]")
            return None, None

    # Random scenario path
    logger.debug(f"Found {len(az_full_paths)} scenarios in AZ_FULL_RESULTS")
    scenario_name = random.choice(list(az_full_paths.keys()))
    console.print(f"[info]Selected random scenario: {scenario_name}[/]")

    return scenario_name, az_full_paths[scenario_name]


# %%
def _configure_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark Pandas vs Polars implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--scenario-path",
        help="Path to scenario data (uses random path from env if not provided)",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs for each implementation",
    )
    parser.add_argument(
        "-p", "--pandas-dir", default="nhpy/data", help="Pandas output directory"
    )
    parser.add_argument(
        "-o", "--polars-dir", default="nhpy/data/pl", help="Polars output directory"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging with detailed memory usage information",
    )

    return parser


# %%
def _setup_logging(debug_mode: bool) -> float:
    """Configure logging based on debug flag.

    Args:
        debug_mode: Whether to enable debug logging

    Returns:
        float: Initial memory usage in MB
    """
    logger.setLevel(DEBUG if debug_mode else INFO)

    if debug_mode:
        console.print("[info]Debug logging enabled[/]")

    initial_memory = get_memory_usage()
    logger.debug(f"Initial memory usage: {initial_memory:.2f} MB")

    return initial_memory


# %%
def _get_scenario_info(scenario_path: str, scenario_key: str | None = None) -> str:
    """Extract scenario name from path.

    Args:
        scenario_path: Path to the scenario data
        scenario_key: Optional scenario key from AZ_FULL_RESULTS

    Returns:
        str: Extracted scenario name
    """
    clean_path = scenario_path.rstrip("/")
    path_parts = Path(clean_path).parts

    # If we don't have the scenario key from AZ_FULL_RESULTS, extract it from the path
    if not scenario_key:
        if len(path_parts) >= MIN_PATH_COMPONENTS:
            scenario_name = path_parts[-2]  # Second to last part is the scenario name
        else:
            # Fallback to parent directory name if path structure is different
            scenario_name = Path(scenario_path).parent.name
    else:
        # Use key from AZ_FULL_RESULTS
        scenario_name = scenario_key

    console.print(f"[heading]Benchmarking scenario path: {scenario_path}[/]")
    console.print(f"[info]Scenario name: {scenario_name}[/]")

    return scenario_name


# %%
def _prepare_directories(
    base_pandas_dir: str, base_polars_dir: str, scenario_name: str
) -> tuple[str, str]:
    """Create necessary output directories.

    Args:
        base_pandas_dir: Base directory for Pandas outputs
        base_polars_dir: Base directory for Polars outputs
        scenario_name: Name of the scenario

    Returns:
        tuple: (pandas_dir, polars_dir) - Paths to scenario-specific directories
    """
    # Create base output directories
    Path(base_pandas_dir).mkdir(parents=True, exist_ok=True)
    Path(base_polars_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directories: {base_pandas_dir}, {base_polars_dir}")

    # Create scenario-specific output directories
    pandas_dir = os.path.join(base_pandas_dir, scenario_name)
    polars_dir = os.path.join(base_polars_dir, scenario_name)
    Path(pandas_dir).mkdir(parents=True, exist_ok=True)
    Path(polars_dir).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created scenario directories: {pandas_dir}, {polars_dir}")

    return pandas_dir, polars_dir


# %%
def _build_commands(
    scenario_path: str, pandas_dir: str, polars_dir: str
) -> tuple[str, str]:
    """Build benchmark commands.

    Args:
        scenario_path: Path to the scenario data
        pandas_dir: Directory for Pandas outputs
        polars_dir: Directory for Polars outputs

    Returns:
        tuple: (pandas_command, polars_command) - Command strings for running benchmarks
    """
    pandas_cmd = (
        f"uv run python -m nhpy.run_detailed_results {scenario_path} "
        f"--output-dir={pandas_dir}"
    )
    polars_cmd = (
        f"uv run python -m nhpy.run_detailed_results_pl {scenario_path} "
        f"--output-dir={polars_dir}"
    )

    return pandas_cmd, polars_cmd


# %%
def calc_improvement(pandas_value: float, polars_value: float) -> float:
    """Calculate percentage improvement between pandas and polars values."""
    if pandas_value <= 0:
        return 0

    return ((pandas_value - polars_value) / pandas_value) * 100


def _calculate_improvements(
    pandas_time: float, polars_time: float, pandas_memory: float, polars_memory: float
) -> tuple[float, float]:
    """Calculate performance improvements.

    Args:
        pandas_time: Execution time for Pandas implementation
        polars_time: Execution time for Polars implementation
        pandas_memory: Memory usage for Pandas implementation
        polars_memory: Memory usage for Polars implementation

    Returns:
        tuple: (time_improvement, memory_improvement) - Percentage improvements
    """
    time_imp = calc_improvement(pandas_time, polars_time)
    memory_imp = calc_improvement(pandas_memory, polars_memory)

    return time_imp, memory_imp


# %%
def _cleanup_output_files(output_dir: str, scenario_name: str) -> None:
    """Clean up existing output files to ensure accurate benchmarking.

    Args:
        output_dir: Directory containing output files
        scenario_name: Name of the scenario for file pattern matching
    """
    if not output_dir or not scenario_name:
        logger.debug("Cannot clean-up: missing output_dir or scenario_name")
        return

    # Get full path
    dir_path = Path(output_dir)
    if not dir_path.exists():
        return

    console.print(f"[info]Cleaning existing result files in {output_dir}[/]")

    # Delete all result files for all activity types
    for activity_type in ACTIVITY_TYPES:
        for ext in [".csv", ".parquet"]:
            pattern = f"{scenario_name}_detailed_{activity_type}_results{ext}"
            file_path = dir_path / pattern
            if file_path.exists():
                try:
                    file_path.unlink()
                    console.print(f"[info]Removed {file_path}[/]")
                except Exception as e:
                    console.print(f"[warning]Failed to remove {file_path}: {str(e)}[/]")

    # Also check for pattern-based matches (in case scenario name is different)
    for activity_type in ACTIVITY_TYPES:
        pattern = f"*_detailed_{activity_type}_results.*"
        for file_path in dir_path.glob(pattern):
            try:
                file_path.unlink()
                console.print(f"[info]Removed pattern-matched file: {file_path}[/]")
            except Exception as e:
                console.print(f"[warning]Failed to remove {file_path}: {str(e)}[/]")


def _run_implementation_benchmark(
    config: RunConfig,
) -> tuple[float, float, dict[str, float]] | int:
    """Run a benchmark for a specific implementation.

    Args:
        config: Configuration for the benchmark run

    Returns:
        tuple: (time, memory, stats) or SIGINT_EXIT_CODE if interrupted
    """
    implementation = config["implementation"]
    cmd = config["cmd"]
    runs = config["runs"]
    result_data = config["result_data"]
    checkpoint_path = config["checkpoint_path"]

    impl_key = f"{implementation}_completed"
    impl_name = f"{implementation.title()} Implementation"

    output_dir_match = re.search(r"--output-dir=([^ ]+)", cmd)
    output_dir = output_dir_match.group(1) if output_dir_match else None

    scenario_name = result_data.get("scenario_name")
    if not scenario_name and "scenario_name" in config:
        scenario_name = config["scenario_name"]

    try:
        console.print(f"[info]Running {implementation.title()} benchmark...[/]")

        times = []
        memory_peaks = []

        for i in range(runs):
            # Clean up before each run to ensure full processing happens every time
            if output_dir and scenario_name:
                _cleanup_output_files(output_dir, scenario_name)
                console.print(
                    f"[info]Run {i + 1}/{runs}: Files cleaned, forcing full processing[/]"
                )
            else:
                msg = (
                    f"[warning]Run {i + 1}/{runs}: Could not clean files "
                    f"(missing output_dir or scenario_name)[/]"
                )
                console.print(msg)

            run_name = f"{impl_name} (Run {i + 1}/{runs})"
            run_time, run_memory, _ = run_benchmark(run_name, cmd, 1)

            times.append(run_time)
            memory_peaks.append(run_memory)

            msg = (
                f"[success]Run {i + 1}/{runs} completed in {run_time:.2f}s "
                f"with {run_memory:.2f}MB peak memory[/]"
            )
            console.print(msg)

        time_result = statistics.median(times) if times else 0
        memory_result = max(memory_peaks) if memory_peaks else 0

        stats_result: dict[str, float] = {
            "min": min(times) if times else 0.0,
            "max": max(times) if times else 0.0,
            "mean": statistics.mean(times) if times else 0.0,
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

        # Log individual run times for visibility
        run_times_str = ", ".join([f"{t:.2f}s" for t in times])
        console.print(f"[info]{implementation.title()} run times: {run_times_str}[/]")

        # Use direct assignment for type safety instead of update
        result_data[Literal[f"{implementation}_time"]] = time_result
        result_data[Literal[f"{implementation}_memory"]] = memory_result
        result_data[Literal[f"{implementation}_stats"]] = stats_result
        result_data[Literal[impl_key]] = True
        _save_checkpoint(checkpoint_path, result_data)

        # Explicit cast to match the return type annotation
        return float(time_result), float(memory_result), stats_result

    except KeyboardInterrupt:
        console.print(
            f"\n[warning]{implementation.title()} benchmark aborted by user. "
            f"Progress saved.[/]"
        )
        _save_checkpoint(checkpoint_path, result_data)
        return SIGINT_EXIT_CODE


def _log_performance_summary_basic(result_data: dict[str, Any]) -> None:
    """Log basic performance summary.

    Args:
        result_data: Dictionary with benchmark results
    """
    # Extract values from the dictionary
    pandas_time = result_data["pandas_time"]
    polars_time = result_data["polars_time"]
    time_improvement = result_data["time_improvement"]

    # Summary table
    table = Table(title="Performance Summary", show_header=True, border_style="heading")

    table.add_column("Implementation", style="bold")
    table.add_column("Median Time (s)", style="metric")

    table.add_row("[pandas]Pandas[/]", f"{pandas_time:.2f}")
    table.add_row("[polars]Polars[/]", f"{polars_time:.2f}")

    imp_style = (
        "improvement_positive" if time_improvement >= 0 else "improvement_negative"
    )
    imp_symbol = "↑" if time_improvement >= 0 else "↓"
    table.add_row(
        "Improvement", f"[{imp_style}]{imp_symbol} {abs(time_improvement):.2f}%[/]"
    )

    console.print(table)


def _check_execution_time_warnings(
    pandas_time: float,
    polars_time: float,
    pandas_stats: dict[str, float],
    polars_stats: dict[str, float],
) -> None:
    """Check for suspiciously fast execution times and show warnings.

    Args:
        pandas_time: Median execution time for Pandas
        polars_time: Median execution time for Polars
        pandas_stats: Dictionary with Pandas time statistics
        polars_stats: Dictionary with Polars time statistics
    """
    # Add enhanced warning about execution time if it's suspiciously low
    if pandas_time < MIN_EXPECTED_TIME or polars_time < MIN_EXPECTED_TIME:
        # Calculate coefficient of variation to detect if some runs used cached results
        pandas_cv = (
            pandas_stats["stdev"] / pandas_stats["mean"]
            if pandas_stats["mean"] > 0
            else 0
        )
        polars_cv = (
            polars_stats["stdev"] / polars_stats["mean"]
            if polars_stats["mean"] > 0
            else 0
        )

        console.print(
            "[warning]⚠️ The measured execution times appear to be very low. "
            "Processing 256 Monte Carlo runs should take minutes to hours.[/]"
        )

        # Print detailed analysis of what might have happened
        if pandas_cv > HIGH_VARIANCE_THRESHOLD or polars_cv > HIGH_VARIANCE_THRESHOLD:
            console.print(
                "[warning]⚠️ High variance in run times detected. Some runs"
                " processed the full dataset while others skipped processing"
                " by using cached results.[/]"
            )
            console.print(
                "[warning]⚠️ Fix this by ensuring output files are completely removed "
                "between benchmark runs. The benchmark has been updated to do this "
                "automatically.[/]"
            )
        else:
            console.print(
                "[warning]⚠️ Consistently fast execution times suggest that all runs "
                "are detecting and using cached results. The benchmark should now clean "
                "files between runs.[/]"
            )

        console.print("[info]For accurate benchmarking, make sure:[/]")
        console.print("[info]1. Output directories are cleaned between runs[/]")
        console.print("[info]2. Each implementation's files are completely isolated[/]")
        console.print("[info]3. Cache directories are cleared if they exist[/]")


def _log_performance_summary_detailed(result_data: dict[str, Any]) -> None:
    """Log detailed performance summary.

    Args:
        result_data: Dictionary with benchmark results
    """
    # Extract values for detailed output
    pandas_time = result_data["pandas_time"]
    polars_time = result_data["polars_time"]
    pandas_memory = result_data["pandas_memory"]
    polars_memory = result_data["polars_memory"]
    memory_improvement = result_data["memory_improvement"]
    time_improvement = result_data["time_improvement"]
    pandas_stats = result_data["pandas_stats"]
    polars_stats = result_data["polars_stats"]

    min_imp = calc_improvement(pandas_stats["min"], polars_stats["min"])
    max_imp = calc_improvement(pandas_stats["max"], polars_stats["max"])
    mean_imp = calc_improvement(pandas_stats["mean"], polars_stats["mean"])

    console.print()
    console.print("[heading]Detailed Statistics[/]")

    time_table = Table(title="Execution Time (seconds)", show_header=True)
    time_table.add_column("Metric")
    time_table.add_column("Pandas", style="pandas")
    time_table.add_column("Polars", style="polars")
    time_table.add_column("Improvement")

    time_table.add_row(
        "Median",
        f"{pandas_time:.2f}",
        f"{polars_time:.2f}",
        f"[{'improvement_positive' if time_improvement >= 0 else 'improvement_negative'}]"
        f"{'↑' if time_improvement >= 0 else '↓'} {abs(time_improvement):.2f}%[/]",
    )

    time_table.add_row(
        "Min",
        f"{pandas_stats['min']:.2f}",
        f"{polars_stats['min']:.2f}",
        f"[{'improvement_positive' if min_imp >= 0 else 'improvement_negative'}]"
        f"{'↑' if min_imp >= 0 else '↓'} {abs(min_imp):.2f}%[/]",
    )

    time_table.add_row(
        "Max",
        f"{pandas_stats['max']:.2f}",
        f"{polars_stats['max']:.2f}",
        f"[{'improvement_positive' if max_imp >= 0 else 'improvement_negative'}]"
        f"{'↑' if max_imp >= 0 else '↓'} {abs(max_imp):.2f}%[/]",
    )

    time_table.add_row(
        "Mean",
        f"{pandas_stats['mean']:.2f}",
        f"{polars_stats['mean']:.2f}",
        f"[{'improvement_positive' if mean_imp >= 0 else 'improvement_negative'}]"
        f"{'↑' if mean_imp >= 0 else '↓'} {abs(mean_imp):.2f}%[/]",
    )

    time_table.add_row(
        "StdDev", f"{pandas_stats['stdev']:.2f}", f"{polars_stats['stdev']:.2f}", ""
    )

    memory_table = Table(title="Memory Usage (MB)", show_header=True)
    memory_table.add_column("Implementation")
    memory_table.add_column("Max Usage")
    memory_table.add_column("Improvement")

    # Improvement text
    positive_improvement = memory_improvement >= 0
    imp_style = "improvement_positive" if positive_improvement else "improvement_negative"
    imp_symbol = "↑" if memory_improvement >= 0 else "↓"
    imp_text = f"[{imp_style}]{imp_symbol} {abs(memory_improvement):.2f}%[/]"

    memory_table.add_row("[pandas]Pandas[/]", f"{pandas_memory:.2f}", "")
    memory_table.add_row("[polars]Polars[/]", f"{polars_memory:.2f}", imp_text)

    console.print(time_table)
    console.print()
    console.print(memory_table)


def _log_performance_summary(result_data: dict[str, Any], debug_mode: bool) -> None:
    """Log performance summary.

    Args:
        result_data: Dictionary with benchmark results
        debug_mode: Whether to show detailed debug info
    """
    _log_performance_summary_basic(result_data)

    # Values needed for warning checks
    pandas_time = result_data["pandas_time"]
    polars_time = result_data["polars_time"]
    pandas_stats = result_data["pandas_stats"]
    polars_stats = result_data["polars_stats"]

    _check_execution_time_warnings(pandas_time, polars_time, pandas_stats, polars_stats)

    if debug_mode:
        _log_performance_summary_detailed(result_data)


# %%
def _show_comparison_and_cleanup(params: ComparisonParams) -> int:
    """Show comparison between Pandas and Polars implementations and clean-up.

    Returns:
        int: 0 for success, 130 if interrupted
    """
    # Extract parameters
    pandas_time = params["pandas_time"]
    polars_time = params["polars_time"]
    pandas_memory = params["pandas_memory"]
    polars_memory = params["polars_memory"]
    pandas_stats = params["pandas_stats"]
    polars_stats = params["polars_stats"]
    pandas_dir = params["pandas_dir"]
    polars_dir = params["polars_dir"]
    scenario_name = params["scenario_name"]
    checkpoint_path = params["checkpoint_path"]
    debug_mode = params["debug_mode"]

    if pandas_time < MIN_EXPECTED_TIME or polars_time < MIN_EXPECTED_TIME:
        # Calculate variance (high variance suggests some runs skipped processing)
        pandas_variance = (
            pandas_stats["stdev"] / pandas_stats["mean"]
            if pandas_stats["mean"] > 0
            else 0
        )
        polars_variance = (
            polars_stats["stdev"] / polars_stats["mean"]
            if polars_stats["mean"] > 0
            else 0
        )

        console.print(
            "[warning]⚠️ At least one implementation completed in less than a minute. "
            "This is suspiciously fast for processing 256 Monte Carlo runs.[/]"
        )

        console.print("[warning]⚠️ Detailed timing analysis:[/]")

        # Pandas analysis
        if (
            pandas_stats["max"] > MIN_EXPECTED_TIME
            and pandas_stats["min"] < SUSPICIOUSLY_FAST_TIME
            and pandas_variance > HIGH_VARIANCE_THRESHOLD
        ):
            msg = (
                f"[warning]  - Pandas: High time variance detected "
                f"(min={pandas_stats['min']:.1f}s, max={pandas_stats['max']:.1f}s). "
                f"Some runs likely processed data, others skipped processing.[/]"
            )
            console.print(msg)
        elif pandas_time < MIN_EXPECTED_TIME:
            msg = (
                "[warning]  - Pandas: All runs completed too quickly. "
                "Make sure files are being cleaned between runs.[/]"
            )
            console.print(msg)

        # Polars analysis
        if (
            polars_stats["max"] > MIN_EXPECTED_TIME
            and polars_stats["min"] < SUSPICIOUSLY_FAST_TIME
            and polars_variance > HIGH_VARIANCE_THRESHOLD
        ):
            msg = (
                f"[warning]  - Polars: High time variance detected "
                f"(min={polars_stats['min']:.1f}s, max={polars_stats['max']:.1f}s). "
                f"Some runs likely processed data, others skipped processing.[/]"
            )
            console.print(msg)
        elif polars_time < MIN_EXPECTED_TIME:
            msg = (
                "[warning]  - Polars: All runs completed too quickly. "
                "Make sure files are being cleaned between runs.[/]"
            )
            console.print(msg)

    time_improvement, memory_improvement = _calculate_improvements(
        pandas_time, polars_time, pandas_memory, polars_memory
    )

    summary_data = {
        "pandas_time": pandas_time,
        "polars_time": polars_time,
        "time_improvement": time_improvement,
        "pandas_memory": pandas_memory,
        "polars_memory": polars_memory,
        "memory_improvement": memory_improvement,
        "pandas_stats": pandas_stats,
        "polars_stats": polars_stats,
    }

    _log_performance_summary(summary_data, debug_mode)

    console.print("[heading]OUTPUT DIRECTORIES[/heading]")
    console.print(f"- [pandas]Pandas[/]: {pandas_dir}")
    console.print(f"- [polars]Polars[/]: {polars_dir}")

    # Compare outputs
    try:
        console.print(
            "[heading]Comparing CSV outputs between Pandas "
            "and Polars implementations[/heading]"
        )
        outputs_match = compare_csv_outputs(pandas_dir, polars_dir, scenario_name)

        if outputs_match:
            # Show a simple success panel
            console.print(
                Panel(
                    "✅ Pandas and Polars implementations are functionally equivalent.",
                    title="SUCCESS",
                    border_style="success",
                )
            )
        else:
            # Show a simple warning panel
            console.print(
                Panel(
                    "⚠️ Some outputs differ. See details above for specific differences.",
                    title="WARNING",
                    border_style="warning",
                )
            )
    except KeyboardInterrupt:
        console.print("\n[warning]Output comparison aborted by user.[/]")
        return SIGINT_EXIT_CODE

    # Clean up checkpoint after successful completion
    try:
        os.remove(checkpoint_path)
        logger.debug(f"Removed checkpoint file: {checkpoint_path}")
    except Exception as e:
        logger.debug(f"Error removing checkpoint: {str(e)}")

    return 0


def _show_incomplete_benchmark_message(
    result_data: dict[str, Any] | BenchmarkData,
) -> None:
    """Show a message for incomplete benchmark runs.

    Args:
        result_data: Dictionary with benchmark results
    """
    console.print(
        "[warning]Benchmark incomplete. Run again to finish remaining steps.[/]"
    )
    if result_data["pandas_completed"]:
        console.print("[info]Pandas benchmark complete. Polars benchmark pending.[/]")
    else:
        console.print("[info]Polars benchmark complete. Pandas benchmark pending.[/]")


def _print_banner() -> None:
    """Print a simple, elegant banner for the benchmark tool."""
    console.print(
        Panel.fit(
            "Pandas vs Polars Benchmark",
            title="NHP Products",
            subtitle="Performance Testing Tool",
            border_style="heading",
            padding=(1, 4),
        )
    )


def _initialize_benchmark(args) -> BenchmarkInit | int:
    """Initialize benchmark environment and load necessary data.

    Args:
        args: Command-line arguments

    Returns:
        tuple containing:
        - result_data: Dictionary with benchmark results
        - pandas_cmd: Command to run Pandas benchmark
        - polars_cmd: Command to run Polars benchmark
        - pandas_dir: Directory for Pandas outputs
        - polars_dir: Directory for Polars outputs
        - checkpoint_path: Path to checkpoint file
        - initial_memory: Initial memory usage
        OR
        - exit code (int) and 6 None values if initialization fails
    """
    initial_memory = _setup_logging(args.debug)

    # Get scenario path from args or environment
    scenario_path = args.scenario_path
    scenario_key = None

    if not scenario_path:
        scenario_key, scenario_path = load_scenario_path()
        if not scenario_path:
            console.print(
                "[error]No scenario path provided and none found in AZ_FULL_RESULTS.[/]"
            )
            return 1

    scenario_name = _get_scenario_info(scenario_path, scenario_key)
    pandas_dir, polars_dir = _prepare_directories(
        args.pandas_dir, args.polars_dir, scenario_name
    )

    checkpoint_path = _get_checkpoint_path(scenario_name)
    checkpoint_data = _load_checkpoint(checkpoint_path)

    result_data: BenchmarkData = {
        "pandas_completed": False,
        "polars_completed": False,
        "pandas_time": 0.0,
        "polars_time": 0.0,
        "pandas_memory": 0.0,
        "polars_memory": 0.0,
        "pandas_stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0},
        "polars_stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0},
        "scenario_name": scenario_name,
    }

    if checkpoint_data:
        # Apply checkpoint data values individually for type safety
        for key, value in checkpoint_data.items():
            result_data[key] = value
        console.print("[info]Resuming from previous benchmark run[/]")

    has_pandas_files = _check_implementation_completed(pandas_dir, "Pandas")
    has_polars_files = _check_implementation_completed(polars_dir, "Polars")

    if has_pandas_files or has_polars_files:
        console.print(
            "[warning]Existing result files will be overwritten for accurate timing[/]"
        )

    # Always force the benchmark to run by setting completed flags to False
    result_data["pandas_completed"] = False
    result_data["polars_completed"] = False

    # Build commands
    pandas_cmd, polars_cmd = _build_commands(scenario_path, pandas_dir, polars_dir)

    # Return all the initialized data as a TypedDict
    return BenchmarkInit(
        result_data=result_data,
        pandas_cmd=pandas_cmd,
        polars_cmd=polars_cmd,
        pandas_dir=pandas_dir,
        polars_dir=polars_dir,
        checkpoint_path=checkpoint_path,
        initial_memory=initial_memory,
        checkpoint_data=checkpoint_data,
        scenario_name=scenario_name,
    )


def _handle_keyboard_interrupt(
    checkpoint_path: str, result_data: dict[str, Any] | BenchmarkData
) -> int:
    """Handle keyboard interrupt during benchmark.

    Args:
        checkpoint_path: Path to checkpoint file
        result_data: Dictionary with benchmark results

    Returns:
        int: SIGINT_EXIT_CODE
    """
    console.print("\n[warning]Benchmark aborted by user. Progress saved.[/]")
    _save_checkpoint(checkpoint_path, result_data)
    return SIGINT_EXIT_CODE


# Type alias for benchmark result
BenchmarkResult = int | tuple[float, float, dict[str, float]]


def _run_benchmarks(
    config: RunBenchmarksConfig,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Run pandas and polars benchmarks.

    Args:
        config: Configuration for running benchmarks

    Returns:
        tuple: (pandas_result, polars_result) - Each result is either an exit code or
            a tuple of (time, memory, stats)
    """
    args = config["args"]
    result_data = config["result_data"]
    pandas_cmd = config["pandas_cmd"]
    polars_cmd = config["polars_cmd"]
    checkpoint_path = config["checkpoint_path"]
    checkpoint_data = config["checkpoint_data"]
    scenario_name = config["scenario_name"]  # Extract scenario_name from config

    try:
        # Configuration for pandas benchmark
        pandas_config: RunConfig = {
            "implementation": "pandas",
            "cmd": pandas_cmd,
            "runs": args.runs,
            "result_data": result_data,
            "checkpoint_data": checkpoint_data,
            "checkpoint_path": checkpoint_path,
            # Pass scenario_name to ensure proper file clean-up
            "scenario_name": scenario_name,
        }
        pandas_result = _run_implementation_benchmark(pandas_config)
        if pandas_result == SIGINT_EXIT_CODE:  # Keyboard interrupt
            return SIGINT_EXIT_CODE, 0

        # Configuration for polars benchmark
        polars_config: RunConfig = {
            "implementation": "polars",
            "cmd": polars_cmd,
            "runs": args.runs,
            "result_data": result_data,
            "checkpoint_data": checkpoint_data,
            "checkpoint_path": checkpoint_path,
            # Pass scenario_name to ensure proper file clean-up
            "scenario_name": scenario_name,
        }
        polars_result = _run_implementation_benchmark(polars_config)
        if polars_result == SIGINT_EXIT_CODE:  # Keyboard interrupt
            return pandas_result, SIGINT_EXIT_CODE

        return pandas_result, polars_result
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(checkpoint_path, result_data), 0


def main() -> int:
    """Main entry point for the benchmark script.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    exit_code = 0

    try:
        _print_banner()

        # Parse arguments and set up environment
        parser = _configure_parser()
        args = parser.parse_args()

        # Initialize benchmark environment
        init_result = _initialize_benchmark(args)

        # If an int is returned, it's an error code
        if isinstance(init_result, int):
            return init_result

        result_data = init_result["result_data"]
        pandas_cmd = init_result["pandas_cmd"]
        polars_cmd = init_result["polars_cmd"]
        pandas_dir = init_result["pandas_dir"]
        polars_dir = init_result["polars_dir"]
        checkpoint_path = init_result["checkpoint_path"]
        initial_memory = init_result["initial_memory"]
        checkpoint_data = init_result["checkpoint_data"]
        scenario_name = init_result["scenario_name"]

        benchmarks_config: RunBenchmarksConfig = {
            "args": args,
            "result_data": result_data,
            "pandas_cmd": pandas_cmd,
            "polars_cmd": polars_cmd,
            "checkpoint_path": checkpoint_path,
            "checkpoint_data": checkpoint_data,
            # Pass scenario_name to ensure proper file clean-up
            "scenario_name": scenario_name,
        }
        pandas_result, polars_result = _run_benchmarks(benchmarks_config)

        # Handle errors in benchmark runs
        if isinstance(pandas_result, int) or isinstance(polars_result, int):
            # Check if either result is the interrupt exit code
            if SIGINT_EXIT_CODE in (pandas_result, polars_result):
                return SIGINT_EXIT_CODE
        else:
            # Unpack benchmark results
            pandas_time, pandas_memory, pandas_stats = pandas_result
            polars_time, polars_memory, polars_stats = polars_result

            # Show comparison if both benchmarks are complete
            if result_data["pandas_completed"] and result_data["polars_completed"]:
                comparison_params: ComparisonParams = {
                    "pandas_time": pandas_time,
                    "polars_time": polars_time,
                    "pandas_memory": pandas_memory,
                    "polars_memory": polars_memory,
                    "pandas_stats": pandas_stats,
                    "polars_stats": polars_stats,
                    "pandas_dir": pandas_dir,
                    "polars_dir": polars_dir,
                    "scenario_name": scenario_name,
                    "checkpoint_path": checkpoint_path,
                    "debug_mode": args.debug,
                }
                exit_code = _show_comparison_and_cleanup(comparison_params)
            elif result_data["pandas_completed"] or result_data["polars_completed"]:
                _show_incomplete_benchmark_message(result_data)

            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            logger.debug(
                f"Final memory usage: {final_memory:.2f} MB "
                f"(delta: {memory_diff:+.2f} MB)"
            )

    except KeyboardInterrupt:
        console.print("\n[warning]Benchmark aborted by user.[/]")
        exit_code = SIGINT_EXIT_CODE
    except Exception as e:
        console.print(f"[error]An error occurred: {str(e)}[/]")
        if "args" in locals() and args.debug:
            console.print(traceback.format_exc())
        exit_code = 1

    return exit_code


# %%
if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code:
            sys.exit(exit_code)
    except KeyboardInterrupt:
        # Final fallback in case other handlers miss it
        console.print("\n[warning]Benchmark aborted by user.[/]")
        sys.exit(130)
