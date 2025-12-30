import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyperf


@dataclass(frozen=True)
class BenchmarkStats:
    name: str
    min: float
    median: float
    max: float


def extract_stats(bench: Any) -> BenchmarkStats:
    """Extract min, median, and max from a pyperf Benchmark object."""
    import statistics

    # Try to use pyperf's built-in methods if available
    try:
        min_val = float(bench.min())
        median_val = float(bench.median())
        max_val = float(bench.max())
    except (AttributeError, TypeError):
        # Fallback: extract all values from runs and compute statistics
        values = []
        for run in bench.get_runs():
            # Skip calibration runs (they don't have 'values', only 'warmups')
            if hasattr(run, "values") and run.values:
                values.extend(run.values)

        if not values:
            # If no values found, try to use mean as fallback
            min_val = median_val = max_val = float(bench.mean())
        else:
            min_val = float(min(values))
            median_val = float(statistics.median(values))
            max_val = float(max(values))

    return BenchmarkStats(
        name=bench.get_name(),
        min=min_val,
        median=median_val,
        max=max_val,
    )


def load_benchmark_stats(path: Path) -> dict[str, BenchmarkStats]:
    """Load a pyperf JSON file and extract statistics for each benchmark."""
    suite = pyperf.BenchmarkSuite.load(str(path))
    stats: dict[str, BenchmarkStats] = {}
    for bench in suite:
        stats[bench.get_name()] = extract_stats(bench)
    return stats


def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.3f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    else:
        return f"{seconds:.3f} s"


def get_time_unit(seconds: float) -> str:
    """Get the appropriate unit for a time value in seconds."""
    if seconds < 1e-6:
        return "ns"
    elif seconds < 1e-3:
        return "μs"
    elif seconds < 1.0:
        return "ms"
    else:
        return "s"


def print_table(
    a_stats: dict[str, BenchmarkStats],
    b_stats: dict[str, BenchmarkStats],
    a_label: str,
    b_label: str,
    csv: bool = False,
) -> None:
    """Print a formatted table comparing two benchmark result sets."""
    # Find common benchmarks
    common_names = sorted(set(a_stats.keys()) & set(b_stats.keys()))
    if not common_names:
        print("ERROR: No common benchmark names found between the two files.")
        return

    if csv:
        # CSV output
        print("Benchmark Name,"
              f"{a_label}_min (s),{a_label}_median (s),{a_label}_max (s),"
              f"{b_label}_min (s),{b_label}_median (s),{b_label}_max (s),"
              f"B/A_ratio")
        for name in common_names:
            a = a_stats[name]
            b = b_stats[name]
            ratio = b.median / a.median if a.median > 0 else float("inf")
            print(f'"{name}",{a.min:.9e},{a.median:.9e},{a.max:.9e},'
                  f'{b.min:.9e},{b.median:.9e},{b.max:.9e},'
                  f'{ratio:.6e}')
        return

    # Determine a common unit for all values (pick based on median values)
    # Use the most common unit across all median values
    all_medians = [a_stats[name].median for name in common_names] + \
                  [b_stats[name].median for name in common_names]
    sample_median = sorted(all_medians)[len(all_medians) // 2]  # Use median of medians
    common_unit = get_time_unit(sample_median)
    
    # Convert seconds to the common unit
    def to_common_unit(seconds: float) -> float:
        if common_unit == "ns":
            return seconds * 1e9
        elif common_unit == "μs":
            return seconds * 1e6
        elif common_unit == "ms":
            return seconds * 1e3
        else:
            return seconds
    
    # Calculate maximum width for median values across ALL benchmarks
    # Using numbers without units - convert to common unit first
    all_med_values = [f"{to_common_unit(a_stats[name].median):.3f}" for name in common_names] + \
                     [f"{to_common_unit(b_stats[name].median):.3f}" for name in common_names]
    
    med_width = max(len(v) for v in all_med_values)

    # Table header - make benchmark name column narrower
    max_name_len = max(len(name) for name in common_names)
    name_col_width = max(25, max_name_len + 1)  # Reduced from 50 to 25
    
    # Calculate stats column width (just median value width)
    stats_col_width = med_width
    
    # Ratio column width - narrow, just enough for values like "2.10x" or "0.50x slower"
    ratio_header = f"{a_label} faster by"
    ratio_col_width = 10  # Fixed at 10 chars max
    
    total_width = name_col_width + stats_col_width + stats_col_width + ratio_col_width + 9  # 9 for separators

    # Build header line to measure its actual length
    header_line = (
        f"{'Benchmark Name':<{name_col_width}} | "
        f"{'A':>{stats_col_width}} | "
        f"{'B':>{stats_col_width}} | "
        f"{ratio_header:>{ratio_col_width}}"
    )
    separator_width = len(header_line)

    print(f"\nBenchmark Comparison (median {common_unit}): {a_label} vs {b_label}")
    print("=" * separator_width)
    
    # Simple column headers: just "A" and "B"
    print(header_line)
    print("-" * separator_width)

    # Table rows with median values only (numbers only, no units)
    for name in common_names:
        a = a_stats[name]
        b = b_stats[name]
        
        # Format median value with right alignment (no units)
        # Convert to common unit
        a_med_str = f"{to_common_unit(a.median):.3f}".rjust(stats_col_width)
        b_med_str = f"{to_common_unit(b.median):.3f}".rjust(stats_col_width)
        
        # Calculate ratio: B/A shows how many times faster A is than B
        # > 1 means A is faster, < 1 means A is slower (B is faster)
        ratio = b.median / a.median if a.median > 0 else float("inf")
        if ratio == float("inf"):
            ratio_str = "inf"
        elif ratio >= 1.0:
            # A is faster: "A is 2.1x faster than B"
            ratio_str = f"{ratio:.2f}x"
        else:
            # A is slower: show as "B is faster" by inverting
            ratio_inv = a.median / b.median
            ratio_str = f"{ratio_inv:.2f}x slower"
        
        print(f"{name:<{name_col_width}} | {a_med_str} | {b_med_str} | {ratio_str:>{ratio_col_width}}")

    print("=" * separator_width)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare two pyperf JSON files and print a statistics table"
    )
    p.add_argument("a", type=Path, help="First pyperf JSON (e.g. arm64.json)")
    p.add_argument("b", type=Path, help="Second pyperf JSON (e.g. x86_64.json)")
    p.add_argument("--a-label", default=None, help="Label for first file (default: filename stem)")
    p.add_argument("--b-label", default=None, help="Label for second file (default: filename stem)")
    p.add_argument("--csv", action="store_true", help="Output as CSV instead of formatted table")
    args = p.parse_args(argv)

    a_stats = load_benchmark_stats(args.a)
    b_stats = load_benchmark_stats(args.b)

    a_label = args.a_label or args.a.stem
    b_label = args.b_label or args.b.stem

    print_table(a_stats, b_stats, a_label, b_label, csv=args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

