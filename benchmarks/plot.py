import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pyperf


@dataclass(frozen=True)
class Series:
    label: str
    means_seconds: dict[str, float]


def load_series(path: Path, label: str | None = None) -> Series:
    suite = pyperf.BenchmarkSuite.load(str(path))
    means: dict[str, float] = {}
    for bench in suite:
        means[bench.get_name()] = float(bench.mean())
    return Series(label=label or path.stem, means_seconds=means)


def common_benchmarks(a: Series, b: Series) -> list[str]:
    names = sorted(set(a.means_seconds) & set(b.means_seconds))
    if not names:
        raise SystemExit("No common benchmark names found between the two files.")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two pyperf JSON files and plot results")
    p.add_argument("a", type=Path, help="First pyperf JSON (e.g. arm64.json)")
    p.add_argument("b", type=Path, help="Second pyperf JSON (e.g. x86_64.json)")
    p.add_argument("--out", type=Path, default=Path("results/compare.png"), help="Output PNG path")
    p.add_argument("--title", default="arm64 vs x86_64 (lower is better)", help="Plot title")
    args = p.parse_args(argv)

    a = load_series(args.a)
    b = load_series(args.b)
    names = common_benchmarks(a, b)

    a_vals = [a.means_seconds[n] for n in names]
    b_vals = [b.means_seconds[n] for n in names]

    # Print a quick text table (b/a speedup).
    print(f"{a.label} vs {b.label}")
    for n, av, bv in zip(names, a_vals, b_vals, strict=True):
        speedup = bv / av if av else float("inf")
        print(f"- {n}: {av:.6f}s vs {bv:.6f}s  (b/a={speedup:.3f}x)")

    # Bar plot of mean seconds
    x = range(len(names))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    ax.bar([i - width / 2 for i in x], a_vals, width=width, label=a.label)
    ax.bar([i + width / 2 for i in x], b_vals, width=width, label=b.label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean time (seconds) — lower is better")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nWrote plot to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


