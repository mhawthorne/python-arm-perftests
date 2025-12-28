import argparse
import json
import os
import platform
import sys

import pyperf

from suites.numpy_suite import add_numpy_benchmarks
from suites.python_suite import add_python_benchmarks
from suites.xgboost_suite import add_xgboost_inference_benchmarks, add_xgboost_training_benchmarks


def _env_threads_metadata() -> dict[str, str]:
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "XGBOOST_NTHREAD",
    ]
    return {k: os.environ.get(k, "") for k in keys if os.environ.get(k) is not None}


def _numpy_config_text(max_chars: int = 4000) -> str:
    """
    Keep this as a single-line string (pyperf metadata forbids literal newlines).

    NumPy 2.x exposes `numpy.__config__.show_config(mode="dicts")` which returns a
    structured representation of build/runtime config. We JSON-encode it to keep
    it single-line.
    """
    try:
        import numpy as np  # local import: allow pure-Python suites without NumPy

        try:
            conf: object = np.__config__.show_config(mode="dicts")
        except TypeError:
            # Older NumPy: no mode kwarg; fall back to a minimal indicator.
            conf = {"numpy.__config__.show_config": "no-dicts-mode"}
    except Exception as exc:  # pragma: no cover
        conf = {"numpy.__config__": f"unavailable: {type(exc).__name__}"}

    text = json.dumps(conf, sort_keys=True, separators=(",", ":"))
    if len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text


def _default_metadata() -> dict[str, object]:
    meta: dict[str, object] = {}
    meta["platform.machine"] = platform.machine()
    meta["platform.processor"] = platform.processor()
    meta["platform.platform"] = platform.platform()
    meta["python.version"] = sys.version.replace("\n", " ")
    try:
        import numpy as np

        meta["numpy.version"] = np.__version__
        meta["numpy.config"] = _numpy_config_text()
    except Exception:  # pragma: no cover
        # Allow running a pure-Python suite in environments without NumPy.
        pass
    try:
        import xgboost as xgb  # local import: allow running without xgboost installed

        meta["xgboost.version"] = getattr(xgb, "__version__", "")
        try:
            # Keep it single-line + bounded.
            info = xgb.build_info()
            text = json.dumps(info, sort_keys=True, separators=(",", ":"))
            meta["xgboost.build_info"] = text[:4000] + ("...(truncated)" if len(text) > 4000 else "")
        except Exception:
            pass
    except Exception:  # pragma: no cover
        pass
    meta.update(_env_threads_metadata())
    return meta


def _add_suite_cmdline_args(cmd: list[str], args: argparse.Namespace) -> None:
    """
    pyperf spawns worker processes with only pyperf's own flags by default.
    Forward our custom CLI options (like --suite) so workers register the same
    benchmarks as the manager process.
    """
    cmd.extend(["--suite", str(getattr(args, "suite", "numpy"))])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pyperf benchmarks (arm64 vs x86_64)")
    parser.add_argument(
        "--suite",
        default="numpy",
        choices=["numpy", "python", "xgboost_train", "xgboost_infer"],
        help="Benchmark suite to run",
    )

    # pyperf will add its own CLI flags to this parser (e.g. --output, --rigorous, --fast, ...).
    runner = pyperf.Runner(
        metadata=_default_metadata(),
        _argparser=parser,
        add_cmdline_args=_add_suite_cmdline_args,
    )
    args = runner.parse_args(argv)

    if args.suite == "numpy":
        add_numpy_benchmarks(runner)
        return 0
    if args.suite == "python":
        add_python_benchmarks(runner)
        return 0
    if args.suite == "xgboost_train":
        try:
            import xgboost as _xgb  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"xgboost suite requested but `import xgboost` failed: {type(exc).__name__}: {exc}"
            ) from exc
        add_xgboost_training_benchmarks(runner)
        return 0
    if args.suite == "xgboost_infer":
        try:
            import xgboost as _xgb  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"xgboost suite requested but `import xgboost` failed: {type(exc).__name__}: {exc}"
            ) from exc
        add_xgboost_inference_benchmarks(runner)
        return 0

    raise SystemExit(f"unknown suite: {args.suite}")


if __name__ == "__main__":
    raise SystemExit(main())


