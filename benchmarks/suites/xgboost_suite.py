import os

import numpy as np
import pyperf


def _xgboost_nthread() -> int:
    """
    XGBoost has multiple thread controls; for the benchmarks we pin `nthread`
    so arch comparisons don't get dominated by different default parallelism.
    """
    try:
        n = int(os.environ.get("XGBOOST_NTHREAD", "1"))
    except ValueError:
        n = 1
    return max(1, n)


def _make_binary_classification(
    *,
    n_samples: int,
    n_features: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    w = rng.standard_normal((n_features,), dtype=np.float32)
    logits = x @ w + np.float32(0.1)  # avoid all-zero decision boundary
    y = (logits > 0).astype(np.int32)
    return x, y


def add_xgboost_training_benchmarks(runner: pyperf.Runner) -> None:
    """
    XGBoost training benchmark(s).

    Notes:
    - We pre-build the `DMatrix` outside the timed region so we primarily
      measure the training kernel.
    - We pin `nthread` (default via XGBOOST_NTHREAD=1) for comparability.
    """
    # If this import fails (missing wheel, missing OpenMP runtime, etc.),
    # we want a hard error when the xgboost suite is explicitly requested.
    import xgboost as xgb

    nthread = _xgboost_nthread()

    # Moderate size: heavy enough to be above timer noise, small enough to run
    # under pyperf without making the suite painfully slow.
    x_train, y_train = _make_binary_classification(n_samples=50_000, n_features=64, seed=0)
    dtrain = xgb.DMatrix(x_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "nthread": nthread,
        "seed": 0,
        "verbosity": 0,
    }
    num_boost_round = 200

    def train_hist() -> float:
        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        # Touch something stable so the training isn't optimized away.
        return float(booster.attributes().get("best_score", "0") or 0.0)

    runner.bench_func(
        f"xgboost.train_hist[{x_train.shape[0]}x{x_train.shape[1]},rounds={num_boost_round},nt={nthread}]",
        train_hist,
        inner_loops=1,
    )


def add_xgboost_inference_benchmarks(runner: pyperf.Runner) -> None:
    """
    XGBoost inference benchmark(s): time `Booster.predict` on a fixed DMatrix.
    """
    # If this import fails (missing wheel, missing OpenMP runtime, etc.),
    # we want a hard error when the xgboost suite is explicitly requested.
    import xgboost as xgb

    nthread = _xgboost_nthread()

    # Train once (not timed), then benchmark predict.
    x_train, y_train = _make_binary_classification(n_samples=50_000, n_features=64, seed=0)
    dtrain = xgb.DMatrix(x_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "nthread": nthread,
        "seed": 0,
        "verbosity": 0,
        "predictor": "cpu_predictor",
    }
    num_boost_round = 200
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    x_test, _y_test = _make_binary_classification(n_samples=50_000, n_features=64, seed=1)
    dtest = xgb.DMatrix(x_test)

    def predict_hist() -> float:
        yhat = booster.predict(dtest)
        return float(yhat[0])

    runner.bench_func(
        f"xgboost.predict_hist[{x_test.shape[0]}x{x_test.shape[1]},rounds={num_boost_round},nt={nthread}]",
        predict_hist,
        inner_loops=10,
    )


