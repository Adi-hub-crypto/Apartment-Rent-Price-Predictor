"""Train the rent model, report MAE, and write submission.csv.

    python3 train.py                 # compare all three estimators, use OLS
    python3 train.py --method gd     # fit with gradient descent instead
    python3 train.py --no-compare    # skip the comparison
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from rent_data import (DATA_DIR, DISTRICTS, TARGET, describe_dataset,
                       load_test, load_train)
from rent_model import METHODS, evaluate, fit_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ols", choices=list(METHODS))
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--model-out", default="model.pkl")
    ap.add_argument("--metrics-out", default="metrics.json")
    args = ap.parse_args()

    train = load_train()
    calibrate = not args.no_calibrate

    print("=" * 66)
    print("DATA")
    print("=" * 66)
    print(describe_dataset(train))

    comparison = {}
    if not args.no_compare:
        print("\n" + "=" * 66)
        print("5-FOLD CROSS-VALIDATION (MAE - lower is better)")
        print("=" * 66)
        for method in METHODS:
            t0 = time.time()
            res = evaluate(train, method=method, calibrate=calibrate)
            comparison[method] = {"mae": res["mae"], "std": res["mae_std"],
                                  "label": METHODS[method]}
            print(f"{METHODS[method]:52s} {res['mae']:8.2f} "
                  f"+/- {res['mae_std']:5.2f}   ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 66)
    print(f"CHOSEN MODEL - {METHODS[args.method]}")
    print("=" * 66)
    res = evaluate(train, method=args.method, calibrate=calibrate)
    print(f"cross-validated MAE : {res['mae']:.2f} +/- {res['mae_std']:.2f}")
    print(f"median error        : {res['median_ae']:.2f}")
    print(f"competition points  : {res['points']:.4f}   (1 - MAE/1500)")

    print("\nMAE by district (out of fold):")
    for name, value in sorted(res["mae_by_district"].items(), key=lambda kv: kv[1]):
        share = int((train["district"] == name).sum())
        print(f"  {name:14s} {value:9.2f}   ({share} rows)")

    print("\n" + "=" * 66)
    print("FINAL FIT ON ALL TRAINING DATA")
    print("=" * 66)
    model = fit_full(train, method=args.method, calibrate=calibrate)
    print(f"trained on {len(train)} rows, {model.n_features_} features")
    if getattr(model.model, "loss_history_", None):
        history = model.model.loss_history_
        print(f"training loss (log space): {history[0]:.5f} -> {history[-1]:.5f}")
    if model.factors_:
        print("per-district calibration: "
              + ", ".join(f"{k}×{v:.3f}" for k, v in model.factors_.items()))
    model.save(args.model_out)
    print(f"model saved to {args.model_out}")

    metrics = {
        "method": args.method,
        "method_label": METHODS[args.method],
        "n_train": len(train),
        "n_features": int(model.n_features_),
        "mae": res["mae"],
        "mae_std": res["mae_std"],
        "median_ae": res["median_ae"],
        "points": res["points"],
        "baseline_median_mae": float((train[TARGET] - train[TARGET].median()).abs().mean()),
        "comparison": comparison,
        "mae_by_district": res["mae_by_district"],
        "district_rows": {d: int((train["district"] == d).sum()) for d in DISTRICTS},
        "district_median_rent": {
            d: float(train.loc[train["district"] == d, TARGET].median())
            for d in DISTRICTS},
        "calibration": model.factors_,
        "reference": model.reference_,
        "loss_history": [float(v) for v in
                         getattr(model.model, "loss_history_", []) or []],
    }
    with open(args.metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"metrics saved to {args.metrics_out}")

    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        print("\n" + "!" * 66)
        print("data/test.csv is missing, so no submission was written.")
        print("Drop the test file there and re-run, or upload it in the browser")
        print("interface (python3 web_app.py) to get submission.csv.")
        print("!" * 66)
        return

    test = load_test()
    preds = model.predict(test)
    pd.DataFrame({"id": test["id"], TARGET: preds}).to_csv(
        args.submission, index=False, lineterminator="\r\n")
    print(f"\npredictions written to {args.submission} ({len(preds)} rows)")
    print(f"predicted rent: min {preds.min():.2f} median {np.median(preds):.2f} "
          f"max {preds.max():.2f}")


if __name__ == "__main__":
    main()
