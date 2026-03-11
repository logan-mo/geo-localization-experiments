"""
step2_smoke_test.py
--------------------
Verifies the evaluation harness works end-to-end using synthetic predictions
(random offsets from GT).  Run this before connecting real matchers to confirm
paths, meta loading, metric math, and plotting all work correctly.

Usage:
    python step2_smoke_test.py --pairs_dir ./pairs --output_dir ./results/smoke
"""

import argparse
import math
import random
from pathlib import Path

import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from step2_evaluation import EvaluationHarness, Timer, THRESHOLDS_M


def run_smoke_test(pairs_dir: str, output_dir: str):
    # pairs_index.csv is written directly into pairs_dir by Step 1
    pairs_index = Path(pairs_dir) / "pairs_index.csv"
    if not pairs_index.exists():
        raise FileNotFoundError(
            f"pairs_index.csv not found at {pairs_index}\n"
            "Run step1_preprocessing.py first."
        )

    index_df = pd.read_csv(pairs_index)
    print(f"[SMOKE] Found {len(index_df)} pairs in index.")

    harness = EvaluationHarness(pairs_dir=pairs_dir, output_dir=output_dir)

    # ── Synthetic method A: "good" matcher — errors mostly < 10m ────────
    random.seed(42)
    for _, row in index_df.iterrows():
        # Simulate a good prediction: small Gaussian error in metres
        err_m = abs(random.gauss(0, 5))    # sigma = 5m
        angle  = random.uniform(0, 2 * math.pi)

        with Timer() as t:
            pass  # simulate matcher call
        t.ms = random.uniform(80, 200)      # fake inference time

        harness.record(
            method       = "Synthetic_Good",
            query_name   = row["query"],
            pred_latlon  = (
                row["gt_dy_m"] / 111_320.0   # back-convert GT + small offset
                    + err_m * math.sin(angle) / 111_320.0
                    + _ref_lat(pairs_dir),
                row["gt_dx_m"] / (111_320.0 * math.cos(math.radians(_ref_lat(pairs_dir))))
                    + err_m * math.cos(angle) / (111_320.0 * math.cos(math.radians(_ref_lat(pairs_dir))))
                    + _ref_lon(pairs_dir),
            ),
            inference_ms = t.ms,
        )

    harness.finalise("Synthetic_Good")

    # ── Synthetic method B: "weak" matcher — larger errors, some failures ─
    random.seed(7)
    for _, row in index_df.iterrows():
        failed = random.random() < 0.15    # 15% failure rate
        if failed:
            harness.record(
                method     = "Synthetic_Weak",
                query_name = row["query"],
                failed     = True,
            )
        else:
            err_m  = abs(random.gauss(0, 20))
            angle  = random.uniform(0, 2 * math.pi)
            harness.record(
                method       = "Synthetic_Weak",
                query_name   = row["query"],
                pred_latlon  = (
                    row["gt_dy_m"] / 111_320.0
                        + err_m * math.sin(angle) / 111_320.0
                        + _ref_lat(pairs_dir),
                    row["gt_dx_m"] / (111_320.0 * math.cos(math.radians(_ref_lat(pairs_dir))))
                        + err_m * math.cos(angle) / (111_320.0 * math.cos(math.radians(_ref_lat(pairs_dir))))
                        + _ref_lon(pairs_dir),
                ),
                inference_ms = random.uniform(20, 50),
            )

    harness.finalise("Synthetic_Weak")

    # ── Comparison across both methods ────────────────────────────────────
    harness.compare()

    print("\n[SMOKE] ✓ All checks passed — harness is ready for real matchers.")
    print(f"[SMOKE] Outputs saved to: {output_dir}")


def _ref_lat(pairs_dir: str) -> float:
    """Read reference image lat from the first pair's meta.json."""
    import json
    first = next((Path(pairs_dir) / "pairs").iterdir())
    with open(first / "meta.json") as f:
        return json.load(f)["ref_lat"]


def _ref_lon(pairs_dir: str) -> float:
    import json
    first = next((Path(pairs_dir) / "pairs").iterdir())
    with open(first / "meta.json") as f:
        return json.load(f)["ref_lon"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_smoke_test(args.pairs_dir, args.output_dir)
