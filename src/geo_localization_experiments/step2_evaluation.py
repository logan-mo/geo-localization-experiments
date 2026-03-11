"""
=============================================================================
STEP 2 — Evaluation Harness
=============================================================================
Geo-Localization Research Pipeline
-------------------------------------
Matcher-agnostic scoring backbone used by all three tiers.

Responsibilities:
  1. Accept predictions from ANY matcher in two formats:
       a) Pixel offset within the reference crop  →  converted to metres
       b) Absolute (lat, lon)                     →  compared to GT lat/lon
  2. Convert pixel predictions → metre errors using GSD from meta.json
  3. Compute standard metrics:
       MAE, RMSE, Median error
       @1m / @5m / @10m / @25m / @50m accuracy (% of queries below threshold)
       Failure rate (matcher returned no result)
  4. Accumulate results across multiple methods for Tier 2 comparison
  5. Generate publication-ready outputs:
       results_<method>.csv       per-pair breakdown
       comparison_table.csv       all methods side by side
       cdf_plot.png               CDF curves matching UAVD/AnyVisLoc style
       error_bars.png             @Xm accuracy bar chart
       scatter_gt_vs_pred.png     per-pair scatter (good for spotting outliers)
  6. Standard-dataset adapters: VIGOR, UAVD/AnyVisLoc

Usage — from a matcher script:
    from step2_evaluation import EvaluationHarness

    harness = EvaluationHarness(pairs_dir="./pairs", output_dir="./results")

    # After running your matcher on a pair:
    harness.record(
        method      = "LoFTR",
        query_name  = "20260310_173657_991.png",
        pred_px     = (512, 480),   # pixel coords in reference_crop_resized
        # — OR —
        pred_latlon = (33.5511, 73.1244),
        inference_ms = 142.3,       # optional timing
        n_matches    = 350,         # optional match count
    )

    harness.finalise("LoFTR")       # writes per-method CSV + plots
    harness.compare()               # writes comparison table across all methods

Standalone CLI (score pre-saved prediction CSVs):
    python step2_evaluation.py \
        --pairs_dir  ./pairs \
        --output_dir ./results \
        --predictions LoFTR:./preds_loftr.csv SuperGlue:./preds_sg.csv
=============================================================================
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats            # for CDF / percentile computations


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Accuracy thresholds in metres — matches UAVD / AnyVisLoc convention
# plus finer thresholds useful for your ~450–500m altitude data
THRESHOLDS_M = [1, 5, 10, 25, 50, 100]

# Colour palette for up to 8 methods — matplotlib-compatible
METHOD_COLOURS = [
    "#E63946",   # red      — RoMa
    "#2196F3",   # blue     — LoFTR
    "#4CAF50",   # green    — LightGlue+SuperPoint
    "#FF9800",   # orange   — LightGlue+DISK
    "#9C27B0",   # purple   — ALIKED+LightGlue
    "#009688",   # teal     — EfficientLoFTR
    "#795548",   # brown    — SIFT
    "#607D8B",   # grey     — Template matching
]

# Maximum error used when a matcher fails completely (no prediction returned).
# Set to a large value so failures don't artificially improve median/MAE.
FAILURE_PENALTY_M = 1000.0


# ---------------------------------------------------------------------------
# 1. UNIT CONVERSION UTILITIES
# ---------------------------------------------------------------------------

def pixel_offset_to_metres(dx_px: float, dy_px: float, gsd_m_px: float) -> tuple:
    """
    Convert a (dx, dy) pixel offset in the reference crop coordinate space
    into (dx_m, dy_m) in metres using the reference GSD.

    Args:
        dx_px, dy_px : pixel offset from crop centre to predicted position
        gsd_m_px     : ground sampling distance of the reference crop (m/px)

    Returns:
        (dx_m, dy_m) in metres
    """
    return dx_px * gsd_m_px, dy_px * gsd_m_px


def latlon_to_metres(pred_lat: float, pred_lon: float,
                     gt_lat:   float, gt_lon:   float) -> tuple:
    """
    Convert (pred_lat, pred_lon) → (dx_m, dy_m) error relative to GT.

    Uses the equirectangular approximation — accurate to < 0.1% for
    displacements under 10 km (well within any single drone flight).

    Returns:
        (dx_m, dy_m) signed components and total Euclidean distance in metres
    """
    dx_m = (pred_lon - gt_lon) * 111_320.0 * math.cos(math.radians(gt_lat))
    dy_m = (pred_lat - gt_lat) * 111_320.0
    dist_m = math.sqrt(dx_m ** 2 + dy_m ** 2)
    return dx_m, dy_m, dist_m


def pred_px_in_crop_to_metres(pred_x: float, pred_y: float,
                               meta: dict) -> tuple:
    """
    Convert a predicted pixel position (x, y) in the resized reference crop
    to a position error in metres against ground truth.

    The ground truth in crop space is the centre of the crop, because
    Step 1 always centres the crop on the GPS prior position.
    Any offset of the predicted point from centre is a localization error.

    The crop has been resized to query image dimensions, so we need to
    scale pixel coordinates back to original GSD space first.

    Args:
        pred_x, pred_y : predicted position in resized crop pixel space
        meta           : loaded meta.json dict for this pair

    Returns:
        (dx_m, dy_m, dist_m)
    """
    crop_box   = meta["crop_bbox_px"]               # [x1,y1,x2,y2] in ref
    crop_w_ref = crop_box[2] - crop_box[0]          # width in ref pixels
    crop_h_ref = crop_box[3] - crop_box[1]          # height in ref pixels

    # GT centre in the resized crop is the GT pixel offset recorded in meta
    # (recorded relative to the crop origin in Step 1)
    gt_x_crop, gt_y_crop = meta["gt_offset_px"]

    # Scale GT to resized-crop pixel space
    # Step 1 resized to query image dims — we need those dims
    # We infer them from gt_offset_px ratio over the original crop
    # Safest: use the scale between original crop and resized crop
    # Original crop dims are crop_w_ref x crop_h_ref (in ref pixels)
    # Resized to query dims — we don't store query dims in meta directly,
    # but we can reconstruct: query_gsd / ref_gsd = scale_factor
    # query footprint / query_gsd = query pixel dims
    scale_x = meta["scale_factor"]   # query_gsd / ref_gsd
    # resized crop pixel = original crop ref pixel / scale_factor
    # (because ref GSD is finer → more pixels; scale_factor < 1 usually)

    # GT in resized crop pixel space:
    gt_x_resized = gt_x_crop / scale_x
    gt_y_resized = gt_y_crop / scale_x

    # Error in resized crop pixels
    err_x_px = pred_x - gt_x_resized
    err_y_px = pred_y - gt_y_resized

    # Convert to metres via query GSD (resized crop has query GSD)
    query_gsd = meta["query_gsd"]
    dx_m  = err_x_px * query_gsd
    dy_m  = err_y_px * query_gsd
    dist_m = math.sqrt(dx_m ** 2 + dy_m ** 2)
    return dx_m, dy_m, dist_m


# ---------------------------------------------------------------------------
# 2. SINGLE-PAIR RESULT RECORD
# ---------------------------------------------------------------------------

class PairResult:
    """Holds the evaluation result for one (query, matcher) combination."""

    __slots__ = [
        "query_name", "method",
        "pred_lat", "pred_lon",
        "pred_x_crop", "pred_y_crop",
        "dx_m", "dy_m", "dist_m",
        "failed",
        "inference_ms", "n_matches",
        "gt_lat", "gt_lon", "gt_dist_from_ref_m",
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.failed = False


# ---------------------------------------------------------------------------
# 3. EVALUATION HARNESS
# ---------------------------------------------------------------------------

class EvaluationHarness:
    """
    Central scoring engine.  Instantiate once, call .record() after each
    matcher prediction, call .finalise(method) when a method is done,
    and .compare() to generate the cross-method comparison.
    """

    def __init__(self, pairs_dir: str, output_dir: str):
        self.pairs_dir  = Path(pairs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffer: method_name → list of PairResult
        self._results: dict[str, list[PairResult]] = {}

        # Cache loaded meta.json files
        self._meta_cache: dict[str, dict] = {}

        # Ordered list of methods (insertion order for consistent colouring)
        self._method_order: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self,
               method:       str,
               query_name:   str,
               pred_px:      Optional[tuple] = None,
               pred_latlon:  Optional[tuple] = None,
               failed:       bool            = False,
               inference_ms: Optional[float] = None,
               n_matches:    Optional[int]   = None) -> PairResult:
        """
        Record one matcher prediction for one pair.

        Provide EITHER pred_px (x, y in resized reference crop pixels)
        OR pred_latlon (lat, lon in decimal degrees).

        If the matcher failed to produce a prediction, set failed=True —
        the error is recorded as FAILURE_PENALTY_M so it counts against
        @Xm accuracy but doesn't hide the failure in aggregate stats.

        Returns the populated PairResult for optional inspection.
        """
        if method not in self._results:
            self._results[method] = []
            self._method_order.append(method)

        meta = self._load_meta(query_name)
        r    = PairResult()
        r.query_name   = query_name
        r.method       = method
        r.inference_ms = inference_ms
        r.n_matches    = n_matches
        r.failed       = failed
        r.gt_lat       = meta["query_lat"]
        r.gt_lon       = meta["query_lon"]
        r.gt_dist_from_ref_m = math.sqrt(
            meta["gt_offset_m"][0] ** 2 + meta["gt_offset_m"][1] ** 2
        )

        if failed or (pred_px is None and pred_latlon is None):
            r.failed = True
            r.dx_m   = FAILURE_PENALTY_M
            r.dy_m   = FAILURE_PENALTY_M
            r.dist_m = FAILURE_PENALTY_M * math.sqrt(2)

        elif pred_latlon is not None:
            r.pred_lat, r.pred_lon = pred_latlon
            r.dx_m, r.dy_m, r.dist_m = latlon_to_metres(
                r.pred_lat, r.pred_lon, r.gt_lat, r.gt_lon
            )

        elif pred_px is not None:
            r.pred_x_crop, r.pred_y_crop = pred_px
            r.dx_m, r.dy_m, r.dist_m = pred_px_in_crop_to_metres(
                r.pred_x_crop, r.pred_y_crop, meta
            )

        self._results[method].append(r)
        return r

    def finalise(self, method: str, print_summary: bool = True) -> pd.DataFrame:
        """
        Compute metrics for one method, save per-pair CSV and per-method plots.
        Returns a DataFrame with one row per pair.
        """
        if method not in self._results:
            raise KeyError(f"No results recorded for method '{method}'")

        df = self._results_to_df(method)
        csv_path = self.output_dir / f"results_{method.replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)

        metrics = self._compute_metrics(df)

        if print_summary:
            self._print_summary(method, metrics, df)

        # Per-method plots
        self._plot_error_histogram(df, method)

        return df

    def compare(self, print_table: bool = True) -> pd.DataFrame:
        """
        Aggregate all recorded methods into a comparison table,
        and generate multi-method CDF + bar chart plots.

        Call this after finalise() has been called for every method.
        Returns a DataFrame with one row per method.
        """
        rows = []
        all_errors: dict[str, np.ndarray] = {}

        for method in self._method_order:
            if method not in self._results:
                continue
            df      = self._results_to_df(method)
            metrics = self._compute_metrics(df)
            all_errors[method] = df["dist_m"].values
            rows.append({"method": method, **metrics})

        comp_df = pd.DataFrame(rows)
        comp_path = self.output_dir / "comparison_table.csv"
        comp_df.to_csv(comp_path, index=False)

        if print_table:
            self._print_comparison_table(comp_df)

        # Multi-method plots
        self._plot_cdf(all_errors)
        self._plot_accuracy_bars(comp_df)
        self._plot_scatter_all(all_errors)

        print(f"\n[COMPARE] Saved → {self.output_dir}")
        return comp_df

    def load_predictions_csv(self, method: str, csv_path: str):
        """
        Load pre-saved predictions from a CSV file produced by a matcher.

        Expected CSV columns:
            query_name, pred_lat, pred_lon [, inference_ms, n_matches, failed]
        OR:
            query_name, pred_x_crop, pred_y_crop [, inference_ms, n_matches, failed]

        This lets you run matchers separately and score them all here.
        """
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            failed = bool(row.get("failed", False))

            pred_latlon = None
            pred_px     = None

            if "pred_lat" in row and not pd.isna(row["pred_lat"]):
                pred_latlon = (float(row["pred_lat"]), float(row["pred_lon"]))
            elif "pred_x_crop" in row and not pd.isna(row["pred_x_crop"]):
                pred_px = (float(row["pred_x_crop"]), float(row["pred_y_crop"]))
            else:
                failed = True

            self.record(
                method       = method,
                query_name   = str(row["query_name"]),
                pred_px      = pred_px,
                pred_latlon  = pred_latlon,
                failed       = failed,
                inference_ms = float(row["inference_ms"]) if "inference_ms" in row else None,
                n_matches    = int(row["n_matches"])   if "n_matches"    in row else None,
            )

    # ------------------------------------------------------------------
    # Internal: meta loading
    # ------------------------------------------------------------------

    def _load_meta(self, query_name: str) -> dict:
        if query_name in self._meta_cache:
            return self._meta_cache[query_name]

        stem = Path(query_name).stem

        # Support two layouts:
        #   Layout A (Step 1 default): pairs_dir/pairs/<stem>/meta.json
        #   Layout B (user passes pairs subdir directly): pairs_dir/<stem>/meta.json
        candidate_a = self.pairs_dir / "pairs" / stem / "meta.json"
        candidate_b = self.pairs_dir / stem / "meta.json"

        if candidate_a.exists():
            meta_path = candidate_a
        elif candidate_b.exists():
            meta_path = candidate_b
        else:
            raise FileNotFoundError(
                f"meta.json not found for query '{query_name}'.\n"
                f"Checked:\n  {candidate_a}\n  {candidate_b}\n"
                f"Make sure Step 1 has been run and --pairs_dir points to "
                f"the output_dir from Step 1 (not the pairs/ subfolder)."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        self._meta_cache[query_name] = meta
        return meta

    # ------------------------------------------------------------------
    # Internal: DataFrame builder
    # ------------------------------------------------------------------

    def _results_to_df(self, method: str) -> pd.DataFrame:
        rows = []
        for r in self._results[method]:
            rows.append({
                "query_name":          r.query_name,
                "method":              r.method,
                "dist_m":              r.dist_m,
                "dx_m":                r.dx_m,
                "dy_m":                r.dy_m,
                "failed":              r.failed,
                "gt_lat":              r.gt_lat,
                "gt_lon":              r.gt_lon,
                "pred_lat":            r.pred_lat,
                "pred_lon":            r.pred_lon,
                "pred_x_crop":         r.pred_x_crop,
                "pred_y_crop":         r.pred_y_crop,
                "gt_dist_from_ref_m":  r.gt_dist_from_ref_m,
                "inference_ms":        r.inference_ms,
                "n_matches":           r.n_matches,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal: metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        errors = df["dist_m"].values
        n      = len(errors)
        n_fail = int(df["failed"].sum())

        # Exclude failures from continuous stats so MAE/median reflect
        # actual matcher quality (failure rate is reported separately)
        valid_errors = errors[~df["failed"].values]

        metrics = {
            "n_pairs":       n,
            "n_failed":      n_fail,
            "failure_rate":  round(n_fail / n, 4) if n > 0 else 0.0,
        }

        if len(valid_errors) > 0:
            metrics["mae_m"]    = round(float(np.mean(valid_errors)),   3)
            metrics["rmse_m"]   = round(float(np.sqrt(np.mean(valid_errors**2))), 3)
            metrics["median_m"] = round(float(np.median(valid_errors)), 3)
            metrics["p75_m"]    = round(float(np.percentile(valid_errors, 75)), 3)
            metrics["p90_m"]    = round(float(np.percentile(valid_errors, 90)), 3)
            metrics["p95_m"]    = round(float(np.percentile(valid_errors, 95)), 3)
        else:
            for k in ["mae_m", "rmse_m", "median_m", "p75_m", "p90_m", "p95_m"]:
                metrics[k] = float("nan")

        # @Xm accuracy — counts ALL pairs (including failures)
        for t in THRESHOLDS_M:
            pct = float(np.sum(errors < t)) / n * 100 if n > 0 else 0.0
            metrics[f"@{t}m"] = round(pct, 2)

        # Timing stats (optional)
        timing = df["inference_ms"].dropna()
        if len(timing) > 0:
            metrics["mean_ms"] = round(float(timing.mean()), 2)
            metrics["p95_ms"]  = round(float(timing.quantile(0.95)), 2)

        return metrics

    # ------------------------------------------------------------------
    # Internal: printing
    # ------------------------------------------------------------------

    def _print_summary(self, method: str, metrics: dict, df: pd.DataFrame):
        print(f"\n{'='*60}")
        print(f"  {method}  ({metrics['n_pairs']} pairs, "
              f"{metrics['n_failed']} failures / "
              f"{metrics['failure_rate']*100:.1f}%)")
        print(f"{'='*60}")
        print(f"  MAE      : {metrics['mae_m']:.2f} m")
        print(f"  RMSE     : {metrics['rmse_m']:.2f} m")
        print(f"  Median   : {metrics['median_m']:.2f} m")
        print(f"  P75 / P90: {metrics['p75_m']:.2f} m  /  {metrics['p90_m']:.2f} m")
        print(f"  ── Accuracy ──────────────────────────────────")
        for t in THRESHOLDS_M:
            bar_len = int(metrics[f'@{t}m'] / 2)
            bar     = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  @{t:>3}m: {metrics[f'@{t}m']:5.1f}%  {bar}")
        if "mean_ms" in metrics:
            print(f"  ── Timing ────────────────────────────────────")
            print(f"  Mean inference: {metrics['mean_ms']:.1f} ms  "
                  f"P95: {metrics['p95_ms']:.1f} ms")
        print(f"{'='*60}")

    def _print_comparison_table(self, comp_df: pd.DataFrame):
        print(f"\n{'='*80}")
        print("  COMPARISON TABLE")
        print(f"{'='*80}")

        # Header
        cols = ["method", "n_pairs", "failure_rate", "mae_m",
                "median_m"] + [f"@{t}m" for t in THRESHOLDS_M]
        available = [c for c in cols if c in comp_df.columns]
        print(comp_df[available].to_string(index=False))
        print(f"{'='*80}")

    # ------------------------------------------------------------------
    # Internal: plotting
    # ------------------------------------------------------------------

    def _colour(self, method: str) -> str:
        idx = self._method_order.index(method) if method in self._method_order else 0
        return METHOD_COLOURS[idx % len(METHOD_COLOURS)]

    def _plot_cdf(self, all_errors: dict[str, np.ndarray]):
        """
        CDF of localization error — matches style of Geo-LoFTR / UAVD papers.
        X-axis: error in metres (0–50m).
        Y-axis: cumulative % of queries.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        max_plot_m = 50
        x_range    = np.linspace(0, max_plot_m, 500)

        for method, errors in all_errors.items():
            # Use all errors including failures (penalty keeps them at tail)
            cdf = np.array([np.mean(errors < t) * 100 for t in x_range])
            at1m = np.mean(errors < 1) * 100

            colour = self._colour(method)
            ax.plot(x_range, cdf,
                    label=f"{method}  (@1m: {at1m:.1f}%)",
                    color=colour, linewidth=2)

        # Threshold reference lines
        for t in [1, 5, 10]:
            ax.axvline(t, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(t + 0.3, 5, f"{t}m", fontsize=8, color="grey")

        ax.set_xlabel("Localization Error [m]", fontsize=12)
        ax.set_ylabel("Cumulative Accuracy [%]", fontsize=12)
        ax.set_title("CDF of Localization Error — All Methods", fontsize=13)
        ax.set_xlim(0, max_plot_m)
        ax.set_ylim(0, 101)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        fig.tight_layout()
        fig.savefig(str(self.output_dir / "cdf_plot.png"), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] cdf_plot.png")

    def _plot_accuracy_bars(self, comp_df: pd.DataFrame):
        """
        Grouped bar chart: @Xm accuracy for each method.
        """
        threshold_cols = [f"@{t}m" for t in THRESHOLDS_M
                          if f"@{t}m" in comp_df.columns]
        methods = comp_df["method"].tolist()
        n_methods = len(methods)
        n_thresh  = len(threshold_cols)

        x      = np.arange(n_thresh)
        width  = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, method in enumerate(methods):
            row    = comp_df[comp_df["method"] == method].iloc[0]
            values = [row[c] for c in threshold_cols]
            offset = (i - n_methods / 2 + 0.5) * width
            bars   = ax.bar(x + offset, values, width * 0.9,
                            label=method, color=self._colour(method),
                            alpha=0.85)
            # Value labels on bars
            for bar, v in zip(bars, values):
                if v > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f"{v:.0f}", ha="center", va="bottom",
                            fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("@", "@\n") for c in threshold_cols])
        ax.set_ylabel("Queries within threshold [%]", fontsize=11)
        ax.set_title("Localization Accuracy @ Distance Thresholds", fontsize=12)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        fig.tight_layout()
        fig.savefig(str(self.output_dir / "error_bars.png"), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] error_bars.png")

    def _plot_scatter_all(self, all_errors: dict[str, np.ndarray]):
        """
        Strip / swarm-style scatter of per-pair errors — one column per method.
        Useful for spotting which specific pairs are hard for all methods.
        """
        fig, ax = plt.subplots(figsize=(max(6, len(all_errors) * 2), 5))

        for i, (method, errors) in enumerate(all_errors.items()):
            # Jitter x to avoid overplotting
            x_jitter = np.random.uniform(-0.2, 0.2, size=len(errors))
            # Cap display at 100m for readability
            display   = np.clip(errors, 0, 100)
            capped    = errors > 100
            ax.scatter(i + x_jitter[~capped], display[~capped],
                       color=self._colour(method), alpha=0.6, s=30, zorder=3)
            if capped.any():
                ax.scatter(i + x_jitter[capped], display[capped],
                           color=self._colour(method), alpha=0.6, s=30,
                           marker="^", zorder=3)  # triangles = capped/failed

        ax.set_xticks(range(len(all_errors)))
        ax.set_xticklabels(list(all_errors.keys()), rotation=15, ha="right")
        ax.set_ylabel("Error [m]  (▲ = capped at 100m)", fontsize=10)
        ax.set_title("Per-pair Error Distribution by Method", fontsize=12)
        ax.axhline(10, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="10m threshold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(self.output_dir / "scatter_per_pair.png"), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] scatter_per_pair.png")

    def _plot_error_histogram(self, df: pd.DataFrame, method: str):
        """
        Per-method histogram of errors (excluding failures).
        """
        valid = df[~df["failed"]]["dist_m"].values
        if len(valid) == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid, bins=20, color=self._colour(method),
                alpha=0.75, edgecolor="white")
        ax.axvline(np.median(valid), color="red", linestyle="--",
                   label=f"Median: {np.median(valid):.1f}m")
        ax.axvline(np.mean(valid), color="orange", linestyle="-.",
                   label=f"MAE: {np.mean(valid):.1f}m")
        ax.set_xlabel("Localization Error [m]", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{method} — Error Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = f"hist_{method.replace(' ', '_')}.png"
        fig.savefig(str(self.output_dir / fname), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. STANDARD DATASET ADAPTERS
# ---------------------------------------------------------------------------

class VIGORAdapter:
    """
    Adapter for the VIGOR dataset (cross-view urban geo-localization).
    VIGOR provides drone images + panoramic street-view references.

    Loads VIGOR's CSV format and wraps EvaluationHarness.record() so
    the same harness scores your custom data AND VIGOR simultaneously.

    VIGOR CSV format (per split):
        panorama_id, latitude, longitude, split
    """

    def __init__(self, harness: EvaluationHarness, vigor_root: str):
        self.harness    = harness
        self.vigor_root = Path(vigor_root)

    def record_prediction(self, method: str, query_id: str,
                          pred_lat: float, pred_lon: float,
                          gt_lat: float,   gt_lon: float,
                          inference_ms: float = None):
        """
        Record a VIGOR prediction directly using lat/lon.
        Bypasses meta.json lookup (VIGOR has its own GT format).
        """
        dx_m, dy_m, dist_m = latlon_to_metres(pred_lat, pred_lon, gt_lat, gt_lon)

        # Inject directly into harness internals (bypasses _load_meta)
        if method not in self.harness._results:
            self.harness._results[method] = []
            self.harness._method_order.append(method)

        r = PairResult()
        r.query_name          = query_id
        r.method              = method
        r.pred_lat            = pred_lat
        r.pred_lon            = pred_lon
        r.gt_lat              = gt_lat
        r.gt_lon              = gt_lon
        r.dx_m                = dx_m
        r.dy_m                = dy_m
        r.dist_m              = dist_m
        r.failed              = False
        r.inference_ms        = inference_ms
        r.gt_dist_from_ref_m  = 0.0   # not applicable for VIGOR
        self.harness._results[method].append(r)


class AnyVisLocAdapter:
    """
    Adapter for UAVD / AnyVisLoc benchmark.
    AnyVisLoc reports @1m, @5m, @10m on a fixed test split.

    Load the benchmark query CSV, run your matcher, call record_prediction(),
    then harness.finalise() to get scores in the same format.
    """

    @staticmethod
    def load_query_list(csv_path: str) -> pd.DataFrame:
        """
        Load AnyVisLoc query CSV.
        Expected columns: image_name, gt_lat, gt_lon [, gt_alt]
        """
        return pd.read_csv(csv_path)

    def __init__(self, harness: EvaluationHarness):
        self.harness = harness

    def record_prediction(self, method: str, query_name: str,
                          pred_lat: float, pred_lon: float,
                          gt_lat: float,   gt_lon: float,
                          inference_ms: float = None):
        dx_m, dy_m, dist_m = latlon_to_metres(pred_lat, pred_lon, gt_lat, gt_lon)

        if method not in self.harness._results:
            self.harness._results[method] = []
            self.harness._method_order.append(method)

        r = PairResult()
        r.query_name         = query_name
        r.method             = method
        r.pred_lat           = pred_lat
        r.pred_lon           = pred_lon
        r.gt_lat             = gt_lat
        r.gt_lon             = gt_lon
        r.dx_m               = dx_m
        r.dy_m               = dy_m
        r.dist_m             = dist_m
        r.failed             = False
        r.inference_ms       = inference_ms
        r.gt_dist_from_ref_m = 0.0
        self.harness._results[method].append(r)


# ---------------------------------------------------------------------------
# 5. TIMING CONTEXT MANAGER  (convenience for matcher scripts)
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple context manager for timing inference.

    Usage:
        with Timer() as t:
            result = matcher.run(query, reference)
        harness.record(..., inference_ms=t.ms)
    """
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._start) * 1000

    @property
    def seconds(self):
        return self.ms / 1000


# ---------------------------------------------------------------------------
# 6. CLI — score pre-saved prediction CSVs
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 2 — Score pre-saved matcher predictions"
    )
    parser.add_argument("--pairs_dir",   required=True,
                        help="pairs/ output directory from Step 1")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save results and plots")
    parser.add_argument("--predictions", nargs="+", required=True,
                        metavar="METHOD:PATH",
                        help="One or more 'MethodName:path/to/preds.csv' entries")
    args = parser.parse_args()

    harness = EvaluationHarness(
        pairs_dir  = args.pairs_dir,
        output_dir = args.output_dir,
    )

    for entry in args.predictions:
        if ":" not in entry:
            print(f"[WARN] Skipping malformed entry '{entry}' "
                  f"(expected 'MethodName:path/to/csv')")
            continue
        method, csv_path = entry.split(":", 1)
        print(f"[LOAD] {method} ← {csv_path}")
        harness.load_predictions_csv(method, csv_path)
        harness.finalise(method)

    harness.compare()


if __name__ == "__main__":
    main()
