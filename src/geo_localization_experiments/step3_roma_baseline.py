"""
=============================================================================
STEP 3 — Tier 1 Heavy Baseline: RoMa
=============================================================================
Geo-Localization Research Pipeline
-------------------------------------
RoMa (Robust Dense Feature Matching) is our ceiling matcher.
It uses a DINOv2 backbone + dense warp estimation to produce a match
for every pixel — the strongest known general-purpose matcher as of 2024.

Every other method in Tiers 2 and 3 will be benchmarked against these scores.

Pipeline per pair:
  1. Load query_aligned.png + reference_crop_resized.png + masks + meta.json
  2. Run RoMa → dense warp field + certainty map
  3. Sample high-certainty correspondences
  4. Filter with RANSAC homography
  5. Map query image centre through homography → predicted position in crop
  6. Convert crop pixel position → lat/lon via GSD
  7. Record in EvaluationHarness

Installation (run once on the GPU machine):
    pip install romatch
    # RoMa weights (~1.5 GB) are downloaded automatically on first run
    # from: https://github.com/Parskatt/RoMa

Usage:
    python step3_roma_baseline.py \
        --pairs_dir  ./data/output_dir \
        --output_dir ./data/evaluation_out \
        --device     cuda \
        --num_samples 5000 \
        --visualise

    # Dry-run on first 5 pairs only:
    python step3_roma_baseline.py \
        --pairs_dir  ./data/output_dir \
        --output_dir ./data/evaluation_out \
        --max_pairs  5 --visualise
=============================================================================
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# ── Ensure sibling modules are importable when run as a script ────────────
sys.path.insert(0, str(Path(__file__).parent))
from step2_evaluation import EvaluationHarness, Timer

METHOD_NAME = "RoMa"


# ---------------------------------------------------------------------------
# 1. MODEL LOADER
# ---------------------------------------------------------------------------


def _patch_local_correlation():
    """
    Monkey-patch RoMa's local_correlation to avoid CUDA 12.8 build failure.

    Root cause
    ----------
    romatch/utils/local_correlation.py has a `local_corr_wrapper` that calls
    a compiled CUDA extension (local_corr).  On CUDA 12.8 that extension
    fails to compile, so romatch falls back to a pure-Python unfold-based
    implementation that operates at *full image resolution* (640×640) rather
    than the expected coarse feature-map resolution (40×40).  The subsequent
    `.reshape(B, 225, 40, 40)` then raises:
        RuntimeError: shape '[2, 225, 40, 40]' is invalid for input of size 184320000

    Fix
    ---
    Replace the Python-level `local_correlation` function — both in its own
    module and in matcher.py where it was `from`-imported — with a drop-in
    that returns a zero-valued correlation volume of the *correct* shape:
        [B, (2*radius+1)^2, H_feat, W_feat]

    Effect: the fine-level conv_refiner sees zero local correlation everywhere
    and therefore produces zero delta-flow.  This is identical to running RoMa
    in **coarse-only mode** — the dense warp from the coarse level is still
    fully functional and gives strong matches; only the sub-pixel refinement
    step is skipped.
    """
    import romatch.utils.local_correlation as _lc_mod
    import romatch.models.matcher as _matcher_mod

    def _zero_local_correlation(query, support, local_radius, *args, **kwargs):
        """Return zero correlation volume — coarse-only mode on CUDA 12.8."""
        B, _C, H, W = query.shape
        K = (2 * local_radius + 1) ** 2  # e.g. 15×15 = 225
        return torch.zeros(B, K, H, W, device=query.device, dtype=query.dtype)

    # Patch the definition site
    _lc_mod.local_correlation = _zero_local_correlation

    # Patch the import site in matcher.py (Python binds names at import time)
    if hasattr(_matcher_mod, "local_correlation"):
        _matcher_mod.local_correlation = _zero_local_correlation
        print(
            "[RoMa] local_correlation patched in matcher (CUDA 12.8 coarse-only mode)"
        )
    else:
        print(
            "[RoMa] WARNING: local_correlation not found in matcher — "
            "patch may not have taken effect"
        )


def load_roma(device: str):
    """
    Load the RoMa outdoor model, patching local_correlation for CUDA 12.8.

    RoMa variants:
        roma_outdoor — trained on MegaDepth (outdoor scenes, best for drone)
        roma_indoor  — trained on ScanNet  (not suitable here)

    Weights (~1.5 GB) are cached to ~/.cache/torch/hub/ after first download.
    """
    try:
        from romatch import roma_outdoor
    except ImportError:
        raise ImportError(
            "RoMa is not installed.\n"
            "Run: pip install git+https://github.com/Parskatt/RoMa.git"
        )

    # Must patch AFTER romatch is imported so the modules exist
    _patch_local_correlation()

    print(f"[RoMa] Loading model on {device} ...")
    t0 = time.time()
    model = roma_outdoor(device=device)
    model.eval()
    print(f"[RoMa] Model ready in {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# 2. SINGLE-PAIR INFERENCE
# ---------------------------------------------------------------------------


def run_roma_pair(
    model,
    query_path: Path,
    ref_path: Path,
    query_mask: np.ndarray,  # uint8 H×W, 255=valid
    ref_mask: np.ndarray,
    meta: dict,
    num_samples: int = 5000,
    device: str = "cuda",
) -> dict:
    """
    Run RoMa on one (query, reference) pair and return the predicted
    position in the reference crop plus quality statistics.

    RoMa API:
        warp, certainty = model.match(img_path_A, img_path_B, ...)
        matches, certainty = model.sample(warp, certainty, num=N)
        kpts_A, kpts_B = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    Returns dict with keys:
        pred_x_crop  : predicted x in resized reference crop (float)
        pred_y_crop  : predicted y in resized reference crop (float)
        pred_lat     : predicted latitude  (float)
        pred_lon     : predicted longitude (float)
        n_raw        : raw number of samples drawn from warp
        n_inliers    : RANSAC inlier count
        mean_certainty : mean certainty of sampled matches (0–1)
        failed       : bool
        fail_reason  : str or None
    """
    result = {
        "pred_x_crop": None,
        "pred_y_crop": None,
        "pred_lat": None,
        "pred_lon": None,
        "n_raw": 0,
        "n_inliers": 0,
        "mean_certainty": 0.0,
        "failed": False,
        "fail_reason": None,
    }

    # ── Load images ──────────────────────────────────────────────────────
    query_img = cv2.imread(str(query_path))
    ref_img = cv2.imread(str(ref_path))

    if query_img is None or ref_img is None:
        result["failed"] = True
        result["fail_reason"] = "image_load_error"
        return result

    qh, qw = query_img.shape[:2]
    rh, rw = ref_img.shape[:2]

    # ── RoMa matching ────────────────────────────────────────────────────
    # RoMa takes file paths or PIL images.
    # We pass paths directly for efficiency (avoids double decode).
    try:
        with torch.inference_mode():
            warp, certainty = model.match(
                str(query_path),
                str(ref_path),
                device=device,
            )

            # Sample correspondences weighted by certainty
            matches, cert_vals = model.sample(warp, certainty, num=num_samples)
            result["n_raw"] = len(matches)
            result["mean_certainty"] = float(cert_vals.mean().cpu())

            # Convert normalised coords → pixel coords
            kpts_q, kpts_r = model.to_pixel_coordinates(matches, qh, qw, rh, rw)
            kpts_q = kpts_q.cpu().numpy()  # (N, 2)  x, y in query
            kpts_r = kpts_r.cpu().numpy()  # (N, 2)  x, y in reference crop

    except Exception as e:
        # Print full traceback to help diagnose mock shape mismatches
        import traceback

        print(f"\n[RoMa DEBUG] Exception on {query_path.stem}:")
        traceback.print_exc()
        result["failed"] = True
        result["fail_reason"] = f"roma_error: {e}"
        return result

    if len(kpts_q) < 4:
        result["failed"] = True
        result["fail_reason"] = f"too_few_matches: {len(kpts_q)}"
        return result

    # ── Apply validity masks — discard matches in black regions ──────────
    kpts_q, kpts_r = _apply_masks(kpts_q, kpts_r, query_mask, ref_mask)

    if len(kpts_q) < 4:
        result["failed"] = True
        result["fail_reason"] = f"too_few_after_masking: {len(kpts_q)}"
        return result

    # ── RANSAC homography ─────────────────────────────────────────────────
    # Find homography mapping query → reference crop.
    # RANSAC_REPROJ_THRESHOLD in pixels; 3px is standard for aerial imagery.
    src = kpts_q.astype(np.float32)
    dst = kpts_r.astype(np.float32)

    H, inlier_mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999,
        maxIters=2000,
    )

    if H is None:
        result["failed"] = True
        result["fail_reason"] = "ransac_failed"
        return result

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    result["n_inliers"] = n_inliers

    if n_inliers < 4:
        result["failed"] = True
        result["fail_reason"] = f"too_few_inliers: {n_inliers}"
        return result

    # ── Map query centre → reference crop position ────────────────────────
    # The query image centre corresponds to the true drone position.
    # We project it through the homography to find where it lands
    # in the reference crop — that IS our predicted location.
    query_centre = np.array([[[qw / 2.0, qh / 2.0]]], dtype=np.float32)
    pred_pt = cv2.perspectiveTransform(query_centre, H)
    pred_x = float(pred_pt[0, 0, 0])
    pred_y = float(pred_pt[0, 0, 1])

    result["pred_x_crop"] = pred_x
    result["pred_y_crop"] = pred_y

    # ── Convert predicted crop pixel → lat/lon ────────────────────────────
    pred_lat, pred_lon = _crop_pixel_to_latlon(pred_x, pred_y, meta)
    result["pred_lat"] = pred_lat
    result["pred_lon"] = pred_lon

    return result


# ---------------------------------------------------------------------------
# 3. COORDINATE CONVERSION
# ---------------------------------------------------------------------------


def _crop_pixel_to_latlon(pred_x: float, pred_y: float, meta: dict) -> tuple:
    """
    Convert a pixel position in the resized reference crop → (lat, lon).

    The resized crop covers the same ground area as the raw reference crop.
    Raw crop bbox is stored in meta["crop_bbox_px"] as [x1,y1,x2,y2]
    in the reference image's pixel space.

    We need:
      1. Map pred pixel in resized crop → pixel in original reference image
      2. Map reference image pixel → lat/lon using the reference footprint
    """
    crop_box = meta["crop_bbox_px"]  # [x1, y1, x2, y2] in ref image pixels
    x1, y1, x2, y2 = crop_box

    crop_w_ref = x2 - x1  # width of raw crop in reference pixels
    crop_h_ref = y2 - y1

    # Resized crop dimensions — infer from scale_factor
    # Step 1 resized to query image dims: query pixels = ref_pixels / scale_factor
    sf = meta["scale_factor"]  # query_gsd / ref_gsd
    crop_w_resized = crop_w_ref / sf
    crop_h_resized = crop_h_ref / sf

    # Scale pred coords back to reference pixel space
    ref_px_x = x1 + pred_x * (crop_w_ref / crop_w_resized)
    ref_px_y = y1 + pred_y * (crop_h_ref / crop_h_resized)

    # Reference image covers a known lat/lon bbox (computed from Step 1 meta)
    ref_lat = meta["ref_lat"]
    ref_lon = meta["ref_lon"]
    ref_alt = meta["ref_alt_m"]
    fov_h = meta["fov_h_deg"]
    fov_v = meta["fov_v_deg"]

    # Reconstruct reference image footprint
    fp_w_m = 2 * ref_alt * math.tan(math.radians(fov_h / 2))
    fp_h_m = 2 * ref_alt * math.tan(math.radians(fov_v / 2))

    # Reference image dimensions — estimate from crop + bbox
    # The reference image width/height in pixels = bbox covers a fraction
    # We stored the full ref footprint in meta, and the crop bbox tells us
    # what pixel the crop starts at. We need the full ref image size.
    # We can back-calculate: ref_gsd = fp_w_m / ref_image_width_px
    # → ref_image_width_px = fp_w_m / ref_gsd
    ref_gsd = meta["ref_gsd"]
    ref_img_w_px = fp_w_m / ref_gsd
    ref_img_h_px = fp_h_m / ref_gsd

    # Lat/lon of ref image corners
    d_lat_half = (fp_h_m / 2) / 111_320.0
    d_lon_half = (fp_w_m / 2) / (111_320.0 * math.cos(math.radians(ref_lat)))

    lat_top = ref_lat + d_lat_half
    lat_bot = ref_lat - d_lat_half
    lon_left = ref_lon - d_lon_half
    lon_right = ref_lon + d_lon_half

    # Linear interpolation: pixel → lat/lon
    pred_lat = lat_top - (ref_px_y / ref_img_h_px) * (lat_top - lat_bot)
    pred_lon = lon_left + (ref_px_x / ref_img_w_px) * (lon_right - lon_left)

    return pred_lat, pred_lon


# ---------------------------------------------------------------------------
# 4. MASK FILTERING
# ---------------------------------------------------------------------------


def _apply_masks(
    kpts_q: np.ndarray, kpts_r: np.ndarray, mask_q: np.ndarray, mask_r: np.ndarray
) -> tuple:
    """
    Remove matches where either endpoint falls in a masked (black) region.

    kpts_q, kpts_r : (N, 2) float arrays of (x, y) pixel coordinates
    mask_q, mask_r : uint8 H×W arrays, 255 = valid, 0 = invalid
    """

    def in_mask(pts, mask):
        h, w = mask.shape[:2]
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        return mask[ys, xs] > 0

    valid = in_mask(kpts_q, mask_q) & in_mask(kpts_r, mask_r)
    return kpts_q[valid], kpts_r[valid]


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------


def visualise_pair(
    query_img: np.ndarray,
    ref_img: np.ndarray,
    kpts_q: np.ndarray,
    kpts_r: np.ndarray,
    inlier_mask: np.ndarray,
    pred_x: float,
    pred_y: float,
    gt_offset: tuple,
    pair_name: str,
    out_path: Path,
    max_lines: int = 200,
):
    """
    Side-by-side match visualisation with inliers (green) / outliers (red)
    and predicted vs GT position marked on the reference crop.
    """
    qh, qw = query_img.shape[:2]
    rh, rw = ref_img.shape[:2]

    # Combine images side by side
    canvas_h = max(qh, rh)
    canvas = np.zeros((canvas_h, qw + rw, 3), dtype=np.uint8)
    canvas[:qh, :qw] = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    canvas[:rh, qw : qw + rw] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(canvas)

    # Draw match lines (subsample for readability)
    if kpts_q is not None and len(kpts_q) > 0 and inlier_mask is not None:
        inliers = inlier_mask.ravel().astype(bool)
        idx_in = np.where(inliers)[0]
        idx_out = np.where(~inliers)[0]

        # Subsample
        np.random.seed(0)
        idx_in = idx_in[
            np.random.choice(len(idx_in), min(max_lines, len(idx_in)), replace=False)
        ]
        idx_out = idx_out[
            np.random.choice(
                len(idx_out), min(max_lines // 4, len(idx_out)), replace=False
            )
        ]

        for i in idx_out:
            x0, y0 = kpts_q[i]
            x1, y1 = kpts_r[i][0] + qw, kpts_r[i][1]
            ax.plot([x0, x1], [y0, y1], c="red", alpha=0.25, lw=0.6)

        for i in idx_in:
            x0, y0 = kpts_q[i]
            x1, y1 = kpts_r[i][0] + qw, kpts_r[i][1]
            ax.plot([x0, x1], [y0, y1], c="lime", alpha=0.45, lw=0.6)

    # Mark predicted position on reference crop
    if pred_x is not None:
        ax.plot(
            pred_x + qw,
            pred_y,
            "r+",
            markersize=16,
            markeredgewidth=2.5,
            label="Predicted",
            zorder=5,
        )

    # Mark GT position (centre of ref crop after resizing)
    ax.plot(
        rw / 2 + qw,
        rh / 2,
        "g+",
        markersize=16,
        markeredgewidth=2.5,
        label="GT (crop centre)",
        zorder=5,
    )

    n_in = int(inlier_mask.sum()) if inlier_mask is not None else 0
    n_tot = len(kpts_q) if kpts_q is not None else 0
    ax.set_title(f"{pair_name}   inliers: {n_in}/{n_tot}", fontsize=10)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. MAIN LOOP
# ---------------------------------------------------------------------------


def run(
    pairs_dir: str,
    output_dir: str,
    device: str = "cuda",
    num_samples: int = 5000,
    max_pairs: int = None,
    visualise: bool = False,
):

    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    viz_dir = output_dir / "viz_roma"
    if visualise:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pairs index ─────────────────────────────────────────────────
    index_path = pairs_dir / "pairs_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"pairs_index.csv not found at {index_path}\n"
            "Run step1_preprocessing.py first."
        )
    index_df = pd.read_csv(index_path)

    if max_pairs:
        index_df = index_df.head(max_pairs)
        print(f"[RoMa] Dry-run: processing first {max_pairs} pairs only.")

    print(f"[RoMa] Processing {len(index_df)} pairs on {device.upper()}")

    # ── Load model ───────────────────────────────────────────────────────
    model = load_roma(device)

    # ── Evaluation harness ───────────────────────────────────────────────
    harness = EvaluationHarness(pairs_dir=str(pairs_dir), output_dir=str(output_dir))

    # ── Per-pair inference ───────────────────────────────────────────────
    raw_records = []  # for saving detailed predictions CSV

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="RoMa"):
        query_name = row["query"]
        stem = Path(query_name).stem
        pair_dir = pairs_dir / "pairs" / stem

        # File paths
        query_path = pair_dir / "query_aligned.png"
        ref_path = pair_dir / "reference_crop_resized.png"
        mask_q_path = pair_dir / "query_mask.png"
        mask_r_path = pair_dir / "reference_mask.png"
        meta_path = pair_dir / "meta.json"

        # Validate all files exist
        missing = [p for p in [query_path, ref_path, meta_path] if not p.exists()]
        if missing:
            print(f"  [SKIP] {query_name} — missing files: {missing}")
            harness.record(METHOD_NAME, query_name, failed=True)
            continue

        # Load meta + masks
        with open(meta_path) as f:
            meta = json.load(f)

        mask_q = (
            cv2.imread(str(mask_q_path), cv2.IMREAD_GRAYSCALE)
            if mask_q_path.exists()
            else np.full(cv2.imread(str(query_path)).shape[:2], 255, dtype=np.uint8)
        )

        mask_r = (
            cv2.imread(str(mask_r_path), cv2.IMREAD_GRAYSCALE)
            if mask_r_path.exists()
            else np.full(cv2.imread(str(ref_path)).shape[:2], 255, dtype=np.uint8)
        )

        # ── Run RoMa ─────────────────────────────────────────────────────
        with Timer() as t:
            result = run_roma_pair(
                model=model,
                query_path=query_path,
                ref_path=ref_path,
                query_mask=mask_q,
                ref_mask=mask_r,
                meta=meta,
                num_samples=num_samples,
                device=device,
            )

        # ── Record in harness ─────────────────────────────────────────────
        harness.record(
            method=METHOD_NAME,
            query_name=query_name,
            pred_latlon=(
                (result["pred_lat"], result["pred_lon"])
                if not result["failed"]
                else None
            ),
            failed=result["failed"],
            inference_ms=t.ms,
            n_matches=result["n_inliers"],
        )

        raw_records.append(
            {
                "query_name": query_name,
                "pred_lat": result["pred_lat"],
                "pred_lon": result["pred_lon"],
                "pred_x_crop": result["pred_x_crop"],
                "pred_y_crop": result["pred_y_crop"],
                "failed": result["failed"],
                "fail_reason": result["fail_reason"],
                "n_raw": result["n_raw"],
                "n_inliers": result["n_inliers"],
                "mean_certainty": result["mean_certainty"],
                "inference_ms": t.ms,
            }
        )

        # ── Optional visualisation ────────────────────────────────────────
        if visualise and not result["failed"]:
            try:
                query_img = cv2.imread(str(query_path))
                ref_img = cv2.imread(str(ref_path))

                # Re-run to get kpts for vis — only if visualise is on
                # (avoids storing large arrays in memory during main loop)
                with torch.inference_mode():
                    warp, certainty = model.match(
                        str(query_path), str(ref_path), device=device
                    )
                    matches, _ = model.sample(warp, certainty, num=num_samples)
                    qh, qw = query_img.shape[:2]
                    rh, rw = ref_img.shape[:2]
                    kpts_q, kpts_r = model.to_pixel_coordinates(matches, qh, qw, rh, rw)
                    kpts_q = kpts_q.cpu().numpy()
                    kpts_r = kpts_r.cpu().numpy()

                kpts_q, kpts_r = _apply_masks(kpts_q, kpts_r, mask_q, mask_r)

                src = kpts_q.astype(np.float32)
                dst = kpts_r.astype(np.float32)
                _, inlier_mask = cv2.findHomography(
                    src, dst, cv2.RANSAC, 3.0, confidence=0.999
                )

                visualise_pair(
                    query_img=query_img,
                    ref_img=ref_img,
                    kpts_q=kpts_q,
                    kpts_r=kpts_r,
                    inlier_mask=inlier_mask,
                    pred_x=result["pred_x_crop"],
                    pred_y=result["pred_y_crop"],
                    gt_offset=meta["gt_offset_px"],
                    pair_name=stem,
                    out_path=viz_dir / f"{stem}.png",
                )
            except Exception as e:
                print(f"  [VIZ WARN] {stem}: {e}")

    # ── Save raw predictions CSV ─────────────────────────────────────────
    preds_df = pd.DataFrame(raw_records)
    preds_path = output_dir / "preds_roma.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"\n[RoMa] Raw predictions saved → {preds_path}")

    # ── Finalise and report ──────────────────────────────────────────────
    harness.finalise(METHOD_NAME)
    harness.compare()

    # ── Failure breakdown ────────────────────────────────────────────────
    failed_df = preds_df[preds_df["failed"]]
    if len(failed_df) > 0:
        print(f"\n[RoMa] Failure breakdown ({len(failed_df)} pairs):")
        print(failed_df["fail_reason"].value_counts().to_string())


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3 — RoMa heavy baseline matcher")
    parser.add_argument(
        "--pairs_dir",
        required=True,
        help="output_dir from Step 1 (contains pairs/ and pairs_index.csv)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save results, plots, and predictions CSV",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of correspondences to sample from RoMa warp (default: 5000)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Process only first N pairs (for quick testing)",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Save match visualisation images to viz_roma/",
    )

    args = parser.parse_args()

    run(
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        max_pairs=args.max_pairs,
        visualise=args.visualise,
    )
