"""
=============================================================================
STEP 1 — Data Loader & Preprocessing Module
=============================================================================
Geo-Localization Research Pipeline
-------------------------------------
This module handles everything before matching begins:
  1. Parse CSV metadata
  2. Extract camera specs from EXIF (focal length, sensor size → FOV)
  3. North-align query images using Gimbal_Yaw + Flight_Yaw
  4. Compute each image's ground footprint (in meters and lat/lon)
  5. Crop the reference image to match each query's footprint area
  6. Build and save (query, reference_crop, ground_truth_offset) triplets
  7. Visualise pairs for sanity checking

Expected CSV columns:
  Image, Latitude, Longitude, Altitude, Gimball_Roll, Gimball_Yaw,
  Gimball_Pitch, Flight_Roll, Flight_Yaw, Flight_Pitch, Time_Stamp

Usage:
  python step1_preprocessing.py \
      --images_dir ./images \
      --csv_path   ./metadata.csv \
      --output_dir ./pairs \
      --reference_image 20260310_173657_991.png \
      --visualise
=============================================================================
"""

import os
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import piexif
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# 1. CAMERA SPECS — extracted from EXIF, with manual fallback
# ---------------------------------------------------------------------------

# Known DJI sensor sizes (mm) by model string found in EXIF Make/Model.
# Add more here as needed.
DJI_SENSOR_DB = {
    "FC220": (6.16, 4.62),  # Mavic Pro
    "FC350": (13.2, 8.8),  # Inspire 1 / X3
    "FC6310": (13.2, 8.8),  # Phantom 4 Pro
    "FC300": (6.16, 4.62),  # Phantom 3
    "FC7303": (17.3, 13.0),  # Zenmuse P1
    "ZH20T": (6.4, 4.8),  # Zenmuse H20T (wide)
    "ZH20": (6.4, 4.8),  # Zenmuse H20
}

# Fallback if EXIF is missing or sensor not in DB
# DJI general wide-camera typical values
DEFAULT_SENSOR_W_MM = 6.4
DEFAULT_SENSOR_H_MM = 4.8
DEFAULT_FOCAL_MM = 4.5  # typical DJI wide


def extract_camera_from_exif(image_path: str) -> dict:
    """
    Read EXIF tags from a JPEG/PNG and return camera intrinsics.

    Returns dict with keys:
        focal_mm       : focal length in mm
        sensor_w_mm    : sensor width in mm
        sensor_h_mm    : sensor height in mm
        fov_h_deg      : horizontal field of view in degrees
        fov_v_deg      : vertical field of view in degrees
        image_w_px     : image width in pixels
        image_h_px     : image height in pixels
        source         : 'exif' | 'db_lookup' | 'default'
    """
    img = Image.open(image_path)
    image_w_px, image_h_px = img.size

    focal_mm = None
    sensor_w_mm = None
    sensor_h_mm = None
    source = "default"

    # --- Try piexif ---
    try:
        exif_dict = piexif.load(image_path)
        exif_ifd = exif_dict.get("Exif", {})

        # Focal length: tag 0x920A
        if piexif.ExifIFD.FocalLength in exif_ifd:
            num, den = exif_ifd[piexif.ExifIFD.FocalLength]
            focal_mm = num / den if den != 0 else None

        # Focal length in 35mm equivalent: tag 0xA405
        focal_35mm = None
        if piexif.ExifIFD.FocalLengthIn35mmFilm in exif_ifd:
            focal_35mm = exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm]

        # Try to look up sensor from Make/Model
        make = (
            exif_dict["0th"]
            .get(piexif.ImageIFD.Make, b"")
            .decode("utf-8", errors="ignore")
            .strip()
        )
        model = (
            exif_dict["0th"]
            .get(piexif.ImageIFD.Model, b"")
            .decode("utf-8", errors="ignore")
            .strip()
        )

        for key, (sw, sh) in DJI_SENSOR_DB.items():
            if key.upper() in model.upper():
                sensor_w_mm = sw
                sensor_h_mm = sh
                source = "db_lookup"
                break

        # If we have focal_35mm and actual focal, derive sensor width
        # sensor_w = 36 * focal_mm / focal_35mm  (35mm film = 36mm wide)
        if focal_mm and focal_35mm and focal_35mm > 0 and sensor_w_mm is None:
            sensor_w_mm = 36.0 * focal_mm / focal_35mm
            # Preserve aspect ratio
            sensor_h_mm = sensor_w_mm * (image_h_px / image_w_px)
            source = "exif"

        if focal_mm:
            source = "exif"

    except Exception:
        pass  # silently fall through to defaults

    # --- Apply defaults if anything is missing ---
    if focal_mm is None:
        focal_mm = DEFAULT_FOCAL_MM
    if sensor_w_mm is None:
        sensor_w_mm = DEFAULT_SENSOR_W_MM
    if sensor_h_mm is None:
        sensor_h_mm = DEFAULT_SENSOR_H_MM

    # --- Compute FOV ---
    fov_h_deg = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_mm)))
    fov_v_deg = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_mm)))

    return {
        "focal_mm": focal_mm,
        "sensor_w_mm": sensor_w_mm,
        "sensor_h_mm": sensor_h_mm,
        "fov_h_deg": fov_h_deg,
        "fov_v_deg": fov_v_deg,
        "image_w_px": image_w_px,
        "image_h_px": image_h_px,
        "source": source,
    }


# ---------------------------------------------------------------------------
# 2. GROUND FOOTPRINT — how many metres/pixels does one image cover?
# ---------------------------------------------------------------------------


def compute_ground_footprint(altitude_m: float, cam: dict) -> dict:
    """
    For a nadir-pointing camera at given AGL altitude, compute:
        footprint_w_m : ground width covered (metres)
        footprint_h_m : ground height covered (metres)
        gsd_m_px      : ground sampling distance (metres per pixel)
    """
    footprint_w_m = 2 * altitude_m * math.tan(math.radians(cam["fov_h_deg"] / 2))
    footprint_h_m = 2 * altitude_m * math.tan(math.radians(cam["fov_v_deg"] / 2))
    gsd_m_px = footprint_w_m / cam["image_w_px"]

    return {
        "footprint_w_m": footprint_w_m,
        "footprint_h_m": footprint_h_m,
        "gsd_m_px": gsd_m_px,
    }


def meters_to_degrees(lat: float, dx_m: float, dy_m: float):
    """
    Convert a (dx_m, dy_m) offset from (lat, lon) centre into
    (delta_lat, delta_lon) in degrees.
    """
    # 1 degree latitude ≈ 111,320 m everywhere
    # 1 degree longitude ≈ 111,320 * cos(lat) m
    d_lat = dy_m / 111_320.0
    d_lon = dx_m / (111_320.0 * math.cos(math.radians(lat)))
    return d_lat, d_lon


def latlon_bbox(lat: float, lon: float, footprint_w_m: float, footprint_h_m: float):
    """
    Given centre (lat, lon) and footprint size, return
    (lat_top, lon_left, lat_bot, lon_right) bounding box.
    """
    d_lat, _ = meters_to_degrees(lat, 0, footprint_h_m / 2)
    _, d_lon = meters_to_degrees(lat, footprint_w_m / 2, 0)

    lat_top = lat + d_lat
    lat_bot = lat - d_lat
    lon_left = lon - d_lon
    lon_right = lon + d_lon

    return lat_top, lon_left, lat_bot, lon_right


# ---------------------------------------------------------------------------
# 3. NORTH ALIGNMENT — rotate image by total yaw so north is always "up"
# ---------------------------------------------------------------------------


def north_align_image(
    image: np.ndarray, gimbal_yaw: float, flight_yaw: float
) -> np.ndarray:
    """
    Rotate image so that north is aligned with the image top edge.

    DJI convention:
      - Gimbal_Yaw is relative to the drone body (0 = forward)
      - Flight_Yaw is absolute heading (0 = north, clockwise positive)
      - Total absolute camera yaw = Flight_Yaw + Gimbal_Yaw

    We rotate the image by -total_yaw to bring north upward.
    OpenCV rotates counter-clockwise for positive angles, so we negate.
    """
    total_yaw = flight_yaw + gimbal_yaw

    if abs(total_yaw) < 0.5:  # already aligned, skip warp
        return image

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # Rotation matrix — rotate by -total_yaw degrees
    M = cv2.getRotationMatrix2D((cx, cy), -total_yaw, 1.0)

    # Expand canvas so corners are not cropped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # BORDER_CONSTANT (black=0) is intentional — reflected content would
    # look like real image data and confuse the validity mask downstream.
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated


# ---------------------------------------------------------------------------
# 3b. CORNER ARTIFACT REMOVAL — inscribed crop + validity mask
# ---------------------------------------------------------------------------


def largest_inscribed_rect(w: int, h: int, angle_deg: float) -> tuple:
    """
    Compute the dimensions of the largest axis-aligned rectangle that fits
    entirely within a w×h image that has been rotated by angle_deg degrees.

    Uses the closed-form solution (Chalkidis 2016 / StackOverflow canonical):
      - "Half-constrained" case: two corners touch the longer side
      - "Fully-constrained" case: crop touches all four sides

    Returns: (rect_w, rect_h) as floats — caller should floor() before slicing.
    """
    angle_rad = math.radians(angle_deg % 180)
    if angle_rad > math.pi / 2:
        angle_rad = math.pi - angle_rad  # solutions are symmetric at 90°

    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)

    if sin_a < 1e-6:  # no rotation → full image
        return float(w), float(h)
    if cos_a < 1e-6:  # exactly 90° → swap dims
        return float(h), float(w)

    # Determine long/short side
    width_is_longer = w >= h
    side_long = w if width_is_longer else h
    side_short = h if width_is_longer else w

    # Half-constrained condition
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        rect_w = x / sin_a if width_is_longer else x / cos_a
        rect_h = x / cos_a if width_is_longer else x / sin_a
    else:
        # Fully-constrained
        cos_2a = cos_a**2 - sin_a**2
        rect_w = (w * cos_a - h * sin_a) / cos_2a
        rect_h = (h * cos_a - w * sin_a) / cos_2a

    return rect_w, rect_h


def crop_to_inscribed_rect(image: np.ndarray, angle_deg: float) -> tuple:
    """
    Crop a rotated image (with black corner artifacts) to its largest
    axis-aligned inscribed rectangle.  The crop is always centred.

    Returns:
        cropped   : np.ndarray — the clean cropped image
        crop_info : dict with keys x1, y1, x2, y2 (in input image coords)
                    and retain_ratio (fraction of pixels kept)
    """
    h, w = image.shape[:2]

    if abs(angle_deg % 180) < 0.5:  # no rotation needed
        return image.copy(), {"x1": 0, "y1": 0, "x2": w, "y2": h, "retain_ratio": 1.0}

    rect_w, rect_h = largest_inscribed_rect(w, h, angle_deg)

    rect_w = max(1, math.floor(rect_w))
    rect_h = max(1, math.floor(rect_h))

    # Centre the crop
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - rect_w // 2)
    y1 = max(0, cy - rect_h // 2)
    x2 = min(w, x1 + rect_w)
    y2 = min(h, y1 + rect_h)

    cropped = image[y1:y2, x1:x2]
    retain_ratio = (x2 - x1) * (y2 - y1) / (w * h)

    return cropped, {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "retain_ratio": retain_ratio,
    }


def compute_validity_mask(image: np.ndarray, black_threshold: int = 10) -> np.ndarray:
    """
    Generate a binary validity mask where 1 = valid pixel, 0 = black artifact.

    After inscribed-rect cropping the mask mainly catches thin residual borders
    and any occasional black pixels from interpolation rounding.

    Args:
        image           : BGR or grayscale uint8 image
        black_threshold : pixels with ALL channels below this are masked out

    Returns:
        mask : uint8 np.ndarray, same H×W as image, values 0 or 255
               (255 = valid, matching OpenCV / LoFTR convention)
    """
    if len(image.shape) == 3:
        # Pixel is invalid only if ALL channels are near-black
        min_channel = np.min(image, axis=2)
    else:
        min_channel = image

    mask = np.where(min_channel > black_threshold, np.uint8(255), np.uint8(0))

    # Clean up tiny isolated invalid pixels with morphological closing
    # (removes salt-and-pepper noise from bilinear interpolation at edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ---------------------------------------------------------------------------
# 4. REFERENCE CROP — cut the reference image to match a query's footprint
# ---------------------------------------------------------------------------


def latlon_to_pixel(
    lat: float,
    lon: float,
    ref_lat_top: float,
    ref_lon_left: float,
    ref_lat_bot: float,
    ref_lon_right: float,
    ref_w_px: int,
    ref_h_px: int,
):
    """
    Map a (lat, lon) coordinate into pixel (x, y) within the reference image.
    Assumes the reference image is an ortho-rectified nadir image with
    linear (equirectangular) mapping between corners.
    """
    x = (lon - ref_lon_left) / (ref_lon_right - ref_lon_left) * ref_w_px
    y = (ref_lat_top - lat) / (ref_lat_top - ref_lat_bot) * ref_h_px
    return int(round(x)), int(round(y))


def crop_reference_for_query(
    ref_image: np.ndarray,
    ref_meta: dict,
    ref_cam: dict,
    ref_fp: dict,
    query_meta: dict,
    query_fp: dict,
    padding_factor: float = 1.5,
) -> dict:
    """
    Cut a region from the reference image that corresponds to where
    the query image is located, with extra padding around it.

    padding_factor > 1 gives the matcher more context and handles
    small GPS errors in the prior position.

    Returns:
        crop          : np.ndarray — the cropped reference tile
        crop_bbox_px  : (x1, y1, x2, y2) in reference pixel coords
        gt_offset_px  : (dx, dy) — true query centre in crop pixel coords
        gt_offset_m   : (dx_m, dy_m) — in metres
        scale_factor  : ref_gsd / query_gsd  (how much to scale the crop)
    """
    ref_lat_top, ref_lon_left, ref_lat_bot, ref_lon_right = latlon_bbox(
        ref_meta["Latitude"],
        ref_meta["Longitude"],
        ref_fp["footprint_w_m"],
        ref_fp["footprint_h_m"],
    )

    ref_h_px, ref_w_px = ref_image.shape[:2]

    # Query centre in reference pixel space
    qcx, qcy = latlon_to_pixel(
        query_meta["Latitude"],
        query_meta["Longitude"],
        ref_lat_top,
        ref_lon_left,
        ref_lat_bot,
        ref_lon_right,
        ref_w_px,
        ref_h_px,
    )

    # How many reference pixels does the query footprint span?
    ref_gsd = ref_fp["gsd_m_px"]
    query_gsd = query_fp["gsd_m_px"]
    scale_factor = query_gsd / ref_gsd  # query pixel = N ref pixels

    crop_w_px = int(query_fp["footprint_w_m"] / ref_gsd * padding_factor)
    crop_h_px = int(query_fp["footprint_h_m"] / ref_gsd * padding_factor)

    x1 = max(0, qcx - crop_w_px // 2)
    y1 = max(0, qcy - crop_h_px // 2)
    x2 = min(ref_w_px, x1 + crop_w_px)
    y2 = min(ref_h_px, y1 + crop_h_px)

    crop = ref_image[y1:y2, x1:x2]
    gt_offset_px = (qcx - x1, qcy - y1)  # ground truth in crop
    gt_offset_m = (
        (query_meta["Longitude"] - ref_meta["Longitude"])
        * 111_320
        * math.cos(math.radians(ref_meta["Latitude"])),
        (query_meta["Latitude"] - ref_meta["Latitude"]) * 111_320,
    )

    return {
        "crop": crop,
        "crop_bbox_px": (x1, y1, x2, y2),
        "gt_offset_px": gt_offset_px,
        "gt_offset_m": gt_offset_m,
        "scale_factor": scale_factor,
        "ref_gsd": ref_gsd,
        "query_gsd": query_gsd,
    }


# ---------------------------------------------------------------------------
# 5. PAIR BUILDER — ties everything together
# ---------------------------------------------------------------------------


def build_pairs(
    images_dir: str,
    csv_path: str,
    output_dir: str,
    reference_image: str,
    padding_factor: float = 1.5,
    visualise: bool = False,
    manual_fov_h: float = None,
    manual_fov_v: float = None,
) -> pd.DataFrame:
    """
    Main entry point.  Reads the CSV, processes every non-reference image,
    and writes:
        output_dir/
            pairs/
                <query_name>/
                    query_aligned.png         (north-aligned query)
                    reference_crop.png        (matching reference crop)
                    meta.json                 (all geometric metadata)
            viz/
                <query_name>.png              (side-by-side sanity check)
            pairs_index.csv                   (summary of all pairs)
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    pairs_dir = output_dir / "pairs"
    viz_dir = output_dir / "viz"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    if visualise:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    # Normalise column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    print(f"[CSV] Loaded {len(df)} rows from {csv_path}")

    # --- Identify reference row ---
    ref_row = df[df["Image"] == reference_image]
    if ref_row.empty:
        raise ValueError(f"Reference image '{reference_image}' not found in CSV.")
    ref_meta = ref_row.iloc[0].to_dict()
    print(
        f"[REF] {reference_image}  alt={ref_meta['Altitude']:.1f}m  "
        f"lat={ref_meta['Latitude']:.6f}  lon={ref_meta['Longitude']:.6f}"
    )

    # --- Load & process reference image ---
    ref_path = images_dir / reference_image
    ref_image = cv2.imread(str(ref_path))
    if ref_image is None:
        raise FileNotFoundError(f"Cannot load reference image: {ref_path}")

    # North-align reference too
    ref_image_aligned = north_align_image(
        ref_image,
        gimbal_yaw=float(ref_meta["Gimball_Yaw"]),
        flight_yaw=float(ref_meta["Flight_Yaw"]),
    )

    # Camera specs for reference
    ref_cam = extract_camera_from_exif(str(ref_path))
    if manual_fov_h:
        ref_cam["fov_h_deg"] = manual_fov_h
    if manual_fov_v:
        ref_cam["fov_v_deg"] = manual_fov_v

    ref_fp = compute_ground_footprint(float(ref_meta["Altitude"]), ref_cam)

    print(
        f"[CAM] Source={ref_cam['source']}  "
        f"focal={ref_cam['focal_mm']:.1f}mm  "
        f"FOV={ref_cam['fov_h_deg']:.1f}°×{ref_cam['fov_v_deg']:.1f}°"
    )
    print(
        f"[REF FP] {ref_fp['footprint_w_m']:.1f}m × {ref_fp['footprint_h_m']:.1f}m  "
        f"GSD={ref_fp['gsd_m_px']:.3f}m/px"
    )

    # --- Process query images ---
    query_rows = df[df["Image"] != reference_image].copy()
    records = []

    for _, row in tqdm(
        query_rows.iterrows(), total=len(query_rows), desc="Building pairs"
    ):
        img_name = row["Image"]
        img_path = images_dir / img_name

        if not img_path.exists():
            print(f"  [SKIP] {img_name} — file not found")
            continue

        query_meta = row.to_dict()

        # Load and north-align query
        query_img = cv2.imread(str(img_path))
        if query_img is None:
            print(f"  [SKIP] {img_name} — cannot read image")
            continue

        total_yaw = float(query_meta["Flight_Yaw"]) + float(query_meta["Gimball_Yaw"])

        query_aligned = north_align_image(
            query_img,
            gimbal_yaw=float(query_meta["Gimball_Yaw"]),
            flight_yaw=float(query_meta["Flight_Yaw"]),
        )

        # ── Step A: inscribed-rectangle crop (removes black corners) ──────
        query_clean, q_crop_info = crop_to_inscribed_rect(query_aligned, total_yaw)

        # ── Step B: validity mask for residual border pixels ──────────────
        query_mask = compute_validity_mask(query_clean)

        # Camera + footprint for query (same camera, different altitude)
        query_cam = extract_camera_from_exif(str(img_path))
        if manual_fov_h:
            query_cam["fov_h_deg"] = manual_fov_h
        if manual_fov_v:
            query_cam["fov_v_deg"] = manual_fov_v
        query_fp = compute_ground_footprint(float(query_meta["Altitude"]), query_cam)

        # Crop reference to match query location + footprint
        result = crop_reference_for_query(
            ref_image=ref_image_aligned,
            ref_meta=ref_meta,
            ref_cam=ref_cam,
            ref_fp=ref_fp,
            query_meta=query_meta,
            query_fp=query_fp,
            padding_factor=padding_factor,
        )

        if result["crop"].size == 0:
            print(f"  [SKIP] {img_name} — query outside reference footprint")
            continue

        # --- Resize reference crop to match CLEAN query image size ---
        # Using clean query dims (post-crop) ensures matcher sees same size.
        qc_h, qc_w = query_clean.shape[:2]
        crop_resized = cv2.resize(
            result["crop"], (qc_w, qc_h), interpolation=cv2.INTER_LINEAR
        )

        # Reference crop also gets a validity mask (accounts for any edge
        # padding that occurred if query was near the border of the reference)
        ref_mask = compute_validity_mask(crop_resized)

        # ── Summary stats for logging ──────────────────────────────────────
        q_valid_ratio = float(np.sum(query_mask > 0)) / query_mask.size
        ref_valid_ratio = float(np.sum(ref_mask > 0)) / ref_mask.size

        # --- Save outputs ---
        pair_dir = pairs_dir / img_name.replace(".png", "").replace(".jpg", "")
        pair_dir.mkdir(exist_ok=True)

        # Full rotated (with black corners) — kept for reference/debugging
        cv2.imwrite(str(pair_dir / "query_aligned_raw.png"), query_aligned)
        # Clean cropped versions — USE THESE for all matchers
        cv2.imwrite(str(pair_dir / "query_aligned.png"), query_clean)
        cv2.imwrite(str(pair_dir / "query_mask.png"), query_mask)
        cv2.imwrite(str(pair_dir / "reference_crop.png"), result["crop"])
        cv2.imwrite(str(pair_dir / "reference_crop_resized.png"), crop_resized)
        cv2.imwrite(str(pair_dir / "reference_mask.png"), ref_mask)

        meta_out = {
            "query_image": img_name,
            "reference_image": reference_image,
            "query_lat": float(query_meta["Latitude"]),
            "query_lon": float(query_meta["Longitude"]),
            "query_alt_m": float(query_meta["Altitude"]),
            "ref_lat": float(ref_meta["Latitude"]),
            "ref_lon": float(ref_meta["Longitude"]),
            "ref_alt_m": float(ref_meta["Altitude"]),
            "gt_offset_m": list(result["gt_offset_m"]),
            "gt_offset_px": list(result["gt_offset_px"]),
            "crop_bbox_px": list(result["crop_bbox_px"]),
            "scale_factor": result["scale_factor"],
            "ref_gsd": result["ref_gsd"],
            "query_gsd": result["query_gsd"],
            "footprint_w_m": query_fp["footprint_w_m"],
            "footprint_h_m": query_fp["footprint_h_m"],
            "fov_h_deg": query_cam["fov_h_deg"],
            "fov_v_deg": query_cam["fov_v_deg"],
            "total_yaw_deg": total_yaw,
            "cam_source": query_cam["source"],
            # ── Crop / mask info ───────────────────────────────────────────
            "inscribed_crop_query": q_crop_info,
            "query_valid_px_ratio": round(q_valid_ratio, 4),
            "ref_valid_px_ratio": round(ref_valid_ratio, 4),
            # Files guide for downstream steps
            "files": {
                "query": "query_aligned.png",
                "query_mask": "query_mask.png",
                "reference": "reference_crop_resized.png",
                "reference_mask": "reference_mask.png",
                "query_raw": "query_aligned_raw.png",
                "reference_full": "reference_crop.png",
            },
        }

        with open(pair_dir / "meta.json", "w") as f:
            json.dump(meta_out, f, indent=2)

        records.append(
            {
                "query": img_name,
                "pair_dir": str(pair_dir),
                "gt_dx_m": result["gt_offset_m"][0],
                "gt_dy_m": result["gt_offset_m"][1],
                "gt_dist_m": math.sqrt(
                    result["gt_offset_m"][0] ** 2 + result["gt_offset_m"][1] ** 2
                ),
                "scale_factor": result["scale_factor"],
                "query_alt_m": float(query_meta["Altitude"]),
                "total_yaw_deg": total_yaw,
                "query_valid_px_ratio": round(q_valid_ratio, 4),
                "ref_valid_px_ratio": round(ref_valid_ratio, 4),
                "inscribed_retain": round(q_crop_info["retain_ratio"], 4),
            }
        )

        # --- Visualise ---
        if visualise:
            _visualise_pair(
                query_clean,
                query_mask,
                crop_resized,
                ref_mask,
                result["gt_offset_px"],
                img_name,
                viz_dir,
            )

    # --- Save index ---
    index_df = pd.DataFrame(records)
    index_path = output_dir / "pairs_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"\n[DONE] Built {len(records)} pairs → {output_dir}")
    print(f"       Index saved to {index_path}")
    print(
        f"       GT distance range: "
        f"{index_df['gt_dist_m'].min():.1f}m – {index_df['gt_dist_m'].max():.1f}m"
    )
    print(
        f"       Inscribed crop retention: "
        f"{index_df['inscribed_retain'].mean()*100:.1f}% avg  "
        f"(min {index_df['inscribed_retain'].min()*100:.1f}%)"
    )
    print(
        f"       Query valid-pixel ratio: "
        f"{index_df['query_valid_px_ratio'].mean()*100:.1f}% avg after masking"
    )
    return index_df


# ---------------------------------------------------------------------------
# 6. VISUALISATION helper
# ---------------------------------------------------------------------------


def _visualise_pair(
    query_clean: np.ndarray,
    query_mask: np.ndarray,
    ref_resized: np.ndarray,
    ref_mask: np.ndarray,
    gt_px: tuple,
    name: str,
    viz_dir: Path,
):
    """
    Save a 4-panel figure:
      [clean query] | [query mask] | [reference crop] | [ref mask]

    The GT centre is marked on the reference crop.
    Valid-pixel ratios are shown in each mask panel title.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(name, fontsize=11)

    axes[0].imshow(cv2.cvtColor(query_clean, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query (north-aligned, cropped)")
    axes[0].axis("off")

    q_valid = np.sum(query_mask > 0) / query_mask.size * 100
    axes[1].imshow(query_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Query mask  ({q_valid:.1f}% valid)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Reference crop (resized to query)")
    axes[2].axis("off")
    # Mark ground truth centre — note gt_px is in original (pre-clean) crop
    # coords; we approximate as centre of the resized crop for visualisation
    rh, rw = ref_resized.shape[:2]
    axes[2].plot(
        rw / 2,
        rh / 2,
        "r+",
        markersize=18,
        markeredgewidth=2,
        label="GT centre (approx)",
    )

    r_valid = np.sum(ref_mask > 0) / ref_mask.size * 100
    axes[3].imshow(ref_mask, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"Reference mask  ({r_valid:.1f}% valid)")
    axes[3].axis("off")

    plt.tight_layout()
    out_name = name.replace(".png", "").replace(".jpg", "") + ".png"
    plt.savefig(str(viz_dir / out_name), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI ENTRY POINT
# ---------------------------------------------------------------------------

# python step1_preprocessing.py --images_dir E:\Github-Repos\geo-localization-experiments\src\data\images --csv_path E:\Github-Repos\geo-localization-experiments\src\data\telemetry.csv --reference_image 20260310_173956_994.png --output_dir E:\Github-Repos\geo-localization-experiments\src\data\output_dir --visualise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 — Build (query, reference_crop) pairs for geo-localization"
    )
    parser.add_argument(
        "--images_dir", required=True, help="Directory containing all drone images"
    )
    parser.add_argument("--csv_path", required=True, help="Path to metadata CSV file")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for pairs"
    )
    parser.add_argument(
        "--reference_image",
        required=True,
        help="Filename of the highest-altitude reference image",
    )
    parser.add_argument(
        "--padding_factor",
        type=float,
        default=1.5,
        help="How much extra context around query footprint (default: 1.5x)",
    )
    parser.add_argument(
        "--manual_fov_h",
        type=float,
        default=None,
        help="Override horizontal FOV in degrees (use if EXIF missing)",
    )
    parser.add_argument(
        "--manual_fov_v",
        type=float,
        default=None,
        help="Override vertical FOV in degrees (use if EXIF missing)",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Save side-by-side visualisation for each pair",
    )

    args = parser.parse_args()

    index = build_pairs(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        reference_image=args.reference_image,
        padding_factor=args.padding_factor,
        visualise=args.visualise,
        manual_fov_h=args.manual_fov_h,
        manual_fov_v=args.manual_fov_v,
    )

    print("\nPairs summary:")
    print(
        index[["query", "gt_dist_m", "scale_factor", "query_alt_m"]].to_string(
            index=False
        )
    )
