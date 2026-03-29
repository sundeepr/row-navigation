#!/usr/bin/env python3
"""
Crop row detection for field video.

Pipeline (derived from petern3/crop_row_detection and PRBonn/visual-crop-row-navigation):

  1. Excess Green Index (ExG = 2G - R - B)  → Otsu threshold  → binary vegetation mask
  2. Divide frame into horizontal strips; find plant-column centres per strip
  3. Accumulate centre points onto a sparse canvas
  4. Probabilistic Hough Transform on that canvas
  5. Three-stage line filter: angle ± 30°, merge near-parallel, merge near-coincident
  6. Draw surviving row lines over the original frame

The strip-based approach (step 2-3) concentrates evidence along actual row axes and
suppresses clutter, which makes the Hough vote much cleaner than running it on a
full vegetation mask.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parameters  (all tunable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_N_STRIPS       = 10       # horizontal strips to divide the frame into
DEFAULT_SUM_THRESH     = 2        # min column-sum to count as "plant present"
DEFAULT_DIFF_NOISE     = 8        # min pixel gap between separate plant regions
DEFAULT_HOUGH_RHO      = 2        # Hough accumulator distance resolution (px)
DEFAULT_HOUGH_ANGLE    = np.pi * 4 / 180   # Hough angle resolution (4°)
DEFAULT_HOUGH_THRESH   = 6        # min Hough votes to accept a line
DEFAULT_ANGLE_THRESH   = np.pi * 30 / 180  # ± 30° around expected row direction
DEFAULT_THETA_SIM      = np.pi * 6 / 180   # merge lines closer than 6° in angle
DEFAULT_RHO_SIM        = 20       # merge lines closer than 20 px in rho
DEFAULT_ALPHA          = 0.6      # overlay opacity for coloured row bands
DEFAULT_SKIP_FRAMES    = 2        # re-run detection every N+1 frames

ROW_COLOURS = [          # BGR — one per detected row
    (0,   200,  50),
    (0,   165, 255),
    (220,   0, 220),
    (0,   220, 220),
    (50,   50, 255),
    (0,   255, 180),
    (255, 200,   0),
    (100,   0, 255),
    (0,   180, 180),
    (255,  50, 180),
]


# ---------------------------------------------------------------------------
# Stage 1 – vegetation mask
# ---------------------------------------------------------------------------

def vegetation_mask(frame_bgr: np.ndarray, otsu_scale: float = 1.0):
    """
    ExG = 2G - R - B  (petern3 formula), then Otsu threshold.
    Returns (mask, exg_u8) where mask is binary uint8 (0/255) and
    exg_u8 is the normalised ExG image before thresholding.
    """
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b

    # Normalise to 0–255 for Otsu
    lo, hi = exg.min(), exg.max()
    if hi == lo:
        empty = np.zeros(exg.shape, dtype=np.uint8)
        return empty, empty
    exg_u8 = ((exg - lo) / (hi - lo) * 255).astype(np.uint8)

    otsu_val, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: tighten threshold by scaling Otsu value
    if otsu_scale != 1.0:
        _, mask = cv2.threshold(exg_u8, int(otsu_val * otsu_scale), 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask, exg_u8


# ---------------------------------------------------------------------------
# Stage 2 – strip-based centre points
# ---------------------------------------------------------------------------

def strip_centre_points(
    mask: np.ndarray,
    n_strips: int,
    sum_thresh: int,
    diff_noise: int,
):
    """
    Divide the mask into n_strips horizontal bands.
    Within each band, sum columns; find contiguous plant regions and mark
    their centre column on a sparse (single-channel) canvas.

    Returns (canvas, canvas_vis) where canvas is the raw dot image used for
    Hough and canvas_vis is an annotated BGR version for debug display.
    """
    h, w = mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    strip_h = h // n_strips

    for s in range(n_strips):
        y0 = s * strip_h
        y1 = y0 + strip_h if s < n_strips - 1 else h
        y_mid = (y0 + y1) // 2

        strip = mask[y0:y1, :]
        col_sum = strip.sum(axis=0) // 255      # count of vegetation pixels per column

        # Find plant regions: contiguous runs where col_sum > sum_thresh
        in_region = False
        region_start = 0
        for x in range(w):
            if col_sum[x] > sum_thresh:
                if not in_region:
                    in_region = True
                    region_start = x
            else:
                if in_region:
                    in_region = False
                    region_end = x - 1
                    width_px = region_end - region_start + 1
                    if width_px >= diff_noise:
                        cx = (region_start + region_end) // 2
                        cv2.circle(canvas, (cx, y_mid), 2, 255, -1)
        # close trailing region
        if in_region:
            region_end = w - 1
            width_px = region_end - region_start + 1
            if width_px >= diff_noise:
                cx = (region_start + region_end) // 2
                cv2.circle(canvas, (cx, y_mid), 2, 255, -1)

    # Debug visualisation: BGR canvas with strip boundaries and enlarged dots
    canvas_vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for s in range(1, n_strips):
        y = s * strip_h
        cv2.line(canvas_vis, (0, y), (w, y), (60, 60, 60), 1)
    # Make the dots more visible
    dot_positions = np.argwhere(canvas > 0)
    for (dy, dx) in dot_positions:
        cv2.circle(canvas_vis, (dx, dy), 5, (0, 255, 0), -1)

    return canvas, canvas_vis


# ---------------------------------------------------------------------------
# Stage 3 – Hough lines on the canvas
# ---------------------------------------------------------------------------

def hough_lines(
    canvas: np.ndarray,
    rho: float,
    angle: float,
    thresh: int,
) -> Optional[np.ndarray]:
    """
    Standard (not probabilistic) Hough on the sparse centre-point canvas.
    Returns array of (rho, theta) pairs or None.
    """
    lines = cv2.HoughLines(canvas, rho, angle, thresh)
    return lines  # shape (N, 1, 2) or None


# ---------------------------------------------------------------------------
# Stage 4 – three-stage line filter  (petern3 method)
# ---------------------------------------------------------------------------

def filter_lines(
    lines: np.ndarray,
    frame_h: int,
    frame_w: int,
    angle_thresh: float,
    theta_sim: float,
    rho_sim: float,
    expected_theta: float,   # radians — expected row direction (default π/2 = vertical)
) -> list[tuple[float, float]]:
    """
    Stage A – angle filter: keep lines within ±angle_thresh of expected_theta.
    Stage B – theta similarity: drop duplicates with similar angle (keep highest-vote first
              since HoughLines returns them sorted by votes).
    Stage C – rho similarity: drop duplicates at similar distance.

    Returns list of (rho, theta) pairs.
    """
    if lines is None:
        return []

    rho_theta = [(float(l[0][0]), float(l[0][1])) for l in lines]

    # Stage A — angle filter
    filtered = []
    for rho_v, theta in rho_theta:
        diff = abs(theta - expected_theta)
        diff = min(diff, np.pi - diff)   # handle wrap-around
        if diff <= angle_thresh:
            filtered.append((rho_v, theta))

    # Stage B — theta similarity (greedy; list is already vote-sorted)
    deduped_theta = []
    for rho_v, theta in filtered:
        too_similar = any(
            min(abs(theta - t), np.pi - abs(theta - t)) < theta_sim
            for _, t in deduped_theta
        )
        if not too_similar:
            deduped_theta.append((rho_v, theta))

    # Stage C — rho similarity
    deduped_rho = []
    for rho_v, theta in deduped_theta:
        too_close = any(abs(rho_v - r) < rho_sim for r, _ in deduped_rho)
        if not too_close:
            deduped_rho.append((rho_v, theta))

    return deduped_rho


# ---------------------------------------------------------------------------
# Stage 5 – draw rows
# ---------------------------------------------------------------------------

def draw_row_lines(
    frame_bgr: np.ndarray,
    lines: list[tuple[float, float]],
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    For each detected (rho, theta) line:
      - Fill a band around the line in a distinct colour (using the vegetation mask).
      - Draw the line itself on top.
    """
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    def x_at_y(rt, y):
        rho_v, theta = rt
        if abs(np.cos(theta)) < 1e-6:
            return float(rho_v)
        return (rho_v - y * np.sin(theta)) / (np.cos(theta) + 1e-9)

    # Sort lines by x at the BOTTOM of the frame — widest separation,
    # most reliable ordering, used only for the drawn line colours.
    lines_sorted = sorted(lines, key=lambda rt: x_at_y(rt, h - 1))

    NEON_GREEN_BGR = np.array((0,   255,  57), dtype=np.uint8)
    BROWN_BGR      = np.array((19,  69, 139), dtype=np.uint8)

    # Paint directly from the vegetation mask — no corridor geometry needed.
    # This avoids all line-intersection and out-of-bounds x issues.
    overlay[mask == 0] = BROWN_BGR       # soil → brown
    overlay[mask >  0] = NEON_GREEN_BGR  # crop → neon green

    result = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)

    return result


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, frame_idx: int, total: int, n_rows: int) -> np.ndarray:
    out = frame.copy()
    lines_text = [
        f"Frame {frame_idx}/{total}  ({100*frame_idx/max(total,1):.1f}%)",
        f"Rows detected: {n_rows}",
        "Q / Esc  to quit",
    ]
    for i, txt in enumerate(lines_text):
        y = 28 + i * 24
        cv2.putText(out, txt, (11, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0),     2, cv2.LINE_AA)
        cv2.putText(out, txt, (10, y  ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Debug mosaic
# ---------------------------------------------------------------------------

def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Burn a stage label into the top-left of a BGR image."""
    out = img_bgr.copy()
    cv2.putText(out, text, (6,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(out, text, (5,  19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_debug_mosaic(
    frame_bgr:   np.ndarray,
    exg_u8:      np.ndarray,
    mask:        np.ndarray,
    canvas_vis:  np.ndarray,
    raw_lines_img: np.ndarray,
    result:      np.ndarray,
    tile_w:      int = 640,
) -> np.ndarray:
    """
    Arrange six stage images in a 2×3 grid:
      [Original]  [ExG heatmap]    [Veg mask]
      [Strip dots][Raw Hough lines][Final result]
    Each tile is resized to tile_w wide (aspect-ratio preserved).
    """
    h, w = frame_bgr.shape[:2]
    tile_h = int(tile_w * h / w)

    def prep(img, label_text):
        # Ensure BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (tile_w, tile_h))
        return _label(resized, label_text)

    exg_colour = cv2.applyColorMap(exg_u8, cv2.COLORMAP_SUMMER)

    tiles = [
        prep(frame_bgr,    "1a. Original"),
        prep(exg_colour,   "1b. ExG heatmap"),
        prep(mask,         "1c. Veg mask (Otsu)"),
        prep(canvas_vis,   "2.  Strip centre dots"),
        prep(raw_lines_img,"3+4. Raw & filtered lines"),
        prep(result,       "5.  Final result"),
    ]

    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    return np.vstack([row1, row2])


def _draw_raw_lines(frame_bgr: np.ndarray, raw, filtered) -> np.ndarray:
    """Grey = all Hough lines before filter. Coloured = survivors."""
    out = frame_bgr.copy()
    if raw is not None:
        for l in raw:
            rho_v, theta = float(l[0][0]), float(l[0][1])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x0 = int(cos_t * rho_v + 1000 * (-sin_t))
            y0 = int(sin_t * rho_v + 1000 * ( cos_t))
            x1 = int(cos_t * rho_v - 1000 * (-sin_t))
            y1 = int(sin_t * rho_v - 1000 * ( cos_t))
            cv2.line(out, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    for idx, (rho_v, theta) in enumerate(filtered):
        colour = ROW_COLOURS[idx % len(ROW_COLOURS)]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0 = int(cos_t * rho_v + 1000 * (-sin_t))
        y0 = int(sin_t * rho_v + 1000 * ( cos_t))
        x1 = int(cos_t * rho_v - 1000 * (-sin_t))
        y1 = int(sin_t * rho_v - 1000 * ( cos_t))
        cv2.line(out, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    n_strips:    int,
    sum_thresh:  int,
    diff_noise:  int,
    hough_rho:   float,
    hough_angle: float,
    hough_thresh:int,
    angle_thresh:float,
    theta_sim:   float,
    rho_sim:     float,
    alpha:       float,
    expected_theta: float,
) -> tuple[np.ndarray, int, dict]:
    """
    Returns (result, n_rows, stages) where stages is a dict of intermediate
    images keyed by stage name (populated for debug display).
    """
    h, w = frame_bgr.shape[:2]

    mask, exg_u8      = vegetation_mask(frame_bgr)
    canvas, canvas_vis = strip_centre_points(mask, n_strips, sum_thresh, diff_noise)
    raw                = hough_lines(canvas, hough_rho, hough_angle, hough_thresh)
    lines              = filter_lines(raw, h, w, angle_thresh, theta_sim, rho_sim, expected_theta)
    result             = draw_row_lines(frame_bgr, lines, mask, alpha)

    stages = {
        "exg_u8":      exg_u8,
        "mask":        mask,
        "canvas_vis":  canvas_vis,
        "raw_lines":   _draw_raw_lines(frame_bgr, raw, lines),
    }
    return result, len(lines), stages


# ---------------------------------------------------------------------------
# Main video loop
# ---------------------------------------------------------------------------

def process_video(
    video_path:    str,
    output_path:   str,
    start_frame:   int,
    skip_frames:   int,
    show:          bool,
    debug:         bool,
    n_strips:      int,
    sum_thresh:    int,
    diff_noise:    int,
    hough_rho:     float,
    hough_angle:   float,
    hough_thresh:  int,
    angle_thresh:  float,
    theta_sim:     float,
    rho_sim:       float,
    alpha:         float,
    expected_theta:float,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Seeking to frame {start_frame} ({start_frame/fps:.1f}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Video  : {width}x{height} @ {fps:.1f} fps  ({total} frames)")
    print(f"Output : {output_path}")

    window       = "Crop Row Detection"
    debug_window = "Pipeline Stages  (Space/Enter = next frame  |  Q/Esc = quit)"

    if show and not debug:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(width, 1280), min(height, 720))
    if debug:
        cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
        # 3 tiles wide × 2 rows; each tile ~640 px wide
        cv2.resizeWindow(debug_window, 3 * 640, 2 * int(640 * height / width))
        print("Debug mode: Space or Enter advances one frame. Q or Esc quits.")

    frame_idx    = start_frame
    last_result  = None
    last_nrows   = 0
    last_stages: dict = {}

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % (skip_frames + 1) == 0:
                last_result, last_nrows, last_stages = process_frame(
                    frame_bgr,
                    n_strips, sum_thresh, diff_noise,
                    hough_rho, hough_angle, hough_thresh,
                    angle_thresh, theta_sim, rho_sim,
                    alpha, expected_theta,
                )

            display = last_result if last_result is not None else frame_bgr
            display = draw_hud(display, frame_idx, total, last_nrows)

            writer.write(display)

            if debug and last_stages:
                mosaic = build_debug_mosaic(
                    frame_bgr,
                    last_stages["exg_u8"],
                    last_stages["mask"],
                    last_stages["canvas_vis"],
                    last_stages["raw_lines"],
                    display,
                )
                cv2.imshow(debug_window, mosaic)
                # Wait indefinitely — Space or Enter advances, Q/Esc quits
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord(" "), 13):   # Space or Enter
                        break
                    if key in (ord("q"), ord("Q"), 27):
                        print("\nAborted by user.")
                        raise StopIteration
            elif show:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nAborted by user.")
                    break

            frame_idx += 1
            print(f"  {frame_idx}/{total}  ({100*frame_idx/max(total,1):.1f}%)",
                  end="\r", flush=True)

    except StopIteration:
        pass
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crop row detection using ExG + strip centres + Hough lines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path (default: <input>_rows_hough.mp4)")

    g = parser.add_argument_group("frame control")
    g.add_argument("--start-frame",  type=int,   default=0,                  help="Skip first N frames")
    g.add_argument("--skip-frames",  type=int,   default=DEFAULT_SKIP_FRAMES, help="Re-run every N+1 frames")
    g.add_argument("--no-display",   action="store_true",                    help="Disable live preview")
    g.add_argument("--debug",        action="store_true",                    help="Show all pipeline stages; advance frame-by-frame with Space/Enter")

    g = parser.add_argument_group("vegetation")
    g.add_argument("--n-strips",    type=int,   default=DEFAULT_N_STRIPS,    help="Horizontal strips")
    g.add_argument("--sum-thresh",  type=int,   default=DEFAULT_SUM_THRESH,  help="Min column veg-pixel count")
    g.add_argument("--diff-noise",  type=int,   default=DEFAULT_DIFF_NOISE,  help="Min plant-region width (px)")

    g = parser.add_argument_group("Hough")
    g.add_argument("--hough-rho",    type=float, default=DEFAULT_HOUGH_RHO,   help="Hough rho resolution (px)")
    g.add_argument("--hough-angle",  type=float, default=DEFAULT_HOUGH_ANGLE, help="Hough angle resolution (rad)")
    g.add_argument("--hough-thresh", type=int,   default=DEFAULT_HOUGH_THRESH,help="Hough vote threshold")

    g = parser.add_argument_group("line filter")
    g.add_argument("--angle-thresh", type=float, default=DEFAULT_ANGLE_THRESH,
                   help="Max deviation from expected row angle (rad)")
    g.add_argument("--theta-sim",    type=float, default=DEFAULT_THETA_SIM,
                   help="Merge lines with angle difference < this (rad)")
    g.add_argument("--rho-sim",      type=float, default=DEFAULT_RHO_SIM,
                   help="Merge lines with rho difference < this (px)")
    g.add_argument("--expected-theta", type=float, default=np.pi / 2,
                   help="Expected row angle in radians (π/2 = vertical rows, default for forward-facing camera)")

    g = parser.add_argument_group("visualisation")
    g.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Row colour overlay opacity")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        p = Path(args.video)
        args.output = str(p.parent / f"{p.stem}_rows_hough.mp4")

    process_video(
        video_path     = args.video,
        output_path    = args.output,
        start_frame    = args.start_frame,
        skip_frames    = args.skip_frames,
        show           = not args.no_display,
        debug          = args.debug,
        n_strips       = args.n_strips,
        sum_thresh     = args.sum_thresh,
        diff_noise     = args.diff_noise,
        hough_rho      = args.hough_rho,
        hough_angle    = args.hough_angle,
        hough_thresh   = args.hough_thresh,
        angle_thresh   = args.angle_thresh,
        theta_sim      = args.theta_sim,
        rho_sim        = args.rho_sim,
        alpha          = args.alpha,
        expected_theta = args.expected_theta,
    )


if __name__ == "__main__":
    main()
