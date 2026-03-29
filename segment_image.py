#!/usr/bin/env python3
"""Intercrop row segmentation in field videos using SegFormer-B0 + Excess Green Index."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# Vegetation-like label keywords used to extract a veg mask from SegFormer output
_VEG_KEYWORDS = {
    "grass", "field", "plant", "flower", "tree", "earth",
    "vegetation", "ground", "crop", "farm", "land", "soil",
    "leaf", "foliage",
}

# Per-row overlay colors (index 0 = soil/background, indices 1+ = rows)
_ROW_COLORS = [
    (0,   0,   0),    # 0  background / soil
    (0,   200,  50),  # 1  green
    (255, 165,   0),  # 2  orange
    (0,   120, 255),  # 3  blue
    (220,   0, 220),  # 4  magenta
    (0,   220, 220),  # 5  cyan
    (255,  50,  50),  # 6  red
    (180, 255,   0),  # 7  lime
    (255, 200,   0),  # 8  yellow
    (100,   0, 255),  # 9  purple
    (0,   180, 180),  # 10 teal
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    model = model.to(device)
    return processor, model


# ---------------------------------------------------------------------------
# Vegetation detection
# ---------------------------------------------------------------------------

def _excess_green(frame_rgb: np.ndarray) -> np.ndarray:
    """Excess Green Index: ExG = 2G - R - B, normalised to [0, 1].

    Works well for green crop canopy against bare-soil backgrounds.
    """
    r = frame_rgb[:, :, 0].astype(np.float32) / 255.0
    g = frame_rgb[:, :, 1].astype(np.float32) / 255.0
    b = frame_rgb[:, :, 2].astype(np.float32) / 255.0
    exg = 2 * g - r - b
    lo, hi = exg.min(), exg.max()
    return (exg - lo) / (hi - lo + 1e-6)


def get_vegetation_mask(
    frame_rgb: np.ndarray,
    seg_map: np.ndarray,
    id2label: dict,
    exg_threshold: float = 0.4,
) -> np.ndarray:
    """Binary vegetation mask combining ExG and SegFormer class labels."""
    # --- ExG mask ---
    exg_mask = (_excess_green(frame_rgb) > exg_threshold).astype(np.uint8)

    # --- SegFormer mask: any class whose label name contains a veg keyword ---
    seg_veg = np.zeros_like(exg_mask)
    for label_id, label_name in id2label.items():
        if any(kw in label_name.lower() for kw in _VEG_KEYWORDS):
            seg_veg[seg_map == label_id] = 1

    combined = np.clip(exg_mask + seg_veg, 0, 1).astype(np.uint8)

    # Morphological cleanup: open to remove noise, close to fill gaps in canopy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    return combined


# ---------------------------------------------------------------------------
# Row detection
# ---------------------------------------------------------------------------

def detect_rows(veg_mask: np.ndarray) -> np.ndarray:
    """Assign each vegetation pixel to a row index (1-based); soil = 0.

    Strategy:
      1. Sum vegetation pixels per column to get a density profile.
      2. Find local maxima (row centres) in the smoothed profile.
      3. Assign each column to its nearest row centre via Voronoi partition.
    """
    h, w = veg_mask.shape

    col_profile = veg_mask.sum(axis=0).astype(np.float32)
    sigma = max(w // 40, 3)
    smoothed = ndimage.gaussian_filter1d(col_profile, sigma=sigma)

    min_dist = max(w // 20, 10)
    peaks, _ = find_peaks(smoothed, distance=min_dist, height=h * 0.05)

    row_label_map = np.zeros((h, w), dtype=np.int32)
    if len(peaks) == 0:
        return row_label_map

    # Voronoi: each column gets the label of its nearest peak
    col_indices = np.arange(w)
    distances = np.abs(col_indices[:, None] - peaks[None, :])  # (W, num_peaks)
    nearest = distances.argmin(axis=1) + 1  # 1-based row label

    # Broadcast to full image and zero out non-vegetation pixels
    row_label_map[:, :] = nearest[np.newaxis, :]
    row_label_map[veg_mask == 0] = 0
    return row_label_map


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def colorize_rows(row_label_map: np.ndarray) -> np.ndarray:
    """RGB image where each row index has a distinct colour."""
    h, w = row_label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    num_colors = len(_ROW_COLORS)
    unique_labels = np.unique(row_label_map)
    for label in unique_labels:
        color_img[row_label_map == label] = _ROW_COLORS[label % num_colors]
    return color_img


def overlay_segmentation(
    frame_bgr: np.ndarray, color_seg_rgb: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    seg_bgr = cv2.cvtColor(color_seg_rgb, cv2.COLOR_RGB2BGR)
    mask = color_seg_rgb.sum(axis=2) > 0  # only paint over vegetation pixels
    result = frame_bgr.copy()
    blend = cv2.addWeighted(frame_bgr, 1 - alpha, seg_bgr, alpha, 0)
    result[mask] = blend[mask]
    return result


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    processor,
    model,
    id2label: dict,
    exg_threshold: float,
    device: str,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

    veg_mask = get_vegetation_mask(frame_rgb, seg_map, id2label, exg_threshold)
    row_label_map = detect_rows(veg_mask)
    return row_label_map


# ---------------------------------------------------------------------------
# Main video loop
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, frame_idx: int, total: int, num_rows: int) -> np.ndarray:
    """Burn a small status overlay onto the frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    pct = 100 * frame_idx / max(total, 1)
    lines = [
        f"Frame {frame_idx}/{total}  ({pct:.1f}%)",
        f"Rows detected: {num_rows}",
        "Press Q to quit",
    ]
    x, y0 = 10, 28
    for i, text in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(out, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(out, text, (x,     y    ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_video(
    video_path: str,
    output_path: str,
    model_name: str,
    exg_threshold: float,
    alpha: float,
    skip_frames: int,
    device: str,
    show: bool = True,
    start_frame: int = 0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping to frame {start_frame} ({start_frame / fps:.1f}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Video : {width}x{height} @ {fps:.1f} fps  ({total} frames)")
    print(f"Model : {model_name}")
    print(f"Device: {device}")
    print("Loading model...")
    processor, model = load_model(model_name, device)
    id2label = model.config.id2label
    print("Model loaded. Processing frames...")

    frame_idx = start_frame
    last_row_map: Optional[np.ndarray] = None
    window_name = "Intercrop Row Segmentation"

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Run inference every (skip_frames + 1) frames; reuse the last result otherwise
            if frame_idx % (skip_frames + 1) == 0:
                last_row_map = process_frame(
                    frame_bgr, processor, model, id2label, exg_threshold, device
                )

            if last_row_map is not None:
                color_seg = colorize_rows(last_row_map)
                annotated = overlay_segmentation(frame_bgr, color_seg, alpha)
                num_rows = int(last_row_map.max())
            else:
                annotated = frame_bgr
                num_rows = 0

            annotated = draw_hud(annotated, frame_idx, total, num_rows)
            out.write(annotated)

            if show:
                cv2.imshow(window_name, annotated)
                # Keep display responsive; Q or Esc quits early
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nAborted by user.")
                    break

            frame_idx += 1
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} frames  ({pct:.1f}%)", end="\r", flush=True)

    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\nDone. Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment intercrop rows in a field video using SegFormer-B0 + ExG"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--output", "-o",
        help="Output video path (default: <input>_rows.mp4)",
    )
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace SegFormer model name",
    )
    parser.add_argument(
        "--exg-threshold",
        type=float,
        default=0.4,
        help="Excess Green Index threshold for vegetation (default: 0.4, lower = more permissive)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay opacity (0 = original only, 1 = segmentation only, default: 0.5)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Re-run segmentation every N+1 frames (default: 2, i.e. every 3rd frame)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: cuda or cpu (auto-detected by default)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip the first N frames before processing (default: 0)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window (useful for headless/server environments)",
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        p = Path(args.video)
        args.output = str(p.parent / f"{p.stem}_rows.mp4")

    process_video(
        args.video,
        args.output,
        args.model,
        args.exg_threshold,
        args.alpha,
        args.skip_frames,
        args.device,
        show=not args.no_display,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()
