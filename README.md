# Row Navigation — Crop Row Detection

Two scripts for detecting and segmenting intercrop rows in field videos.

---

## Scripts

| Script | Method | Use case |
|---|---|---|
| `crop_row_hough.py` | ExG + Hough transform | Fast, no GPU needed |
| `segment_image.py` | SegFormer-B0 + ExG | Deep learning segmentation |

---

## Installation

Python 3.10+ required.

```bash
python -m venv rnav
source rnav/bin/activate        # Windows: rnav\Scripts\activate
pip install -r requirements.txt
```

`torch` and `transformers` are only needed if you intend to run `segment_image.py`.

---

## crop_row_hough.py

Detects crop rows using:
1. **Excess Green Index** (ExG = 2G − R − B) + Otsu threshold → vegetation mask
2. **Strip centre points** — divides the frame into horizontal strips, finds the midpoint of each plant region per strip
3. **Hough transform** on the sparse centre points → row centrelines
4. **Three-stage filter** — angle, theta-similarity, rho-similarity

Vegetation is overlaid in **neon green**, soil in **brown**.

### Basic usage

```bash
python crop_row_hough.py field_video.mp4
```

Output is saved as `field_video_rows_hough.mp4` alongside the input.

### All options

```
positional:
  video                   Input video path

frame control:
  --start-frame N         Skip the first N frames (default: 0)
  --skip-frames N         Re-run detection every N+1 frames (default: 2)
  --no-display            Disable the live preview window
  --debug                 Show all pipeline stages; advance frame-by-frame
                          with Space/Enter, quit with Q or Esc

vegetation:
  --n-strips N            Number of horizontal strips (default: 10)
  --sum-thresh N          Min vegetation pixels per column to count as a
                          plant region (default: 2)
  --diff-noise N          Min plant region width in pixels (default: 8)

Hough:
  --hough-rho F           Distance resolution in pixels (default: 2)
  --hough-angle F         Angle resolution in radians (default: 0.0698)
  --hough-thresh N        Min Hough votes to accept a line (default: 6)

line filter:
  --angle-thresh F        Max deviation from expected row direction in
                          radians (default: 0.5236 = 30°)
  --theta-sim F           Merge lines closer than this angle (default: 0.1047)
  --rho-sim F             Merge lines closer than this distance (default: 20)
  --expected-theta F      Expected row angle in radians. π/2 = vertical rows
                          for a forward-facing camera (default: 1.5708)

visualisation:
  --alpha F               Overlay opacity 0.0–1.0 (default: 0.6)
  --output PATH           Output video path
```

### Examples

```bash
# Skip the first 5 seconds of a 30fps video
python crop_row_hough.py field.mp4 --start-frame 150

# Step through frames one at a time to inspect each pipeline stage
python crop_row_hough.py field.mp4 --debug

# Adjust for rows that are not perfectly vertical in the frame
python crop_row_hough.py field.mp4 --expected-theta 1.2 --angle-thresh 0.7

# Reduce overlay opacity
python crop_row_hough.py field.mp4 --alpha 0.4

# Run without a display (e.g. on a server)
python crop_row_hough.py field.mp4 --no-display
```

### Debug mode

Pass `--debug` to open a 2×3 mosaic window showing every pipeline stage for each frame:

```
[ Original      ]  [ ExG heatmap        ]  [ Vegetation mask ]
[ Strip dots    ]  [ Raw + filtered lines]  [ Final result    ]
```

- **Space / Enter** — advance to next frame
- **Q / Esc** — quit

---

## segment_image.py

Segments a field video using **SegFormer-B0** (HuggingFace) combined with ExG vegetation detection. Identifies individual crop rows by finding peaks in the column-wise vegetation density profile.

### Basic usage

```bash
python segment_image.py field_video.mp4
```

Output is saved as `field_video_rows.mp4`.

### Options

```
  video                   Input video path
  --model NAME            HuggingFace model name
                          (default: nvidia/segformer-b0-finetuned-ade-512-512)
  --exg-threshold F       ExG threshold for vegetation (default: 0.4)
  --alpha F               Overlay opacity (default: 0.5)
  --skip-frames N         Re-run every N+1 frames (default: 2)
  --start-frame N         Skip first N frames (default: 0)
  --device cuda|cpu       Compute device (auto-detected)
  --no-display            Disable live preview
  --output PATH           Output video path
```

### Example

```bash
# Use GPU, process every frame
python segment_image.py field.mp4 --device cuda --skip-frames 0

# Adjust vegetation sensitivity
python segment_image.py field.mp4 --exg-threshold 0.35
```

> **Note:** SegFormer-B0 is pretrained on ADE20K (150 general classes) and has
> no specific "crop row" class. The ExG index carries the row detection. For
> best results on a specific crop/soil type, fine-tuning on a small labelled
> dataset is recommended.
