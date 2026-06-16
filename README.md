# poc-burst-shot

A proof-of-concept pipeline that groups a folder of photos into burst-shot clusters. It combines EXIF metadata, visual embeddings, geometric verification, and timestamp analysis to accurately separate images taken in rapid succession from unrelated shots.

## How it works

The pipeline runs in four stages:

1. **EXIF fast path** — images that share a `BurstUUID` tag are immediately grouped together.
2. **Visual clustering** — remaining images are embedded with a pretrained EfficientNet-B0 and clustered with HDBSCAN.
3. **Geometric verification** — each visual cluster is checked with ORB feature matching + RANSAC homography to detect and split false merges.
4. **Time sub-split** — within every group, large timestamp gaps (> 3× the median intra-group gap) trigger a further split.

Each output group carries a `grouping_method` label: `exif_uuid`, `visual`, or `singleton`.

## Installation

Requires Python ≥ 3.11. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

```bash
poc-burst-shot
```

You will be prompted for a path to a folder of images. Supported formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tiff`, `.tif`, `.webp`, `.bmp`.

Example output:

```
Found 12 image(s) in /photos/trip

Found 3 group(s):

  Group 1 [exif_uuid] — 5 image(s)
    /photos/trip/IMG_0042.jpg
    ...
  Group 2 [visual] — 4 image(s)
    /photos/trip/IMG_0101.jpg
    ...
  Group 3 [singleton] — 1 image(s)
    /photos/trip/IMG_0200.jpg
```

## Using the pipeline programmatically

```python
from pathlib import Path
from poc_burst_shot.grouping.pipeline import run

paths = list(Path("/photos/trip").glob("*.jpg"))
groups = run(paths)

for group in groups:
    print(group.group_id, group.grouping_method, len(group.members))
```

## Dependencies

| Package | Purpose |
|---|---|
| `timm` + `torch` | EfficientNet-B0 visual embeddings |
| `hdbscan` | Density-based clustering |
| `opencv-python-headless` | ORB feature matching and RANSAC verification |
| `Pillow` | Image loading and conversion |
| `piexif` | EXIF metadata extraction |
| `numpy` / `scikit-learn` | Numerical utilities |
