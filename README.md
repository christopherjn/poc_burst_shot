# poc-burst-shot

A proof-of-concept pipeline that groups a folder of photos into series: bursts and posed shots taken a few frames apart. It combines self-supervised visual embeddings with EXIF capture times.

## How it works

The pipeline runs in three stages:

1. **Visual clustering**: images are embedded with DINOv2 ViT-S/14 and clustered with HDBSCAN.
2. **Geometric verification** (optional, off by default): each visual cluster is checked with ORB feature matching + RANSAC homography to split false merges.
3. **Time sub-split**: a group is split wherever consecutive frames are more than 60s apart.

Images are decoded from the camera's embedded preview JPEG when present, with EXIF orientation applied. See [`eval/README.md`](eval/README.md) for the benchmark behind these choices.

Each output group carries a `grouping_method` label: `visual` or `singleton`.

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

  Group 1 [visual] — 5 image(s)
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
| `timm` + `torch` | DINOv2 ViT-S/14 visual embeddings (Apache-2.0 weights) |
| `hdbscan` | Density-based clustering |
| `opencv-python-headless` | ORB feature matching and RANSAC verification |
| `Pillow` | Image loading and conversion |
| `numpy` / `scikit-learn` | Numerical utilities |
