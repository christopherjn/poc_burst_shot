# poc-burst-shot

A proof-of-concept pipeline that groups a folder of photos into series: bursts and posed shots taken a few frames apart. It is built for cards from professional camera bodies.

## How it works

A series is a run of frames in capture order, so the pipeline segments the camera's timeline instead of clustering the whole folder:

1. **Metadata**: capture time (`DateTimeOriginal` + `SubSecTimeOriginal`) and camera body (make, model, serial when written) are read from EXIF.
2. **Embedding**: each image is embedded with DINOv2 ViT-S/14. The camera's embedded preview JPEG is decoded instead of the full frame when present, with EXIF orientation applied.
3. **Sequence linking**: frames from each camera are sorted by capture time. Each frame is compared with the previous 5 frames. Two frames join the same series when they are at most 60s apart and their cosine similarity is at least 0.86. Linked frames are unioned into series.

Looking back 5 frames lets a series survive an interloper (a detail shot, a second shooter's frame on the same timeline) and lets the photographer alternate between framings. See [`eval/README.md`](eval/README.md) for the benchmark, parameter sweep and stress tests behind these choices.

Groups come back in capture order. Each carries a `grouping_method` label: `series` or `singleton`.

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
poc-burst-shot /path/to/card
```

Without an argument you are prompted for the folder. Supported formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tiff`, `.tif`, `.webp`, `.bmp`.

Example output:

```
Found 12 image(s) in /photos/trip

Found 3 group(s):

  Group 1 [series] — 5 image(s)
    /photos/trip/IMG_0042.jpg
    ...
  Group 2 [series] — 4 image(s)
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

| Package | Purpose | Licence |
|---|---|---|
| `torch` + `torchvision` | Inference | BSD-3-Clause |
| `timm` | DINOv2 ViT-S/14 model and weights | Apache-2.0 (code and weights) |
| `Pillow` | Image loading, EXIF | MIT-CMU (HPND) |
| `numpy` | Numerical utilities | BSD-3-Clause |

The DINOv2 weights are downloaded from the Hugging Face hub (`timm/vit_small_patch14_dinov2.lvd142m`) on first run.
