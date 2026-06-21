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

## Web UI

A Streamlit frontend (in `ui/`) shows the resulting clusters as image grids. It
talks to a small FastAPI backend that wraps the grouping pipeline.

Start both processes and wire them together with one command:

```bash
make burst
```

This installs the UI dependency group, launches the backend
(`http://127.0.0.1:8000`), waits for it to become healthy, then opens the
Streamlit frontend (`http://127.0.0.1:8501`). Enter an absolute folder path,
click **Group images**, and a live **progress bar (0–100%)** tracks the pipeline
stages while it works; the clusters then render as thumbnail grids. Quitting the
frontend (Ctrl-C) shuts the backend down too.

A **Grouping aggressiveness** slider (labelled *Create fewer groups* ◀ … ▶
*Create more groups*) controls how eagerly the visual stage splits: drag it lower
if separate bursts are being split apart too much (this raises the cluster merge
distance, producing fewer, larger groups). It maps onto the
`cluster_selection_epsilon` knob described under
[Tuning the grouping](#tuning-the-grouping).

An **Embedding model** dropdown selects the backbone used by the visual stage.
EfficientNet-B0 is the default; ResNet-50, MobileNetV3-Large, ConvNeXt-Tiny,
DenseNet-121, and ViT-Base/16 are also available (the backend lists them via
`GET /models`). Switch to **Compare models** mode to pick two or more backbones
and group the same folder with each, rendering the results side by side so you
can see how the choice of model changes the clustering.

Under the hood, `POST /group` (with an optional `model_name`) starts the job in a
background thread and returns a job id; the frontend polls `GET /progress/{job_id}`
for the 0–1 progress fraction and the final result. Compare mode simply runs one
job per selected model.

Other targets:

```bash
make backend    # run only the FastAPI service
make frontend   # run only the Streamlit app (expects a backend already up)
make install    # install core + UI dependencies
```

Ports are configurable: `make burst BACKEND_PORT=9000 FRONTEND_PORT=9001`.

## Tuning the grouping

If groups come out too fragmented (clusters that should be one burst getting
split apart), raise `CLUSTER_SELECTION_EPSILON` in
`src/poc_burst_shot/grouping/clusterer.py`. Because embeddings are
L2-normalised, the distance lives in `[0, 2]`; a larger epsilon merges more
nearby micro-clusters into a single group (less aggressive splitting), while
`0.0` restores HDBSCAN's default finer-grained behaviour.

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
