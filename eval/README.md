# Grouping benchmark

Three scripts, all run from the repo root:

```bash
# score the pipeline defaults (and a neighbours-only variant) per embedding model
uv run python eval/evaluate.py --images /path/to/photos \
    --labels eval/labels/sony_a7iv_portrait_100.json \
    --models vit_small_patch14_dinov2.lvd142m efficientnet_b0 \
    --show                 # list every predicted group with its true labels

# sweep threshold / window / max gap per model
uv run python eval/sweep.py --images /path/to/photos --labels eval/labels/sony_a7iv_portrait_100.json

# stress tests against the tuned HDBSCAN pipeline from research/grouping-approach
uv run --with hdbscan python eval/stress.py --images /path/to/photos --labels eval/labels/sony_a7iv_portrait_100.json
```

`--models` takes any `timm` model name, or `tinyimage` (32x32 grayscale pixels, no learned
model) as a floor. Embeddings are cached in `eval/.cache/`, keyed on the file list, the model
and the source of `utils/image_loader.py`.

## Labels

A labels file (see `labels/sony_a7iv_portrait_100.json`) holds:

- `groups`: atomic series, i.e. images that must end up together, keyed by file stem;
- `optional_links`: pairs of atomic groups where merging is a judgement call (same pose shot
  wide and tight, a colour and a B&W take of the same frame). Pairs across an optional link
  count neither for nor against a prediction.

Metrics (`metrics.py`):

- **pair P/R/F1** over image pairs;
- **B-cubed P/R/F1**, averaged per image, so large series do not dominate;
- **merges**: pairs wrongly put together; **splits**: pairs wrongly separated.

To benchmark a new card, list its files by series in a new JSON file. Shoots with repeated
backdrops (wedding stage, mandap) are the most valuable ones to add.

## Current labelled set

`sony_a7iv_portrait_100.json`: 100 JPEGs, Sony ILCE-7M4, 33MP, one pre-wedding portrait
session. There are 26 atomic series (sizes 1-17) with 19 optional links. The hard part is that
the same person is on the same few backdrops throughout, so series differ in pose and framing
only. Gaps between frames are 0.17s (continuous drive) to 10s within a series.

## Results: HDBSCAN pipeline (branch `research/grouping-approach`)

Embedding models on the original pipeline (HDBSCAN `min_cluster_size=2`, no ORB), with the
original squashed full-resolution loader:

| Embedding | Licence (weights) | Pair F1 | + median time split |
|---|---|---|---|
| `tinyimage` (no model) | n/a | 0.912 | 0.868 |
| `efficientnet_b0.ra_in1k` (original) | Apache-2.0; trained on ImageNet-1k | 0.911 | 0.873 |
| `vit_small_patch14_dinov2.lvd142m` | Apache-2.0 | **0.988** | 0.938 |
| `vit_small_patch14_reg4_dinov2.lvd142m` | Apache-2.0 | 0.911 | 0.873 |
| `vit_base_patch14_dinov2.lvd142m` | Apache-2.0 | 0.911 | 0.873 |
| `vit_base_patch16_clip_224.openai` | MIT | 0.718 | 0.771 |
| `vit_base_patch16_siglip_224.v2_webli` | Apache-2.0 | 0.738 | 0.790 |

Findings:

1. **EfficientNet-B0 adds nothing over raw 32x32 pixels.** ImageNet class features describe
   *what* is in the picture, not *which shot* it is.
2. **CLIP/SigLIP are worse still.** Two frames of "a woman in a pink dress by a wall" match
   whatever the pose. This agrees with the DINOv2 paper (Oquab et al., 2023, Table 9). On
   instance-level retrieval, Oxford-Hard mAP is 43.2 for ViT-S/14 against 19.7 for OpenCLIP
   ViT-G/14.
3. **DINOv2 ViT-S/14 is the best model and also the smallest DINOv2.** The larger ViT-B and the
   register variants landed on a worse HDBSCAN partition. Tuning HDBSCAN (`min_samples`,
   `leaf`/`eom`, `cluster_selection_epsilon`) got EfficientNet to 0.950 at the cost of 33 wrong
   merges. For DINOv2-S the defaults were already the best setting.
4. **ORB + RANSAC verification changed no group** and cost ~0.8s per image, because it
   re-decodes every member at full resolution. It is now off by default.
5. **The median-relative time split hurt every model.** Continuous-drive bursts (0.17s) pull
   the median down, so ordinary 1-6s gaps inside one pose get split. With DINOv2-S it cut
   pair F1 from 0.988 to 0.938. A fixed 10-120s ceiling scores the same as no split (0.986)
   and still stops the same backdrop shot an hour apart from merging. Replaced with a 60s ceiling.

Loader changes (no accuracy change, 9x faster embedding):

- EXIF orientation is applied. Portrait frames were being embedded sideways.
- The camera's embedded ~1600px MPF preview is decoded instead of the full 33MP frame
  (8.5ms vs 300ms per image), with JPEG draft-mode downscaling.
- DINOv2-S embedding of 100 photos on a laptop CPU: 117s → 13s.

**Tuned pipeline: DINOv2-S + HDBSCAN defaults + 60s time ceiling: pair F1 0.986, B-cubed F1
0.973.** The only wrong merge left puts a bow-tie detail close-up with the two bow-tie framings
around it. Every other mixed group spans an optional link.

## Results: sequence grouping (this branch)

### Why a different approach

Photo-collection research has segmented photos **in capture order** for over twenty years.
Event clustering (Loui & Savakis 2003, IEEE TMM; Platt et al. 2003, PhotoTOC; Cooper et al.
2005, ACM TOMM) and "photo-worthy moment" segmentation (Gozali et al. 2012, HMM over time +
EXIF + visual features) all compare *adjacent* photos in time and content, rather than
clustering a collection globally. Recent open-source culling work reaches the same design,
with capture time as the constraint and visual similarity as the link decision. It also
reports that duplicates at shoots are mostly pose sets 2-30s apart, not sub-second bursts.
That matches this card.

HDBSCAN, by contrast, clusters the whole folder at once. It ignores the strongest signal a
camera card offers (order), its result for one series depends on the density of everything
else in the folder, and it has parameters with no photographic meaning.

### Sweep (window 5, max gap 60s; pair F1 by cosine threshold)

| Embedding | Best pair F1 | Thresholds within 0.02 of best |
|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` | **1.000** (0.84-0.88) | 0.80-0.94 |
| `vit_base_patch14_dinov2.lvd142m` | 1.000 (0.88) | 0.84-0.92 |
| `efficientnet_b0.ra_in1k` | 1.000 (0.80-0.92) | 0.76-0.92 |
| `tinyimage` | 0.992 | 0.72-0.82 |
| `vit_base_patch16_clip_224.openai` | 0.980 | 0.90 only |
| `vit_base_patch16_siglip_224.v2_webli` | 0.986 | 0.96 only |

Once capture order is used, the embedding matters much less: even raw pixels reach 0.99.
DINOv2-S is kept as the default for three reasons:

1. It has the widest safe threshold band.
2. It is clearly the best model under global clustering, where order cannot help.
3. Its weights are released under Apache-2.0 by the party that trained them, on Meta's own
   LVD-142M. EfficientNet's weights were trained on ImageNet-1k, whose terms of access limit the
   dataset to non-commercial research. Whether that reaches trained weights is legally unsettled.

The default threshold is 0.86, the middle of the perfect band. When unsure, prefer a higher
threshold: a false split shows a photographer two stacks, but a false merge hides a distinct
shot inside a stack.

### Stress tests (`stress.py`, DINOv2-S, pair F1)

| Scenario | Sequence (defaults) | window 1 | window 30 | HDBSCAN (tuned) |
|---|---|---|---|---|
| Benchmark as shot | **1.000** | 0.994 | 0.855 | 0.986 |
| Two shooters interleaved, bodies distinguishable | **1.000** | 0.994 | 0.855 | 0.986 |
| Two shooters interleaved, same camera id | **0.994** | 0.940 | 0.855 | 0.986 |
| No EXIF timestamps (file-name order) | **1.000** | 0.994 | 0.855 | 0.986 |
| Grouping time, 5,000 images | **0.23s** | 0.20s | 0.43s | 25.9s |

- **Same camera id** matters because Sony bodies write no `BodySerialNumber`. Two A7 IVs on one
  card folder share an id, so their frames interleave on one timeline. An earlier version
  linked any two frames under 0.4s apart without a visual check (a "burst rule"). Here that
  rule merged the two shooters' series (pair F1 0.850, 109 wrong merges), so it was removed.
- **Window** trades bridging against chaining. With 1 frame, any interloper splits a series.
  With 30 frames, similar-looking series chain together. 3-10 frames all score ≥ 0.994.
- **Locality**: adding session B changed no grouping in session A, for either method.
- **Runtime**: embedding dominates either way. DINOv2-S takes ~0.13s/image on a laptop CPU
  (about 11 minutes for a 5,000-photo wedding, including preview decoding).

### Limits of this evidence

This is one 100-photo portrait session. It has no wedding stage or mandap shots, where
different families pose in front of the same backdrop within seconds of each other. The
threshold will most likely need checking there. Label one such card with the format above and
rerun `sweep.py` before trusting the 0.86 default in production.
