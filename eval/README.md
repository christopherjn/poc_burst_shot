# Grouping benchmark

`evaluate.py` scores the pipeline against a hand-labelled folder of photos.

```bash
uv run python eval/evaluate.py \
    --images /path/to/photos \
    --labels eval/labels/sony_a7iv_portrait_100.json \
    --models vit_small_patch14_dinov2.lvd142m efficientnet_b0 \
    --stages fast          # skip the slow ORB-verification variants
    --show                 # list every predicted group with its true labels
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

## Results (HDBSCAN pipeline, this branch)

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
