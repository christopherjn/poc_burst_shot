"""Sweep the sequence-grouping parameters per embedding model.

    uv run python eval/sweep.py --images DIR --labels eval/labels/sony_a7iv_portrait_100.json \
        --models vit_small_patch14_dinov2.lvd142m efficientnet_b0

For each model, prints pair F1 across similarity thresholds at the default window/gaps (to
show how wide the good range is), then the best setting found over the full grid.
"""
import argparse
import copy
import itertools
import logging
from pathlib import Path

import numpy as np

from evaluate import records_with_embeddings
from metrics import load_labels, score
from poc_burst_shot.grouping import pipeline

THRESHOLDS = np.round(np.arange(0.60, 0.981, 0.02), 2)
WINDOWS = [1, 3, 5, 10]
MAX_GAPS = [10.0, 30.0, 60.0, 120.0]


def _pred(records, **kwargs) -> list[list[str]]:
    groups = pipeline.group_records(copy.deepcopy(records), **kwargs)
    return [[m.path.stem for m in g.members] for g in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=[pipeline.DEFAULT_MODEL])
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    labels = load_labels(args.labels)
    paths = sorted(p for p in args.images.iterdir() if p.stem in labels.group_of)

    for model_name in args.models:
        records, _ = records_with_embeddings(paths, model_name)
        print(f"== {model_name}")

        curve = []
        for t in THRESHOLDS:
            s = score(_pred(records, sim_threshold=float(t)), labels)
            curve.append(s.pair_f1)
        print("  threshold " + " ".join(f"{t:5.2f}" for t in THRESHOLDS))
        print("  pair F1   " + " ".join(f"{f:5.3f}" for f in curve))
        good = [t for t, f in zip(THRESHOLDS, curve) if f >= max(curve) - 0.02]
        print(f"  within 0.02 of best ({max(curve):.3f}): {min(good):.2f}-{max(good):.2f}")

        best = None
        for t, w, g in itertools.product(THRESHOLDS, WINDOWS, MAX_GAPS):
            kw = dict(sim_threshold=float(t), window=w, max_gap_s=g)
            s = score(_pred(records, **kw), labels)
            if best is None or s.pair_f1 > best[0].pair_f1:
                best = (s, kw)
        print(f"  grid best {best[1]}\n    {best[0].row()}\n")


if __name__ == "__main__":
    main()
