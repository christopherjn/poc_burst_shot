"""Score the grouping pipeline against a hand-labelled folder.

    uv run python eval/evaluate.py --images DIR --labels eval/labels/sony_a7iv_portrait_100.json \
        --models efficientnet_b0 vit_small_patch14_dinov2.lvd142m

Embeddings are cached under eval/.cache so re-runs only pay for grouping.
"""
import argparse
import copy
import hashlib
import inspect
import logging
import time
from pathlib import Path

import numpy as np

from metrics import load_labels, score
from poc_burst_shot.grouping.embedder import DEFAULT_MODEL, Embedder
from poc_burst_shot.grouping.pipeline import build_records, embed_records, group_records
from poc_burst_shot.utils import image_loader

CACHE_DIR = Path(__file__).parent / ".cache"

# classic "tiny images" baseline: 32x32 grayscale pixels, no learned model
TINY_IMAGE = "tinyimage"


def records_with_embeddings(paths: list[Path], model_name: str):
    records = build_records(paths)
    # the loader source is part of the key so preprocessing changes invalidate the cache
    key_src = "|".join(str(p) for p in paths) + model_name + inspect.getsource(image_loader)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{model_name.replace('/', '_')}-{key}.npz"

    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        by_stem = dict(zip(data["stems"], data["embeddings"]))
        for r in records:
            r.embedding = by_stem.get(r.path.stem)
        return records, 0.0

    start = time.perf_counter()
    if model_name == TINY_IMAGE:
        for r in records:
            img = image_loader.load_image(r.path)
            if img is not None:
                vec = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32).ravel()
                vec -= vec.mean()
                r.embedding = vec / np.linalg.norm(vec)
    else:
        embed_records(records, Embedder(model_name))
    elapsed = time.perf_counter() - start
    done = [r for r in records if r.embedding is not None]
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, stems=np.array([r.path.stem for r in done]), embeddings=np.stack([r.embedding for r in done]))
    return records, elapsed


def show_groups(pred: list[list[str]], labels) -> None:
    for group in sorted(pred, key=min):
        names = sorted({labels.group_of[s] for s in group})
        flag = "  <-- mixed" if len(names) > 1 else ""
        print(f"      {min(group)}..{max(group)} ({len(group)}): {', '.join(names)}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--show", action="store_true", help="print each predicted group with its true labels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    labels = load_labels(args.labels)
    paths = sorted(p for p in args.images.iterdir() if p.stem in labels.group_of)
    print(f"{len(paths)} labelled images\n")

    variants = {
        "defaults": {},
        "window 1 (neighbours only)": dict(window=1),
    }

    for model_name in args.models:
        records, embed_s = records_with_embeddings(paths, model_name)
        print(f"== {model_name}" + (f"  (embedding {embed_s:.1f}s)" if embed_s else "  (cached embeddings)"))
        for name, kwargs in variants.items():
            start = time.perf_counter()
            groups = group_records(copy.deepcopy(records), **kwargs)
            elapsed = time.perf_counter() - start
            pred = [[m.path.stem for m in g.members] for g in groups]
            print(f"  {name:28s} {score(pred, labels).row()}  [{elapsed:.1f}s]")
            if args.show:
                show_groups(pred, labels)
        print()


if __name__ == "__main__":
    main()
