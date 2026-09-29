"""Stress tests that the 100-photo benchmark alone does not exercise, run against both
the sequence grouper (this branch) and the tuned HDBSCAN pipeline (research/grouping-approach).

    uv run --with hdbscan python eval/stress.py --images DIR --labels eval/labels/sony_a7iv_portrait_100.json

1. Two shooters: the two sessions in the set (09:54-09:58 and 10:38-10:47) are shifted onto the
   same clock so their frames interleave, as when a main and a second shooter's cards are
   grouped together. Run with distinct camera ids and with one shared id (two Sony bodies
   write no BodySerialNumber).
2. No timestamps: EXIF stripped (e.g. files re-exported by messaging apps).
3. Locality: does grouping session A change when session B is added to the input?
4. Runtime at wedding scale: 5,000 synthetic embeddings.
"""
import argparse
import copy
import logging
import time
import uuid
from datetime import timedelta
from pathlib import Path

import hdbscan
import numpy as np

from evaluate import records_with_embeddings
from metrics import load_labels, score
from poc_burst_shot.grouping import pipeline
from poc_burst_shot.models.burst_group import BurstGroup

MODEL = "vit_small_patch14_dinov2.lvd142m"
SESSION_B_START = "DSC01412"


def hdbscan_pipeline(records, max_gap_s: float = 60.0) -> list[BurstGroup]:
    """research/grouping-approach defaults: HDBSCAN(min_cluster_size=2), no ORB, 60s ceiling."""
    labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(np.stack([r.embedding for r in records]))
    clusters: dict[int, list] = {}
    groups = []
    for r, label in zip(records, labels):
        if label == -1:
            groups.append([r])
        else:
            clusters.setdefault(label, []).append(r)
    for members in clusters.values():
        members.sort(key=lambda m: (m.timestamp is None, m.timestamp or 0))
        current = [members[0]]
        for a, b in zip(members, members[1:]):
            if a.timestamp and b.timestamp and (b.timestamp - a.timestamp).total_seconds() > max_gap_s:
                groups.append(current)
                current = []
            current.append(b)
        groups.append(current)
    return [BurstGroup(str(uuid.uuid4()), g, "visual") for g in groups]


def stems(groups) -> list[list[str]]:
    return [[m.path.stem for m in g.members] for g in groups]


def methods():
    return {
        "sequence (defaults)": lambda rs: pipeline.group_records(rs),
        **{f"sequence window={w}": (lambda rs, w=w: pipeline.group_records(rs, window=w))
           for w in (1, 3, 10, 30)},
        "hdbscan (tuned)": hdbscan_pipeline,
    }


def report(title: str, records, labels) -> None:
    print(f"-- {title}")
    for name, fn in methods().items():
        print(f"  {name:22s} {score(stems(fn(copy.deepcopy(records))), labels).row()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    labels = load_labels(args.labels)
    paths = sorted(p for p in args.images.iterdir() if p.stem in labels.group_of)
    base, _ = records_with_embeddings(paths, MODEL)

    report("baseline (one camera, real clock)", base, labels)

    # 1. two shooters on one clock
    session_a = [r for r in base if r.path.stem < SESSION_B_START]
    session_b = [r for r in base if r.path.stem >= SESSION_B_START]
    shift = session_b[0].timestamp - session_a[0].timestamp - timedelta(seconds=7)
    interleaved = copy.deepcopy(base)
    for r in interleaved:
        if r.path.stem >= SESSION_B_START:
            r.timestamp -= shift
            r.camera_id = "SONY ILCE-7M4 #2"
    report("two shooters, distinct camera ids", interleaved, labels)
    for r in interleaved:
        r.camera_id = "SONY ILCE-7M4"
    report("two shooters, same camera id", interleaved, labels)

    # 2. no timestamps
    no_time = copy.deepcopy(base)
    for r in no_time:
        r.timestamp = None
    report("no EXIF timestamps (file-name order)", no_time, labels)

    # 3. locality: groups of session A alone vs. with session B present
    print("-- locality: pairs in session A whose grouping changes when session B is added")
    a_stems = {r.path.stem for r in session_a}
    for name, fn in methods().items():
        def together(groups):
            of = {s: i for i, g in enumerate(stems(groups)) for s in g}
            ordered = sorted(a_stems)
            return {(x, y) for i, x in enumerate(ordered) for y in ordered[i + 1:] if of[x] == of[y]}
        alone = together(fn(copy.deepcopy(session_a)))
        full = together(fn(copy.deepcopy(base)))
        print(f"  {name:22s} {len(alone ^ full):4d} pairs changed")

    # 4. runtime at wedding scale
    print("-- runtime, 5000 images (synthetic: real embeddings + noise, 2s apart)")
    rng = np.random.default_rng(0)
    big = []
    for i in range(5000):
        r = copy.deepcopy(base[i % len(base)])
        v = r.embedding + rng.normal(0, 0.02, r.embedding.shape).astype(np.float32)
        r.embedding = v / np.linalg.norm(v)
        r.timestamp = base[0].timestamp + timedelta(seconds=2 * i)
        r.path = r.path.with_name(f"SYN{i:05d}.JPG")
        big.append(r)
    for name, fn in methods().items():
        start = time.perf_counter()
        fn(copy.deepcopy(big))
        print(f"  {name:22s} {time.perf_counter() - start:6.2f}s")


if __name__ == "__main__":
    main()
