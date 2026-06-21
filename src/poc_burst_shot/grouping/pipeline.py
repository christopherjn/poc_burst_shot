import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from poc_burst_shot.grouping.clusterer import cluster
from poc_burst_shot.grouping.embedder import DEFAULT_MODEL, Embedder
from poc_burst_shot.grouping.exif_extractor import extract
from poc_burst_shot.grouping.time_splitter import split_by_time
from poc_burst_shot.grouping.verifier import verify_and_split
from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord
from poc_burst_shot.utils.image_loader import load_image

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}

# Called with (fraction_complete in [0, 1], human-readable message).
ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


def _build_records(
    paths: list[Path],
    on_each: Callable[[int, int], None] | None = None,
) -> list[ImageRecord]:
    records = []
    total = len(paths)
    for i, path in enumerate(paths, 1):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type, skipping: %s", path)
            if on_each:
                on_each(i, total)
            continue
        exif = extract(path)
        records.append(ImageRecord(
            path=path,
            burst_uuid=exif["burst_uuid"],
            timestamp=exif["timestamp"],
        ))
        if on_each:
            on_each(i, total)
    return records


def _group_by_exif_uuid(
    records: list[ImageRecord],
) -> tuple[list[BurstGroup], list[ImageRecord]]:
    """Stage 0: group by BurstUUID. Returns (exif_groups, unresolved_records)."""
    uuid_map: dict[str, list[ImageRecord]] = defaultdict(list)
    unresolved: list[ImageRecord] = []

    for record in records:
        if record.burst_uuid:
            uuid_map[record.burst_uuid].append(record)
        else:
            unresolved.append(record)

    exif_groups = [
        BurstGroup(
            group_id=str(uuid.uuid4()),
            members=members,
            grouping_method="exif_uuid",
        )
        for members in uuid_map.values()
    ]

    logger.info(
        "Stage 0: %d EXIF groups, %d unresolved images",
        len(exif_groups),
        len(unresolved),
    )
    return exif_groups, unresolved


def _visual_cluster(
    records: list[ImageRecord],
    cluster_selection_epsilon: float | None = None,
    on_load: Callable[[int, int], None] | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[BurstGroup]:
    """Stage 1: embed + HDBSCAN cluster on records that had no BurstUUID."""
    embedder = Embedder(model_name=model_name)

    images = []
    valid_records = []
    total = len(records)
    for i, record in enumerate(records, 1):
        img = load_image(record.path)
        if on_load:
            on_load(i, total)
        if img is None:
            logger.warning("Could not load image, treating as singleton: %s", record.path)
            continue
        images.append(img)
        valid_records.append(record)

    if not valid_records:
        return []

    embeddings = embedder.embed_batch(images)

    for record, vec in zip(valid_records, embeddings):
        record.embedding = vec

    if cluster_selection_epsilon is None:
        labels = cluster(embeddings)
    else:
        labels = cluster(embeddings, cluster_selection_epsilon=cluster_selection_epsilon)

    label_map: dict[int, list[ImageRecord]] = defaultdict(list)
    for record, label in zip(valid_records, labels):
        label_map[label].append(record)

    groups = []
    for label, members in label_map.items():
        method = "singleton" if label == -1 else "visual"
        # singletons each get their own group
        if label == -1:
            for member in members:
                groups.append(BurstGroup(
                    group_id=str(uuid.uuid4()),
                    members=[member],
                    grouping_method="singleton",
                ))
        else:
            groups.append(BurstGroup(
                group_id=str(uuid.uuid4()),
                members=members,
                grouping_method="visual",
            ))

    logger.info("Stage 1: %d visual groups from %d images", len(groups), len(valid_records))
    return groups


def run(
    paths: list[Path],
    cluster_selection_epsilon: float | None = None,
    progress_callback: ProgressCallback | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[BurstGroup]:
    """
    Full grouping pipeline. Returns a list of BurstGroups.
    Each group has grouping_method: 'exif_uuid' | 'visual' | 'singleton'

    `cluster_selection_epsilon` controls how aggressively the visual stage
    splits: larger values merge more nearby micro-clusters (fewer, larger
    groups). When None, the clusterer's default is used.

    `model_name` selects the timm embedding backbone used by the visual stage
    (see `embedder.AVAILABLE_MODELS`).

    `progress_callback(fraction, message)` is invoked as work proceeds, with
    `fraction` rising from 0.0 to 1.0. When None, progress is ignored.
    """
    progress = progress_callback or _noop_progress

    # Stage 0 — read metadata (0% → 10%)
    progress(0.0, "Reading image metadata")
    records = _build_records(
        paths,
        on_each=lambda i, n: progress(0.10 * i / n, f"Reading metadata ({i}/{n})"),
    )
    exif_groups, unresolved = _group_by_exif_uuid(records)

    # Stage 1 — load + embed + cluster images without a BurstUUID (10% → 70%)
    progress(0.10, "Loading images")
    visual_groups = _visual_cluster(
        unresolved,
        cluster_selection_epsilon,
        on_load=lambda i, n: progress(0.10 + 0.40 * i / n, f"Loading images ({i}/{n})"),
        model_name=model_name,
    )
    progress(0.70, "Clustering")

    # Stage 2 — verify visual groups, split false merges (70% → 90%)
    progress(0.70, "Verifying groups")
    verified_groups = []
    candidates = [g for g in visual_groups if g.grouping_method != "singleton"]
    verified_groups.extend(g for g in visual_groups if g.grouping_method == "singleton")
    for i, group in enumerate(candidates, 1):
        verified_groups.extend(verify_and_split(group))
        if candidates:
            progress(0.70 + 0.20 * i / len(candidates), f"Verifying groups ({i}/{len(candidates)})")

    logger.info("Stage 2: %d groups after verification", len(verified_groups))

    # Stage 3 — time sub-split within every group (90% → 100%)
    progress(0.90, "Splitting by time")
    final_groups = []
    for group in exif_groups + verified_groups:
        final_groups.extend(split_by_time(group))

    logger.info("Stage 3: %d final groups", len(final_groups))
    progress(1.0, "Complete")
    return final_groups
