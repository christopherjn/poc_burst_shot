import logging
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from poc_burst_shot.grouping.clusterer import cluster
from poc_burst_shot.grouping.embedder import Embedder
from poc_burst_shot.grouping.exif_extractor import extract
from poc_burst_shot.grouping.time_splitter import split_by_time
from poc_burst_shot.grouping.verifier import verify_and_split
from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord
from poc_burst_shot.utils.image_loader import load_image

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}


def _build_records(paths: list[Path]) -> list[ImageRecord]:
    records = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type, skipping: %s", path)
            continue
        exif = extract(path)
        records.append(ImageRecord(
            path=path,
            burst_uuid=exif["burst_uuid"],
            timestamp=exif["timestamp"],
        ))
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


def _visual_cluster(records: list[ImageRecord]) -> list[BurstGroup]:
    """Stage 1: embed + HDBSCAN cluster on records that had no BurstUUID."""
    embedder = Embedder()

    images = []
    valid_records = []
    for record in records:
        img = load_image(record.path)
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

    labels = cluster(embeddings)

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


def run(paths: list[Path]) -> list[BurstGroup]:
    """
    Full grouping pipeline. Returns a list of BurstGroups.
    Each group has grouping_method: 'exif_uuid' | 'visual' | 'singleton'
    """
    records = _build_records(paths)

    # Stage 0 — EXIF fast path
    exif_groups, unresolved = _group_by_exif_uuid(records)

    # Stage 1 — visual clustering for images without a BurstUUID
    visual_groups = _visual_cluster(unresolved)

    # Stage 2 — verify visual groups, split false merges (skip singletons)
    verified_groups = []
    for group in visual_groups:
        if group.grouping_method == "singleton":
            verified_groups.append(group)
        else:
            verified_groups.extend(verify_and_split(group))

    logger.info("Stage 2: %d groups after verification", len(verified_groups))

    # Stage 3 — time sub-split within every group
    final_groups = []
    for group in exif_groups + verified_groups:
        final_groups.extend(split_by_time(group))

    logger.info("Stage 3: %d final groups", len(final_groups))
    return final_groups
