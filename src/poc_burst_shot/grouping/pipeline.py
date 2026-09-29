import logging
import uuid
from collections import defaultdict
from pathlib import Path

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

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp"}


def _new_group(members: list[ImageRecord], method: str) -> BurstGroup:
    return BurstGroup(group_id=str(uuid.uuid4()), members=members, grouping_method=method)


def build_records(paths: list[Path]) -> list[ImageRecord]:
    records = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type, skipping: %s", path)
            continue
        exif = extract(path)
        records.append(ImageRecord(path=path, timestamp=exif["timestamp"]))
    return records


def embed_records(records: list[ImageRecord], embedder: Embedder) -> None:
    """Set `embedding` on every record whose image can be loaded; others keep None."""
    images = []
    loaded = []
    for record in records:
        img = load_image(record.path)
        if img is None:
            logger.warning("Could not load image, treating as singleton: %s", record.path)
            continue
        images.append(img)
        loaded.append(record)

    if not loaded:
        return

    for record, vec in zip(loaded, embedder.embed_batch(images)):
        record.embedding = vec


def _visual_cluster(records: list[ImageRecord], min_cluster_size: int) -> list[BurstGroup]:
    """Stage 1: HDBSCAN on embeddings. Records without an embedding become singletons."""
    embedded = [r for r in records if r.embedding is not None]
    groups = [_new_group([r], "singleton") for r in records if r.embedding is None]

    if embedded:
        labels = cluster(np.stack([r.embedding for r in embedded]), min_cluster_size)
        label_map: dict[int, list[ImageRecord]] = defaultdict(list)
        for record, label in zip(embedded, labels):
            label_map[label].append(record)

        for label, members in label_map.items():
            if label == -1:
                groups.extend(_new_group([m], "singleton") for m in members)
            else:
                groups.append(_new_group(members, "visual"))

    logger.info("Stage 1: %d visual groups from %d images", len(groups), len(records))
    return groups


def group_records(
    records: list[ImageRecord],
    min_cluster_size: int = 2,
    verify: bool = False,
    time_split: bool = True,
) -> list[BurstGroup]:
    """Stages 1-3 on records that already carry embeddings.

    ORB verification is off by default: on the labelled benchmark it changed no group while
    costing ~0.8s per image, since it re-decodes every member at full resolution.
    """
    groups = _visual_cluster(records, min_cluster_size)

    # Stage 2 — verify visual groups, split false merges (skip singletons)
    if verify:
        verified = []
        for group in groups:
            if group.grouping_method == "singleton":
                verified.append(group)
            else:
                verified.extend(verify_and_split(group))
        groups = verified
        logger.info("Stage 2: %d groups after verification", len(groups))

    # Stage 3 — time sub-split within every group
    if time_split:
        groups = [sub for group in groups for sub in split_by_time(group)]
        logger.info("Stage 3: %d final groups", len(groups))

    for group in groups:
        if len(group.members) == 1:
            group.grouping_method = "singleton"
    return groups


def run(paths: list[Path], model_name: str = DEFAULT_MODEL, **group_kwargs) -> list[BurstGroup]:
    """
    Full grouping pipeline. Returns a list of BurstGroups.
    Each group has grouping_method: 'visual' | 'singleton'
    """
    records = build_records(paths)
    embed_records(records, Embedder(model_name))
    return group_records(records, **group_kwargs)
