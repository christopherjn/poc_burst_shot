import logging
from pathlib import Path

from poc_burst_shot.grouping.embedder import DEFAULT_MODEL, Embedder
from poc_burst_shot.grouping.exif_extractor import extract
from poc_burst_shot.grouping.sequencer import group_sequence
from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord
from poc_burst_shot.utils.image_loader import load_image

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp"}

# Defaults chosen on eval/labels/sony_a7iv_portrait_100.json; see eval/README.md for the sweep.
SIM_THRESHOLD = 0.86    # cosine similarity (DINOv2-S) above which two nearby frames are one series
MAX_GAP_S = 60.0        # frames further apart than this are never linked
WINDOW = 5              # compare each frame with this many previous frames from the same camera


def build_records(paths: list[Path]) -> list[ImageRecord]:
    records = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type, skipping: %s", path)
            continue
        meta = extract(path)
        records.append(ImageRecord(path=path, timestamp=meta["timestamp"], camera_id=meta["camera_id"]))
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


def group_records(
    records: list[ImageRecord],
    sim_threshold: float = SIM_THRESHOLD,
    max_gap_s: float = MAX_GAP_S,
    window: int = WINDOW,
) -> list[BurstGroup]:
    """Group records that already carry embeddings into series, in capture order."""
    return group_sequence(records, sim_threshold, max_gap_s, window)


def run(paths: list[Path], model_name: str = DEFAULT_MODEL, **group_kwargs) -> list[BurstGroup]:
    """
    Full grouping pipeline. Returns a list of BurstGroups in capture order.
    Each group has grouping_method: 'series' | 'singleton'
    """
    records = build_records(paths)
    embed_records(records, Embedder(model_name))
    return group_records(records, **group_kwargs)
