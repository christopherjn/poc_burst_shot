import logging
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np

from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord

logger = logging.getLogger(__name__)


def _capture_order(record: ImageRecord) -> tuple:
    # images without a timestamp go last, in file-name order
    return (record.timestamp is None, record.timestamp or datetime.min, record.path.name)


def _gap_seconds(a: ImageRecord, b: ImageRecord) -> float | None:
    if a.timestamp is None or b.timestamp is None:
        return None
    return abs((b.timestamp - a.timestamp).total_seconds())


def group_sequence(
    records: list[ImageRecord],
    sim_threshold: float,
    max_gap_s: float,
    window: int,
) -> list[BurstGroup]:
    """
    Link each frame to the previous `window` frames from the same camera body and union the links.

    Two frames are linked when they are no more than `max_gap_s` apart and the cosine similarity
    of their embeddings is at least `sim_threshold`. There is deliberately no "close in time
    means same burst" shortcut: when two bodies cannot be told apart (Sony writes no serial
    number) their frames interleave, and a timing-only link joins the two shooters' series.

    Looking back more than one frame lets a series survive an interloper (a detail shot, a
    second shooter's frame) and lets a photographer alternate between two framings. Much more
    than ~10 frames lets similar-looking series chain together.
    """
    parent = {id(r): id(r) for r in records}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    streams: dict[str | None, list[ImageRecord]] = defaultdict(list)
    for record in records:
        streams[record.camera_id].append(record)

    for stream in streams.values():
        stream.sort(key=_capture_order)
        for i, current in enumerate(stream):
            for prev in stream[max(0, i - window):i]:
                gap = _gap_seconds(prev, current)
                if gap is not None and gap > max_gap_s:
                    continue
                if prev.embedding is None or current.embedding is None:
                    continue
                if float(np.dot(prev.embedding, current.embedding)) >= sim_threshold:
                    parent[find(id(prev))] = find(id(current))

    components: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        components[find(id(record))].append(record)

    groups = []
    for members in components.values():
        members.sort(key=_capture_order)
        groups.append(BurstGroup(
            group_id=str(uuid.uuid4()),
            members=members,
            grouping_method="series" if len(members) > 1 else "singleton",
        ))
    groups.sort(key=lambda g: _capture_order(g.members[0]))

    logger.info("Sequence grouping: %d groups from %d images", len(groups), len(records))
    return groups
