import uuid
import logging
import numpy as np

from poc_burst_shot.models.burst_group import BurstGroup

logger = logging.getLogger(__name__)

# sub-split a visual cluster if an internal gap exceeds this multiple of the median gap
GAP_MULTIPLIER = 3.0


def split_by_time(group: BurstGroup) -> list[BurstGroup]:
    """
    Within a verified visual cluster, sub-split on large timestamp gaps.
    Returns the original group unchanged if timestamps are missing or no split needed.
    """
    members = group.members

    if len(members) <= 1:
        return [group]

    members_with_ts = [m for m in members if m.timestamp is not None]
    if len(members_with_ts) < len(members):
        logger.warning(
            "Group %s has %d/%d members without timestamps — skipping time split",
            group.group_id,
            len(members) - len(members_with_ts),
            len(members),
        )
        return [group]

    sorted_members = sorted(members, key=lambda m: m.timestamp)
    timestamps = [m.timestamp.timestamp() for m in sorted_members]  # seconds since epoch

    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    # adaptive threshold: GAP_MULTIPLIER × median gap
    median_gap = float(np.median(gaps))
    threshold = GAP_MULTIPLIER * median_gap

    split_indices = [i for i, gap in enumerate(gaps) if gap > threshold]

    if not split_indices:
        return [group]

    # split at each large gap
    sub_groups = []
    start = 0
    for split_at in split_indices:
        sub_groups.append(sorted_members[start : split_at + 1])
        start = split_at + 1
    sub_groups.append(sorted_members[start:])

    logger.info(
        "Time-split group %s into %d sub-groups (threshold=%.2fs)",
        group.group_id,
        len(sub_groups),
        threshold,
    )

    return [
        BurstGroup(
            group_id=str(uuid.uuid4()),
            members=sub,
            grouping_method=group.grouping_method,
        )
        for sub in sub_groups
    ]
