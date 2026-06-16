from dataclasses import dataclass

from poc_burst_shot.models.image_record import ImageRecord

@dataclass
class BurstGroup:
    group_id: str
    members: list[ImageRecord]
    grouping_method: str