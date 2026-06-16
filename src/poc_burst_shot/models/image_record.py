from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np


@dataclass
class ImageRecord:
    path: Path
    burst_uuid: str | None = None
    timestamp: datetime | None = None
    embedding: np.ndarray | None = None