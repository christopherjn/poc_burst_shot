from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np


@dataclass
class ImageRecord:
    path: Path
    timestamp: datetime | None = None
    camera_id: str | None = None
    embedding: np.ndarray | None = None