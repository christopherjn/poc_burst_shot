import logging
from datetime import datetime, timedelta
from pathlib import Path

from PIL import ExifTags, Image

logger = logging.getLogger(__name__)


def _parse_timestamp(dt_str: str | None, subsec: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        # standard EXIF format: "2024:01:15 10:23:45"
        dt = datetime.strptime(dt_str.strip("\x00 "), "%Y:%m:%d %H:%M:%S")
        if subsec and subsec.strip("\x00 ").isdigit():
            # subsec is a string like "123" meaning 0.123 seconds
            dt += timedelta(seconds=float(f"0.{subsec.strip(chr(0) + ' ')}"))
        return dt
    except ValueError:
        return None


def extract(path: Path) -> dict:
    """
    Extract the capture timestamp from a single image file.

    Returns a dict with keys:
      - timestamp : datetime | None
    """
    result = {"timestamp": None}
    try:
        with Image.open(path) as img:
            exif_ifd = img.getexif().get_ifd(ExifTags.IFD.Exif)
        result["timestamp"] = _parse_timestamp(
            exif_ifd.get(ExifTags.Base.DateTimeOriginal),
            exif_ifd.get(ExifTags.Base.SubsecTimeOriginal),
        )
    except Exception as e:
        logger.warning("EXIF read failed for %s: %s", path, e)
    return result
