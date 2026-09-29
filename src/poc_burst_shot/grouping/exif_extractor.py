import logging
from datetime import datetime, timedelta
from pathlib import Path

from PIL import ExifTags, Image

logger = logging.getLogger(__name__)


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip("\x00 ")
    return text or None


def _parse_timestamp(dt_str: str | None, subsec: str | None) -> datetime | None:
    dt_str, subsec = _clean(dt_str), _clean(subsec)
    if not dt_str:
        return None
    try:
        # standard EXIF format: "2024:01:15 10:23:45"
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        if subsec and subsec.isdigit():
            # subsec is a string like "123" meaning 0.123 seconds
            dt += timedelta(seconds=float(f"0.{subsec}"))
        return dt
    except ValueError:
        return None


def extract(path: Path) -> dict:
    """
    Extract capture metadata from a single image file.

    Returns a dict with keys:
      - timestamp : datetime | None
      - camera_id : str | None   (make + model + body serial, identifies one camera body)
    """
    result = {"timestamp": None, "camera_id": None}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        result["timestamp"] = _parse_timestamp(
            exif_ifd.get(ExifTags.Base.DateTimeOriginal),
            exif_ifd.get(ExifTags.Base.SubsecTimeOriginal),
        )
        parts = [
            _clean(exif.get(ExifTags.Base.Make)),
            _clean(exif.get(ExifTags.Base.Model)),
            _clean(exif_ifd.get(ExifTags.Base.BodySerialNumber)),
        ]
        if any(parts):
            result["camera_id"] = " ".join(p for p in parts if p)
    except Exception as e:
        logger.warning("EXIF read failed for %s: %s", path, e)
    return result
