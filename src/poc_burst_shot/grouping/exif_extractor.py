import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# EXIF tag IDs from the Exif spec
_TAG_DATETIME_ORIGINAL = 36867
_TAG_SUBSEC_TIME_ORIGINAL = 37521


def _run_exiftool(path: Path) -> dict:
    """Call exiftool and return its JSON output as a dict."""
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-BurstUUID", "-ContentIdentifier",
             "-DateTimeOriginal", "-SubSecTimeOriginal", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        return data[0] if data else {}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def _parse_timestamp(dt_str: str | None, subsec: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        # standard EXIF format: "2024:01:15 10:23:45"
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        if subsec:
            # subsec is a string like "123" meaning 0.123 seconds
            fraction = float(f"0.{subsec.strip()}")
            from datetime import timedelta
            dt = dt + timedelta(seconds=fraction)
        return dt
    except ValueError:
        return None


def extract(path: Path) -> dict:
    """
    Extract burst UUID and timestamp from a single image file.

    Returns a dict with keys:
      - burst_uuid : str | None
      - timestamp  : datetime | None
    """
    result = {"burst_uuid": None, "timestamp": None}

    # try exiftool first — handles HEIC Apple MakerNotes reliably
    exif_data = _run_exiftool(path)
    if exif_data:
        result["burst_uuid"] = (
            exif_data.get("BurstUUID") or exif_data.get("ContentIdentifier")
        )
        result["timestamp"] = _parse_timestamp(
            exif_data.get("DateTimeOriginal"),
            exif_data.get("SubSecTimeOriginal"),
        )
        return result

    # fallback: Pillow for JPEG/PNG when exiftool is unavailable
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(path)
        raw_exif = img._getexif()
        if raw_exif:
            dt_str = raw_exif.get(_TAG_DATETIME_ORIGINAL)
            subsec = raw_exif.get(_TAG_SUBSEC_TIME_ORIGINAL)
            result["timestamp"] = _parse_timestamp(dt_str, subsec)
    except Exception as e:
        logger.warning("Pillow EXIF fallback failed for %s: %s", path, e)

    return result
