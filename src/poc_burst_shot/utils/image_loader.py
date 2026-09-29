import logging
from pathlib import Path
from PIL import ExifTags, Image

logger = logging.getLogger(__name__)

RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}

TARGET_SIZE = (224, 224)

# decode at no less than twice the target size so the final resize still has detail to work with
_MIN_DECODE = (TARGET_SIZE[0] * 2, TARGET_SIZE[1] * 2)

# EXIF Orientation -> transpose that makes the image upright (same table as ImageOps.exif_transpose)
_ORIENTATION_FIX = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _use_embedded_preview(image: Image.Image, min_size: tuple[int, int]) -> None:
    """Switch to the camera's embedded preview JPEG (MPF frame) when it is big enough.

    Camera JPEGs from Sony/Canon/Nikon/Fuji carry a ~1600px preview next to the full frame;
    decoding it is ~35x cheaper than entropy-decoding a 30MP+ image.
    """
    if getattr(image, "n_frames", 1) < 2:
        return
    image.seek(1)
    if image.width < min_size[0] or image.height < min_size[1]:
        image.seek(0)


def open_upright(path: Path, min_size: tuple[int, int]) -> Image.Image | None:
    """Decode an upright RGB image at least `min_size` large (when the file is), or None."""
    if path.suffix.lower() in RAW_SUFFIXES:
        logger.warning(f"Raw image file detected: {path}. Skipping...")
        return None

    try:
        image = Image.open(path)
        # cameras store portrait shots sideways with an Orientation tag on the primary image
        orientation = image.getexif().get(ExifTags.Base.Orientation)
        _use_embedded_preview(image, min_size)
        # let the JPEG decoder downscale by up to 8x while decoding
        image.draft("RGB", min_size)
        image.load()
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

    if orientation in _ORIENTATION_FIX:
        image = image.transpose(_ORIENTATION_FIX[orientation])

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def load_image(path: Path) -> Image.Image | None:
    """Return an upright 224 x 224 image or None if the file can't be loaded."""
    image = open_upright(path, _MIN_DECODE)
    return None if image is None else image.resize(TARGET_SIZE, Image.LANCZOS)
