import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}

TARGET_SIZE = (224, 224)

def load_image(path: Path) -> Image.Image | None:

    # Return a 224 x 224 imaage or none if file not loaded

    if path.suffix.lower() in RAW_SUFFIXES:
        logger.warning(f"Raw image file detected: {path}. Skipping...")
        return None

    try: 
        image = Image.open(path)
        image.load()
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image.resize(TARGET_SIZE, Image.LANCZOS) # LANCZOS is a better resampling algorithm in this case else we can go with BICUBIC but results will be less good