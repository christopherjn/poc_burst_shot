import logging
import sys
from pathlib import Path

from poc_burst_shot.grouping.pipeline import SUPPORTED_SUFFIXES, run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        try:
            raw = input("Enter path to image folder: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

    folder = Path(raw).expanduser().resolve()

    if not folder.exists():
        print(f"Path does not exist: {folder}")
        sys.exit(1)

    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)

    if not paths:
        print(f"No supported images found in: {folder}")
        sys.exit(1)

    print(f"Found {len(paths)} image(s) in {folder}\n")
    groups = run(paths)

    print(f"\nFound {len(groups)} group(s):\n")
    for i, group in enumerate(groups, 1):
        print(f"  Group {i} [{group.grouping_method}] — {len(group.members)} image(s)")
        for member in group.members:
            print(f"    {member.path}")
