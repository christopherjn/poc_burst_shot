"""Browse the series found in a folder of photos: uv run --group ui streamlit run ui/app.py"""

import io
import math
import time
from pathlib import Path

import streamlit as st

from poc_burst_shot.grouping.embedder import DEFAULT_MODEL, Embedder
from poc_burst_shot.grouping.pipeline import (
    MAX_GAP_S,
    SIM_THRESHOLD,
    SUPPORTED_SUFFIXES,
    WINDOW,
    build_records,
    embed_records,
    group_records,
)
from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord
from poc_burst_shot.utils.image_loader import RAW_SUFFIXES, open_upright

THUMB_PX = 360
EMBED_CHUNK = 32

st.set_page_config(page_title="Series grouping", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> Embedder:
    return Embedder(DEFAULT_MODEL)


@st.cache_data(show_spinner=False, max_entries=20_000)
def thumbnail(path: str, mtime_ns: int) -> bytes | None:
    image = open_upright(Path(path), (THUMB_PX, THUMB_PX))
    if image is None:
        return None
    image.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def list_photos(folder: Path) -> tuple[list[Path], int]:
    files = [p for p in folder.iterdir() if p.is_file()]
    photos = sorted(p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES)
    raws = sum(p.suffix.lower() in RAW_SUFFIXES for p in files)
    return photos, raws


def signature(paths: list[Path]) -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def analyse(paths: list[Path]) -> list[ImageRecord]:
    bar = st.progress(0.0, text="Reading capture times…")
    records = build_records(paths)
    embedder = get_embedder()
    for start in range(0, len(records), EMBED_CHUNK):
        embed_records(records[start:start + EMBED_CHUNK], embedder)
        done = min(start + EMBED_CHUNK, len(records))
        bar.progress(done / len(records), text=f"Analysing photos {done} / {len(records)}")
    bar.empty()
    return records


def group_header(index: int, group: BurstGroup, show_camera: bool) -> str:
    members = group.members
    kind = f"Series {index}" if len(members) > 1 else f"Single {index}"
    parts = [f"**{kind}**", f"{len(members)} photo{'s' if len(members) > 1 else ''}"]
    times = [m.timestamp for m in members if m.timestamp is not None]
    if times:
        first, last = min(times), max(times)
        span = (last - first).total_seconds()
        when = first.strftime("%H:%M:%S")
        if span > 0:
            when += f" → {last.strftime('%H:%M:%S')} ({span:.0f}s)" if span >= 1 else f" ({span:.2f}s)"
        parts.append(when)
    else:
        parts.append("no capture time")
    if show_camera and members[0].camera_id:
        parts.append(members[0].camera_id)
    return " · ".join(parts)


def captions(members: list[ImageRecord]) -> list[str]:
    out = []
    for prev, member in zip([None, *members], members):
        text = member.path.name
        if prev is not None and prev.timestamp and member.timestamp:
            text += f"  +{(member.timestamp - prev.timestamp).total_seconds():.1f}s"
        if member.embedding is None:
            text += "  (unreadable)"
        out.append(text)
    return out


def render_group(index: int, group: BurstGroup, show_camera: bool, thumb_width: int) -> None:
    with st.container(border=True):
        st.markdown(group_header(index, group, show_camera))
        images, labels = [], []
        for member, label in zip(group.members, captions(group.members)):
            data = thumbnail(str(member.path), member.path.stat().st_mtime_ns)
            if data is None:
                st.warning(f"Could not display {member.path.name}")
                continue
            images.append(data)
            labels.append(label)
        if images:
            st.image(images, caption=labels, width=thumb_width)


st.title("Series grouping")

with st.sidebar:
    folder_text = st.text_input("Photo folder", placeholder="/path/to/card/DCIM/100MSDCF")
    run_clicked = st.button("Group photos", type="primary", disabled=not folder_text.strip())

    st.subheader("Grouping")
    sim_threshold = st.slider(
        "Similarity threshold", 0.70, 0.98, SIM_THRESHOLD, 0.01,
        help="How alike two nearby frames must look to join one series. "
             "Higher gives more, smaller groups.",
    )
    window = st.slider(
        "Look-back (frames)", 1, 15, WINDOW,
        help="Each frame is compared with this many earlier frames from the same camera. "
             "Lets a series survive an odd frame in between; too high chains similar series.",
    )
    max_gap_s = st.slider(
        "Max gap (seconds)", 5, 300, int(MAX_GAP_S), 5,
        help="Frames further apart than this are never grouped.",
    )
    st.caption("Changing these regroups instantly; photos are only analysed once per folder.")

    st.subheader("Display")
    series_only = st.toggle("Series only", value=False, help="Hide photos that stand alone.")
    thumb_width = st.slider("Thumbnail size", 120, THUMB_PX, 200, 20)
    per_page = st.select_slider("Groups per page", [10, 25, 50, 100], value=25)

if run_clicked:
    folder = Path(folder_text.strip()).expanduser().resolve()
    if not folder.is_dir():
        st.error(f"Not a folder: {folder}")
        st.stop()
    paths, raw_count = list_photos(folder)
    if not paths:
        st.error(f"No supported photos in {folder}")
        st.stop()
    sig = signature(paths)
    cached = st.session_state.get("analysis")
    if cached is None or cached["folder"] != folder or cached["signature"] != sig:
        started = time.monotonic()
        records = analyse(paths)
        st.session_state["analysis"] = {
            "folder": folder,
            "signature": sig,
            "records": records,
            "raw_count": raw_count,
            "seconds": time.monotonic() - started,
        }
        st.session_state["page"] = 1

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("Enter the path of a folder of photos in the sidebar and press **Group photos**.")
    st.stop()

records: list[ImageRecord] = analysis["records"]
groups = group_records(records, sim_threshold=sim_threshold, max_gap_s=float(max_gap_s), window=window)
series = [g for g in groups if len(g.members) > 1]

st.caption(f"`{analysis['folder']}` · analysed in {analysis['seconds']:.1f}s")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Photos", len(records))
c2.metric("Series", len(series))
c3.metric("Photos in series", sum(len(g.members) for g in series))
c4.metric("Single photos", len(groups) - len(series))

if analysis["raw_count"]:
    st.info(f"{analysis['raw_count']} RAW files were skipped; only JPEG/HEIF/PNG/TIFF are grouped.")
unreadable = sum(r.embedding is None for r in records)
if unreadable:
    st.warning(f"{unreadable} photos could not be read and are shown as single photos.")

show_camera = len({r.camera_id for r in records}) > 1
numbered = list(enumerate(groups, start=1))
shown = [(i, g) for i, g in numbered if len(g.members) > 1] if series_only else numbered

n_pages = max(1, math.ceil(len(shown) / per_page))
if st.session_state.get("page", 1) > n_pages:
    st.session_state["page"] = n_pages
page = 1
if n_pages > 1:
    page = st.columns([1, 5])[0].number_input(
        f"Page (of {n_pages})", min_value=1, max_value=n_pages, key="page"
    )

for index, group in shown[(page - 1) * per_page:page * per_page]:
    render_group(index, group, show_camera, thumb_width)
