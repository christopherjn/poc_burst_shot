"""Streamlit frontend for poc-burst-shot.

Talks to the FastAPI backend (see ``ui/backend/server.py``): it sends a folder
path to be grouped and renders the returned burst clusters as image grids.
"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BURST_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
COLUMNS = 4
REQUEST_TIMEOUT = 600  # grouping can be slow on first run (model download)
POLL_INTERVAL = 0.5  # seconds between progress polls

# Embeddings are L2-normalised, so the merge distance lives in [0, 2]. We map an
# intuitive 0–100 "aggressiveness" slider onto epsilon: high aggressiveness ->
# small epsilon (split more), low aggressiveness -> large epsilon (merge more).
MAX_EPSILON = 0.7


def aggressiveness_to_epsilon(aggressiveness: int) -> float:
    return round(MAX_EPSILON * (1 - aggressiveness / 100), 3)

st.set_page_config(page_title="Burst Shot Grouping", page_icon="📸", layout="wide")


@st.cache_data(show_spinner=False)
def fetch_image(path: str) -> bytes | None:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/image", params={"path": path}, timeout=30
        )
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def start_job(folder: str, epsilon: float) -> str:
    resp = requests.post(
        f"{BACKEND_URL}/group",
        json={"folder": folder, "cluster_selection_epsilon": epsilon},
        timeout=30,
    )
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(detail)
    return resp.json()["job_id"]


def poll_progress(job_id: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/progress/{job_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=5).ok
    except requests.RequestException:
        return False


st.title("📸 Burst Shot Grouping")
st.caption(f"Connected to backend at `{BACKEND_URL}`")

if not backend_alive():
    st.error(
        f"Cannot reach the backend at {BACKEND_URL}. "
        "Start everything together with `make burst`."
    )
    st.stop()

folder = st.text_input(
    "Image folder",
    placeholder="/path/to/folder/of/photos",
    help="Absolute path to a folder of images on the machine running the backend.",
)

aggressiveness = st.slider(
    "Grouping aggressiveness",
    min_value=0,
    max_value=100,
    value=50,
    help=(
        "Controls how eagerly the visual stage splits clusters. "
        "Lower it if separate bursts are being split apart too eagerly."
    ),
)
left, right = st.columns(2)
left.caption("◀ Create fewer groups")
right.markdown(
    "<p style='text-align:right;color:gray;font-size:0.8rem;margin:0'>"
    "Create more groups ▶</p>",
    unsafe_allow_html=True,
)
epsilon = aggressiveness_to_epsilon(aggressiveness)
st.caption(f"Merge distance (epsilon) = {epsilon}")

if st.button("Group images", type="primary", disabled=not folder.strip()):
    progress_bar = st.progress(0, text="Starting…")
    try:
        job_id = start_job(folder.strip(), epsilon)
        while True:
            status = poll_progress(job_id)
            percent = int(status["progress"] * 100)
            if status["status"] == "running":
                # hold at 99% until the result is actually ready
                progress_bar.progress(min(percent, 99), text=status["message"])
            elif status["status"] == "done":
                progress_bar.progress(100, text="Complete")
                st.session_state["result"] = status["result"]
                break
            elif status["status"] == "error":
                progress_bar.empty()
                st.session_state.pop("result", None)
                st.error(status.get("error") or "Grouping failed")
                break
            time.sleep(POLL_INTERVAL)
    except (RuntimeError, requests.RequestException) as exc:
        progress_bar.empty()
        st.session_state.pop("result", None)
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    groups = result["groups"]
    multi = [g for g in groups if len(g["members"]) > 1]
    singles = [g for g in groups if len(g["members"]) == 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Images", result["image_count"])
    c2.metric("Groups", len(groups))
    c3.metric("Burst clusters (2+)", len(multi))

    st.divider()

    for idx, group in enumerate(multi + singles, start=1):
        members = group["members"]
        label = f"Group {idx} · {group['grouping_method']} · {len(members)} image(s)"
        with st.expander(label, expanded=len(members) > 1):
            cols = st.columns(COLUMNS)
            for i, member in enumerate(members):
                col = cols[i % COLUMNS]
                data = fetch_image(member["path"])
                if data is not None:
                    col.image(data, caption=member["name"], use_container_width=True)
                else:
                    col.warning(f"Could not load {member['name']}")
