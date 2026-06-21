"""Streamlit frontend for poc-burst-shot.

Talks to the FastAPI backend (see ``ui/backend/server.py``): it sends a folder
path to be grouped and renders the returned burst clusters as image grids.

Two modes:

* **Single model** — group with one embedding backbone (EfficientNet-B0 by
  default) and browse the resulting bursts.
* **Compare models** — group the same folder with two or more backbones and lay
  the results out side by side to compare how each one clusters.
"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BURST_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
COLUMNS = 4
COMPARE_COLUMNS = 2  # image grid is narrower when several models sit side by side
REQUEST_TIMEOUT = 600  # grouping can be slow on first run (model download)
POLL_INTERVAL = 0.5  # seconds between progress polls

# Fallback model list if the backend is too old to expose /models.
FALLBACK_MODELS = [{"name": "efficientnet_b0", "label": "EfficientNet-B0 (default)"}]
FALLBACK_DEFAULT = "efficientnet_b0"

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


@st.cache_data(show_spinner=False)
def fetch_models() -> tuple[list[dict], str]:
    """Return (models, default_model_name) from the backend, with a fallback."""
    try:
        resp = requests.get(f"{BACKEND_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["models"], data["default"]
    except (requests.RequestException, KeyError, ValueError):
        return FALLBACK_MODELS, FALLBACK_DEFAULT


def start_job(folder: str, epsilon: float, model_name: str) -> str:
    resp = requests.post(
        f"{BACKEND_URL}/group",
        json={
            "folder": folder,
            "cluster_selection_epsilon": epsilon,
            "model_name": model_name,
        },
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


def run_grouping(folder: str, epsilon: float, model_name: str, progress_bar) -> dict:
    """Start a job and block until it finishes, driving `progress_bar`.

    Returns the result payload, or raises RuntimeError on backend failure.
    """
    job_id = start_job(folder, epsilon, model_name)
    while True:
        status = poll_progress(job_id)
        percent = int(status["progress"] * 100)
        if status["status"] == "running":
            # hold at 99% until the result is actually ready
            progress_bar.progress(min(percent, 99), text=status["message"])
        elif status["status"] == "done":
            progress_bar.progress(100, text="Complete")
            return status["result"]
        elif status["status"] == "error":
            raise RuntimeError(status.get("error") or "Grouping failed")
        time.sleep(POLL_INTERVAL)


def backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=5).ok
    except requests.RequestException:
        return False


def render_result(result: dict, container, columns: int) -> None:
    """Render a single grouping result (metrics + group grids) into `container`."""
    groups = result["groups"]
    multi = [g for g in groups if len(g["members"]) > 1]
    singles = [g for g in groups if len(g["members"]) == 1]

    c1, c2, c3 = container.columns(3)
    c1.metric("Images", result["image_count"])
    c2.metric("Groups", len(groups))
    c3.metric("Burst clusters (2+)", len(multi))

    container.divider()

    for idx, group in enumerate(multi + singles, start=1):
        members = group["members"]
        label = f"Group {idx} · {group['grouping_method']} · {len(members)} image(s)"
        is_multi = len(members) > 1
        with container.expander(label, expanded=is_multi):
            cols = st.columns(columns)
            for i, member in enumerate(members):
                col = cols[i % columns]
                data = fetch_image(member["path"])
                if data is not None:
                    col.image(data, caption=member["name"], use_container_width=True)
                else:
                    col.warning(f"Could not load {member['name']}")


st.title("📸 Burst Shot Grouping")
st.caption(f"Connected to backend at `{BACKEND_URL}`")

if not backend_alive():
    st.error(
        f"Cannot reach the backend at {BACKEND_URL}. "
        "Start everything together with `make burst`."
    )
    st.stop()

models, default_model = fetch_models()
model_names = [m["name"] for m in models]
model_labels = {m["name"]: m["label"] for m in models}
default_index = model_names.index(default_model) if default_model in model_names else 0

folder = st.text_input(
    "Image folder",
    placeholder="/path/to/folder/of/photos",
    help="Absolute path to a folder of images on the machine running the backend.",
)

mode = st.radio(
    "Mode",
    ["Single model", "Compare models"],
    horizontal=True,
    help="Compare mode runs the same folder through several models side by side.",
)

if mode == "Single model":
    selected_models = [
        st.selectbox(
            "Embedding model",
            options=model_names,
            index=default_index,
            format_func=lambda name: model_labels.get(name, name),
            help="The neural network used to turn each image into a feature vector.",
        )
    ]
else:
    selected_models = st.multiselect(
        "Embedding models to compare",
        options=model_names,
        default=[default_model] if default_model in model_names else model_names[:1],
        format_func=lambda name: model_labels.get(name, name),
        help="Pick two or more models to group the same folder and compare results.",
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

compare_mode = mode == "Compare models"
button_label = "Compare models" if compare_mode else "Group images"
# Compare mode needs at least two distinct models selected.
ready = bool(folder.strip()) and (
    len(selected_models) >= 2 if compare_mode else len(selected_models) == 1
)
if compare_mode and folder.strip() and len(selected_models) < 2:
    st.info("Select at least two models to compare.")

if st.button(button_label, type="primary", disabled=not ready):
    st.session_state.pop("results", None)
    results: list[dict] = []
    failed = False
    for model_name in selected_models:
        st.write(f"**{model_labels.get(model_name, model_name)}**")
        progress_bar = st.progress(0, text="Starting…")
        try:
            result = run_grouping(folder.strip(), epsilon, model_name, progress_bar)
            results.append(result)
        except (RuntimeError, requests.RequestException) as exc:
            progress_bar.empty()
            st.error(f"{model_labels.get(model_name, model_name)}: {exc}")
            failed = True
            break
    if not failed and results:
        st.session_state["results"] = results

results = st.session_state.get("results")
if results:
    if len(results) == 1:
        render_result(results[0], st, COLUMNS)
    else:
        st.subheader("Side-by-side comparison")
        columns = st.columns(len(results))
        for result, col in zip(results, columns):
            label = model_labels.get(result["model_name"], result["model_name"])
            col.markdown(f"### {label}")
            render_result(result, col, COMPARE_COLUMNS)
