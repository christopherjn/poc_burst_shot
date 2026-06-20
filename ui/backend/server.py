"""FastAPI service that wraps the poc-burst-shot grouping pipeline.

It exposes what the Streamlit frontend needs:

* ``POST /group``        — start a grouping job in the background, returns a job id.
* ``GET  /progress/{id}`` — poll a job's progress (0.0–1.0), and its result once done.
* ``GET  /image``        — stream a single image file by absolute path so the
  frontend can render thumbnails without needing direct filesystem access.
"""

import logging
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from poc_burst_shot.grouping.pipeline import run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tiff", ".tif", ".webp", ".bmp",
}

app = FastAPI(title="poc-burst-shot", version="0.1.0")


class GroupRequest(BaseModel):
    folder: str
    # Larger values merge more nearby micro-clusters (less aggressive grouping).
    # None falls back to the clusterer's built-in default.
    cluster_selection_epsilon: float | None = None


class GroupMember(BaseModel):
    path: str
    name: str
    timestamp: str | None = None


class Group(BaseModel):
    group_id: str
    grouping_method: str
    members: list[GroupMember]


class GroupResponse(BaseModel):
    folder: str
    image_count: int
    groups: list[Group]


class StartResponse(BaseModel):
    job_id: str


class ProgressResponse(BaseModel):
    status: str  # "running" | "done" | "error"
    progress: float  # 0.0 – 1.0
    message: str
    error: str | None = None
    result: GroupResponse | None = None


# In-memory job registry. Fine for a single-process POC backend.
_jobs: dict[str, ProgressResponse] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        current = _jobs[job_id]
        _jobs[job_id] = current.model_copy(update=fields)


def _build_payload(folder: Path, paths: list[Path], groups) -> GroupResponse:
    return GroupResponse(
        folder=str(folder),
        image_count=len(paths),
        groups=[
            Group(
                group_id=group.group_id,
                grouping_method=group.grouping_method,
                members=[
                    GroupMember(
                        path=str(member.path),
                        name=member.path.name,
                        timestamp=member.timestamp.isoformat() if member.timestamp else None,
                    )
                    for member in group.members
                ],
            )
            for group in groups
        ],
    )


def _run_job(job_id: str, folder: Path, paths: list[Path], epsilon: float | None) -> None:
    def report(fraction: float, message: str) -> None:
        _set_job(job_id, progress=fraction, message=message)

    try:
        groups = run(paths, cluster_selection_epsilon=epsilon, progress_callback=report)
        _set_job(
            job_id,
            status="done",
            progress=1.0,
            message="Complete",
            result=_build_payload(folder, paths, groups),
        )
    except Exception as exc:  # surface failures to the frontend
        logger.exception("Grouping job %s failed", job_id)
        _set_job(job_id, status="error", message="Failed", error=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/group", response_model=StartResponse)
def group_images(request: GroupRequest) -> StartResponse:
    folder = Path(request.folder).expanduser().resolve()

    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise HTTPException(
            status_code=400, detail=f"No supported images found in: {folder}"
        )

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = ProgressResponse(
            status="running", progress=0.0, message="Starting"
        )

    logger.info(
        "Job %s: grouping %d image(s) in %s (epsilon=%s)",
        job_id,
        len(paths),
        folder,
        request.cluster_selection_epsilon,
    )
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, folder, paths, request.cluster_selection_epsilon),
        daemon=True,
    )
    thread.start()

    return StartResponse(job_id=job_id)


@app.get("/progress/{job_id}", response_model=ProgressResponse)
def get_progress(job_id: str) -> ProgressResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job


@app.get("/image")
def get_image(path: str = Query(...)) -> FileResponse:
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {file_path}")
    if file_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path}")

    return FileResponse(file_path)
