"""
cy_hia — one HTML page over two ComfyUI GPU backends.

  Animate tab  -> comfy-animate (GPU 1)  WAN 2.2 Animate
  Replace tab  -> comfy-replace (GPU 0)  SCAIL-2 character replacement

A small FastAPI server holds the jobs; the page (static/index.html) uploads,
then polls. Each backend gets its own single-worker executor, so the two GPUs
work in parallel while jobs *for one GPU* run strictly one at a time — a second
concurrent ComfyUI job would only fight for VRAM.

Jobs live in memory. A page reload resumes tracking (the page remembers its job
ids), but restarting this server forgets running jobs — ComfyUI keeps executing
them and the result still lands in that backend's output/ dir.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .comfy_client import ComfyClient, ComfyError, ComfyTimeout
from .workflows import ANIMATE, REPLACE, Adapter

OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs"))
UPLOAD_DIR = OUT_DIR / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"
# assets/gallery/<service>/<case>/{image.*, video.*, result.*, meta.json}; see assets/gallery/README.md.
GALLERY_DIR = Path(os.environ.get("GALLERY_DIR",
                                  Path(__file__).resolve().parents[1] / "assets" / "gallery"))

# Replacement costs ~3 min fixed plus ~3.3 s/frame (measured), so a full
# 1140-frame job runs 65-70 min. Three hours leaves room for a cold model load
# and a queued job ahead of it without stranding the user on a hang.
TIMEOUTS = {"replace": 3 * 3600.0, "animate": 2 * 3600.0}

# Per backend: at ~1 h a job, a longer queue is a day of waiting nobody wants.
MAX_WAITING = 10
# Finished jobs kept for the status endpoint; their output files are not deleted.
KEEP_FINISHED = 200

BACKENDS = {
    "animate": ComfyClient(os.environ.get("ANIMATE_URL", "http://comfy-animate:8134"),
                           name="animate"),
    "replace": ComfyClient(os.environ.get("REPLACE_URL", "http://comfy-replace:8136"),
                           name="replace"),
}
ADAPTERS = {"animate": ANIMATE, "replace": REPLACE}
LABELS = {"animate": "Animate (GPU 1)", "replace": "Replace (GPU 0)"}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ── jobs ─────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    id: str
    service: str
    image: Path
    video: Path
    state: str = "queued"            # queued | running | done | error
    stage: str = "Waiting for the GPU"
    step: int = 0
    steps: int = 0
    created: float = 0.0
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: Path | None = None

    def public(self) -> dict:
        now = time.time()
        return {
            "id": self.id,
            "service": self.service,
            "state": self.state,
            "stage": self.stage,
            "step": self.step,
            "steps": self.steps,
            # Seconds rather than timestamps, so client clock skew is irrelevant.
            "elapsed": ((self.finished or now) - self.started) if self.started else 0.0,
            "ahead": _ahead(self),
            "error": self.error,
            "result": f"/api/jobs/{self.id}/video" if self.result else None,
            "filename": self.result.name if self.result else None,
        }


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()
EXECUTORS = {name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
             for name in BACKENDS}


def _ahead(job: Job) -> int:
    """Jobs from this UI that will run on the same GPU before this one."""
    if job.state != "queued":
        return 0
    with LOCK:
        return sum(1 for j in JOBS.values()
                   if j.service == job.service and j.created < job.created
                   and j.state in ("queued", "running"))


def _prune() -> None:
    with LOCK:
        done = sorted((j for j in JOBS.values() if j.state in ("done", "error")),
                      key=lambda j: j.created)
        for j in done[:-KEEP_FINISHED]:
            del JOBS[j.id]


def _execute(job: Job) -> None:
    adapter: Adapter = ADAPTERS[job.service]
    client = BACKENDS[job.service]
    job.state, job.started = "running", time.time()
    try:
        if not client.is_up():
            raise ComfyError(f"The {LABELS[job.service]} backend is not reachable "
                             f"({client.base_url}). The other tab may still work.")

        job.stage = "Uploading inputs"
        image_name = client.upload(job.image)
        video_name = client.upload(job.video)

        # Unset in production: the workflow's own cap (1140 frames for replace)
        # decides length. Set MAX_FRAMES on a staging deployment to keep jobs short.
        cap = os.environ.get("MAX_FRAMES")
        workflow = adapter.build(image_name, video_name,
                                 frame_cap=int(cap) if cap else None)
        titles = client.node_titles(workflow)

        def on_progress(p):
            # No honest overall fraction exists: the graph is ~100 nodes of very
            # uneven cost. Report the live stage; the page adds elapsed time.
            job.stage = titles.get(p.node_id or "", "Working")
            job.step, job.steps = p.step, p.steps

        job.stage = "Submitting"
        result = client.run(workflow, timeout=TIMEOUTS[job.service],
                            on_progress=on_progress)

        finals = [v for v in result.videos if v["node_id"] == adapter.output_node]
        if not finals:
            got = ", ".join(sorted({v["node_id"] for v in result.videos})) or "nothing"
            raise ComfyError(f"The job finished but produced no video on node "
                             f"{adapter.output_node} (got: {got}).")

        job.stage, job.step, job.steps = "Downloading result", 0, 0
        job.result = Path(client.download(finals[0], OUT_DIR / job.service))
        job.state, job.stage = "done", "Done"
    except ComfyTimeout as e:
        job.state, job.error = "error", f"Timed out: {e}"
    except ComfyError as e:
        job.state, job.error = "error", str(e)
    except Exception as e:                       # noqa: BLE001 - surfaced to the user
        traceback.print_exc()
        job.state, job.error = "error", f"Unexpected failure: {e}"
    finally:
        job.finished = time.time()
        for p in (job.image, job.video):
            p.unlink(missing_ok=True)
        _prune()


def _save(upload: UploadFile, allowed: set[str], kind: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported {kind} type '{suffix or '?'}'. "
                                 f"Use one of: {', '.join(sorted(allowed))}.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, 1 << 20)
    return dest


# ── http ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="cy_hia", docs_url=None, redoc_url=None)


@app.get("/api/status")
def status() -> dict:
    out = {}
    for name, client in BACKENDS.items():
        if not client.is_up():
            out[name] = {"state": "offline", "label": LABELS[name],
                         "detail": f"offline ({client.base_url})"}
            continue
        running, pending = client.queue_depth()
        out[name] = {"state": "busy" if running else "ready", "label": LABELS[name],
                     "detail": (f"busy, {pending} queued" if pending else "busy")
                     if running else "ready"}
    return out


@app.post("/api/{service}/jobs")
def create_job(service: str, image: UploadFile = File(...),
               video: UploadFile = File(...)) -> dict:
    if service not in BACKENDS:
        raise HTTPException(404, f"Unknown service '{service}'.")
    with LOCK:
        waiting = sum(1 for j in JOBS.values()
                      if j.service == service and j.state in ("queued", "running"))
    if waiting >= MAX_WAITING:
        raise HTTPException(429, f"{waiting} jobs are already waiting for "
                                 f"{LABELS[service]}. Try again later.")
    img = _save(image, IMAGE_EXT, "image")
    try:
        vid = _save(video, VIDEO_EXT, "video")
    except HTTPException:
        img.unlink(missing_ok=True)
        raise

    job = Job(id=uuid.uuid4().hex, service=service, image=img, video=vid,
              created=time.time())
    with LOCK:
        JOBS[job.id] = job
    EXECUTORS[service].submit(_execute, job)
    return job.public()


def _job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job — the UI server may have restarted.")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job(job_id).public()


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job = _job(job_id)
    if not job.result or not job.result.is_file():
        raise HTTPException(404, "No result for this job.")
    return FileResponse(job.result, media_type="video/mp4", filename=job.result.name,
                        content_disposition_type="inline")


def _gallery() -> dict[str, list[dict]]:
    """Scan GALLERY_DIR on every call, so cases can be added without a restart."""
    out: dict[str, list[dict]] = {name: [] for name in BACKENDS}
    for service in out:
        for case in sorted(p for p in (GALLERY_DIR / service).glob("*") if p.is_dir()):
            def find(stem: str, exts: set[str]) -> str | None:
                for f in sorted(case.glob(f"{stem}.*")):
                    if f.suffix.lower() in exts:
                        return f"/gallery/{service}/{case.name}/{f.name}"
                return None
            result = find("result", VIDEO_EXT)
            if not result:
                continue
            meta = {}
            try:
                meta = json.loads((case / "meta.json").read_text())
            except (OSError, ValueError):
                pass
            out[service].append({
                "title": str(meta.get("title") or case.name.replace("_", " ")),
                "note": str(meta.get("note") or ""),
                "image": find("image", IMAGE_EXT),
                "video": find("video", VIDEO_EXT),
                "result": result,
            })
    return out


@app.get("/api/gallery")
def gallery() -> dict:
    return _gallery()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if GALLERY_DIR.is_dir():
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    import uvicorn

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, client in BACKENDS.items():
        print(f"{name:>8}: {client.base_url}  {'up' if client.is_up() else 'DOWN'}")
    uvicorn.run(app, host=os.environ.get("UI_HOST", "0.0.0.0"),
                port=int(os.environ.get("UI_PORT", "7860")))


if __name__ == "__main__":
    main()
