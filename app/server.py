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
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import auth
from .auth import User
from .mp4 import faststart
from .comfy_client import ComfyClient, ComfyError, ComfyTimeout
from .workflows import ANIMATE, REPLACE, Adapter

OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs"))
UPLOAD_DIR = OUT_DIR / "uploads"
INPUT_DIR = OUT_DIR / "inputs"  # inputs of finished jobs, for the History tab
STATIC_DIR = Path(__file__).resolve().parent / "static"
# assets/gallery/<service>/<case>/{image.*, video.*, result.*, meta.json}; see assets/gallery/README.md.
GALLERY_DIR = Path(
    os.environ.get(
        "GALLERY_DIR", Path(__file__).resolve().parents[1] / "assets" / "gallery"
    )
)
# Stills and clips illustrating the Documentation tab. The tab's text lives in
# index.html; this directory only supplies the media it points at, so a missing
# directory degrades to a text-only tab rather than breaking it.
DOCS_DIR = Path(
    os.environ.get(
        "DOCS_DIR", Path(__file__).resolve().parents[1] / "assets" / "documentation"
    )
)

# Replacement costs ~3 min fixed plus ~3.3 s/frame (measured), so a full
# 1040-frame job runs about 60 min. Three hours leaves room for a cold model load
# and a queued job ahead of it without stranding the user on a hang.
TIMEOUTS = {"replace": 3 * 3600.0, "animate": 2 * 3600.0}

# Per backend: at ~1 h a job, a longer queue is a day of waiting nobody wants.
MAX_WAITING = 10
# Finished jobs kept for the status endpoint; their output files are not deleted.
KEEP_FINISHED = 200

# Warm-up frames cut from the start of the result AND of the saved input video,
# per service. 0 turns trimming off for that service. At the workflows' 24 fps
# these 6 frames are 0.25 s.
TRIM_FRAMES = {"replace": 10, "animate": 10}

BACKENDS = {
    "animate": ComfyClient(
        os.environ.get("ANIMATE_URL", "http://comfy-animate:8134"), name="animate"
    ),
    "replace": ComfyClient(
        os.environ.get("REPLACE_URL", "http://comfy-replace:8136"), name="replace"
    ),
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
    user_id: int = 0  # who submitted it; only they can see it or its result
    state: str = "queued"  # queued | running | done | error
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
EXECUTORS = {
    name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
    for name in BACKENDS
}


def _ahead(job: Job) -> int:
    """Jobs from this UI that will run on the same GPU before this one."""
    if job.state != "queued":
        return 0
    with LOCK:
        return sum(
            1
            for j in JOBS.values()
            if j.service == job.service
            and j.created < job.created
            and j.state in ("queued", "running")
        )


def _prune() -> None:
    with LOCK:
        done = sorted(
            (j for j in JOBS.values() if j.state in ("done", "error")),
            key=lambda j: j.created,
        )
        for j in done[:-KEEP_FINISHED]:
            del JOBS[j.id]


_NAME_LOCK = threading.Lock()


def _next_number(service: str, prefix: str) -> int:
    """
    The checker: look in ui_outputs/<service>/ (and the saved-inputs folder),
    find the highest number used by files named `<prefix>_<number>...`, and
    return that plus one. Looking at the inputs folder too means a number is
    never reused while any file from an older run still carries it.
    """
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)")
    used = [
        int(m.group(1))
        for folder in (OUT_DIR / service, INPUT_DIR / service)
        if folder.is_dir()
        for f in folder.iterdir()
        if (m := pat.match(f.name))
    ]
    return max(used, default=0) + 1


def _place_result(service: str, downloaded: Path, prefix: str) -> Path:
    """
    Move a freshly downloaded result into ui_outputs/<service>/ under the next
    free name (Cy_Replace_00013.mp4, ...). It can never overwrite an older file,
    whatever name ComfyUI gave it.
    """
    folder = OUT_DIR / service
    folder.mkdir(parents=True, exist_ok=True)
    with _NAME_LOCK:
        dest = (
            folder / f"{prefix}_{_next_number(service, prefix):05d}{downloaded.suffix}"
        )
        downloaded.replace(dest)  # same disk, so this is instant and atomic
    return dest


def _duration(path: Path) -> float | None:
    """Length of a video in seconds, or None if ffprobe cannot tell."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return None


def _fps(path: Path) -> float | None:
    """Frame rate of the first video stream, or None if ffprobe cannot tell."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception:  # noqa: BLE001
        return None


def _trim(src: Path, dst: Path, seconds: float) -> bool:
    """
    Write a copy of `src` without its first `seconds` to `dst` (src and dst may
    be the same path). Returns True on success. On any failure, or if the clip
    is too short to trim, the original is left untouched and False is returned,
    so a trim problem never fails a job that already cost an hour of GPU time.

    Takes seconds rather than a frame count on purpose: the result and the saved
    input have different frame rates (the workflow resamples the driving video to
    its own rate), so only a duration cuts the same moment out of both.
    """
    if seconds <= 0:
        return False
    dur = _duration(src)
    if dur is not None and dur <= seconds + 0.5:
        print(f"trim skipped for {src.name}: only {dur:.1f}s long")
        return False

    tmp = dst.with_name(dst.name + ".part")  # History ignores .part files
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ss",
        str(seconds),  # -ss after -i = frame-accurate
        # yuv420p needs even width/height; this changes nothing if already even
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        tmp.replace(dst)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"trim skipped for {src.name}: {e}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _trim_warmup(result: Path, frames: int) -> float:
    """
    Cut `frames` warm-up frames off the front of a finished result, in place.
    Returns the number of seconds actually removed, or 0.0 if nothing was cut.

    The frame count is read against the result's own frame rate, which is the rate
    the workflow renders at. That duration is what the caller reuses for the saved
    input video, whose native rate is usually different.
    """
    fps = _fps(result)
    if not fps:
        print(f"trim skipped for {result.name}: cannot read its frame rate")
        return 0.0
    seconds = frames / fps
    return seconds if _trim(result, result, seconds) else 0.0


def _execute(job: Job) -> None:
    adapter: Adapter = ADAPTERS[job.service]
    client = BACKENDS[job.service]
    job.state, job.started = "running", time.time()
    try:
        if not client.is_up():
            raise ComfyError(
                f"The {LABELS[job.service]} backend is not reachable "
                f"({client.base_url}). The other tab may still work."
            )

        job.stage = "Uploading inputs"
        image_name = client.upload(job.image)
        video_name = client.upload(job.video)

        # Unset in production: the workflow's own cap (1040 frames for replace)
        # decides length. Set MAX_FRAMES on a staging deployment to keep jobs short.
        cap = os.environ.get("MAX_FRAMES")
        workflow = adapter.build(
            image_name, video_name, frame_cap=int(cap) if cap else None
        )
        titles = client.node_titles(workflow)

        def on_progress(p):
            # No honest overall fraction exists: the graph is ~100 nodes of very
            # uneven cost. Report the live stage; the page adds elapsed time.
            job.stage = titles.get(p.node_id or "", "Working")
            job.step, job.steps = p.step, p.steps

        job.stage = "Submitting"
        result = client.run(
            workflow, timeout=TIMEOUTS[job.service], on_progress=on_progress
        )

        finals = [v for v in result.videos if v["node_id"] == adapter.output_node]
        if not finals:
            got = ", ".join(sorted({v["node_id"] for v in result.videos})) or "nothing"
            raise ComfyError(
                f"The job finished but produced no video on node "
                f"{adapter.output_node} (got: {got})."
            )

        job.stage, job.step, job.steps = "Downloading result", 0, 0
        incoming = OUT_DIR / "incoming" / job.id  # History never looks here
        downloaded = Path(client.download(finals[0], incoming))
        # Finish the file while it is still in the private holding folder, so
        # nothing untrimmed is ever visible in History.
        # How much was actually cut off the result, in seconds. The saved input is
        # then cut by the same duration, so the two line up in History.
        trimmed = 0.0
        if TRIM_FRAMES.get(job.service, 0) > 0:
            job.stage = "Trimming"
            trimmed = _trim_warmup(downloaded, TRIM_FRAMES[job.service])
        if not trimmed:
            faststart(downloaded)  # so the browser can start playing at once
            # (a successful trim already wrote faststart)
        # Only now does it get its final name and appear in ui_outputs/<service>/
        job.result = _place_result(job.service, downloaded, adapter.output_prefix)
        auth.set_owner(job.service, job.result.name, job.user_id)
        _keep_inputs(job, trimmed)  # History shows what produced each result
        job.state, job.stage = "done", "Done"
    except ComfyTimeout as e:
        job.state, job.error = "error", f"Timed out: {e}"
    except ComfyError as e:
        job.state, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001 - surfaced to the user
        traceback.print_exc()
        job.state, job.error = "error", f"Unexpected failure: {e}"
    finally:
        job.finished = time.time()
        shutil.rmtree(OUT_DIR / "incoming" / job.id, ignore_errors=True)
        for p in (job.image, job.video):  # already moved on success
            p.unlink(missing_ok=True)
        _prune()


def _keep_inputs(job: Job, trimmed: float = 0.0) -> None:
    """
    Save the uploads next to the result, named after it: <stem>__image.<ext>.

    `trimmed` is how many seconds came off the front of the result; the driving
    video is cut by the same duration so the pair stays in sync in History.
    """
    dest = INPUT_DIR / job.service
    dest.mkdir(parents=True, exist_ok=True)
    stem = job.result.stem
    for kind, src in (("image", job.image), ("video", job.video)):
        if not src.is_file():
            continue
        if kind == "video" and trimmed > 0:
            if _trim(src, dest / f"{stem}__video.mp4", trimmed):
                continue  # the original upload is deleted in _execute
        shutil.move(src, dest / f"{stem}__{kind}{src.suffix}")


def _history_input(service: str, stem: str, kind: str, exts: set[str]) -> Path | None:
    for ext in sorted(exts):
        f = INPUT_DIR / service / f"{stem}__{kind}{ext}"
        if f.is_file():
            return f
    return None


def _save(upload: UploadFile, allowed: set[str], kind: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            400,
            f"Unsupported {kind} type '{suffix or '?'}'. "
            f"Use one of: {', '.join(sorted(allowed))}.",
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, 1 << 20)
    return dest


# ── http ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="cy_hia", docs_url=None, redoc_url=None)


def current_user(request: Request) -> User:
    user = auth.user_for(request.cookies.get(auth.COOKIE))
    if user is None:
        raise HTTPException(401, "Please sign in.")
    return user


def _signed_in(user: User) -> JSONResponse:
    resp = JSONResponse({"username": user.username})
    resp.set_cookie(
        auth.COOKIE,
        auth.new_session(user),
        max_age=auth.SESSION_DAYS * 86400,
        httponly=True,
        samesite="lax",
        # Set COOKIE_SECURE=1 once the UI is served over HTTPS.
        secure=os.environ.get("COOKIE_SECURE") == "1",
    )
    return resp


@app.post("/api/auth/signup")
def signup(
    username: str = Body(...), password: str = Body(...), code: str = Body("")
):
    try:
        return _signed_in(auth.signup(username, password, code))
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e)) from None


@app.post("/api/auth/login")
def login(username: str = Body(...), password: str = Body(...)):
    try:
        return _signed_in(auth.login(username, password))
    except auth.AuthError as e:
        raise HTTPException(e.status, str(e)) from None


@app.post("/api/auth/logout")
def logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE)
    return resp


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"username": user.username}


@app.get("/api/auth/config")
def auth_config() -> dict:
    """What the sign-in page should offer."""
    return {"signup": auth.ALLOW_SIGNUP, "code": bool(auth.SIGNUP_CODE)}


@app.get("/api/status")
def status(_: User = Depends(current_user)) -> dict:
    out = {}
    for name, client in BACKENDS.items():
        if not client.is_up():
            out[name] = {
                "state": "offline",
                "label": LABELS[name],
                "detail": f"offline ({client.base_url})",
            }
            continue
        running, pending = client.queue_depth()
        out[name] = {
            "state": "busy" if running else "ready",
            "label": LABELS[name],
            "detail": (
                (f"busy, {pending} queued" if pending else "busy")
                if running
                else "ready"
            ),
        }
    return out


@app.post("/api/{service}/jobs")
def create_job(
    service: str,
    image: UploadFile = File(...),
    video: UploadFile = File(...),
    user: User = Depends(current_user),
) -> dict:
    if service not in BACKENDS:
        raise HTTPException(404, f"Unknown service '{service}'.")
    with LOCK:
        waiting = sum(
            1
            for j in JOBS.values()
            if j.service == service and j.state in ("queued", "running")
        )
    if waiting >= MAX_WAITING:
        raise HTTPException(
            429,
            f"{waiting} jobs are already waiting for "
            f"{LABELS[service]}. Try again later.",
        )
    img = _save(image, IMAGE_EXT, "image")
    try:
        vid = _save(video, VIDEO_EXT, "video")
    except HTTPException:
        img.unlink(missing_ok=True)
        raise

    job = Job(
        id=uuid.uuid4().hex,
        service=service,
        image=img,
        video=vid,
        user_id=user.id,
        created=time.time(),
    )
    with LOCK:
        JOBS[job.id] = job
    EXECUTORS[service].submit(_execute, job)
    return job.public()


def _job(job_id: str, user: User) -> Job:
    job = JOBS.get(job_id)
    # Someone else's job is reported exactly like a missing one.
    if job is None or job.user_id != user.id:
        raise HTTPException(404, "Unknown job — the UI server may have restarted.")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user)) -> dict:
    return _job(job_id, user).public()


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str, user: User = Depends(current_user)):
    job = _job(job_id, user)
    if not job.result or not job.result.is_file():
        raise HTTPException(404, "No result for this job.")
    return FileResponse(
        job.result,
        media_type="video/mp4",
        filename=job.result.name,
        content_disposition_type="inline",
    )


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
            out[service].append(
                {
                    "title": str(meta.get("title") or case.name.replace("_", " ")),
                    "note": str(meta.get("note") or ""),
                    "image": find("image", IMAGE_EXT),
                    "video": find("video", VIDEO_EXT),
                    "result": result,
                }
            )
    return out


@app.get("/api/gallery")
def gallery(_: User = Depends(current_user)) -> dict:
    return _gallery()


@app.get("/api/history")
def history(user: User = Depends(current_user)) -> list[dict]:
    """The caller's finished results on disk, newest first. Read from OUT_DIR
    rather than JOBS, so it survives a restart of the UI container; ownership
    comes from the auth database."""
    mine = auth.owned(user.id)
    rows = []
    for service in BACKENDS:
        for f in (OUT_DIR / service).glob("*"):
            if (
                f.is_file()
                and f.suffix.lower() in VIDEO_EXT
                and (service, f.name) in mine
            ):
                st = f.stat()
                row = {
                    "service": service,
                    "name": f.name,
                    "size": st.st_size,
                    "finished": st.st_mtime,
                    "url": f"/api/history/{service}/{quote(f.name)}",
                    "image_url": None,
                    "video_url": None,
                }
                for kind, exts in (("image", IMAGE_EXT), ("video", VIDEO_EXT)):
                    if _history_input(service, f.stem, kind, exts):
                        row[f"{kind}_url"] = (
                            f"/api/history/{service}/" f"{quote(f.name)}/input/{kind}"
                        )
                rows.append(row)
    return sorted(rows, key=lambda r: r["finished"], reverse=True)


@app.get("/api/history/{service}/{name}")
def history_video(service: str, name: str, user: User = Depends(current_user)):
    if service not in BACKENDS:
        raise HTTPException(404, f"Unknown service '{service}'.")
    root = (OUT_DIR / service).resolve()
    f = (root / name).resolve()
    if (
        f.parent != root
        or not f.is_file()
        or f.suffix.lower() not in VIDEO_EXT
        or not auth.owns(user.id, service, f.name)
    ):
        raise HTTPException(404, "No such result.")
    return FileResponse(
        f, media_type="video/mp4", filename=f.name, content_disposition_type="inline"
    )


@app.get("/api/history/{service}/{name}/input/{kind}")
def history_input(
    service: str, name: str, kind: str, user: User = Depends(current_user)
):
    if (
        service not in BACKENDS
        or kind not in ("image", "video")
        or not auth.owns(user.id, service, name)
    ):
        raise HTTPException(404, "No such input.")
    f = _history_input(
        service, Path(name).stem, kind, IMAGE_EXT if kind == "image" else VIDEO_EXT
    )
    if f is None:
        raise HTTPException(404, "No such input.")
    return FileResponse(f, content_disposition_type="inline")


@app.get("/")
def index(request: Request):
    if auth.user_for(request.cookies.get(auth.COOKIE)) is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
def login_page(request: Request):
    if auth.user_for(request.cookies.get(auth.COOKIE)) is not None:
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


if GALLERY_DIR.is_dir():
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")
if DOCS_DIR.is_dir():
    app.mount("/docs-assets", StaticFiles(directory=DOCS_DIR), name="docs-assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    import uvicorn

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auth.init()
    # Half-finished downloads from a container that was killed mid-job. The
    # per-job cleanup in _execute cannot run in that case, and nothing else
    # ever looks in here, so the files would sit on disk forever.
    shutil.rmtree(OUT_DIR / "incoming", ignore_errors=True)
    if any(TRIM_FRAMES.values()) and not (
        shutil.which("ffmpeg") and shutil.which("ffprobe")
    ):
        print(
            "WARNING: ffmpeg/ffprobe not found in this container: results will "
            "NOT be trimmed. Install ffmpeg in the UI image or set TRIM_FRAMES to 0."
        )
    for f in OUT_DIR.glob("*/*.mp4"):  # results saved before faststart was added
        try:
            faststart(f)
        except (OSError, ValueError) as e:
            print(f"faststart skipped for {f.name}: {e}")
    for name, client in BACKENDS.items():
        print(f"{name:>8}: {client.base_url}  {'up' if client.is_up() else 'DOWN'}")
    uvicorn.run(
        app,
        host=os.environ.get("UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("UI_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
