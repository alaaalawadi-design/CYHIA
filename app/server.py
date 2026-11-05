"""
cy_hia — one HTML page over any number of ComfyUI GPU backends.

Which workflows exist (one tab each) comes from workflows/workflows.json; which
ComfyUI servers run them, and on which GPU, from generated/services.json, which
./start.sh writes after counting the GPUs. Without that file the old two-GPU
layout is used (replace on GPU 0, animate on GPU 1), so the hand-written
docker-compose.yml still works.

Scheduling: every job goes into ONE waiting list, in the order users submitted
it. Each GPU has one worker thread, which takes the oldest waiting job that a
server on its GPU can run. So:
  - one GPU shared by several workflows runs their jobs one at a time, strictly
    in submission order, whatever the tab;
  - separate GPUs run in parallel, and a job never waits behind a job for a
    different GPU;
  - a workflow on several GPUs runs that many of its jobs at once.
Before a job starts on a GPU that another server last used, that other server
is told to unload its models, so the job gets the whole card. And after every
job the server that ran it unloads its own models (free_after_run in
workflows.json), so an idle GPU holds no weights.

Jobs live in memory. A page reload resumes tracking (the page asks the server
for the user's jobs), but restarting this server forgets them — ComfyUI keeps
executing a running one and its result still lands in that backend's output/.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import auth, pipelines, workflow_settings
from .auth import User
from .mp4 import faststart
from .comfy_client import ComfyClient, ComfyError, ComfyTimeout
from .workflows import ADAPTERS, Adapter

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

# ── limits ───────────────────────────────────────────────────────────────────
# How many jobs ONE user may have waiting or running in ONE tab at the same
# time. Pressing Generate again beyond this is refused until one finishes.
MAX_PER_USER = 5
# How many jobs ALL users together may have waiting or running in one tab.
# At ~1 h a replacement job, a longer queue is a day of waiting nobody wants.
MAX_WAITING = 10
# Finished jobs kept for the status endpoint; their output files are not deleted.
KEEP_FINISHED = 200
# After a server is told to unload (before a job starts on a GPU another server
# used last, and after every job), wait up to this many seconds for the memory
# to come back...
FREE_VRAM_WAIT = 60.0
# ...until at least this share of the card is free.
FREE_VRAM_SHARE = 0.85

# Timeouts and warm-up trimming are per workflow: see workflows/workflows.json.

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
# Script pipelines whose main upload is audio (lipsync). ComfyUI tabs never use these.
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
MEDIA_EXT = {"video": VIDEO_EXT, "audio": AUDIO_EXT}


# ── backends: which ComfyUI server runs which workflow on which GPU ──────────
SERVICES_FILE = Path(
    os.environ.get(
        "SERVICES_FILE",
        Path(__file__).resolve().parents[1] / "generated" / "services.json",
    )
)


@dataclass
class Backend:
    name: str  # compose service name: comfy-replace, comfy-replace-2, ...
    workflow: str  # key in workflows.json
    gpu: int  # host GPU index
    client: ComfyClient
    up: bool = False  # refreshed every few seconds by _health_loop


def _load_backends() -> tuple[list[Backend], list[dict], str]:
    """Backends from generated/services.json, else the old fixed layout."""
    if SERVICES_FILE.is_file():
        spec = json.loads(SERVICES_FILE.read_text())
        rows, gpu_info, source = spec["backends"], spec.get("gpus", []), str(SERVICES_FILE)
    else:
        # No start.sh run: one server per workflow at its default GPU and URL,
        # overridable per workflow with <NAME>_URL / <NAME>_GPU.
        rows = [
            {
                "name": f"comfy-{n}",
                "workflow": n,
                "gpu": int(os.environ.get(f"{n.upper()}_GPU", a.default_gpu)),
                "url": os.environ.get(f"{n.upper()}_URL", f"http://comfy-{n}:{a.port}"),
            }
            for n, a in ADAPTERS.items()
        ]
        gpu_info, source = [], "built-in defaults (no generated/services.json)"
    out = []
    for r in rows:
        if r["workflow"] not in ADAPTERS:
            print(f"WARNING: backend {r['name']} runs unknown workflow {r['workflow']!r}; ignored")
            continue
        out.append(
            Backend(r["name"], r["workflow"], int(r["gpu"]), ComfyClient(r["url"], name=r["name"]))
        )
    return out, gpu_info, source


BACKENDS, GPU_INFO, BACKENDS_SOURCE = _load_backends()

# Script pipelines (app/pipelines.py, workflows/pipelines.json): tabs whose
# backend runs a script, not ComfyUI. They are added after the ComfyUI tabs are
# built, so those are exactly as before; the new tabs join the same waiting
# list and GPU workers. No pipelines.json = nothing added.
ADAPTERS = {**ADAPTERS, **pipelines.ADAPTERS}
BACKENDS += [Backend(a.backend, a.name, a.gpu, c) for a, c in pipelines.clients()]
# Workflows that have at least one server: these get a tab. Catalog order.
SERVICES = [n for n in ADAPTERS if any(b.workflow == n for b in BACKENDS)]
BY_WORKFLOW = {n: [b for b in BACKENDS if b.workflow == n] for n in SERVICES}
GPUS = {g: [b for b in BACKENDS if b.gpu == g] for g in sorted({b.gpu for b in BACKENDS})}
GPUS_FOR = {n: {b.gpu for b in bs} for n, bs in BY_WORKFLOW.items()}


def _label(service: str) -> str:
    """The tab's name. Which GPU it runs on, and who shares it, is not shown to users."""
    return ADAPTERS[service].label


# ── jobs ─────────────────────────────────────────────────────────────────────
# Node titles are shown to users as the live stage; keep model names out of them.
_STAGE_NAMES = [
    (r"SCAIL.*(Infinity|window)|WanAnimateToVideo|Wan.*Animate", "Generating video"),
    (r"Colored Mask", "Preparing mask"),
    (r"SAM\s?3", "Tracking subject"),
    (r"Qwen", "Analysing lighting"),
    (r"Flux.*Prompt", "Analysing pose"),
    (r"Empty Flux", "Preparing canvas"),
    (r"LTXV.*(Decode)", "Decoding video"),
    (r"LTXV.*Context", "Restoring video"),
]
_STAGE_SCRUB = re.compile(r"[\U0001F150-\U0001F1FF\U0001F3A5]+|\b(?:wan|ltxv?|scail-?\d*|flux ?\d*|klein|qwen\w*|sam ?\d|vitpose)\b", re.I)


def _public_stage(title: str) -> str:
    for pat, name in _STAGE_NAMES:
        if re.search(pat, title, re.I):
            return name
    title = re.sub(r"dwpose", "Pose", title, flags=re.I)
    return re.sub(r"\s+", " ", _STAGE_SCRUB.sub("", title)).strip(" -·") or "Working"


@dataclass
class Job:
    id: str
    service: str
    image: Path | None  # None for video-only workflows (restore)
    video: Path
    user_id: int = 0  # who submitted it; only they can see it or its result
    seq: int = 0  # submission order across ALL tabs; the queue runs by this
    state: str = "queued"  # queued | running | done | error
    stage: str = "Waiting for the GPU"
    step: int = 0
    steps: int = 0
    created: float = 0.0
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: Path | None = None
    settings: dict | None = None  # admin overrides this job ran with
    backend: str | None = None  # server that ran it
    gpu: int | None = None
    dismissed: bool = False  # the user removed it from their job list
    prompt: str | None = None  # script pipelines only (lipsync); None for ComfyUI tabs
    audio: Path | None = None  # script pipelines only: extra audio upload (dubbing)

    def public(self) -> dict:
        now = time.time()
        return {
            "id": self.id,
            "service": self.service,
            "seq": self.seq,
            "state": self.state,
            "stage": self.stage,
            "step": self.step,
            "steps": self.steps,
            # Seconds rather than timestamps, so client clock skew is irrelevant.
            "elapsed": ((self.finished or now) - self.started) if self.started else 0.0,
            "waited": ((self.started or now) - self.created),
            "ahead": _ahead(self),
            "gpu": self.gpu,
            "error": self.error,
            "result": f"/api/jobs/{self.id}/video" if self.result else None,
            "filename": self.result.name if self.result else None,
        }


JOBS: dict[str, Job] = {}
# One lock for JOBS and the queue. A Condition so GPU workers can sleep until a
# job is added or one finishes; re-entrant because public() takes it again.
LOCK = threading.Condition(threading.RLock())
_SEQ = itertools.count(1)
ACTIVE = ("queued", "running")


def _ahead(job: Job) -> int:
    """
    Jobs submitted earlier that will use one of this job's GPUs first, from
    every tab and every user: on a shared GPU an Animate job ahead of a Replace
    job counts, on separate GPUs it does not.
    """
    if job.state != "queued":
        return 0
    mine = GPUS_FOR.get(job.service, set())
    with LOCK:
        return sum(
            1
            for j in JOBS.values()
            if j.seq < job.seq
            and (
                (j.state == "running" and j.gpu in mine)
                or (j.state == "queued" and GPUS_FOR.get(j.service, set()) & mine)
            )
        )


# ── the scheduler: one worker per GPU over one shared waiting list ───────────
def _fail(job: Job, message: str) -> None:
    job.state, job.error, job.finished = "error", message, time.time()
    for p in (job.image, job.video, job.audio):
        if p:
            p.unlink(missing_ok=True)


def _pick(gpu: int) -> tuple[Job, Backend] | None:
    """The oldest waiting job a live server on `gpu` can run. Caller holds LOCK."""
    for job in sorted((j for j in JOBS.values() if j.state == "queued"), key=lambda j: j.seq):
        if not any(b.up for b in BY_WORKFLOW.get(job.service, [])):
            # Nothing anywhere can run it; failing it now keeps it from
            # holding up the jobs behind it.
            _fail(
                job,
                f"The {_label(job.service)} server is not reachable. "
                "Other tabs may still work.",
            )
            continue
        here = [b for b in GPUS[gpu] if b.workflow == job.service and b.up]
        if here:
            return job, here[0]
        # Otherwise a server on another GPU will take it: look further down.
    return None


def _wait_for_vram(backend: Backend) -> tuple[bool, tuple[int, int] | None]:
    """Wait up to FREE_VRAM_WAIT for FREE_VRAM_SHARE of `backend`'s GPU to be free.
    Returns (reached, last reading); the reading is None if it cannot tell."""
    deadline = time.time() + FREE_VRAM_WAIT
    while True:
        v = backend.client.vram()
        if v is None:
            return False, None
        if v[0] >= FREE_VRAM_SHARE * v[1]:
            return True, v
        if time.time() >= deadline:
            return False, v
        time.sleep(2)


def _gib(v: tuple[int, int]) -> str:
    return f"{v[0] / 2**30:.1f} of {v[1] / 2**30:.1f} GiB free"


def _make_room(gpu: int, backend: Backend, job: Job) -> None:
    """Have every other server on this GPU unload its models before `job` runs."""
    others = [b for b in GPUS[gpu] if b is not backend]
    if not others:
        return
    job.stage = "Freeing GPU memory"
    for b in others:
        b.client.free()
    ok, v = _wait_for_vram(backend)
    if not ok:
        print(
            f"GPU {gpu}: only {_gib(v)} after {FREE_VRAM_WAIT:.0f}s; "
            f"starting {job.service} job {job.id} anyway"
            if v
            else f"GPU {gpu}: cannot read free memory; starting job {job.id} anyway"
        )


def _release(gpu: int, backend: Backend, job: Job) -> None:
    """
    After `job`, have the server that ran it unload its models and drop its
    caches, so the GPU sits empty until the next job. Runs on the GPU worker,
    after the job is already marked done, so it never delays a result. A
    timed-out job may still be running inside ComfyUI; /free then waits for it.
    """
    before = backend.client.vram()
    if not backend.client.free():
        print(f"GPU {gpu}: {backend.name} did not accept /free after job {job.id}")
        return
    ok, v = _wait_for_vram(backend)
    if ok and before and v:
        print(
            f"GPU {gpu}: {backend.name} released "
            f"{(v[0] - before[0]) / 2**30:.1f} GiB after job {job.id}; {_gib(v)}"
        )
    elif v:
        # The reading is for the whole card, so it may be another server holding it.
        print(
            f"GPU {gpu}: only {_gib(v)} {FREE_VRAM_WAIT:.0f}s after "
            f"{backend.name} was told to unload (job {job.id})"
        )


def _gpu_worker(gpu: int) -> None:
    last: str | None = None  # server that ran the previous job on this GPU
    while True:
        try:
            with LOCK:
                pick = _pick(gpu)
                while pick is None:
                    LOCK.wait(timeout=5.0)
                    pick = _pick(gpu)
                job, backend = pick
                job.state, job.started = "running", time.time()
                job.backend, job.gpu = backend.name, gpu
                job.stage = "Starting"
            # `last` is None after a restart: the other server may still hold
            # memory from before, so free it then too.
            if last != backend.name:
                _make_room(gpu, backend, job)
            last = backend.name
            _execute(job, backend)
            if ADAPTERS[job.service].free_after_run:
                _release(gpu, backend, job)
        except Exception:  # noqa: BLE001 - a worker must never die
            traceback.print_exc()
            time.sleep(1)
        finally:
            with LOCK:
                LOCK.notify_all()


def _health_loop() -> None:
    while True:
        for b in BACKENDS:
            b.up = b.client.is_up()
        with LOCK:
            LOCK.notify_all()  # a server that came back may unblock a GPU
        time.sleep(10)


def _start_scheduler() -> None:
    for b in BACKENDS:  # before any worker looks at .up
        b.up = b.client.is_up()
    threading.Thread(target=_health_loop, name="health", daemon=True).start()
    for gpu in GPUS:
        threading.Thread(target=_gpu_worker, args=(gpu,), name=f"gpu{gpu}", daemon=True).start()


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


def _uses_video_audio(adapter: Adapter) -> bool:
    """True if some node reads the driving video's audio output (VHS output 2)."""
    try:
        wf = adapter.load()
    except Exception:  # noqa: BLE001
        return False
    for node in wf.values():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == adapter.video_node and v[1] == 2:
                return True
    return False


def _ensure_audio(path: Path) -> None:
    """
    VHS_LoadVideo fails ("Output file does not contain any stream") when asked
    for the audio of a video that has none. Give such a video a silent track
    (video stream copied, so this is fast). Never fails a job: on any error the
    original file is left as it was.
    """
    try:
        has_audio = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.strip()
        if has_audio:
            return
        tmp = path.with_name(path.stem + ".silent" + path.suffix)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
             "-shortest", "-movflags", "+faststart", str(tmp)],
            capture_output=True, check=True, timeout=300,
        )
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        print(f"could not add silent audio to {path.name}: {exc}")


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


def _execute(job: Job, backend: Backend) -> None:
    """Run one job on `backend`. The GPU worker has already marked it running."""
    adapter: Adapter = ADAPTERS[job.service]
    client = backend.client
    try:
        if not client.is_up():
            raise ComfyError(
                f"The {_label(job.service)} server is not reachable "
                f"({client.base_url}). Other tabs may still work."
            )

        job.stage = "Uploading inputs"
        image_name = client.upload(job.image) if adapter.needs_image else None
        if _uses_video_audio(adapter):
            _ensure_audio(job.video)
        video_name = client.upload(job.video)

        # Unset in production: the workflow's own cap (1040 frames for replace)
        # decides length. Set MAX_FRAMES on a staging deployment to keep jobs short.
        cap = os.environ.get("MAX_FRAMES")
        job.settings = workflow_settings.overrides(job.service)
        # Only script pipelines take a prompt; ComfyUI tabs are built exactly as before.
        extra = {"prompt": job.prompt} if job.prompt is not None else {}
        if job.audio is not None:  # dubbing: the extra audio upload
            extra["audio_name"] = client.upload(job.audio)
        workflow = adapter.build(
            image_name, video_name, frame_cap=int(cap) if cap else None, **extra
        )
        titles = client.node_titles(workflow)

        def on_progress(p):
            # No honest overall fraction exists: the graph is ~100 nodes of very
            # uneven cost. Report the live stage; the page adds elapsed time.
            job.stage = _public_stage(titles.get(p.node_id or "", "Working"))
            job.step, job.steps = p.step, p.steps

        job.stage = "Submitting"
        result = client.run(workflow, timeout=adapter.timeout, on_progress=on_progress)

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
        if adapter.trim_frames > 0:
            job.stage = "Trimming"
            trimmed = _trim_warmup(downloaded, adapter.trim_frames)
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
        for p in (job.image, job.video, job.audio):  # already moved on success
            if p:
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
    if job.settings:
        # Which admin overrides produced this result; absent = workflow as saved.
        (dest / f"{stem}__settings.json").write_text(
            json.dumps(job.settings, indent=2, ensure_ascii=False)
        )
    if job.prompt is not None:  # script pipelines: the prompt, for History
        (dest / f"{stem}__prompt.txt").write_text(job.prompt, encoding="utf-8")
    # The main upload is saved as __audio.* for audio pipelines, __video.* otherwise.
    media = getattr(ADAPTERS[job.service], "media", "video")
    for kind, src in (("image", job.image), (media, job.video), ("audio", job.audio)):
        if src is None or not src.is_file():
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
@asynccontextmanager
async def _lifespan(_app):
    _start_scheduler()  # here, not at import, so importing this module is cheap
    yield


app = FastAPI(title="cy_hia", docs_url=None, redoc_url=None, lifespan=_lifespan)


def current_user(request: Request) -> User:
    user = auth.user_for(request.cookies.get(auth.COOKIE))
    if user is None:
        raise HTTPException(401, "Please sign in.")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admins only.")
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
    return {"username": user.username, "admin": user.is_admin}


@app.get("/api/auth/config")
def auth_config() -> dict:
    """What the sign-in page should offer."""
    return {"signup": auth.ALLOW_SIGNUP, "code": bool(auth.SIGNUP_CODE)}


@app.get("/api/services")
def services(_: User = Depends(current_user)) -> dict:
    """What the page builds its tabs from."""
    return {
        "max_per_user": MAX_PER_USER,
        "gpus": GPU_INFO or [{"index": g} for g in GPUS],
        "services": [
            {
                "name": n,
                "label": ADAPTERS[n].label,
                "blurb": ADAPTERS[n].blurb,
                "gallery_blurb": ADAPTERS[n].gallery_blurb,
                "needs_image": ADAPTERS[n].needs_image,
                # Script pipelines only; ComfyUI tabs get the defaults (video, no prompt).
                "media": getattr(ADAPTERS[n], "media", "video"),
                "needs_prompt": getattr(ADAPTERS[n], "needs_prompt", False),
                # A fixed prompt (needs_prompt false) is never sent to the page.
                "default_prompt": getattr(ADAPTERS[n], "default_prompt", "")
                if getattr(ADAPTERS[n], "needs_prompt", False) else "",
                "max_prompt_chars": getattr(ADAPTERS[n], "max_prompt_chars", 0),
                "needs_audio": getattr(ADAPTERS[n], "needs_audio", False),
                "max_input_minutes": getattr(ADAPTERS[n], "max_input_minutes", 0),
                "labels": getattr(ADAPTERS[n], "labels", {}),
                "pipeline": n in pipelines.ADAPTERS,  # script pipeline tab (Img/Video Lipsync)
                "notes": getattr(ADAPTERS[n], "notes", []),  # points under the description
                "gpus": sorted(GPUS_FOR[n]),
                "servers": len(BY_WORKFLOW[n]),
            }
            for n in SERVICES
        ],
        # Workflows with results on disk but no server right now: History only.
        "history_only": [
            {"name": n, "label": ADAPTERS[n].label} for n in ADAPTERS if n not in SERVICES
        ],
    }


@app.get("/api/status")
def status(_: User = Depends(current_user)) -> dict:
    with LOCK:
        running = [j for j in JOBS.values() if j.state == "running"]
        waiting = Counter(j.service for j in JOBS.values() if j.state == "queued")
    busy_gpus = {j.gpu: j for j in running}
    out = {}
    for name in SERVICES:
        servers = BY_WORKFLOW[name]
        live = [b for b in servers if b.up]
        if not live:
            out[name] = {
                "state": "offline",
                "label": _label(name),
                "detail": "offline",
            }
            continue
        idle = [b for b in live if b.gpu not in busy_gpus]
        if idle:
            detail = "ready"
        else:
            detail = "busy"
        if waiting[name]:
            detail += f" · {waiting[name]} waiting"
        out[name] = {
            "state": "ready" if idle else "busy",
            "label": _label(name),
            "detail": detail,
        }
    return out


def _count(service: str, user_id: int | None = None) -> int:
    """Jobs waiting or running in one tab, for one user or for everyone."""
    with LOCK:
        return sum(
            1
            for j in JOBS.values()
            if j.service == service
            and j.state in ACTIVE
            and (user_id is None or j.user_id == user_id)
        )


def _check_limits(service: str, user: User) -> None:
    mine = _count(service, user.id)
    if mine >= MAX_PER_USER:
        raise HTTPException(
            429,
            f"You already have {mine} {ADAPTERS[service].label} job"
            f"{'s' if mine != 1 else ''} waiting or running (the limit is "
            f"{MAX_PER_USER}). Generate again when one finishes.",
        )
    everyone = _count(service)
    if everyone >= MAX_WAITING:
        raise HTTPException(
            429,
            f"{everyone} jobs are already waiting for {_label(service)}. Try again later.",
        )


@app.post("/api/{service}/jobs")
def create_job(
    service: str,
    video: UploadFile = File(...),
    image: UploadFile | None = File(None),
    prompt: str | None = Form(None),  # script pipelines only; ignored by ComfyUI tabs
    audio: UploadFile | None = File(None),  # script pipelines with needs_audio (dubbing)
    user: User = Depends(current_user),
) -> dict:
    if service not in SERVICES:
        raise HTTPException(404, f"Unknown service '{service}'.")
    _check_limits(service, user)
    if not any(b.client.is_up() for b in BY_WORKFLOW[service]):
        raise HTTPException(
            503, f"The {_label(service)} server is not reachable. Other tabs may still work."
        )
    needs_image = ADAPTERS[service].needs_image
    if needs_image and image is None:
        raise HTTPException(400, f"{_label(service)} needs a reference image.")
    if getattr(ADAPTERS[service], "needs_prompt", False):
        prompt = (prompt or "").strip()
        limit = ADAPTERS[service].max_prompt_chars
        if not prompt:
            raise HTTPException(400, f"{_label(service)} needs a prompt.")
        if limit and len(prompt) > limit:
            raise HTTPException(400, f"The prompt is too long ({len(prompt)} characters, the limit is {limit}).")
    else:
        prompt = None
    # The main upload is a video for ComfyUI tabs, audio for audio pipelines.
    media = getattr(ADAPTERS[service], "media", "video")
    needs_audio = getattr(ADAPTERS[service], "needs_audio", False)
    if needs_audio and audio is None:
        raise HTTPException(400, f"{_label(service)} needs an audio file.")
    img = _save(image, IMAGE_EXT, "image") if needs_image else None
    vid = aud = None
    try:
        vid = _save(video, MEDIA_EXT[media], media)
        if needs_audio:  # dubbing: video + audio; ComfyUI tabs never get here
            aud = _save(audio, AUDIO_EXT, "audio")
    except HTTPException:
        for p in (img, vid):
            if p:
                p.unlink(missing_ok=True)
        raise

    with LOCK:
        # Again under the lock: two uploads finishing at once must not both
        # squeeze past the limits checked above.
        try:
            _check_limits(service, user)
        except HTTPException:
            for p in (img, vid, aud):
                if p:
                    p.unlink(missing_ok=True)
            raise
        job = Job(
            id=uuid.uuid4().hex,
            service=service,
            image=img,
            video=vid,
            user_id=user.id,
            seq=next(_SEQ),
            created=time.time(),
            prompt=prompt,
            audio=aud,
        )
        JOBS[job.id] = job
        LOCK.notify_all()  # wake the GPU workers
    return job.public()


@app.get("/api/jobs")
def my_jobs(user: User = Depends(current_user)) -> list[dict]:
    """The caller's jobs this server still knows about, oldest first: everything
    waiting or running, plus finished ones the user has not removed."""
    with LOCK:
        mine = sorted(
            (j for j in JOBS.values() if j.user_id == user.id and not j.dismissed),
            key=lambda j: j.seq,
        )
        return [j.public() for j in mine]


@app.post("/api/jobs/{job_id}/dismiss")
def dismiss_job(job_id: str, user: User = Depends(current_user)) -> dict:
    """Remove a finished job from the caller's list. Its result stays in History."""
    job = _job(job_id, user)
    if job.state in ACTIVE:
        raise HTTPException(409, "That job has not finished yet.")
    job.dismissed = True
    return {"ok": True}


# ── admin: workflow controls ─────────────────────────────────────────────────
def _settings_view(service: str) -> dict:
    if service not in workflow_settings.CONTROLS:
        raise HTTPException(404, f"No editable controls for '{service}'.")
    view = workflow_settings.describe(service, ADAPTERS[service].load())
    view["label"] = ADAPTERS[service].label
    return view


@app.get("/api/admin/workflows")
def admin_workflows(_: User = Depends(admin_user)) -> list[dict]:
    return [
        {"service": s, "label": ADAPTERS[s].label}
        for s in workflow_settings.CONTROLS
        if s in ADAPTERS
    ]


@app.get("/api/admin/workflows/{service}")
def admin_get_workflow(service: str, _: User = Depends(admin_user)) -> dict:
    return _settings_view(service)


@app.put("/api/admin/workflows/{service}")
def admin_put_workflow(
    service: str, values: dict = Body(..., embed=True), user: User = Depends(admin_user)
) -> dict:
    _settings_view(service)  # 404 for an unknown service
    try:
        workflow_settings.save(service, ADAPTERS[service].load(), values, user.username)
    except workflow_settings.SettingsError as e:
        raise HTTPException(400, str(e)) from None
    return _settings_view(service)


@app.delete("/api/admin/workflows/{service}")
def admin_reset_workflow(service: str, user: User = Depends(admin_user)) -> dict:
    _settings_view(service)
    workflow_settings.save(service, ADAPTERS[service].load(), {}, user.username)
    return _settings_view(service)


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
    out: dict[str, list[dict]] = {name: [] for name in ADAPTERS}
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
    for service in ADAPTERS:  # also workflows with no server right now
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
                if service in pipelines.ADAPTERS:  # script pipelines: audio + prompt
                    row["audio_url"], row["prompt"] = None, None
                    if _history_input(service, f.stem, "audio", AUDIO_EXT):
                        row["audio_url"] = (
                            f"/api/history/{service}/{quote(f.name)}/input/audio"
                        )
                    p = INPUT_DIR / service / f"{f.stem}__prompt.txt"
                    if p.is_file():
                        row["prompt"] = p.read_text(encoding="utf-8", errors="replace")
                rows.append(row)
    return sorted(rows, key=lambda r: r["finished"], reverse=True)


@app.get("/api/history/{service}/{name}")
def history_video(service: str, name: str, user: User = Depends(current_user)):
    if service not in ADAPTERS:
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


@app.delete("/api/history/{service}/{name}")
def history_delete(service: str, name: str, user: User = Depends(current_user)) -> dict:
    """Remove one of the caller's results, and the inputs saved with it."""
    if service not in ADAPTERS:
        raise HTTPException(404, f"Unknown service '{service}'.")
    root = (OUT_DIR / service).resolve()
    f = (root / name).resolve()
    if (
        f.parent != root
        or f.suffix.lower() not in VIDEO_EXT
        or not auth.owns(user.id, service, f.name)
    ):
        raise HTTPException(404, "No such result.")
    with _NAME_LOCK:
        auth.remove_owner(user.id, service, f.name)
        f.unlink(missing_ok=True)
        folder = INPUT_DIR / service
        if folder.is_dir():
            for p in folder.glob(f"{f.stem}__*"):
                p.unlink(missing_ok=True)
        # Keep the run's number taken: _next_number also reads the inputs folder,
        # so a later result can never reuse this name.
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{f.stem}__removed.txt").write_text("removed by its owner\n")
    return {"removed": f.name}


@app.get("/api/history/{service}/{name}/input/{kind}")
def history_input(
    service: str, name: str, kind: str, user: User = Depends(current_user)
):
    if (
        service not in ADAPTERS
        or kind not in ("image", "video", "audio")
        or not auth.owns(user.id, service, name)
    ):
        raise HTTPException(404, "No such input.")
    exts = {"image": IMAGE_EXT, "video": VIDEO_EXT, "audio": AUDIO_EXT}[kind]
    f = _history_input(service, Path(name).stem, kind, exts)
    if f is None:
        raise HTTPException(404, "No such input.")
    return FileResponse(f, content_disposition_type="inline")


@app.get("/")
def index(request: Request):
    if auth.user_for(request.cookies.get(auth.COOKIE)) is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page(request: Request):
    user = auth.user_for(request.cookies.get(auth.COOKIE))
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if not user.is_admin:
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "admin.html")


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
    if any(a.trim_frames for a in ADAPTERS.values()) and not (
        shutil.which("ffmpeg") and shutil.which("ffprobe")
    ):
        print(
            "WARNING: ffmpeg/ffprobe not found in this container: results will "
            "NOT be trimmed. Install ffmpeg in the UI image or set trim_frames to 0 "
            "in workflows/workflows.json."
        )
    for f in OUT_DIR.glob("*/*.mp4"):  # results saved before faststart was added
        try:
            faststart(f)
        except (OSError, ValueError) as e:
            print(f"faststart skipped for {f.name}: {e}")
    print(f"backends from {BACKENDS_SOURCE}:")
    for gpu, servers in GPUS.items():
        for b in servers:
            print(
                f"  GPU {gpu}  {b.workflow:>8}  {b.name:<18} {b.client.base_url}  "
                f"{'up' if b.client.is_up() else 'DOWN'}"
            )
    print(f"limit: {MAX_PER_USER} jobs per user per tab, {MAX_WAITING} per tab in all")
    uvicorn.run(
        app,
        host=os.environ.get("UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("UI_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
