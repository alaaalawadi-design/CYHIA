"""
Script pipelines: tabs whose backend runs a script instead of ComfyUI.

  lipsync  LongCat-Video-Avatar 1.5: image + audio -> talking video
           (pipelines/lipsync/worker.py runs run_avatar.sh)
  dubbing  TBDub: video + audio -> the video with re-synced lips
           (pipelines/dubbing/worker.py runs episode_dub.py)

Each worker runs its script on the GPU and answers the same small HTTP API.
This module is the UI's side of it:

  PipelineAdapter  plays the part of workflows.Adapter for the scheduler and
                   the page (label, blurb, timeout, output prefix, ...), and
                   build() turns the uploads + prompt into a job spec.
  PipelineClient   plays the part of comfy_client.ComfyClient: is_up, vram,
                   free, upload, run (with progress), download. So
                   server._execute() and the shared GPU queue drive a script
                   pipeline exactly as they drive a ComfyUI server.

Config comes from workflows/pipelines.json, which workflows.json, start.sh and
scripts/gen_config.py never read: adding or changing a pipeline cannot change
the ComfyUI tabs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .comfy_client import ComfyError, ComfyTimeout, JobResult, Progress

CATALOG_FILE = Path(__file__).resolve().parent.parent / "workflows" / "pipelines.json"

# How often run() asks the worker for progress, and how long the worker may be
# unreachable in the middle of a job before the job is given up.
POLL_SECONDS = 3.0
UNREACHABLE_LIMIT = 300.0


@dataclass
class PipelineAdapter:
    """What the scheduler and the page need to know about one script pipeline."""

    name: str
    label: str
    blurb: str = ""
    gallery_blurb: str = ""
    media: str = "audio"            # what the main upload is: "audio" (lipsync) or "video" (dubbing)
    needs_image: bool = True
    needs_audio: bool = False       # an extra audio upload next to the video (dubbing)
    needs_prompt: bool = True       # false = default_prompt is used, users never see it
    default_prompt: str = ""
    max_prompt_chars: int = 4000
    max_input_minutes: float = 0    # longest video/audio accepted; 0 = no limit
    labels: dict = field(default_factory=dict)   # captions: image, video, audio, result
    notes: list = field(default_factory=list)    # short points shown under the description
    args: dict = field(default_factory=dict)     # fixed script settings
    stages: dict = field(default_factory=dict)   # stage code -> label shown in the UI
    prefix: str = ""
    timeout: float = 4 * 3600.0
    trim_frames: int = 0            # results are never trimmed
    free_after_run: bool = True
    gpu: int = 0
    backend: str = ""
    url: str = ""

    pipeline = True                 # server.py: this tab is a script pipeline
    output_node = "output"          # JobResult.videos[*]["node_id"] of the result
    default_gpu = property(lambda self: self.gpu)

    @property
    def output_prefix(self) -> str:
        return self.prefix or f"Cy_{self.name.capitalize()}"

    def load(self) -> dict:
        """No workflow graph: there is nothing for /admin to edit."""
        return {}

    def build(self, image_name: str | None, video_name: str, *,
              frame_cap: int | None = None, prompt: str | None = None,
              audio_name: str | None = None) -> dict:
        """
        The job spec the worker runs. frame_cap is a ComfyUI-only setting.
          lipsync: {"image", "audio" (the main upload), "prompt", ...}
          dubbing: {"video" (the main upload), "audio" (the extra upload), ...}
        """
        spec = {"args": dict(self.args), "output_prefix": self.output_prefix}
        if self.needs_image:
            spec["image"] = image_name
        spec[self.media] = video_name
        if self.needs_audio:
            spec["audio"] = audio_name
        text = (prompt or self.default_prompt).strip()
        if text:
            spec["prompt"] = text
        return spec


class PipelineClient:
    """Talks to one pipeline worker; same methods server.py uses on ComfyClient."""

    def __init__(self, base_url: str, name: str, stages: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.stages = dict(stages or {})

    # ── health and GPU memory ────────────────────────────────────────────────
    def is_up(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def vram(self) -> tuple[int, int] | None:
        """(free, total) bytes on the worker's GPU, or None if it cannot tell."""
        try:
            r = httpx.get(f"{self.base_url}/vram", timeout=10)
            r.raise_for_status()
            v = r.json()
            return int(v["free"]), int(v["total"])
        except Exception:
            return None

    def free(self) -> bool:
        """The script exits after every job, so its memory is already back; the
        worker answers at once. Kept so the scheduler can treat it like ComfyUI."""
        try:
            return httpx.post(f"{self.base_url}/free", timeout=30).status_code == 200
        except Exception:
            return False

    # ── inputs ───────────────────────────────────────────────────────────────
    def upload(self, path: str | os.PathLike) -> str:
        src = Path(path)
        if not src.is_file():
            raise ComfyError(f"input file not found: {src}")
        name = f"{uuid.uuid4().hex}{src.suffix.lower()}"

        def chunks():  # streamed: a 30-minute video can be over a gigabyte
            with src.open("rb") as fh:
                while chunk := fh.read(1 << 20):
                    yield chunk

        r = httpx.post(
            f"{self.base_url}/upload",
            content=chunks(),
            headers={"X-Filename": name, "Content-Type": "application/octet-stream",
                     "Content-Length": str(src.stat().st_size)},
            timeout=900.0,
        )
        if r.status_code != 200:
            raise ComfyError(f"{self.name} refused the upload: {_detail(r)}")
        return r.json()["name"]

    # ── running ──────────────────────────────────────────────────────────────
    def node_titles(self, spec: dict) -> dict[str, str]:
        """Stage code -> label; server.py shows the label of the current stage."""
        return dict(self.stages)

    def run(self, spec: dict, timeout: float = 4 * 3600.0,
            on_progress=None) -> JobResult:
        r = httpx.post(f"{self.base_url}/jobs", json=spec, timeout=30)
        if r.status_code != 200:
            raise ComfyError(f"{self.name} rejected the job: {_detail(r)}")
        job_id = r.json()["id"]
        started = time.time()
        deadline = started + timeout
        last_seen = time.time()
        prog = Progress()

        while True:
            if time.time() >= deadline:
                self._cancel(job_id)
                raise ComfyTimeout(f"{self.name}: job did not finish within {timeout:.0f}s")
            time.sleep(POLL_SECONDS)
            try:
                st = httpx.get(f"{self.base_url}/jobs/{job_id}", timeout=15)
                if st.status_code == 404:
                    raise ComfyError(
                        f"{self.name} no longer knows this job (the worker restarted)."
                    )
                st.raise_for_status()
                s = st.json()
                last_seen = time.time()
            except ComfyError:
                raise
            except Exception:
                if time.time() - last_seen > UNREACHABLE_LIMIT:
                    raise ComfyError(f"{self.name} has been unreachable for "
                                     f"{UNREACHABLE_LIMIT / 60:.0f} minutes.") from None
                continue

            if s.get("state") == "done":
                return JobResult(
                    prompt_id=job_id,
                    videos=[{"node_id": "output", "filename": s["output"]}],
                    elapsed=time.time() - started,
                )
            if s.get("state") == "error":
                raise ComfyError(f"{self.name}: {s.get('error') or 'the script failed'}")

            stage = s.get("stage") or "queued"
            prog.node_id = stage
            prog.stage = self.stages.get(stage, stage)
            prog.step, prog.steps = int(s.get("step") or 0), int(s.get("steps") or 0)
            if on_progress:
                on_progress(prog)

    def _cancel(self, job_id: str) -> None:
        """Stop a timed-out job so it does not keep the GPU busy."""
        try:
            httpx.post(f"{self.base_url}/jobs/{job_id}/cancel", timeout=30)
        except Exception:
            pass

    # ── results ──────────────────────────────────────────────────────────────
    def download(self, item: dict, dest_dir: str | os.PathLike) -> str:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = Path(item["filename"]).name
        dest = dest_dir / name
        with httpx.stream("GET", f"{self.base_url}/files/{name}", timeout=600.0) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        return str(dest)


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error") or r.text[:300])
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:300]}"


def _load() -> dict[str, PipelineAdapter]:
    if not CATALOG_FILE.is_file():
        return {}
    try:
        catalog = json.loads(CATALOG_FILE.read_text())
    except (OSError, ValueError) as e:
        # A broken pipelines.json must never take the ComfyUI tabs down with it.
        print(f"WARNING: {CATALOG_FILE.name} unreadable ({e}); no script pipelines loaded")
        return {}
    out = {}
    for name, s in (catalog.get("pipelines") or {}).items():
        out[name] = PipelineAdapter(
            name=name,
            label=s.get("label") or name.capitalize(),
            blurb=s.get("blurb", ""),
            gallery_blurb=s.get("gallery_blurb", ""),
            media=s.get("media", "audio"),
            needs_image=bool(s.get("needs_image", True)),
            needs_audio=bool(s.get("needs_audio", False)),
            needs_prompt=bool(s.get("needs_prompt", True)),
            default_prompt=s.get("default_prompt", ""),
            max_prompt_chars=int(s.get("max_prompt_chars", 4000)),
            max_input_minutes=float(s.get("max_input_minutes", 0) or 0),
            labels=dict(s.get("labels") or {}),
            notes=[str(n) for n in (s.get("notes") or [])],
            args=dict(s.get("args") or {}),
            stages=dict(s.get("stages") or {}),
            prefix=s.get("prefix", ""),
            timeout=float(s.get("timeout_hours", 4)) * 3600.0,
            free_after_run=bool(s.get("free_after_run", True)),
            # <NAME>_URL / <NAME>_GPU override the file, like the ComfyUI tabs.
            gpu=int(os.environ.get(f"{name.upper()}_GPU", s.get("gpu", 0))),
            backend=s.get("backend") or f"comfy-{name}",
            url=os.environ.get(f"{name.upper()}_URL", s.get("url") or f"http://comfy-{name}:8138"),
        )
    return out


ADAPTERS: dict[str, PipelineAdapter] = _load()


def clients() -> list[tuple[PipelineAdapter, PipelineClient]]:
    """One worker per pipeline."""
    return [(a, PipelineClient(a.url, name=a.backend, stages=a.stages)) for a in ADAPTERS.values()]
