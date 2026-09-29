"""
Minimal ComfyUI API client, scoped to one ComfyUI instance.

Two of these are live at once — one per GPU service — so every piece of state
(base URL, client id) belongs to the instance and nothing is module-global.

Lifted from the pattern already proven in comfyui-wan-docker/gradio_app_gpu1.py:
upload inputs, POST /prompt, follow progress over the websocket, then pull the
result out of /history. The websocket is treated as an optimisation only; if it
drops, polling /history still finishes the job.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import httpx
import websocket  # websocket-client


class ComfyError(RuntimeError):
    """ComfyUI refused the prompt or failed while executing it."""


class ComfyTimeout(ComfyError):
    pass


@dataclass
class Progress:
    """What the UI needs to draw a progress bar, without knowing about nodes."""
    stage: str = ""          # human-readable: the executing node's title
    node_id: str | None = None
    step: int = 0            # sampler step within the current node
    steps: int = 0
    queue_position: int | None = None

    @property
    def fraction(self) -> float:
        return (self.step / self.steps) if self.steps else 0.0


@dataclass
class JobResult:
    prompt_id: str
    videos: list[dict] = field(default_factory=list)   # {filename, subfolder, type}
    images: list[dict] = field(default_factory=list)
    elapsed: float = 0.0


ProgressCb = Callable[[Progress], None]


class ComfyClient:
    def __init__(self, base_url: str, name: str = "comfy", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ── plumbing ─────────────────────────────────────────────────────────────
    @property
    def ws_url(self) -> str:
        return (self.base_url.replace("https://", "wss://").replace("http://", "ws://")
                + f"/ws?clientId={self.client_id}")

    def _get(self, path: str, timeout: float | None = None):
        r = httpx.get(f"{self.base_url}{path}", timeout=timeout or self.timeout)
        r.raise_for_status()
        return r.json()

    # ── health ───────────────────────────────────────────────────────────────
    def is_up(self) -> bool:
        try:
            self._get("/system_stats", timeout=5)
            return True
        except Exception:
            return False

    def queue_depth(self) -> tuple[int, int]:
        """(running, pending) — used for the per-GPU status badge."""
        try:
            q = self._get("/queue", timeout=5)
            return len(q.get("queue_running", [])), len(q.get("queue_pending", []))
        except Exception:
            return (0, 0)

    def wait_until_up(self, deadline: float = 600.0, poll: float = 5.0) -> bool:
        end = time.time() + deadline
        while time.time() < end:
            if self.is_up():
                return True
            time.sleep(poll)
        return False

    # ── inputs ───────────────────────────────────────────────────────────────
    def upload(self, path: str | os.PathLike, subfolder: str = "") -> str:
        """
        Upload an input file and return the name to put in the workflow.

        The name is made unique per upload: the two ComfyUI instances keep
        separate input dirs, but concurrent users of one instance would still
        collide on a plain basename.
        """
        src = Path(path)
        if not src.is_file():
            raise ComfyError(f"input file not found: {src}")
        name = f"{uuid.uuid4().hex}{src.suffix.lower()}"
        mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        with src.open("rb") as fh:
            r = httpx.post(
                f"{self.base_url}/upload/image",
                files={"image": (name, fh, mime)},
                data={"overwrite": "false", "subfolder": subfolder},
                timeout=300.0,
            )
        r.raise_for_status()
        body = r.json()
        stored = body.get("name", name)
        sub = body.get("subfolder") or ""
        return f"{sub}/{stored}" if sub else stored

    # ── running ──────────────────────────────────────────────────────────────
    def submit(self, workflow: dict) -> str:
        r = httpx.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise ComfyError(f"{self.name} rejected the prompt: {_explain(r)}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 7200.0,
             on_progress: ProgressCb | None = None, poll: float = 10.0) -> JobResult:
        """
        Block until the job lands in /history, then return its outputs.

        Progress arrives over the websocket, but completion is always confirmed
        against /history so a dropped socket cannot strand the caller.
        """
        started = time.time()
        deadline = started + timeout
        titles = {}
        prog = Progress()
        ws = None
        next_poll = 0.0

        try:
            while time.time() < deadline:
                if ws is None:
                    try:
                        ws = websocket.create_connection(self.ws_url, timeout=10)
                        ws.settimeout(1.0)
                    except Exception:
                        ws = None

                if ws is not None:
                    try:
                        raw = ws.recv()
                        if isinstance(raw, str):
                            self._apply_message(json.loads(raw), prompt_id, prog,
                                                titles, on_progress)
                    except websocket.WebSocketTimeoutException:
                        pass
                    except Exception:
                        try:
                            ws.close()
                        except Exception:
                            pass
                        ws = None
                else:
                    time.sleep(1.0)

                # /history is the source of truth for "is it done".
                if time.time() >= next_poll:
                    next_poll = time.time() + poll
                    done = self._history_entry(prompt_id)
                    if done is not None:
                        _raise_if_failed(done, self.name)
                        return self._collect(prompt_id, done, time.time() - started)

            raise ComfyTimeout(
                f"{self.name}: job {prompt_id} did not finish within {timeout:.0f}s"
            )
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    def run(self, workflow: dict, timeout: float = 7200.0,
            on_progress: ProgressCb | None = None) -> JobResult:
        return self.wait(self.submit(workflow), timeout=timeout, on_progress=on_progress)

    # ── results ──────────────────────────────────────────────────────────────
    def _history_entry(self, prompt_id: str) -> dict | None:
        try:
            hist = self._get(f"/history/{prompt_id}", timeout=15)
        except Exception:
            return None
        entry = hist.get(prompt_id)
        if not entry:
            return None
        status = entry.get("status", {})
        # 'completed' is absent on older builds; a terminal status_str is enough.
        if status.get("completed") or status.get("status_str") in ("success", "error"):
            return entry
        return None

    def _collect(self, prompt_id: str, entry: dict, elapsed: float) -> JobResult:
        res = JobResult(prompt_id=prompt_id, elapsed=elapsed)
        for node_id, out in (entry.get("outputs") or {}).items():
            for item in out.get("gifs", []) or []:
                res.videos.append({**item, "node_id": node_id})
            for item in out.get("images", []) or []:
                if item.get("type") == "temp":
                    continue            # previews are not deliverables
                res.images.append({**item, "node_id": node_id})
        return res

    def view_url(self, item: dict) -> str:
        q = httpx.QueryParams({
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", "") or "",
            "type": item.get("type", "output") or "output",
        })
        return f"{self.base_url}/view?{q}"

    def download(self, item: dict, dest_dir: str | os.PathLike) -> str:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item["filename"]
        with httpx.stream("GET", self.view_url(item), timeout=600.0) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        return str(dest)

    # ── websocket messages ───────────────────────────────────────────────────
    def _apply_message(self, msg: dict, prompt_id: str, prog: Progress,
                       titles: dict, on_progress: ProgressCb | None) -> None:
        kind, data = msg.get("type"), msg.get("data") or {}
        if data.get("prompt_id") not in (None, prompt_id):
            return

        if kind == "executing":
            node = data.get("node")
            if node:
                prog.node_id = node
                prog.stage = titles.get(node, f"node {node}")
                prog.step = prog.steps = 0
        elif kind == "progress":
            prog.step = int(data.get("value") or 0)
            prog.steps = int(data.get("max") or 0)
        elif kind == "status":
            remaining = (((data.get("status") or {}).get("exec_info") or {})
                         .get("queue_remaining"))
            if remaining is not None:
                prog.queue_position = int(remaining)
        else:
            return

        if on_progress:
            on_progress(prog)

    def node_titles(self, workflow: dict) -> dict[str, str]:
        return {nid: (n.get("_meta") or {}).get("title") or n.get("class_type", nid)
                for nid, n in workflow.items()}


def _explain(resp: httpx.Response) -> str:
    """Turn ComfyUI's validation payload into one readable line."""
    try:
        body = resp.json()
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:400]}"

    bits: list[str] = []
    err = body.get("error")
    if isinstance(err, dict):
        bits.append(err.get("message") or "")
        if err.get("details"):
            bits.append(str(err["details"]))
    elif err:
        bits.append(str(err))
    for node_id, info in (body.get("node_errors") or {}).items():
        for e in info.get("errors", []) or []:
            bits.append(f"node {node_id} ({info.get('class_type','?')}): "
                        f"{e.get('message')} {e.get('details','')}".strip())
    return " | ".join(b for b in bits if b) or f"HTTP {resp.status_code}"


def _raise_if_failed(entry: dict, name: str) -> None:
    status = entry.get("status", {})
    if status.get("status_str") != "error":
        return
    for kind, data in status.get("messages", []) or []:
        if kind == "execution_error":
            raise ComfyError(
                f"{name}: {data.get('node_type')} (node {data.get('node_id')}) failed: "
                f"{data.get('exception_type')}: {data.get('exception_message')}"
            )
    raise ComfyError(f"{name}: execution failed")
