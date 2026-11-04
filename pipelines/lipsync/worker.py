#!/usr/bin/env python3
"""
Lip-sync worker: runs LongCat-Video-Avatar 1.5 (run_avatar.sh) for the CYHIA UI.

Standard library only, so it runs on the LongCat env's own Python with nothing
installed. One job at a time, in arrival order; the UI's shared GPU queue
already sends one at a time, this is just a safety net.

HTTP API (JSON unless noted), used by app/pipelines.py:
  GET  /health              200 {"ok": true, "busy": bool}
  GET  /vram                {"free": bytes, "total": bytes}   (nvidia-smi)
  POST /free                200: the script exits after each job, nothing stays loaded
  POST /upload              raw file body, header X-Filename -> {"name": stored name}
  POST /jobs                {"image","audio","prompt","args","output_prefix"} -> {"id"}
  GET  /jobs/<id>           {"state","stage","step","steps","error","output"}
  POST /jobs/<id>/cancel    stop a queued or running job
  GET  /files/<name>        a finished video (binary)

Progress comes from the script's own output:
  "... / N segments / ..."         -> stage "models", N segments
  "Segment k/N, attempt ..."       -> stage "generate"
  "Denoising: ... i/S [..."        -> step = (k-1)*S + i of N*S
  "Saved: ..."                     -> stage "save"
  "RESULT_JSON {...}" (last line)  -> done / error

Environment:
  LIPSYNC_PORT        default 8138
  LIPSYNC_DATA        writable folder (input/, output/, jobs/), default /data
  LIPSYNC_SCRIPT      default /home/administrator/Disk/alaa-work/LongCat-Video/run_avatar.sh
  LIPSYNC_MAX_HOURS   a job running longer is killed, default 6
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PORT = int(os.environ.get("LIPSYNC_PORT", "8138"))
DATA = Path(os.environ.get("LIPSYNC_DATA", "/data"))
SCRIPT = os.environ.get(
    "LIPSYNC_SCRIPT", "/home/administrator/Disk/alaa-work/LongCat-Video/run_avatar.sh"
)
MAX_SECONDS = float(os.environ.get("LIPSYNC_MAX_HOURS", "6")) * 3600.0
INPUT, OUTPUT, JOBS_DIR = DATA / "input", DATA / "output", DATA / "jobs"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
MAX_UPLOAD = 1 << 30  # 1 GiB
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Script options the job spec may set (pipelines.json "args"), with their types.
# image / audio / prompt / output-dir / gpu are set by the worker itself.
ARGS = {
    "crf": int, "preset": str, "audio_bitrate": str, "no_segments": bool,
    "precision": str, "lora_multiplier": float, "resolution": str, "steps": int,
    "text_guidance": float, "audio_guidance": float, "negative_prompt": str,
    "seed": int, "no_distill": bool, "ref_img_index": int, "mask_frame_range": int,
    "frames_per_segment": int, "overlap": int, "max_seconds": float,
    "no_vocal_separation": bool, "mlp_chunks": str,
}

# ── jobs ─────────────────────────────────────────────────────────────────────
LOCK = threading.Condition()
JOBS: dict[str, dict] = {}
QUEUE: deque[str] = deque()
KEEP_FINISHED = 200


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "state", "stage", "step", "steps", "error", "output")}


def _cli_args(args: dict) -> list[str]:
    out = []
    for key, value in (args or {}).items():
        if key not in ARGS:
            raise ValueError(f"unknown setting {key!r}")
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        kind = ARGS[key]
        if kind is bool:
            if value is True:
                out.append(flag)
            elif value is not False:
                raise ValueError(f"{key} must be true or false")
            continue
        try:
            out += [flag, str(kind(value))]
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be {kind.__name__}") from None
    return out


def _input(name: str | None, exts: set[str], what: str) -> Path:
    if not name or not SAFE_NAME.match(name) or Path(name).suffix.lower() not in exts:
        raise ValueError(f"bad {what} name {name!r}")
    path = INPUT / name
    if not path.is_file():
        raise ValueError(f"{what} {name!r} was not uploaded")
    return path


def create_job(spec: dict) -> dict:
    try:
        image = _input(spec.get("image"), IMAGE_EXT, "image")
        audio = _input(spec.get("audio"), AUDIO_EXT, "audio")
        prompt = str(spec.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("the prompt is empty")
        extra = _cli_args(spec.get("args") or {})
    except ValueError:
        # A rejected job's uploads would otherwise sit in input/ until a restart.
        for key in ("image", "audio"):
            name = str(spec.get(key) or "")
            if SAFE_NAME.match(name):
                (INPUT / name).unlink(missing_ok=True)
        raise
    prefix = re.sub(r"[^A-Za-z0-9_-]", "", str(spec.get("output_prefix") or "Cy_LipSync"))[:40]
    job = {
        "id": uuid.uuid4().hex, "state": "queued", "stage": "queued", "step": 0, "steps": 0,
        "error": None, "output": None, "image": image, "audio": audio, "prompt": prompt,
        "prefix": prefix or "Cy_LipSync", "extra": extra, "proc": None, "cancelled": False,
        "created": time.time(),
    }
    with LOCK:
        JOBS[job["id"]] = job
        QUEUE.append(job["id"])
        LOCK.notify_all()
    return job


def _prune() -> None:
    with LOCK:
        done = sorted((j for j in JOBS.values() if j["state"] in ("done", "error")),
                      key=lambda j: j["created"])
        for j in done[:-KEEP_FINISHED]:
            del JOBS[j["id"]]


RE_PLAN = re.compile(r"/\s*(\d+) segments /")
RE_SEG = re.compile(r"Segment (\d+)/(\d+), attempt")
RE_SEG_DONE = re.compile(r"Segment (\d+)/(\d+) done")
RE_STEP = re.compile(r"Denoising:.*?(\d+)/(\d+) \[")


def _follow(job: dict, line: str, state: dict) -> None:
    """Turn one line (or tqdm \\r-chunk) of the script's output into progress."""
    if m := RE_PLAN.search(line):
        state["segments"] = int(m.group(1))
        job["stage"] = "models"
    elif m := RE_SEG.search(line):
        state["segment"], state["segments"] = int(m.group(1)), int(m.group(2))
        job["stage"] = "generate"
    elif m := RE_STEP.search(line):
        i, s = int(m.group(1)), int(m.group(2))
        n, k = state.get("segments") or 1, state.get("segment") or 1
        job["step"], job["steps"] = (k - 1) * s + i, n * s
    elif m := RE_SEG_DONE.search(line):
        k, n = int(m.group(1)), int(m.group(2))
        if job["steps"]:
            job["step"] = job["steps"] * k // n
    elif line.startswith("Saved: "):
        job["stage"], job["step"], job["steps"] = "save", 0, 0
    elif line.startswith("RESULT_JSON "):
        try:
            state["result"] = json.loads(line[len("RESULT_JSON "):])
        except ValueError:
            pass


def run_job(job: dict) -> None:
    folder = JOBS_DIR / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(job["prompt"], encoding="utf-8")
    name = f"{job['prefix']}_{job['id'][:12]}"
    cmd = [
        SCRIPT, "--image", str(job["image"]), "--audio", str(job["audio"]),
        "--prompt-file", str(folder / "prompt.txt"), "--output-dir", str(folder),
        "--output-name", name, "--gpu", "0", *job["extra"],
    ]
    job["stage"] = "audio"
    state: dict = {}
    log = (folder / "run.log").open("w", encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=40)
    try:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,  # its own process group, so cancel stops all of it
        )
        job["proc"] = proc
        killer = threading.Timer(MAX_SECONDS, _kill, args=(job, "ran longer than the limit"))
        killer.daemon = True
        killer.start()
        buf = b""
        try:
            while chunk := proc.stdout.read1(65536):
                log.write(chunk.decode("utf-8", "replace"))
                log.flush()
                buf += chunk
                # tqdm redraws with \r, normal prints end with \n: both end a "line".
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    text = raw.decode("utf-8", "replace").strip()
                    if text:
                        tail.append(text)
                        _follow(job, text, state)
            if buf.strip():
                text = buf.decode("utf-8", "replace").strip()
                tail.append(text)
                _follow(job, text, state)
            code = proc.wait()
        finally:
            killer.cancel()

        result = state.get("result") or {}
        if job["cancelled"]:
            raise RuntimeError(job.get("error") or "cancelled")
        if code != 0 or result.get("status") != "ok":
            why = result.get("error") or f"the script exited with code {code}"
            raise RuntimeError(f"{why} (last output: {' | '.join(list(tail)[-3:])})")
        produced = Path(result["output"])
        OUTPUT.mkdir(parents=True, exist_ok=True)
        final = OUTPUT / produced.name
        shutil.move(str(produced), final)
        job["output"], job["state"], job["stage"] = final.name, "done", "save"
    except Exception as e:  # noqa: BLE001 - reported to the UI
        job["state"], job["error"] = "error", str(e)[:2000]
        log.write("\n" + traceback.format_exc())
    finally:
        log.close()
        job["proc"] = None
        for p in (job["image"], job["audio"]):
            Path(p).unlink(missing_ok=True)
        shutil.rmtree(folder / "segments", ignore_errors=True)
        _prune()


def _kill(job: dict, why: str) -> None:
    job["cancelled"], job["error"] = True, why
    proc = job.get("proc")
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(20)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def cancel(job: dict) -> None:
    with LOCK:
        if job["state"] == "queued":
            if job["id"] in QUEUE:
                QUEUE.remove(job["id"])
            job["state"], job["error"] = "error", "cancelled"
            for p in (job["image"], job["audio"]):
                Path(p).unlink(missing_ok=True)
            return
    if job["state"] == "running":
        threading.Thread(target=_kill, args=(job, "cancelled"), daemon=True).start()


def runner() -> None:
    while True:
        with LOCK:
            while not QUEUE:
                LOCK.wait()
            job = JOBS[QUEUE.popleft()]
            job["state"] = "running"
        run_job(job)


def busy() -> bool:
    with LOCK:
        return any(j["state"] in ("queued", "running") for j in JOBS.values())


def vram() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        free, total = (int(x.strip()) * 1024 * 1024 for x in out.split(","))
        return {"free": free, "total": total}
    except Exception:
        return None


# ── http ─────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "cyhia-lipsync/1"

    def log_message(self, fmt, *args):  # health checks every few seconds: keep quiet
        if "/health" not in (args[0] if args else "") and "/vram" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("upload too large")
        return self.rfile.read(n)

    def _job(self, parts: list[str]) -> dict | None:
        return JOBS.get(parts[1]) if len(parts) >= 2 else None

    def do_GET(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if parts == ["health"]:
            return self._json(200, {"ok": True, "busy": busy()})
        if parts == ["vram"]:
            v = vram()
            return self._json(200, v) if v else self._json(503, {"error": "nvidia-smi failed"})
        if len(parts) == 2 and parts[0] == "jobs":
            job = self._job(parts)
            return self._json(200, _public(job)) if job else self._json(404, {"error": "unknown job"})
        if len(parts) == 2 and parts[0] == "files" and SAFE_NAME.match(parts[1]):
            f = OUTPUT / parts[1]
            if f.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                with f.open("rb") as fh:
                    shutil.copyfileobj(fh, self.wfile, 1 << 20)
                return None
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        try:
            if parts == ["free"]:
                self._body()
                return self._json(200, {"ok": True})
            if parts == ["upload"]:
                name = self.headers.get("X-Filename", "")
                ext = Path(name).suffix.lower()
                if not SAFE_NAME.match(name) or ext not in IMAGE_EXT | AUDIO_EXT:
                    return self._json(400, {"error": f"bad file name {name!r}"})
                INPUT.mkdir(parents=True, exist_ok=True)
                (INPUT / name).write_bytes(self._body())
                return self._json(200, {"name": name})
            if parts == ["jobs"]:
                spec = json.loads(self._body() or b"{}")
                return self._json(200, _public(create_job(spec)))
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                job = self._job(parts)
                if not job:
                    return self._json(404, {"error": "unknown job"})
                cancel(job)
                return self._json(200, _public(job))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})


def main() -> None:
    for d in (INPUT, OUTPUT, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Uploads of jobs that died with a previous run of the worker.
    for f in INPUT.iterdir():
        f.unlink(missing_ok=True)
    if not os.access(SCRIPT, os.X_OK):
        print(f"WARNING: {SCRIPT} is missing or not executable; every job will fail")
    threading.Thread(target=runner, name="runner", daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"lipsync worker on :{PORT}, script {SCRIPT}, data {DATA}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
