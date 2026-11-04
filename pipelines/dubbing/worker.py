#!/usr/bin/env python3
"""
Dubbing worker: runs TBDub (episode_dub.py) for the CYHIA UI.

Video + dubbed audio -> the same video with the speaker's lips re-synced to the
new audio. Same HTTP API as the lip-sync worker (pipelines/lipsync/worker.py),
used by app/pipelines.py; kept as a separate file so the two pipelines never
affect each other. Standard library only. One job at a time.

  GET  /health              200 {"ok": true, "busy": bool}
  GET  /vram                {"free": bytes, "total": bytes}   (nvidia-smi)
  POST /free                200: the script exits after each job, nothing stays loaded
  POST /upload              raw file body, header X-Filename -> {"name": stored name}
  POST /jobs                {"video","audio","args","output_prefix"} -> {"id"}
                            400 if a file is longer than DUBBING_MAX_MINUTES
  GET  /jobs/<id>           {"state","stage","step","steps","error","output"}
  POST /jobs/<id>/cancel    stop a queued or running job
  GET  /files/<name>        a finished video (binary)

Progress comes from episode_dub.py's own output:
  [scan] 1200/~9000 frames            -> stage "scan", step 1200 of 9000
  [speaker] ...                       -> stage "speaker"
  [plan] ...                          -> stage "plan"
  [episode] segment 3/12: ...         -> stage "generate", step 2 of 12
  [episode] progress 3/12 segments    -> step 3 of 12
  [episode] assembling 9000 frames    -> stage "assemble"
  [episode] assembled 2500/9000       -> step 2500 of 9000
  [episode] output saved to ...       -> stage "save"

Environment:
  DUBBING_PORT          default 8139
  DUBBING_DATA          writable folder (input/, output/, jobs/), default /data
  TBDUB_ROOT            default /home/administrator/Disk/alaa-work/TBDub-dubbing
  TBDUB_PYTHON          default <TBDUB_ROOT>/venv/bin/python
  DUBBING_MAX_MINUTES   longest video/audio accepted, default 30
  DUBBING_MAX_HOURS     a job running longer is killed, default 8
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

PORT = int(os.environ.get("DUBBING_PORT", "8139"))
DATA = Path(os.environ.get("DUBBING_DATA", "/data"))
ROOT = Path(os.environ.get("TBDUB_ROOT", "/home/administrator/Disk/alaa-work/TBDub-dubbing"))
PYTHON = os.environ.get("TBDUB_PYTHON", str(ROOT / "venv" / "bin" / "python"))
MAX_MINUTES = float(os.environ.get("DUBBING_MAX_MINUTES", "30"))
MAX_SECONDS = float(os.environ.get("DUBBING_MAX_HOURS", "8")) * 3600.0
INPUT, OUTPUT, JOBS_DIR = DATA / "input", DATA / "output", DATA / "jobs"

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
MAX_UPLOAD = 8 << 30  # 8 GiB: a 30-minute video can be large
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# episode_dub.py / inference.py options the job spec may set (pipelines.json "args").
# video / audio / output / work-dir / cleanup are set by the worker itself.
ARGS = {
    "inference_mode": str, "num_student_steps": int, "num_inference_steps": int,
    "sigma_shift": float, "motion_from_latents": bool, "seed": int,
    "cpu_offload": bool, "crf": int, "preset": str,
    "min_segment_seconds": float, "max_segment_seconds": float,
    "no_speaker_detection": bool,
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


def duration(path: Path) -> float | None:
    """Length in seconds, or None if ffprobe cannot read it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def create_job(spec: dict) -> dict:
    try:
        video = _input(spec.get("video"), VIDEO_EXT, "video")
        audio = _input(spec.get("audio"), AUDIO_EXT, "audio")
        for what, path in (("video", video), ("audio", audio)):
            seconds = duration(path)
            if seconds is None:
                raise ValueError(f"the {what} file cannot be read; is it a valid {what}?")
            if MAX_MINUTES and seconds > MAX_MINUTES * 60:
                raise ValueError(f"the {what} is {seconds / 60:.1f} min long; "
                                 f"the limit is {MAX_MINUTES:g} min")
        extra = _cli_args(spec.get("args") or {})
    except ValueError:
        # A rejected job's uploads would otherwise sit in input/ until a restart.
        for key in ("video", "audio"):
            name = str(spec.get(key) or "")
            if SAFE_NAME.match(name):
                (INPUT / name).unlink(missing_ok=True)
        raise
    prefix = re.sub(r"[^A-Za-z0-9_-]", "", str(spec.get("output_prefix") or "Cy_Dubbing"))[:40]
    job = {
        "id": uuid.uuid4().hex, "state": "queued", "stage": "queued", "step": 0, "steps": 0,
        "error": None, "output": None, "video": video, "audio": audio,
        "prefix": prefix or "Cy_Dubbing", "extra": extra, "proc": None, "cancelled": False,
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


RE_SCAN = re.compile(r"^\[scan\] (\d+)/~?(\d+) frames")
RE_SEGMENT = re.compile(r"^\[episode\] segment (\d+)/(\d+)")
RE_PROGRESS = re.compile(r"^\[episode\] progress (\d+)/(\d+) segments")
RE_ASSEMBLING = re.compile(r"^\[episode\] assembling (\d+) frames")
RE_ASSEMBLED = re.compile(r"^\[episode\] assembled (\d+)/(\d+) frames")


def _follow(job: dict, line: str) -> None:
    """Turn one line of episode_dub.py's output into progress."""
    if m := RE_SCAN.search(line):
        job["stage"], job["step"], job["steps"] = "scan", int(m.group(1)), int(m.group(2))
    elif line.startswith("[scan]"):
        if job["stage"] != "scan":
            job["stage"], job["step"], job["steps"] = "scan", 0, 0
    elif line.startswith("[speaker]"):
        if job["stage"] != "speaker":
            job["stage"], job["step"], job["steps"] = "speaker", 0, 0
    elif line.startswith("[plan]"):
        if job["stage"] != "plan":
            job["stage"], job["step"], job["steps"] = "plan", 0, 0
    elif m := RE_SEGMENT.search(line):
        job["stage"], job["step"], job["steps"] = "generate", int(m.group(1)) - 1, int(m.group(2))
    elif m := RE_PROGRESS.search(line):
        job["stage"], job["step"], job["steps"] = "generate", int(m.group(1)), int(m.group(2))
    elif m := RE_ASSEMBLING.search(line):
        job["stage"], job["step"], job["steps"] = "assemble", 0, int(m.group(1))
    elif m := RE_ASSEMBLED.search(line):
        job["stage"], job["step"], job["steps"] = "assemble", int(m.group(1)), int(m.group(2))
    elif line.startswith("[episode] output saved"):
        job["stage"], job["step"], job["steps"] = "save", 0, 0


def run_job(job: dict) -> None:
    folder = JOBS_DIR / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    produced = folder / f"{job['prefix']}_{job['id'][:12]}.mp4"
    cmd = [
        PYTHON, "-u", "episode_dub.py",
        "--video", str(job["video"]), "--audio", str(job["audio"]),
        "--output", str(produced), "--work-dir", str(folder / "work"), "--cleanup",
        *job["extra"],
    ]
    job["stage"] = "prepare"
    log = (folder / "run.log").open("w", encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=40)
    try:
        log.write("$ cd " + str(ROOT) + " && " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
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
                        _follow(job, text)
            if buf.strip():
                text = buf.decode("utf-8", "replace").strip()
                tail.append(text)
                _follow(job, text)
            code = proc.wait()
        finally:
            killer.cancel()

        if job["cancelled"]:
            raise RuntimeError(job.get("error") or "cancelled")
        if code != 0 or not produced.is_file() or produced.stat().st_size == 0:
            raise RuntimeError(f"episode_dub.py exited with code {code} "
                               f"(last output: {' | '.join(list(tail)[-3:])})")
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
        for p in (job["video"], job["audio"]):
            Path(p).unlink(missing_ok=True)
        # Scan, plan and per-segment files: large, and only useful to resume.
        shutil.rmtree(folder / "work", ignore_errors=True)
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
            for p in (job["video"], job["audio"]):
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
    server_version = "cyhia-dubbing/1"

    def log_message(self, fmt, *args):  # health checks every few seconds: keep quiet
        first = args[0] if args else ""
        if "/health" not in first and "/vram" not in first:
            super().log_message(fmt, *args)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
        n = int(self.headers.get("Content-Length") or 0)
        try:
            if n > MAX_UPLOAD:
                return self._json(413, {"error": "upload too large"})
            if parts == ["upload"]:
                name = self.headers.get("X-Filename", "")
                ext = Path(name).suffix.lower()
                if not SAFE_NAME.match(name) or ext not in VIDEO_EXT | AUDIO_EXT:
                    return self._json(400, {"error": f"bad file name {name!r}"})
                INPUT.mkdir(parents=True, exist_ok=True)
                # Streamed to disk: a 30-minute video does not fit comfortably in memory.
                with (INPUT / name).open("wb") as fh:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        fh.write(chunk)
                        left -= len(chunk)
                return self._json(200, {"name": name})
            body = self.rfile.read(n) if n else b""
            if parts == ["free"]:
                return self._json(200, {"ok": True})
            if parts == ["jobs"]:
                return self._json(200, _public(create_job(json.loads(body or b"{}"))))
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
    for f in INPUT.iterdir():  # uploads of jobs that died with a previous run
        f.unlink(missing_ok=True)
    if not (ROOT / "episode_dub.py").is_file():
        print(f"WARNING: {ROOT}/episode_dub.py is missing; every job will fail")
    if not os.access(PYTHON, os.X_OK):
        print(f"WARNING: {PYTHON} is missing or not executable; every job will fail")
    threading.Thread(target=runner, name="runner", daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"dubbing worker on :{PORT}, TBDub {ROOT}, limit {MAX_MINUTES:g} min, data {DATA}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
