# cy_hia

A web UI for two video-generation pipelines, each running on its own ComfyUI backend and its own GPU:

- **Replace** puts the person from a reference image into a driving video (SCAIL-2 character replacement).
- **Animate** animates a reference image with the motion of a driving video (WAN 2.2 Animate).

Users sign in, upload an image and a video, and get back an MP4. Admins also get an
[admin page](#admin-page) where they can tune the replacement workflow without editing JSON.

```
            browser ──► :7860  ui  (FastAPI + plain HTML, no GPU)
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
     comfy-replace  :8136              comfy-animate  :8134
     GPU 0 · ComfyUI 0.27.0            GPU 1 · ComfyUI 0.20.1
     workflows/replace_workflow.json   workflows/animate_workflow.json
```

The backends run ComfyUI and nothing else. Because the UI is a separate container, you can
restart or edit it without evicting ~15 GB of model weights from either GPU.

> **Maintainers:** [CLAUDE.md](CLAUDE.md) holds the design notes, the history behind each
> decision and the traps in the workflow graph. Read it before you change the Dockerfiles, the
> mounts or `replace_workflow.json`.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [GPUs and the job queue](#gpus-and-the-job-queue)
- [Using the UI](#using-the-ui)
- [Accounts](#accounts)
- [Admin page](#admin-page)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Where data lives](#where-data-lives)
- [Editing the replacement workflow](#editing-the-replacement-workflow)
- [Testing](#testing)
- [Timings](#timings)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)

---

## Requirements

- Linux with Docker, Docker Compose v2 and the NVIDIA Container Toolkit.
- One or more NVIDIA GPUs. The services were built for RTX 5090s (32 GB, sm_120). `./start.sh`
  counts them and places the workflows; see [GPUs and the job queue](#gpus-and-the-job-queue).
- The host model and node trees that the backends mount read-only:
  - `nouran_Rep/` (`REP_ROOT`): ComfyUI `models/`, `custom_nodes/`, the patched `comfy/ops.py`
    and the `venv` that `comfy-replace` runs in.
  - `nouran_wan22/ComfyUI/` (`WAN_ROOT`): models and custom nodes for `comfy-animate`.
  - `/usr/local/cuda-12.8` on the host.
- The `wan-animate:latest` image, which `comfy-animate` reuses and does not build.

## Quick start

```bash
# 1. Build the replace backend (~45 min; Docker Hub is the bottleneck here) and the UI.
docker compose build comfy-replace ui

# 2. Count the GPUs, place the workflows on them and start everything.
./start.sh                 # ./start.sh --plan only prints where each workflow would go

# 3. Create the first admin account.
docker compose -f generated/docker-compose.yml exec ui python -m app.auth create-admin <username> <password>
```

The old way, `docker compose up -d` with the hand-written `docker-compose.yml`, still works and
uses the original two-GPU layout (replace on GPU 0, animate on GPU 1).

Then open `http://<host>:7860`. Only port 7860 is published. The backends are reachable only on
the internal `cyhia` network.

Follow startup with `docker logs -f cyhia_comfy_replace`. A cold start loads about 60 custom
nodes, so allow a few minutes before the health check passes.

The UI starts even when a backend is down. That tab shows a status badge, and the other tab keeps
working.

### Running the UI without Docker

```bash
../venv/bin/python -m app.server     # has fastapi, uvicorn, httpx, websocket-client
```

Point it at the backends with `SERVICES_FILE` (a `services.json` like the one `start.sh` writes)
or with `REPLACE_URL` / `ANIMATE_URL`, and choose a writable `OUT_DIR`.

## GPUs and the job queue

**Which workflows exist** is set in `workflows/workflows.json`: one entry per tab, with its label,
node ids, output prefix, timeout and warm-up trim. **Where they run** is decided by `./start.sh`,
which counts the GPUs with `nvidia-smi` and writes two files into `generated/` (gitignored):
`services.json`, which the UI reads, and `docker-compose.yml`, with one ComfyUI container per
server, each pinned to its GPU. Those containers are copies of the ones in the hand-written
`docker-compose.yml`, so mounts and images are only ever edited there.

| GPUs found | Layout |
|---|---|
| 1 | replace and animate both on GPU 0, sharing one queue |
| 2 | replace on GPU 0, animate on GPU 1, in parallel |
| 3 or more | one each, then the spare GPUs get extra servers for the workflows in `spare_gpus` (replace by default) |

To pin workflows by hand, set `gpu_plan` in `workflows/workflows.json`, for example
`{"replace": [0, 1], "animate": [1]}`, then run `./start.sh` again. Run it again too whenever a GPU
is added, removed or fails. If the layout changed, it restarts the UI, which forgets the jobs
waiting in it, so run it when nothing is queued.

**How jobs are queued.** Every job from every tab and every user goes into one waiting list, in
the order it was submitted. Each GPU takes the oldest job that a server on it can run:

- On a **shared GPU**, jobs run one at a time in exact submission order, whatever the tab. Before a
  job starts on a server other than the one that ran the last job there, the other servers on that
  GPU are told to unload their models, and the UI waits up to 60 s for the memory to come free.
  Switching workflows therefore costs a model reload, about 1–2 minutes.
- On **separate GPUs**, jobs run in parallel. A job never waits behind a job for a different GPU.
- A workflow with **several servers** runs that many of its jobs at once.

**Limits.** `MAX_PER_USER` at the top of `app/server.py` (**5**) is how many jobs one user may have
waiting or running in one tab. `MAX_WAITING` next to it (10) is the same for all users together.
Change either and restart the UI (`docker restart cyhia_ui`).

## Using the UI

The page has four tabs:

| Tab | What it is for |
|---|---|
| **Replace** | Upload a reference image and a driving video, then run SCAIL-2 replacement. The result is side-by-side with the original audio. |
| **Animate** | The same inputs, run through WAN 2.2 Animate. |
| **History** | Your own finished results, with the inputs that produced each one. |
| **Documentation** | The model evaluation: input criteria, strengths, limitations and worked examples. |

How a job behaves:

- **Generate can be pressed again while jobs run.** Each press queues one more job, up to 5 per
  user per tab. A "Your jobs" list shows each one: its place in the queue, the running stage, and a
  Show button for finished results. ✕ removes a finished job from the list; its result stays in
  History. See [GPUs and the job queue](#gpus-and-the-job-queue) for the order jobs run in.
- **Progress is shown as the current stage and elapsed minutes, not a percentage.** The graph has
  about 100 nodes of very uneven cost, so a percentage would mislead.
- **Reloading the page is safe.** The job list comes from the server, so it is the same after a
  reload or in another browser. Restarting the UI container, however, forgets jobs, because they
  are held in memory.
  ComfyUI still finishes the job, and the file stays in `data/<service>/output/`.
- **Results are deterministic.** Seeds are taken from the workflow, or from the admin page, so the
  same image, video and settings always produce the same video. To get a different result, change
  an input or a setting.
- **Results are named by the UI**, for example `Cy_Replace_00013.mp4` and `Cy_Animate_00013.mp4`.
  Numbers are never reused, so an older result is never overwritten.
- **The first `trim_frames` frames (set per workflow in `workflows/workflows.json`) are cut from
  every result** to drop warm-up frames. The saved
  copy of the driving video is cut by the same duration, so the two stay in sync. Trimming needs
  `ffmpeg`/`ffprobe`, which the UI image includes. If trimming fails, the job still succeeds and
  the result is returned untrimmed.

Input recommendations, such as the ~15 s stability window and the supported resolutions, are on
the Documentation tab. They are advice, not upload rules.

## Accounts

Everything except `/login`, `/api/auth/*`, `/gallery` and `/docs-assets` needs a session cookie.

- Users sign up and sign in at `/login`. You can restrict sign-up with `SIGNUP_CODE` (an invite
  code) or close it with `ALLOW_SIGNUP=0`. See [Configuration](#configuration).
- Each user sees only their own jobs and History. Asking for another user's result returns 404.
- Accounts are stored in SQLite at `data/ui_outputs/users.db`. Passwords are hashed with scrypt,
  and only hashes of session tokens are stored. The code is standard library only (`app/auth.py`).

Manage accounts from the command line. Inside the container, prefix each command with
`docker compose exec ui`:

```bash
python -m app.auth users                                # list users, admin flag, result counts
python -m app.auth create-admin <username> <password>   # new account with admin rights
python -m app.auth set-admin <username> on|off          # grant or revoke admin
python -m app.auth assign-unowned <username>            # give pre-accounts results to a user
```

Results made before accounts were added have no owner. They stay hidden until you run
`assign-unowned`.

## Admin page

**`/admin`** lets an admin change the inputs of the **replacement workflow** from the browser,
without editing `replace_workflow.json` or restarting anything.

### Access

- Only admins can open it. Signed-out visitors are sent to `/login`, and signed-in users who are
  not admins are sent back to `/`.
- Admin rights can only be granted from the CLI: `create-admin` or `set-admin <user> on` (see
  [Accounts](#accounts)). The web UI cannot grant them.

### What you can change

About 60 controls, grouped in the order the pipeline runs:

| Group | Examples |
|---|---|
| Driving video | max frames (the frame cap), frames to skip at the start |
| 1 · Clean first frame | positive and negative prompt, RefControl pose and depth LoRA strengths, sampler settings |
| 2 · Pose retarget | keypoint confidence, pose stick width, negative prompts for the frontal reference and frontal motion frame, sampler settings |
| 3 · Character in pose | positive and negative prompt, sampler settings |
| 4 · Relight | relight instruction, the QwenVL lighting-analysis prompt and its seed, Klein-Migration LoRA strength |
| 5 · Segmentation (SAM3) | target text, detection threshold and max objects, for both the driving video and the reference |
| 6 · Video (SCAIL-2) | prompts, window length and overlap, vary seed per window, pose strength/start/end, sampling shift, DPO LoRA strength, resolution in megapixels |
| Output | video quality (CRF) |

Sampler settings means seed, steps, CFG, sampler and scheduler, for each sampling stage.

Every control shows the workflow's own default next to its current value and marks the ones that
differ. Values are checked on the server before they are saved: type, min/max, step (for example,
window length must stay 4n+1), and allowed sampler and scheduler names. An invalid value is
rejected with a message and nothing is saved. **Reset** clears every override for that workflow.

### How it takes effect

- **Only values that differ from the workflow JSON are stored**, in
  `data/ui_outputs/workflow_settings.json`, together with who changed them and when. The workflow
  file stays the source of truth for every default.
- **Overrides are applied when a job starts, not when it is submitted.** A change therefore
  applies to every job that has not started yet, for all users, including jobs already waiting in
  the queue.
- **Every result records the settings it ran with**, in
  `data/ui_outputs/inputs/<service>/<stem>__settings.json`, so you can always tell which values
  produced a given video.
- If the settings file is missing or damaged, the service logs a warning and runs the workflow as
  saved. A bad settings file never stops jobs.

### API

The page calls these endpoints. All of them require an admin session.

| Method | Path | Does |
|---|---|---|
| `GET` | `/api/admin/workflows` | lists the workflows that have controls (currently `replace`) |
| `GET` | `/api/admin/workflows/{service}` | returns every control with its default, current value and `modified` flag |
| `PUT` | `/api/admin/workflows/{service}` | body `{"values": {"<node>.<input>": value, ...}}`; validates and saves |
| `DELETE` | `/api/admin/workflows/{service}` | resets the workflow to its defaults |

### Adding a control

Add one `Control(node, input, label, group, type, min, max, step, ...)` row to
`REPLACE_CONTROLS` in [app/workflow_settings.py](app/workflow_settings.py). The page renders it
automatically, with no HTML change. Take the node id and input name from
`workflows/replace_workflow.json`. To give the Animate workflow its own controls, add an entry to
`CONTROLS`.

## Configuration

Set these in `.env` next to `docker-compose.yml` (the file is gitignored) or in your shell
environment.

| Variable | Default | Purpose |
|---|---|---|
| `REP_ROOT` | `/home/administrator/Disk/nouran_Rep` | host install mirrored by `comfy-replace` |
| `WAN_ROOT` | `/home/administrator/Disk/nouran_wan22` | host models and nodes for `comfy-animate` |
| `UI_PORT` | `7860` | published port of the UI |
| `REPLACE_URL` | `http://comfy-replace:8136` | replace backend, only without `generated/services.json` |
| `ANIMATE_URL` | `http://comfy-animate:8134` | animate backend, only without `generated/services.json` |
| `SIGNUP_CODE` | *(empty)* | if set, sign-up requires this invite code |
| `ALLOW_SIGNUP` | `1` | `0` disables sign-up entirely |
| `COOKIE_SECURE` | `0` | set to `1` once the UI is served over HTTPS |
| `MAX_FRAMES` | *(unset)* | caps frames per job; for staging only, production uses the workflow's cap |

Inside the UI container, `OUT_DIR`, `GALLERY_DIR` and `DOCS_DIR` are already set by the compose
file, and `SERVICES_FILE` by `generated/docker-compose.yml`.

Per workflow, in `workflows/workflows.json`: label and descriptions, node ids, output prefix,
`timeout_hours`, `trim_frames`, ComfyUI port and the compose service it is copied from. GPU
placement: `spare_gpus` and `gpu_plan` in the same file. Job limits: `MAX_PER_USER` and
`MAX_WAITING` in `app/server.py`.

## Repository layout

```
app/
  server.py             FastAPI: jobs, one shared queue with one worker per GPU, /api/*, pages
  comfy_client.py       ComfyClient: upload, submit, websocket progress, /history, /view
  workflows.py          one adapter per workflow, built from workflows/workflows.json
  workflow_settings.py  admin controls: list, validation, overrides
  auth.py               accounts, sessions, admin CLI
  mp4.py                trimming and faststart of results
  static/               index.html (4 tabs), login.html, admin.html; no build step, no CDN
workflows/
  workflows.json              every workflow the UI offers: node ids, labels, limits
  replace_workflow_src.json   the graph as exported from ComfyUI
  replace_workflow.json       the hardened graph the service runs (generated)
  animate_workflow.json
scripts/
  harden_replace_workflow.py  src → hardened workflow, validated
  gen_requirements.py         regenerates docker/requirements-lock.txt
  smoke.py                    node-class check or one end-to-end job
  gen_config.py               counts GPUs, writes generated/ (run by start.sh)
start.sh                start on however many GPUs there are
generated/              services.json + docker-compose.yml for this machine (gitignored)
docker/                 Dockerfile.replace, Dockerfile.ui, entrypoint-comfy.sh, lock files
assets/                 gallery/ and documentation/ media (gitignored)
data/                   runtime bind mounts (gitignored)
```

## Where data lives

Everything under `data/` is created at runtime and is gitignored.

| Path | Contents |
|---|---|
| `data/replace/{input,output,temp,user}` | the replace backend's ComfyUI dirs |
| `data/animate/{input,output,temp}` | the animate backend's ComfyUI dirs |
| `data/ui_outputs/replace/`, `.../animate/` | finished, named results (`Cy_*_NNNNN.mp4`) |
| `data/ui_outputs/inputs/<service>/` | the image, video and `__settings.json` behind each result |
| `data/ui_outputs/users.db` | accounts, sessions and result ownership |
| `data/ui_outputs/workflow_settings.json` | admin overrides |
| `data/ui_outputs/incoming/` | in-progress downloads; cleared at startup |

The UI fetches results from the backends over HTTP (`/view`), so the UI container does not need
access to the backends' folders.

## Editing the replacement workflow

Do not edit `replace_workflow.json` by hand. Edit or re-export `replace_workflow_src.json`, then
regenerate:

```bash
python3 scripts/harden_replace_workflow.py            # rewrites replace_workflow.json
python3 scripts/harden_replace_workflow.py --check    # validate only
```

The hardener refuses to write a broken graph. To change a *value* rather than the graph, use the
[admin page](#admin-page) instead.

Node ids for the inputs and the final video are defined in `app/workflows.py` and nowhere else.
Several nodes that look like debug nodes are actually in the data path, for example `500`, `167`,
`208`, `235`, `377` and `412`. [CLAUDE.md](CLAUDE.md) lists them.

## Testing

```bash
# Does a backend resolve every node class the workflow needs? Seconds, no GPU work.
python3 scripts/smoke.py --service replace --check-only --image x --video y

# One real job, end to end (~6 min with 49 frames).
python3 scripts/smoke.py --service replace --url http://127.0.0.1:8136 \
    --image ref.jpg --video drive.mp4 --frames 49 --out /tmp/out
```

Run these with a Python environment that has `httpx` and `websocket-client`, such as
`../venv/bin/python`. Use frame counts of the form 4n+1, such as 49, so the Wan latent stays
aligned. To reach a backend from the host, uncomment its `ports:` entry in `docker-compose.yml`.

## Timings

Measured on 49-frame runs:

- **Replace:** about 3 min fixed, plus about 3.3 s per frame. At the default 1040-frame cap, a job
  takes **about 60 min**.
- **Animate:** about 6 min for 49 frames.

The UI waits up to 3 h for a replace job and 2 h for an animate job before giving up.

## Troubleshooting

| Symptom | Explanation |
|---|---|
| `comfy-animate` takes minutes to boot | A custom node runs `pip install` at startup (`comfyui-logicutils`, ~139 s). This is known and needs network access. |
| `IMPORT FAILED` lines in the animate log | Expected for a few nodes (was-node-suite, impact-pack, …). Run `smoke.py --check-only` to confirm every class the workflow needs still resolves. |
| `[Errno 30] Read-only file system` in the replace log | `comfyui-lora-manager` tries to write metadata next to the read-only models. Harmless. |
| A job vanished after a UI restart | Jobs are kept in memory. The result is still in `data/<service>/output/`. |
| Results are not trimmed | The log shows a startup warning: `ffmpeg`/`ffprobe` is missing from the UI image. |
| A tab shows its backend as offline | That container is down or still loading. Check `docker compose ps` and its logs. The other tab is unaffected. |
| A change in `nouran_Rep` broke replace | `comfy-replace` runs that host's venv and nodes directly. A `pip install` there takes effect on the next restart. |

## Security notes

- **Never mount `nouran_Rep/ComfyUI` as a whole directory, and never clone its remote in a
  Dockerfile.** Its `.git/config` contains a GitHub token. The compose file mounts only the
  subdirectories it needs.
- Set `COOKIE_SECURE=1` and serve the UI over HTTPS before exposing it beyond a trusted network.
- Restrict sign-up with `SIGNUP_CODE`, or close it with `ALLOW_SIGNUP=0`, once your users have
  accounts.
- `/gallery` and `/docs-assets` are public by design. Put only curated content there.
