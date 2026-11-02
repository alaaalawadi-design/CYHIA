# cy_hia

One HTML UI over ComfyUI backends on however many GPUs the machine has (32 GB RTX 5090s).

| Service | GPU | ComfyUI port | Workflow | What it does |
|---|---|---|---|---|
| `comfy-replace` | set by `start.sh` (0 in `docker-compose.yml`) | 8136 | `workflows/replace_workflow.json` | SCAIL-2 character replacement |
| `comfy-animate` | set by `start.sh` (1 in `docker-compose.yml`) | 8134 | `workflows/animate_workflow.json` | WAN 2.2 Animate |
| `ui` | none | **7860** (the only published port) | — | one tab per workflow |

Since 2026-10-21 the machine has **one GPU** (GPU 1 dropped out), and `./start.sh` runs both
backends on GPU 0 behind one shared queue. See "GPU layout and the shared queue" below.

The backends run **ComfyUI only** — no UI inside them. That is the point of the split: the
UI can be restarted without evicting ~15 GB of weights from either GPU.

## Two separate ComfyUI installs — do not merge them

**Each container bakes its own ComfyUI core.** `models/` and `custom_nodes/` come from outside,
and since 2026-10-04 `comfy-replace` also runs the **host's venv** (see "comfy-replace mirrors
nouran_Rep" below). Verified from the running containers:

| | `comfy-replace` | `comfy-animate` |
|---|---|---|
| ComfyUI | **0.27.0** (upstream tag `v0.27.0`) | **0.20.1** (commit `025e679`, 2026-05-03) |
| Python | 3.12.3, **host venv mounted** | 3.11.0rc1 (a release candidate) |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu128 |
| Baked in | `scail-replace` | `wan-animate` (pre-existing image) |

Host trees supply models and custom nodes only:
- animate → `../ComfyUI` — its **source is unused** by the container (that install is 0.22.0,
  the container runs 0.20.1). Only `models/` and `custom_nodes/` are mounted.
- replace → `../../nouran_Rep/ComfyUI` (0.27.0) — `models/`, the whole `custom_nodes/` tree,
  `comfy/ops.py`, plus `../../nouran_Rep/venv` and `/usr/local/cuda-12.8`. All read-only.

They need different ComfyUI versions and node sets. `../ComfyUI` **cannot** run the
replacement workflow at all — it is missing the nodes and every model.

In `nouran_Rep/ComfyUI`, only `custom_nodes/` and `user/` are under version control (a private
repo); the ComfyUI core there is upstream v0.27.0 with **one local patch**: `comfy/ops.py`
(two places, tolerates an all-zero `layer_conf` instead of `json.loads` failing; the original is
kept as `ops.py.bak`). Everything else matches the tag (`diff -r`, 2026-10-04). So
`SCAIL2ColoredMask` (`comfy_extras.nodes_scail`) and `SAM3_VideoTrack` (`comfy_extras.nodes_sam3`)
are **core** nodes, not custom ones — that surprises people.

> **Security:** `nouran_Rep/ComfyUI/.git/config` has a GitHub PAT embedded in the `origin` URL.
> Never clone that remote in a Dockerfile or copy that config into an image. It should be rotated.
> For the same reason **never bind-mount the `nouran_Rep/ComfyUI` directory whole** — that puts
> `.git/config` inside the container. Mount its subdirectories, as `docker-compose.yml` does.

## comfy-replace mirrors nouran_Rep (user's choice, 2026-10-04)

The user asked for the container to be identical to `nouran_Rep/ComfyUI`. It is done by mounting,
not by rebuilding the environment, because host and image are both Ubuntu 24.04 / Python 3.12.3:

- `nouran_Rep/venv` is mounted **at its own host path** (venv scripts and `pyvenv.cfg` carry that
  path) and put first on `PATH`. The image's `/opt/venv` is unused but kept as a fallback.
- the whole `custom_nodes/` tree (~60 nodes), `models/`, and the patched `comfy/ops.py`.
- `/usr/local/cuda-12.8`, added to `LD_LIBRARY_PATH`: bitsandbytes, sam2 and the ONNX CUDA
  provider link against CUDA 12 libraries that the host finds through ldconfig.
- the last layer of `Dockerfile.replace` adds the system pieces that venv expects:
  `/usr/bin/python3`, `libegl1 libgles2` (mediapipe), `libtbb12` (numba), `libibverbs1`.

Measured against a host run of the same install: **3178 node classes on both, none differing;
`pip freeze` byte-identical (427 packages); same `ops.py` checksum; no `IMPORT FAILED`**.

Consequences to know about:
- A `pip install` or node change in `nouran_Rep` takes effect here on the next container restart.
  That is the point, and also the risk: breaking that venv breaks this service.
- The mounts are read-only, so nothing here can alter the host install. `comfyui-lora-manager`
  logs `[Errno 30] Read-only file system` for each `*.metadata.json` it tries to write next to a
  model at startup. Harmless.
- Not mirrored: `user/` (the service keeps its own in `data/replace/user`, so ComfyUI-Manager
  settings are separate), `HF_HUB_OFFLINE=1`, and the launch flags in `entrypoint-comfy.sh`.
- To go back to the self-contained image: restore the 7 individual `custom_nodes` mounts and drop
  the venv, `ops.py`, cuda-12.8 mounts and the `PATH`/`VIRTUAL_ENV`/`LD_LIBRARY_PATH` overrides.

## GPU layout and the shared queue (2026-10-21, branch nouran/gpu-executors)

Nothing about GPUs or workflows is fixed in the code any more:

- `workflows/workflows.json` — every workflow (= tab): label, blurbs, node ids, output prefix,
  `timeout_hours`, `trim_frames`, ComfyUI `port`, `compose_template` (the service in
  `docker-compose.yml` it is copied from), `default_gpu`. Also `spare_gpus` and `gpu_plan`.
- `./start.sh` → `scripts/gen_config.py` counts GPUs with `nvidia-smi`, places the workflows
  (shared if fewer GPUs than workflows; one each; spare GPUs cycle through `spare_gpus` —
  replace, animate, replace, … — so neither workflow is starved; or `gpu_plan` by hand) and writes `generated/services.json` + `generated/docker-compose.yml`.
  The generated compose is `docker compose config` of the hand-written file with, per copy, only
  the GPU (`device_ids` + `NVIDIA_VISIBLE_DEVICES`), container name and `data/<wf>-<n>/` folders
  changed. Same project name (`cy_hia`) and network, so unchanged containers keep running.
  `start.sh` deliberately does **not** pass `--remove-orphans`: `cyhia_comfy_replace2` (stopped,
  from the `dual-replace-gpu1` branch) belongs to the same project and must not be deleted silently.
- `app/server.py` reads `generated/services.json` (`SERVICES_FILE`); without it, it falls back to
  one server per workflow at `default_gpu` / `http://comfy-<name>:<port>` (`<NAME>_URL`/`_GPU`
  override), which is exactly the old layout, so the hand-written compose still works.

Scheduler (`server.py`): one waiting list (`Job.seq` = submission order across all tabs and users),
one worker thread per GPU (`_gpu_worker`), which takes the oldest queued job a live server on its
GPU can run (`_pick`). Before running on a server other than the one that ran the GPU's previous
job (or after a restart), `_make_room` POSTs ComfyUI's `/free` (`unload_models` + `free_memory`)
to the other servers on that GPU and waits up to `FREE_VRAM_WAIT` (60 s) for `FREE_VRAM_SHARE`
(85 %) of the card to be free, then starts anyway with a log line. `/free` never interrupts a
running prompt, so a job submitted to ComfyUI directly (smoke.py, port 8136) can still collide.

**VRAM is released after every job (2026-11-02).** With `free_after_run` (per workflow in
`workflows.json`, default true) the GPU worker calls `_release` after each job, done or failed:
`/free` on the server that just ran it, then the same wait, logging how many GiB came back. It
runs after the job is marked done, so it never delays a result; the next job on that GPU waits
for it. Measured: an idle animate server went from 11.3 GB to 0.8 GB, and replace sits at 0.5 GB
after a `/free` (what remains is the CUDA context). So custom nodes do not hold weights
outside ComfyUI's model manager, and `--disable-smart-memory` or restarting containers is not
needed. The cost is a model reload at the start of every job (not yet timed). `_make_room` stays as
the safety net for UI restarts and jobs sent to ComfyUI directly.

Server health is cached (`_health_loop`, every 10 s); a job whose workflow has no live server
anywhere is failed rather than left blocking the queue, and a submit to such a tab gets a 503.

Limits: `MAX_PER_USER = 5` per user per tab, `MAX_WAITING = 10` per tab for everyone, both at
the top of `server.py`. The page reads the per-user limit from `/api/services`.

Page: tabs and the History filter come from `/api/services`; each tab has a "Your jobs" list fed
by one poll of `/api/jobs` (the caller's jobs, oldest first), so it survives reloads and other
browsers; finished jobs are removed from it with `/api/jobs/{id}/dismiss`. Generate stays enabled
while jobs run.

Tested with fake ComfyUI servers (strict order on a shared GPU, parallel on two, two Replace
servers, offline server, limit of 5, per-user isolation) and in headless Chromium.

## Layout

```
workflows/workflows.json  every workflow the UI offers (see above)
start.sh              count GPUs, write generated/, start; --plan only prints the layout
scripts/gen_config.py what start.sh runs; --gpus N to plan for another machine
generated/            services.json + docker-compose.yml for this machine (gitignored)
app/comfy_client.py   ComfyClient — one instance per backend (upload, submit, ws progress, /history, /free)
app/workflows.py      Adapter per workflow, built from workflows.json: patches inputs and the output prefix
app/server.py         FastAPI: in-memory jobs, one shared queue, one worker thread per GPU, /api/*
app/static/index.html the whole front end — plain HTML/CSS/JS, no build step, no CDN
scripts/harden_replace_workflow.py   src workflow -> service workflow (declarative + validated)
scripts/gen_requirements.py          derives docker/requirements-lock.txt
scripts/smoke.py                     one end-to-end job, either service; --check-only
workflows/            replace_workflow_src.json (as exported) -> replace_workflow.json (used)
docker/               Dockerfile.replace, Dockerfile.ui, entrypoint-comfy.sh, lock files
data/                 bind-mount targets (gitignored): per-service input/output/temp
assets/documentation/ stills and clips for the Documentation tab (gitignored), served at
                      /docs-assets; assets/gallery/ is the per-tab Gallery, served at /gallery
```

## Workflow node ids

Set in `workflows/workflows.json` (`nodes`), read by `app/workflows.py`. Never hardcode these elsewhere.

| | replace | animate |
|---|---|---|
| reference image | `188` | `311` |
| driving video | `384` | `417` |
| **final video** | `335` (side-by-side + audio) | `462` |

The adapter also overwrites `filename_prefix` on the final-video node (`Cy_Replace` /
`Cy_Animate`). Seed nodes — `169 212 239 373 484 408 411` for replace, `324` for animate — are
**no longer touched by the adapter**; see "Results are deterministic" below. The harden script
still validates that they exist.

## Editing the replacement workflow

> **The drift is resolved (2026-10-06).** `replace_workflow_src.json` is now the re-export of the
> tuned graph, and `replace_workflow.json` was regenerated from it by `harden_replace_workflow.py`.
> The service runs a **hardened** 108-node graph again; `--check` passes with 0 inert nodes and all
> 37 classes resolve against `comfy-replace`. Running the hardener is safe — it is idempotent and
> no longer reverts anything.
>
> The tuned values the old source would have reverted are now *in* the source: `484.steps` 15,
> `408.window_length` 125, `481.strength_model` 0.8, `311`/`317.detection_threshold` 0.5,
> `499.upscale_method` `nearest-exact`, node `525` present, `384.frame_load_cap` 1040. Nodes `527`
> and `528` (debug `SaveImageAdvanced` of the mask previews, consumed by nothing) are gone with the
> re-export, as are the duplicate loaders — `482`→`113`, `483`→`116`, `478`→`117`, so the Klein 9B
> UNET, the Qwen 8B CLIP and the Flux2 VAE each load once instead of twice.
>
> One semantic nuance survives from hardening: node `493`, where was-ns `Text Concatenate`
> (`clean_whitespace=true`, which strips each operand and drops empty ones) becomes core
> `StringConcatenate` (a plain `delimiter.join`). The `", "` delimiter is preserved and `492` has
> no stray whitespace, so the only possible difference is trailing whitespace from the QwenVL
> output — almost certainly invisible to the tokenizer, but not provably identical.
>
> **Note `487` is no longer in the data path.** The tuned graph feeds `317.images`, `331.image` and
> `408.reference_image` from `477` (the VAE decode) rather than from the `SaveImage` node, so the
> old trap note below applies only to `500`.

Edit `replace_workflow_src.json` (or replace it with a fresh export), then:

```bash
python3 scripts/harden_replace_workflow.py            # rewrites replace_workflow.json
python3 scripts/harden_replace_workflow.py --check    # validate without rewriting
```

It is declarative and refuses to emit a broken graph: every wire must resolve, the input/output/seed
nodes must survive, and it will not drop a node something still consumes. It applies six things —
QwenVL `keep_model_loaded=False`, duplicate loader merges, debug-output removal, a core-node
substitution, pass-through bypass, and pruning of nodes ComfyUI would never execute.

**Traps when editing this graph:**
- `500` looks like a debug `SaveImage` node but its **image output is consumed downstream**
  (`499.image`). Removing it breaks the pipeline. `487` was in the same position before the
  2026-10-06 re-export and no longer is.
- `167`, `208`, `235`, `377` are `PreviewImage` nodes used as **pass-throughs in the data path**.
- `412` is `PreviewAny`, which passes STRING through in this ComfyUI version.
- Inert nodes are not harmless: each drags in its custom node (node `233` alone pulled in
  `ComfyUI-WanAnimatePreprocess` and an onnxruntime stack).

## Dependencies — do not use `pip freeze`

> Since 2026-10-04 `comfy-replace` runs the mounted host venv, so `requirements-lock.txt` only
> defines the image's fallback `/opt/venv`. The rest of this section applies to that fallback.

`nouran_Rep/venv` serves **60** custom nodes; this workflow needs **7**. A full freeze pulls in
`dlib` (long source build) and `diffusers==0.40.0.dev0` (never published to PyPI, so the build
fails). Regenerate with the validated interpreter:

```bash
/home/administrator/Disk/nouran_Rep/venv/bin/python scripts/gen_requirements.py
```

It walks the dependency closure from ComfyUI core plus those 7 node dirs and pins each package to
the version installed in that venv: **147 packages, not 399**. The 7 dirs are listed in
`NEEDED_NODES` there and mounted individually in `docker-compose.yml` — keep the two in step.

Derive the list from ComfyUI itself rather than guessing, via `python_module` in `/object_info`.

## Building and running

```bash
./start.sh                                  # normal start: counts GPUs, see above
./start.sh --plan [--gpus N]                # just print the layout
docker compose build comfy-replace          # ~45 min, see note below
docker compose up -d comfy-replace          # old fixed layout: GPU 0
docker compose up -d                       # + the UI on :7860 (service `ui`; was `gradio-ui`)
                                           # no --remove-orphans: it would delete cyhia_comfy_replace2
docker compose --profile phase3 up -d comfy-animate   # GPU 1
docker compose logs -f comfy-replace
```

**Docker Hub pulls at ~1.2 MB/s from this machine** while PyPI and GitHub run at 1–2.3 MB/s. Base
image size therefore dominates build time — that is why `Dockerfile.replace` uses the
**`runtime`** CUDA base, not `devel`. Nothing builds from source, so `devel` bought nothing and
cost ~35 extra minutes. Do not "upgrade" it back.

Other Dockerfile constraints:
- torch `2.11.0+cu130` from the CUDA 13.0 wheel index; the 5090 is sm_120.
- Packages go into a **venv at `/opt/venv`**, not the system Python. Ubuntu 24.04's
  Debian-packaged pip has no `RECORD` file, so `pip install --upgrade pip` dies with
  `Cannot uninstall pip 24.0`; `PIP_BREAK_SYSTEM_PACKAGES` does not save you. The venv also
  matches the venv the lock was generated from.
- `HF_HOME` must stay **writable** (`/app/hf`). `models/` is a read-only mount.
- ENV lines placed before the pip layers invalidate the whole cache when edited — put late
  corrections near the bottom.

## Known quirks in the animate service (pre-existing, not introduced here)

- Booting `comfy-animate` runs a **pip install at startup**: a custom node
  (`comfyui-logicutils`, ~139 s) fetches `dghs-imgutils` and downgrades numpy to 1.26.4. Startup
  is therefore network-dependent and not reproducible. It happens with the old
  `comfyui-wan-docker` entrypoint too. Consider pinning those deps into the image.
- Several custom nodes log `IMPORT FAILED` (`was-node-suite`, `comfyui-impact-pack`,
  `comfyui_layerstyle`, `ComfyUI-Crystools`, `comfyui-logicutils`). **This is fine** - the same
  ones fail in the old running stack, and all 33 classes the animate workflow needs still
  resolve. Check with `smoke.py --check-only` instead of reading the log and panicking.

## ONNX execution provider: now CPU, same as the host

The host venv has both `onnxruntime` and `onnxruntime-gpu`; the plain build shadows the GPU one,
so DWPose runs on **CPU** there (`AzureExecutionProvider`, `CPUExecutionProvider`). Since
`comfy-replace` uses that venv, it now reports exactly the same providers.

History: from 2026-09-29 to 2026-10-04 the container ran its own lock with only
`onnxruntime-gpu`, so DWPose ran on CUDA and output differed from the host for the same seed
(SSIM 0.78 on the generated half; container vs container was bit-identical). Outputs made in that
window will not match outputs made now. The user's 2026-10-04 request for a full host copy
supersedes the earlier choice to keep GPU ONNX - do not switch it back without asking.

Timing is not a factor either way: DWPose runs on single frames, well under a second on both.

## Testing

```bash
# does a backend resolve every node class? (seconds, no GPU work)
python3 scripts/smoke.py --service replace --check-only --image x --video y

# one real job end to end
python3 scripts/smoke.py --service replace --url http://127.0.0.1:8136 \
    --image ref.jpg --video drive.mp4 --frames 49 --out /tmp/out
```

Use `--frames 49` (4n+1 keeps the Wan latent aligned) for a ~6 min run instead of ~65 min.
Run these with a venv that has `httpx` and `websocket-client` — `../venv/bin/python` (it also has
`fastapi`/`uvicorn`, so `python -m app.server` runs the UI outside Docker).

## Timings (measured, 49-frame runs)

- **replace**: ~3 min fixed (the Flux2 Klein stages) + **~3.3 s/frame** (`WanSCAILInfinity`,
  windowed at 125 frames so VRAM stays bounded). The default 1040-frame cap ⇒ **about 60 min/job**.
- **animate**: 6.3 min for 49 frames.

UI timeouts are set from this: 3 h for replace, 2 h for animate. Do not lower them.

## Conventions

- The UI is **not Gradio** (replaced 2026-09-29). `app/server.py` + `app/static/index.html`; the page
  uploads with XHR, then polls `/api/jobs` (all of the user's jobs) every 2 s while any is active.
  A reload reconnects because the list comes from the server. Jobs are in memory: restarting the UI
  container forgets them (ComfyUI still finishes, the output stays in that backend's `data/*/output`).
- **Accounts (added 2026-10-19):** `/login` (`static/login.html`) signs up / signs in; `/` and every
  `/api/*` route except `/api/auth/*` need a session cookie (`app/auth.py`, stdlib only: SQLite
  at `ui_outputs/users.db`, scrypt passwords, only token hashes stored). Each result is recorded
  against its submitter when `_place_result()` names it, and History, `/api/history/...` and
  `/api/jobs/...` return only the caller's own; anyone else's is a 404. Results from before
  accounts have no owner and are hidden until `python -m app.auth assign-unowned <user>`.
  The `/gallery` and `/docs-assets` static mounts stay public (curated content). Env:
  `SIGNUP_CODE`, `ALLOW_SIGNUP`, `COOKIE_SECURE`.
- **Admin workflow controls (added 2026-10-19):** `/admin` (`static/admin.html`) lets an admin
  edit ~60 replace-workflow inputs (prompts, seeds, steps/cfg/sampler per stage, LoRA strengths,
  SAM3 thresholds, SCAIL window/pose/shift, resolution, frame cap, CRF). The list and its
  validation live in `CONTROLS` in `app/workflow_settings.py`; defaults are read from the workflow
  JSON, and only values that differ from it are stored, in `ui_outputs/workflow_settings.json`.
  `Adapter.build()` applies them when a job **starts**, for every user. Each result's overrides
  are saved as `inputs/<service>/<stem>__settings.json`. Admins are made only from the CLI:
  `python -m app.auth create-admin <user> <password>` / `set-admin <user> on|off`. To expose
  another control, add a `Control(node, input, ...)` row; the page renders it automatically.
- The UI shows **stage name + elapsed minutes, not a percentage** — ~100 nodes of uneven cost mean
  any fraction would be a lie.
- The UI must start even when a backend is down (`condition: service_started`, not
  `service_healthy`), so one dead GPU service never takes the other tab offline.
- One job at a time per **GPU** (one worker thread each), from one shared list in submission
  order. At most `MAX_PER_USER` (5) per user per tab and `MAX_WAITING` (10) per tab in all.
- Results are fetched over `/view`, never through a shared filesystem, so the UI container needs no
  mounts from the backends.
- Uploads get UUID filenames; the two backends have separate `input/` dirs.
- **Results are deterministic.** `app/workflows.py` leaves every seed exactly as saved in the
  workflow JSON (or as set on `/admin`), so the same image + video + settings always produce the
  same video. There is no re-roll:
  to get a different take, change an input. (Until 2026-10-06 the adapter randomised the seven
  replace seeds and the one animate seed on every run.)
- **Results are named by the UI, not by ComfyUI**: `_place_result()` moves each download to
  `Cy_Replace_00013.mp4` / `Cy_Animate_00013.mp4`, numbered one past the highest number already
  present in *either* `ui_outputs/<service>/` or the saved-inputs folder, under a lock. So a
  number is never reused and an older result can never be overwritten, whatever name ComfyUI gave
  the file.
- **Downloads land in `ui_outputs/incoming/<job id>/` first**, are trimmed and faststarted there,
  and only then get their final name. Nothing half-finished is ever visible to `/api/history`.
  The folder is removed per job, and the whole `incoming/` tree is cleared at startup to catch a
  container killed mid-download.
- **The UI has four tabs**: Replace, Animate, History and Documentation. The Documentation tab
  is the model evaluation — input criteria, strengths, limitations and worked examples — written
  as plain data (`DOCS`) in `index.html` and rendered on first open. Its media comes from
  `assets/documentation/` via `DOCS_DIR`, mounted read-only and served at `/docs-assets`. Each
  case is one reference image, the driving video it was run against, and the result; a figure
  whose file is missing removes itself, so the tab still reads with the directory absent.
- **Input criteria appear only in the Documentation tab** (removed from the Replace and Animate
  tabs 2026-10-19). Their source is `GUIDE` in `index.html`, rendered by `guideCols()`. `limits` there is what the pipeline enforces (formats,
  frame cap, timings); `disclaimers` is what the evaluation observed — a ~15 s stability window,
  the three animation resolutions, 2/66 segmentation and 1/66 relighting failures. Those are
  recommendations, not acceptance rules: do not turn them into upload validation.
- **The first `trim_frames` frames are cut off each result** (`workflows.json`; 10 for both
  services, so ~0.42 s at the workflows' 24 fps) to drop warm-up frames, and the saved driving video is cut by the same
  **duration** — not the same frame count, since the upload's native rate is usually not 24.
  This needs `ffmpeg` and `ffprobe`, which `Dockerfile.ui` installs; without them the server logs
  a warning at startup and every job returns untrimmed. A trim failure never fails a job. Set
  `trim_frames` to 0 per workflow to turn it off. **As of 2026-10-21 the running `cyhia-ui:latest`
  image has no ffmpeg** (built before the Dockerfile line was added), so nothing is trimmed until
  `docker compose build ui` is run.

## State as of 2026-09-29

Validated end to end on GPU:
- **replace**: hardened workflow, 4 runs; `comfy-replace` container healthy on GPU 0, all 7 custom
  nodes import cleanly, all 37 node classes resolve, output correct (2104x1920, audio, node 335).
- **animate**: 33 classes resolve, full job in the old stack *and* in a container using
  `entrypoint-comfy.sh` with the new mounts (5.2 min / 49 frames).
- **UI**: image builds, container serves, both tabs render, a real job ran through its handler.
  It correctly comes up with one backend down.

Update 2026-10-04: `comfy-replace` now mirrors `nouran_Rep` (host venv + all custom nodes, see
above). Re-validated: healthy on GPU 0, 3178 node classes identical to the host, one real 49-frame
job in 268 s with video + audio from node 335. The "7 custom nodes / 37 classes" line above
describes the earlier self-contained setup. All three containers of this compose file are up.

Remaining: the cutover. `comfy-animate` needs GPU 1 and ports 8135/7835, held by the old
`wan_animate_gpu1` container, which is still serving and must be stopped first.
`../comfyui-wan-docker/docker-compose.yml` is the untouched rollback path.

Interim option that needs no cutover: point the UI's `ANIMATE_URL` at the old stack
(`http://<docker-gateway>:8135`) so the animate tab works against the existing container while
`comfy-replace` serves the replace tab.
