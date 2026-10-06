# cy_hia

One HTML UI over two ComfyUI backends, one GPU each. Both GPUs are 32 GB RTX 5090s.

| Service | GPU | ComfyUI port | Workflow | What it does |
|---|---|---|---|---|
| `comfy-replace` | **0** | 8136 | `workflows/replace_workflow.json` | SCAIL-2 character replacement |
| `comfy-animate` | **1** | 8134 | `workflows/animate_workflow.json` | WAN 2.2 Animate |
| `ui` | none | **7860** (the only published port) | — | both tabs |

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

## Layout

```
app/comfy_client.py   ComfyClient — one instance per backend (upload, submit, ws progress, /history)
app/workflows.py      Adapter per service: owns node ids, patches inputs and the output prefix
app/server.py         FastAPI: in-memory jobs, one worker thread per backend, /api/*
app/static/index.html the whole front end — plain HTML/CSS/JS, no build step, no CDN
scripts/harden_replace_workflow.py   src workflow -> service workflow (declarative + validated)
scripts/gen_requirements.py          derives docker/requirements-lock.txt
scripts/smoke.py                     one end-to-end job, either service; --check-only
workflows/            replace_workflow_src.json (as exported) -> replace_workflow.json (used)
docker/               Dockerfile.replace, Dockerfile.ui, entrypoint-comfy.sh, lock files
data/                 bind-mount targets (gitignored): per-service input/output/temp
```

## Workflow node ids

Both adapters in `app/workflows.py`. Never hardcode these elsewhere.

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

> **The two files have drifted (as of 2026-10-06). Do not run the hardener before reading this.**
> `replace_workflow.json` was committed in c775706 as a hand-tuned raw export that never went
> through `harden_replace_workflow.py`, and `replace_workflow_src.json` was not updated to match.
> The service therefore runs an **unhardened** graph: node `411` still has
> `keep_model_loaded=True`, the duplicate loaders and ~17 debug/inert nodes are still present, and
> node `514` (`easy saveText`) writes into the working directory on every run.
>
> Regenerating from the current source would **revert live tuning**: `484.steps` 15→8,
> `408.window_length` 125→149, `481.strength_model` 0.8→1, `311`/`317.detection_threshold`
> 0.5→0.61, `499.upscale_method` `nearest-exact`→`lanczos`, three prompt strings, and the loss of
> nodes `525`/`527`/`528`. To fix the drift properly, re-export the tuned graph over the source
> file first, then harden.
>
> Hardening the *tuned* graph was checked for output safety: comparing the compute DAG from node
> `335` node by node, the only semantic change is node `493`, where was-ns `Text Concatenate`
> (`clean_whitespace=true`, which strips each operand and drops empty ones) becomes core
> `StringConcatenate` (a plain `delimiter.join`). The `", "` delimiter is preserved and `492` has
> no stray whitespace, so the only possible difference is trailing whitespace from the QwenVL
> output — almost certainly invisible to the tokenizer, but not provably identical.

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
- `487` and `500` look like debug `SaveImage` nodes but their **image output is consumed
  downstream**. Removing them breaks the pipeline.
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
docker compose build comfy-replace          # ~45 min, see note below
docker compose up -d comfy-replace          # GPU 0
docker compose up -d --remove-orphans      # + the UI on :7860 (service `ui`; was `gradio-ui`)
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
  windowed at 149 frames so VRAM stays bounded). The default 1140-frame cap ⇒ **65–70 min/job**.
- **animate**: 6.3 min for 49 frames.

UI timeouts are set from this: 3 h for replace, 2 h for animate. Do not lower them.

## Conventions

- The UI is **not Gradio** (replaced 2026-09-29). `app/server.py` + `app/static/index.html`; the page
  uploads with XHR, then polls `/api/jobs/{id}` every 2 s. The page keeps its job id in
  `localStorage`, so a reload reconnects. Jobs are in memory: restarting the UI container forgets
  them (ComfyUI still finishes, the output stays in that backend's `data/*/output`).
- The UI shows **stage name + elapsed minutes, not a percentage** — ~100 nodes of uneven cost mean
  any fraction would be a lie.
- The UI must start even when a backend is down (`condition: service_started`, not
  `service_healthy`), so one dead GPU service never takes the other tab offline.
- One job at a time per backend (a single-worker executor each): one GPU each. At most 10
  waiting per backend.
- Results are fetched over `/view`, never through a shared filesystem, so the UI container needs no
  mounts from the backends.
- Uploads get UUID filenames; the two backends have separate `input/` dirs.
- **Results are deterministic.** `app/workflows.py` leaves every seed exactly as saved in the
  workflow JSON, so the same image + video always produce the same video. There is no re-roll:
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
- **The first `TRIM_FRAMES` frames are cut off each result** (10 for both services, so 0.42 s at
  the workflows' 24 fps) to drop warm-up frames, and the saved driving video is cut by the same
  **duration** — not the same frame count, since the upload's native rate is usually not 24.
  This needs `ffmpeg` and `ffprobe`, which `Dockerfile.ui` installs; without them the server logs
  a warning at startup and every job returns untrimmed. A trim failure never fails a job. Set
  `TRIM_FRAMES` to 0 per service to turn it off.

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
