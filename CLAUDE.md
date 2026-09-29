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

**Each container bakes its own ComfyUI.** Neither mounts ComfyUI *source* from the host - only
`models/` and `custom_nodes/` come from outside. Verified from the running containers:

| | `comfy-replace` | `comfy-animate` |
|---|---|---|
| ComfyUI | **0.27.0** (upstream tag `v0.27.0`) | **0.20.1** (commit `025e679`, 2026-05-03) |
| Python | 3.12.3 | 3.11.0rc1 (a release candidate) |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu128 |
| Baked in | `scail-replace` | `wan-animate` (pre-existing image) |

Host trees supply models and custom nodes only:
- animate → `../ComfyUI` — its **source is unused** by the container (that install is 0.22.0,
  the container runs 0.20.1). Only `models/` and `custom_nodes/` are mounted.
- replace → `../../nouran_Rep/ComfyUI` (0.27.0) — `models/` and 7 `custom_nodes/` dirs.

They need different ComfyUI versions and node sets. `../ComfyUI` **cannot** run the
replacement workflow at all — it is missing the nodes and every model.

In `nouran_Rep/ComfyUI`, only `custom_nodes/` and `user/` are under version control (a private
repo); the ComfyUI core there is unmodified upstream v0.27.0, verified by checksum. So
`SCAIL2ColoredMask` (`comfy_extras.nodes_scail`) and `SAM3_VideoTrack` (`comfy_extras.nodes_sam3`)
are **core** nodes, not custom ones — that surprises people.

> **Security:** `nouran_Rep/ComfyUI/.git/config` has a GitHub PAT embedded in the `origin` URL.
> Never clone that remote in a Dockerfile or copy that config into an image. It should be rotated.

## Layout

```
app/comfy_client.py   ComfyClient — one instance per backend (upload, submit, ws progress, /history)
app/workflows.py      Adapter per service: owns node ids, patches inputs, randomises seeds
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
| seeds | `169 212 239 373 484 408 411` | `324` |

## Editing the replacement workflow

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

## ONNX execution provider: the container is deliberately NOT identical to the host

`scripts/gen_requirements.py` excludes plain `onnxruntime`, leaving only `onnxruntime-gpu`, so the
container runs DWPose on CUDA:

| | providers |
|---|---|
| `nouran_Rep/venv` (host) | `AzureExecutionProvider`, `CPUExecutionProvider` - **CPU** |
| `comfy-replace` container | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |

The host has *both* `onnxruntime` and `onnxruntime-gpu` installed and the plain build shadows the
GPU one, so DWPose has always run on CPU there. CUDA and CPU ONNX give slightly different pose
keypoints, which propagates through the retarget, so **the container's output differs from the
host's for the same seed** (SSIM 0.78 on the generated half; the passed-through source half matches
at 0.994, confirming the wiring is identical).

This was measured, not assumed:

- container vs container, same seed: **SSIM 1.000000, PSNR inf** - bit-identical
- container vs host, same seed: SSIM 0.78

So the pipeline is fully deterministic *within* an environment. **The user chose to keep GPU ONNX**
(2026-09-29). Do not "fix" this by adding `onnxruntime` to the lock to match the host, and do not
treat a host-vs-container pixel difference as a bug. If output ever must match the old host
results exactly, adding `onnxruntime==1.26.0` in a layer after the lock install is the lever.

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

## State as of 2026-09-29

Validated end to end on GPU:
- **replace**: hardened workflow, 4 runs; `comfy-replace` container healthy on GPU 0, all 7 custom
  nodes import cleanly, all 37 node classes resolve, output correct (2104x1920, audio, node 335).
- **animate**: 33 classes resolve, full job in the old stack *and* in a container using
  `entrypoint-comfy.sh` with the new mounts (5.2 min / 49 frames).
- **UI**: image builds, container serves, both tabs render, a real job ran through its handler.
  It correctly comes up with one backend down.

Remaining: the cutover. `comfy-animate` needs GPU 1 and ports 8135/7835, held by the old
`wan_animate_gpu1` container, which is still serving and must be stopped first.
`../comfyui-wan-docker/docker-compose.yml` is the untouched rollback path.

Interim option that needs no cutover: point the UI's `ANIMATE_URL` at the old stack
(`http://<docker-gateway>:8135`) so the animate tab works against the existing container while
`comfy-replace` serves the replace tab.
