# Img-Lipsync tab (internal name `lipsync`; LongCat-Video-Avatar 1.5): a script pipeline

Image + speech audio + prompt → talking video, as a CYHIA tab next to Replace,
Animate and Restore. Users set only those three; every other setting is fixed in
`workflows/pipelines.json` → `args`.

Unlike the other tabs this is **not ComfyUI**: a small worker runs
`/home/administrator/Disk/alaa-work/LongCat-Video/run_avatar.sh` once per job.
The UI drives it through the same shared queue, so it gets the same job list,
History, gallery and GPU sharing, and never runs at the same time as another job
on GPU 0.

```
browser ─► cyhia_ui (app/server.py) ─ shared GPU queue ─► app/pipelines.py (PipelineClient)
                                                            │ HTTP: upload, jobs, progress, files
                                                            ▼
                                     cyhia_lipsync (pipelines/lipsync/worker.py)
                                                            │ subprocess
                                                            ▼
                                     run_avatar.sh → avatar_generate.py (GPU)
```

## Files

| File | New / changed | What |
|---|---|---|
| `workflows/pipelines.json` | new | the tab: label, texts, default prompt, fixed script args, timeout, GPU, worker URL |
| `app/pipelines.py` | new | `PipelineAdapter` + `PipelineClient`: the UI side of the worker |
| `pipelines/lipsync/worker.py` | new | the worker (standard library only; runs on the LongCat env's Python) |
| `pipelines/lipsync/up.sh` | new | create `data/lipsync/` and start the worker |
| `docker-compose.lipsync.yml` | new | the worker container, its own compose project |
| `app/server.py` | changed | hooks, all inactive for the ComfyUI tabs (see below) |
| `app/static/index.html` | changed | audio slot, prompt box, History audio/prompt (only on tabs that ask for them) |
| `assets/gallery/lipsync/` | new, not in git | 7 gallery cases (gitignored like the other galleries) |

Not touched: `docker-compose.yml`, `workflows/workflows.json`, `start.sh`,
`scripts/gen_config.py`, `app/comfy_client.py`, `app/workflows.py`, the ComfyUI
containers.

`app/server.py` hooks:
- the tabs in `pipelines.json` are added to the tab list and the shared queue after the ComfyUI tabs are built;
- `create_job` takes an optional `prompt` and accepts audio for audio tabs; the ComfyUI tabs ignore the prompt and still accept video only;
- `_execute` passes the prompt only when the job has one;
- `_keep_inputs` / History save and show `__audio.*` and `__prompt.txt` for pipeline tabs;
- `/api/services` reports `media`, `needs_prompt`, `default_prompt`, `max_prompt_chars` (defaults for the ComfyUI tabs).

Tested locally against the original `main`: for Replace, Animate and Restore,
`/api/services` is identical apart from the four new keys, and so are the
gallery and the submit/validation behaviour. The only visible difference is the
status line of the GPU-0 tabs, which now lists Lip sync among the tabs sharing
GPU 0 (as it does for Restore).

## Deploy (server .12)

Verified on the server (2026-11-04): network `cy_hia_cyhia` exists, image
`scail-replace:latest` is present, `administrator` is uid/gid 1000, the
`longcat` env is a self-contained conda env, and `LongCat-Video/weights` has no
symlinks. If any of these change, set `CYHIA_NETWORK`, `LIPSYNC_IMAGE` or
`LIPSYNC_UID`/`LIPSYNC_GID`.

Then, in the CYHIA folder with this branch checked out:

```bash
pipelines/lipsync/up.sh                                      # 1. the worker only
docker logs cyhia_lipsync                                    #    "lipsync worker on :8138 ..."
rsync assets/gallery/lipsync/ → <cy_hia>/assets/gallery/lipsync/   # 2. gallery (no restart needed)
docker compose restart ui                                    # 3. the UI loads the new tab
```

Step 3 forgets waiting/running jobs of every tab (jobs live in memory), so do it
when nothing is queued.

## Roll back

```bash
docker compose -f docker-compose.lipsync.yml down     # stop the worker
git checkout main -- app/server.py app/static/index.html && rm workflows/pipelines.json
docker compose restart ui
```
Without `workflows/pipelines.json` the UI has no Lip sync tab and runs exactly as before.

## Timing and limits

About 8–10 min per segment (the first is 93 frames = 3.72 s, then 80 frames =
3.2 s each, at 25 fps) at 720p: 20 s of audio took 59 min, 35 s ~70 min. There is
no audio length limit: the whole audio is used (`args.max_seconds` is not set).
`timeout_hours` (72) in pipelines.json and `LIPSYNC_MAX_HOURS` (72) in
docker-compose.lipsync.yml only stop a job that hangs.
