# Video_Lipsync tab (internal name `dubbing`; TBDub): a script pipeline

Video + dubbed speech audio → the same video with the speaker's lips re-synced
to the new audio, as a CYHIA tab next to Lip sync. Built the same way as Lip
sync (see `pipelines/lipsync/README.md`): a small worker runs TBDub's
`episode_dub.py` once per job, and the UI drives it through the shared GPU queue.

- **Inputs:** a video and an audio file, each **up to 30 minutes** (checked in the
  page and again by the worker with ffprobe). No image, no prompt.
- **Fixed settings** (`workflows/pipelines.json` → `dubbing.args`), the ones that made
  the gallery samples: `--inference-mode student --num-student-steps 2
  --sigma-shift 1.0 --motion-from-latents --seed 42`, plus `--cleanup`.
  Speaker detection for shots with several faces is on (episode_dub's default).
- **TBDub install:** `/home/administrator/Disk/alaa-work/TBDub-dubbing` on .12, copied
  from .22 (`/home/administrator/alaa-workspace/TBDub`): code, `checkpoints/`, and the
  venv that produced the samples (from `/home/administrator/abdelrahman/v_envs/TBDub_env`)
  as `venv/`. Test data and results were not copied. Mounted read-only.
- **Progress:** "Scanning faces" (frames), "Finding who speaks", "Planning segments",
  "Dubbing segments" (n of N), "Assembling video" (frames).

## Files

| File | New / changed | What |
|---|---|---|
| `pipelines/dubbing/worker.py` | new | the worker (standard library only, its own file) |
| `pipelines/dubbing/up.sh` | new | create `data/dubbing/` and start the worker |
| `docker-compose.dubbing.yml` | new | the worker container, its own compose project `cyhia-dubbing` |
| `workflows/pipelines.json` | changed | `dubbing` entry; `lipsync` prompt now fixed and hidden |
| `app/pipelines.py` | changed | extra audio upload, length limit, per-tab captions; streamed uploads |
| `app/server.py` | changed | optional `audio` upload for tabs with `needs_audio`; fixed prompts are not sent to the page |
| `app/static/index.html` | changed | second audio card (removed on other tabs), 30-minute check, captions |
| `assets/gallery/dubbing/` | new, not in git | 13 samples under 10 minutes (input + student output) |

Not touched: `docker-compose.yml`, `workflows.json`, `start.sh`, `gen_config.py`,
`comfy_client.py`, `workflows.py`, the ComfyUI containers, `pipelines/lipsync/worker.py`.

## Deploy (server .12)

```bash
git checkout dubbingv1_integrate
pipelines/dubbing/up.sh                 # the Dubbing worker only
docker restart cyhia_ui                 # when nothing is queued: forgets waiting jobs
```

## Roll back to Lip sync only

```bash
docker compose -f docker-compose.dubbing.yml down
git checkout lipsyncv1_integrate
docker restart cyhia_ui
```
