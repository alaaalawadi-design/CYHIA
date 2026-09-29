#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI-only entrypoint, shared by both GPU services.
#
#  Unlike comfyui-wan-docker/entrypoint.sh this starts no Gradio: the UI is its
#  own container talking to both backends over HTTP. That means the UI can be
#  restarted without evicting 14 GB of models from either GPU.
#
#  Env:
#    COMFY_PORT    HTTP port                                   (default 8188)
#    CUDA_DEVICE   CUDA index *inside* the container           (default 0)
#    RESERVE_VRAM  GiB held back from the model allocator      (default unset)
#    EXTRA_ARGS    anything else to pass to main.py
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COMFY_PORT="${COMFY_PORT:-8188}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
COMFY_DIR="${COMFY_DIR:-/app/ComfyUI}"

# The source, models and custom nodes are all bind-mounted. If a mount is
# missing the container would otherwise start and fail confusingly on the first
# job, so check the things that cannot be recovered from.
for path in "$COMFY_DIR/main.py" "$COMFY_DIR/models" "$COMFY_DIR/custom_nodes"; do
    if [ ! -e "$path" ]; then
        echo "FATAL: $path is missing — check the bind mounts in docker-compose.yml" >&2
        exit 1
    fi
done

# Written into by ComfyUI at run time, so they must not be read-only mounts.
mkdir -p "$COMFY_DIR/input" "$COMFY_DIR/output" "$COMFY_DIR/temp"

args=(
    --listen 0.0.0.0
    --port "$COMFY_PORT"
    --cuda-device "$CUDA_DEVICE"
    --disable-auto-launch
    --preview-method none
)
[ -n "${RESERVE_VRAM:-}" ] && args+=(--reserve-vram "$RESERVE_VRAM")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})

echo "════════════════════════════════════════════════"
echo "  ComfyUI  : http://0.0.0.0:${COMFY_PORT}"
echo "  GPU      : cuda:${CUDA_DEVICE}  (host GPU = NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-?})"
echo "  Source   : ${COMFY_DIR}"
echo "════════════════════════════════════════════════"

cd "$COMFY_DIR"
# exec: ComfyUI becomes PID 1, so docker stop and `restart: unless-stopped`
# (the OOM recovery path) work without a supervising shell swallowing signals.
exec python main.py "${args[@]}"
