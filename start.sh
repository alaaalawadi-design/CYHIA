#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Start cy_hia on however many GPUs this machine has.
#
#    ./start.sh              count GPUs, place the workflows, start everything
#    ./start.sh --plan       only show where each workflow would go
#    ./start.sh --gpus 2     plan for 2 GPUs whatever nvidia-smi says
#
#  1. scripts/gen_config.py counts the GPUs with nvidia-smi and writes
#     generated/services.json (for the UI) and generated/docker-compose.yml
#     (one ComfyUI container per server, each pinned to its GPU).
#  2. docker compose starts that file. Containers whose settings did not change
#     keep running untouched; only changed ones are recreated.
#  3. If the GPU layout changed, the UI is restarted to read it. That forgets
#     the UI's waiting jobs, so run this when nothing is queued.
#
#  Run it again whenever a GPU is added, removed or fails.
#  The hand-written docker-compose.yml still works on its own (old layout:
#  replace on GPU 0, animate on GPU 1) and is the template for step 1.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

GEN=generated
COMPOSE=(docker compose -f "$GEN/docker-compose.yml")

if [[ "${1:-}" == "--plan" ]]; then
    shift
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    python3 scripts/gen_config.py --out "$tmp" "$@" | grep -v '^wrote '
    exit 0
fi

before=""
[[ -f "$GEN/services.json" ]] && before=$(sha256sum < "$GEN/services.json")
python3 scripts/gen_config.py "$@"
after=$(sha256sum < "$GEN/services.json")

"${COMPOSE[@]}" up -d --remove-orphans

if [[ -n "$before" && "$before" != "$after" ]]; then
    echo "GPU layout changed: restarting the UI to pick it up"
    "${COMPOSE[@]}" restart ui
fi
"${COMPOSE[@]}" ps --format 'table {{.Name}}\t{{.Status}}'
