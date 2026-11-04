#!/usr/bin/env bash
# Start (or update) the Lip sync worker only. Never touches the other services.
#   pipelines/lipsync/up.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
# Created here, as this user, so the worker (running as the same uid) can write.
mkdir -p data/lipsync/{input,output,jobs,home,cache}
export LIPSYNC_UID="${LIPSYNC_UID:-$(id -u)}" LIPSYNC_GID="${LIPSYNC_GID:-$(id -g)}"
docker compose -f docker-compose.lipsync.yml up -d
docker compose -f docker-compose.lipsync.yml ps --format 'table {{.Name}}\t{{.Status}}'
