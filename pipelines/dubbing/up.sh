#!/usr/bin/env bash
# Start (or update) the Dubbing worker only. Never touches the other services.
#   pipelines/dubbing/up.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
# Created here, as this user, so the worker (running as the same uid) can write.
mkdir -p data/dubbing/{input,output,jobs,home,cache}
export DUBBING_UID="${DUBBING_UID:-$(id -u)}" DUBBING_GID="${DUBBING_GID:-$(id -g)}"
docker compose -f docker-compose.dubbing.yml up -d
docker compose -f docker-compose.dubbing.yml ps --format 'table {{.Name}}\t{{.Status}}'
