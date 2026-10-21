#!/usr/bin/env python3
"""
Count this machine's GPUs and decide which ComfyUI server runs on which.

Writes two files into cy_hia/generated/ (both gitignored, one set per machine):

  services.json        read by the UI: every server, its workflow, GPU and URL
  docker-compose.yml   one container per server, each pinned to its GPU

The containers are copied from the hand-written docker-compose.yml, so mounts,
images and health checks are only ever written once, there. This script only
changes, per copy: the GPU, the container name and the data/ folders.

Run by ./start.sh. On its own:

    python3 scripts/gen_config.py              # count GPUs with nvidia-smi
    python3 scripts/gen_config.py --gpus 4     # pretend there are 4 (dry planning)
    python3 scripts/gen_config.py --gpus 4 --out /tmp/plan   # and write elsewhere

How GPUs are assigned (workflows/workflows.json can override with gpu_plan):
  fewer GPUs than workflows -> workflows share GPUs, round robin
  one GPU per workflow      -> each gets its own
  more GPUs than workflows  -> spare GPUs get extra servers, in spare_gpus order
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "workflows" / "workflows.json"
TEMPLATE = ROOT / "docker-compose.yml"
DATA = ROOT / "data"


def detect_gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"cannot run nvidia-smi ({e}); pass --gpus N to plan by hand")
    gpus = []
    for line in out.strip().splitlines():
        index, name, mem, uuid = [p.strip() for p in line.split(",")]
        gpus.append({"index": int(index), "name": name, "memory_mb": int(mem), "uuid": uuid})
    if not gpus:
        sys.exit("nvidia-smi lists no GPUs")
    return gpus


def plan(workflows: list[str], gpus: list[int], spare: list[str],
         pinned: dict | None) -> dict[str, list[int]]:
    """workflow -> the GPUs it gets one server on."""
    out: dict[str, list[int]] = {w: [] for w in workflows}
    if pinned:
        for w, wanted in pinned.items():
            if w not in out:
                print(f"  gpu_plan: unknown workflow {w!r}, ignored")
                continue
            have = [g for g in wanted if g in gpus]
            for g in set(wanted) - set(have):
                print(f"  gpu_plan: {w} wants GPU {g}, which this machine does not have")
            out[w] = sorted(set(have))
    unplaced = [w for w in workflows if not out[w]]
    if pinned and not unplaced:
        return out
    if pinned:
        # Workflows the plan left without a GPU go where the fewest servers are.
        load = {g: sum(g in v for v in out.values()) for g in gpus}
        for w in unplaced:
            g = min(gpus, key=lambda g: (load[g], g))
            out[w], load[g] = [g], load[g] + 1
            print(f"  gpu_plan: {w} had no usable GPU, placed on GPU {g}")
        return out

    if len(gpus) < len(workflows):
        for i, w in enumerate(workflows):
            out[w] = [gpus[i % len(gpus)]]
        return out
    for i, w in enumerate(workflows):
        out[w] = [gpus[i]]
    order = [w for w in spare if w in out] or workflows
    for i, g in enumerate(gpus[len(workflows):]):
        out[order[i % len(order)]].append(g)
    return out


def resolved_template() -> dict:
    """docker-compose.yml with variables and anchors resolved by compose itself."""
    try:
        out = subprocess.run(
            ["docker", "compose", "-f", str(TEMPLATE), "config", "--format", "json"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"docker compose config failed:\n{e.stderr}")
    return json.loads(out)


def backend_service(template: dict, copy_no: int, gpu: int, dirs: list[Path]) -> dict:
    """A copy of `template` pinned to `gpu`. Data folders it needs go in `dirs`."""
    svc = copy.deepcopy(template)
    if copy_no > 1:
        svc.pop("build", None)  # the first copy builds the image; the rest reuse it
        svc["container_name"] = f"{template['container_name']}_{copy_no}"
        for v in svc.get("volumes", []):
            src = Path(v.get("source", ""))
            if v.get("type") == "bind" and DATA in src.parents:
                # data/replace/input -> data/replace-2/input: own inputs and outputs
                rel = src.relative_to(DATA)
                v["source"] = str(DATA / f"{rel.parts[0]}-{copy_no}" / Path(*rel.parts[1:]))
                dirs.append(Path(v["source"]))
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    for d in devices:
        if "gpu" in d.get("capabilities", []):
            d["device_ids"] = [str(gpu)]
    svc["environment"]["NVIDIA_VISIBLE_DEVICES"] = str(gpu)
    return svc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", type=int, help="plan for this many GPUs instead of asking nvidia-smi")
    ap.add_argument("--out", type=Path, default=ROOT / "generated")
    args = ap.parse_args()

    gpus = (
        [{"index": i, "name": "assumed (--gpus)"} for i in range(args.gpus)]
        if args.gpus else detect_gpus()
    )
    catalog = json.loads(CATALOG.read_text())
    workflows = list(catalog["workflows"])
    placed = plan(workflows, [g["index"] for g in gpus], catalog.get("spare_gpus") or [],
                  catalog.get("gpu_plan"))

    tpl = resolved_template()
    services, backends, dirs = {}, [], []
    for w in workflows:
        spec = catalog["workflows"][w]
        base = spec["compose_template"]
        if base not in tpl["services"]:
            sys.exit(f"{w}: compose_template {base!r} is not a service in docker-compose.yml")
        for n, gpu in enumerate(placed[w], start=1):
            name = base if n == 1 else f"{base}-{n}"
            services[name] = backend_service(tpl["services"][base], n, gpu, dirs)
            backends.append({"name": name, "workflow": w, "gpu": gpu,
                             "url": f"http://{name}:{spec['port']}"})

    ui = copy.deepcopy(tpl["services"]["ui"])
    for k in ("ANIMATE_URL", "REPLACE_URL"):  # superseded by services.json
        ui["environment"].pop(k, None)
    ui["environment"]["SERVICES_FILE"] = "/app/generated/services.json"
    ui.setdefault("volumes", []).append(
        {"type": "bind", "source": str(args.out.resolve()), "target": "/app/generated",
         "read_only": True})
    ui["depends_on"] = {b["name"]: {"condition": "service_started", "required": True}
                        for b in backends}
    services["ui"] = ui

    compose = {"name": tpl["name"], "services": services, "networks": tpl["networks"]}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.out.resolve() == (ROOT / "generated").resolve():
        # Created here so they belong to this user, not to root via docker.
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    head = ("# GENERATED by scripts/gen_config.py from docker-compose.yml and\n"
            "# workflows/workflows.json. Do not edit: change those and run ./start.sh.\n")
    (args.out / "docker-compose.yml").write_text(
        head + yaml.safe_dump(compose, sort_keys=False, default_flow_style=False))
    (args.out / "services.json").write_text(json.dumps(
        {"_about": "GENERATED by scripts/gen_config.py; read by the UI. Do not edit.",
         "gpus": gpus, "backends": backends}, indent=2) + "\n")

    print(f"{len(gpus)} GPU(s): " + ", ".join(f"{g['index']} {g['name']}" for g in gpus))
    for g in gpus:
        here = [b for b in backends if b["gpu"] == g["index"]]
        what = ", ".join(f"{b['workflow']} ({b['name']})" for b in here) or "unused"
        note = "  <- shared: one job at a time, in submission order" if len(here) > 1 else ""
        print(f"  GPU {g['index']}: {what}{note}")
    print(f"wrote {args.out / 'services.json'} and {args.out / 'docker-compose.yml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
