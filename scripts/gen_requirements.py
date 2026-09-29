#!/usr/bin/env python3
"""
Generate docker/requirements-lock.txt for the replacement image.

Must run with the validated interpreter:
    /home/administrator/Disk/nouran_Rep/venv/bin/python scripts/gen_requirements.py

A plain `pip freeze` of that venv is the wrong input: the venv serves all 60
custom nodes in that install, so it carries dlib (a long source build),
diffusers==0.40.0.dev0 (never published to PyPI) and a long tail of packages
belonging to nodes this workflow does not touch.

Instead this walks the dependency graph outward from a small root set — ComfyUI
core plus the 7 custom node directories the hardened workflow actually needs —
and pins every package it reaches to the version installed in that venv. So the
image gets exactly the validated versions, and nothing else.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY = Path("/home/administrator/Disk/nouran_Rep/ComfyUI")
OUT = ROOT / "docker" / "requirements-lock.txt"

# The custom node dirs the workflow needs, derived from ComfyUI's own
# python_module mapping (see the plan); keep in step with docker-compose.yml.
NEEDED_NODES = [
    "comfyui_controlnet_aux",
    "ComfyUI-VideoHelperSuite",
    "ComfyUI-KJNodes",
    "comfyui_essentials",
    "comfyui-scail2-infinity",
    "dwpose_retarget",
    "ComfyUI-QwenVL",
]

# Installed separately from the CUDA 13.0 wheel index, with their nvidia-* deps.
SEPARATE = {"torch", "torchvision", "torchaudio"}

# Pulled in as an extra by some requirement but useless in a headless service,
# and each one is a large or slow install.
EXCLUDE = {
    "onnxruntime",          # onnxruntime-gpu is the one that gets used
    "comfyui-manager",
}


def canon(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def parse_requirements(path: Path) -> list[str]:
    names = []
    if not path.is_file():
        return names
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if line.startswith("git+") or "://" in line:
            continue                      # VCS pins: not reproducible here
        name = line
        for sep in ("[", "==", ">=", "<=", "~=", "!=", ">", "<", ";", " "):
            name = name.split(sep, 1)[0]
        if name:
            names.append(canon(name))
    return names


def requires_of(dist_name: str) -> list[str]:
    """Direct runtime dependencies of an installed distribution."""
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return []
    out = []
    for raw in dist.requires or []:
        # "foo (>=1) ; extra == 'bar'" — skip optional extras, keep the rest.
        if ";" in raw:
            req, marker = raw.split(";", 1)
            if "extra" in marker:
                continue
            try:
                from packaging.markers import Marker
                if not Marker(marker.strip()).evaluate():
                    continue
            except Exception:
                pass                       # unparseable marker: keep the dep
        else:
            req = raw
        name = req.strip()
        for sep in ("[", "(", "==", ">=", "<=", "~=", "!=", ">", "<", " "):
            name = name.split(sep, 1)[0]
        if name:
            out.append(canon(name))
    return out


def main() -> int:
    installed = {canon(d.metadata["Name"]): d.version
                 for d in metadata.distributions()
                 if d.metadata and d.metadata.get("Name")}
    if "torch" not in installed:
        return int(print("run this with the nouran_Rep venv python") or 1)

    roots: set[str] = set()
    sources: dict[str, str] = {}

    for name in parse_requirements(COMFY / "requirements.txt"):
        roots.add(name)
        sources.setdefault(name, "ComfyUI core")
    for node in NEEDED_NODES:
        for name in parse_requirements(COMFY / "custom_nodes" / node / "requirements.txt"):
            roots.add(name)
            sources.setdefault(name, node)

    # Not declared by any requirements.txt, but the service needs them:
    # websocket support for ComfyUI's /ws, and the image runs headless.
    for extra in ("opencv-python-headless",):
        roots.add(canon(extra))
        sources.setdefault(canon(extra), "service")

    # Walk outward.
    seen: set[str] = set()
    stack = sorted(roots)
    missing_roots = []
    while stack:
        name = stack.pop()
        if name in seen or name in SEPARATE or name in EXCLUDE:
            continue
        seen.add(name)
        if name not in installed:
            if name in roots:
                missing_roots.append(name)
            continue
        for dep in requires_of(name):
            if dep not in seen:
                stack.append(dep)

    resolved = sorted(n for n in seen if n in installed)

    header = [
        "# Generated by scripts/gen_requirements.py from the validated install at",
        "# /home/administrator/Disk/nouran_Rep/venv.",
        "#",
        "# This is the dependency closure of ComfyUI core plus the 7 custom node",
        "# directories the hardened replacement workflow needs - NOT a full freeze",
        "# of that venv, which also serves 53 other custom nodes.",
        "#",
        "# torch/torchvision/torchaudio and their nvidia-* deps come from the",
        "# CUDA 13.0 wheel index instead; see the Dockerfile.",
        "",
    ]
    body = [f"{name}=={installed[name]}" for name in resolved]
    OUT.write_text("\n".join(header + body) + "\n")

    print(f"roots: {len(roots)}  ->  closure: {len(resolved)} packages")
    if missing_roots:
        print(f"declared but not installed (skipped): {', '.join(sorted(missing_roots))}")
    for probe in ("dlib", "diffusers", "insightface", "fairscale", "mediapipe",
                  "onnxruntime-gpu", "transformers", "numpy", "rembg"):
        mark = "IN " if probe in resolved else "out"
        ver = installed.get(probe, "-")
        print(f"  [{mark}] {probe} {ver}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
