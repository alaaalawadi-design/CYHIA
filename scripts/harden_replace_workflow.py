#!/usr/bin/env python3
"""
Harden the SCAIL-2 character-replacement workflow for unattended service use.

Input :  workflows/replace_workflow_src.json   (exported as-is from ComfyUI)
Output:  workflows/replace_workflow.json       (what the UI service loads)

The source graph was built interactively, so it carries three kinds of baggage
that hurt a long-running service:

  1. QwenVL pins itself in VRAM while the Wan 14B sampler runs afterwards.
  2. Several loaders are duplicated, so the same weights land in memory twice.
  3. A dozen debug/no-op outputs write files on every single run.

Every change is declared below and validated afterwards; nothing is edited by
hand. Run with --check to verify an already-hardened file without rewriting it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows" / "replace_workflow_src.json"
DST = ROOT / "workflows" / "replace_workflow.json"

# ── The nodes the service drives. Kept here so the app and this script agree. ──
NODE_REF_IMAGE = "188"        # LoadImage      — reference character
NODE_DRIVE_VIDEO = "384"      # VHS_LoadVideo  — driving video
NODE_FINAL_VIDEO = "335"      # VHS_VideoCombine — side-by-side result + audio
SEED_NODES = ["169", "212", "239", "373", "484", "408", "411"]

# ── 1 · Free QwenVL's VRAM before the Wan sampler loads ───────────────────────
SET_INPUTS = {
    "411": {"keep_model_loaded": False},
}

# ── 2 · Collapse duplicate loaders onto one instance each ─────────────────────
# Each entry: drop this loader, and rewire its consumers to the twin that loads
# the identical file. Verified pairwise on class_type + every literal input.
MERGE_LOADERS = [
    ("482", "113"),   # UNETLoader   flux-2-klein-base-9b-fp8.safetensors
    ("483", "116"),   # CLIPLoader   qwen_3_8b_fp8mixed.safetensors
    ("478", "117"),   # VAELoader    flux2-vae.safetensors
]

# ── 3 · Drop debug outputs that write files but feed nothing ──────────────────
# Load-bearing lookalikes deliberately NOT listed: 487 and 500 are SaveImage
# nodes whose image output is consumed downstream, and 167/208/235/377 are
# PreviewImage nodes used as pass-throughs in the data path.
DROP_NODES = [
    "164",   # SavePoseKpsAsJsonFile — pose keypoint dump
    "185",   # SaveImageAdvanced     — 5-up debug contact sheet
    "488",   # Image Comparer        — also carries stale temp URLs from authoring
    "509",   # SaveImage             — "Relighted image"
    "512",   # SaveImage             — "Relighted" (same batch as 509)
    "514",   # easy saveText         — empty output_file_path, writes to cwd
    "398",   # Flux2KleinRefLatentWeight — orphaned, never executed
]

# ── 4 · Swap a custom node for the core node that does the same job ──────────
# was-ns exists in this graph purely to join two strings. It is a 71 MB tree
# pulling fairscale, rembg, img2texture and more, so the replacement service
# would have to carry all of it for one concatenation. Core StringConcatenate
# is equivalent here; clean_whitespace has no meaningful effect on these two
# operands (a fixed instruction and a generated paragraph).
SUBSTITUTE = {
    "493": {
        "from_class": "Text Concatenate",
        "to_class": "StringConcatenate",
        "rename": {"text_a": "string_a", "text_b": "string_b"},
        "drop": ["clean_whitespace"],
    },
}

# ── 5 · Bypass pass-through nodes that only exist to show a value in the UI ───
# ShowText returns its input unchanged; in a headless service it is a no-op
# that costs the whole comfyui-custom-scripts dependency. Consumers are rewired
# to whatever fed it. Value: the input holding the upstream wire.
BYPASS = {
    "136": "text",          # ShowText|pysssss — displays the retarget prompt
}

# ── 6 · Drop nodes ComfyUI would never execute ───────────────────────────────
# Leftovers from authoring. They are inert by definition (nothing reachable from
# an output depends on them), but each one still drags in its custom node: node
# 233 DrawViTPose is the only reason this graph would need
# ComfyUI-WanAnimatePreprocess and its onnxruntime-gpu stack.
PRUNE_INERT = True

# Class types that ComfyUI treats as outputs: the graph is executed backwards
# from these, so anything not reachable from one never runs.
OUTPUT_CLASSES = {
    "SaveImage",
    "SaveImageAdvanced",
    "PreviewImage",
    "PreviewAny",
    "VHS_VideoCombine",
    "SavePoseKpsAsJsonFile",
    "ShowText|pysssss",
    "easy saveText",
    "Image Comparer (rgthree)",
}


def iter_refs(node):
    """Yield (input_name, [node_id, slot]) for every wired input of a node."""
    for name, value in node.get("inputs", {}).items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
            yield name, value


def literal_inputs(node):
    return {k: v for k, v in node.get("inputs", {}).items() if not isinstance(v, list)}


def reachable_from_outputs(wf):
    """Node ids ComfyUI would actually execute."""
    stack = [nid for nid, n in wf.items() if n["class_type"] in OUTPUT_CLASSES]
    seen = set(stack)
    while stack:
        nid = stack.pop()
        for _, (dep, _slot) in iter_refs(wf.get(nid, {})):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def validate(wf, label):
    """Every wire resolves, required nodes survive, and nothing is orphaned."""
    problems = []

    for nid, node in wf.items():
        for name, (dep, _slot) in iter_refs(node):
            if dep not in wf:
                problems.append(f"{nid}.{name} -> missing node {dep}")

    for nid, what in [
        (NODE_REF_IMAGE, "reference image input"),
        (NODE_DRIVE_VIDEO, "driving video input"),
        (NODE_FINAL_VIDEO, "final video output"),
    ]:
        if nid not in wf:
            problems.append(f"{what} (node {nid}) is gone")

    for nid in SEED_NODES:
        if nid not in wf:
            problems.append(f"seed node {nid} is gone")
        elif "seed" not in wf[nid].get("inputs", {}):
            problems.append(f"node {nid} has no seed input to randomise")

    live = reachable_from_outputs(wf)
    if NODE_FINAL_VIDEO not in live:
        problems.append(f"final video node {NODE_FINAL_VIDEO} is not an executed output")

    dead = sorted(set(wf) - live, key=lambda k: (len(k), k))
    if problems:
        print(f"\n✗ {label}: {len(problems)} problem(s)")
        for p in problems:
            print(f"    {p}")
        return False, dead

    print(f"\n✓ {label}: {len(wf)} nodes, all wires resolve, "
          f"{len(live)} executed, {len(dead)} inert")
    return True, dead


def harden(wf):
    wf = json.loads(json.dumps(wf))  # never mutate the caller's copy
    changes = []

    # 1 · literal input overrides
    for nid, overrides in SET_INPUTS.items():
        if nid not in wf:
            sys.exit(f"cannot set inputs on missing node {nid}")
        for key, new in overrides.items():
            old = wf[nid]["inputs"].get(key)
            if old == new:
                changes.append(f"  = {nid}.{key} already {new!r}")
                continue
            wf[nid]["inputs"][key] = new
            changes.append(f"  ~ {nid}.{key}: {old!r} -> {new!r}")

    # 2 · duplicate loaders
    for dup, keep in MERGE_LOADERS:
        if dup not in wf:
            changes.append(f"  = loader {dup} already merged")
            continue
        if keep not in wf:
            sys.exit(f"cannot merge {dup} into missing node {keep}")
        a, b = wf[dup], wf[keep]
        if a["class_type"] != b["class_type"] or literal_inputs(a) != literal_inputs(b):
            sys.exit(f"refusing to merge {dup} into {keep}: they are not identical")
        rewired = []
        for nid, node in wf.items():
            for name, ref in iter_refs(node):
                if ref[0] == dup:
                    ref[0] = keep
                    rewired.append(f"{nid}.{name}")
        del wf[dup]
        changes.append(f"  - {dup} ({a['class_type']}) merged into {keep}"
                       f"; rewired {', '.join(rewired) or 'nothing'}")

    # 3 · debug outputs
    # Dropped as one set: some feed each other (512 -> 514), so the only thing
    # that matters is that no *survivor* consumes anything in the set.
    doomed = {nid for nid in DROP_NODES if nid in wf}
    for nid in DROP_NODES:
        if nid not in wf:
            changes.append(f"  = node {nid} already dropped")
    for other, node in wf.items():
        if other in doomed:
            continue
        for name, ref in iter_refs(node):
            if ref[0] in doomed:
                sys.exit(f"refusing to drop {ref[0]}: still feeds surviving {other}.{name}")
    for nid in sorted(doomed, key=lambda k: (len(k), k)):
        cls = wf.pop(nid)["class_type"]
        changes.append(f"  - {nid} ({cls}) dropped")

    # 4 · core-node substitutions
    for nid, spec in SUBSTITUTE.items():
        node = wf.get(nid)
        if node is None:
            sys.exit(f"cannot substitute missing node {nid}")
        if node["class_type"] == spec["to_class"]:
            changes.append(f"  = {nid} already {spec['to_class']}")
            continue
        if node["class_type"] != spec["from_class"]:
            sys.exit(f"node {nid} is {node['class_type']}, "
                     f"expected {spec['from_class']}")
        inputs = node["inputs"]
        for old, new in spec["rename"].items():
            if old not in inputs:
                sys.exit(f"node {nid} has no input {old} to rename")
            inputs[new] = inputs.pop(old)
        for key in spec["drop"]:
            inputs.pop(key, None)
        node["class_type"] = spec["to_class"]
        changes.append(f"  ~ {nid}: {spec['from_class']} -> {spec['to_class']}")

    # 5 · pass-through bypass
    for nid, feed in BYPASS.items():
        node = wf.get(nid)
        if node is None:
            changes.append(f"  = passthrough {nid} already bypassed")
            continue
        source = node["inputs"].get(feed)
        if not (isinstance(source, list) and len(source) == 2):
            sys.exit(f"node {nid} input {feed!r} is not wired; cannot bypass")
        rewired = []
        for other, n in wf.items():
            if other == nid:
                continue
            for name, ref in iter_refs(n):
                if ref[0] == nid:
                    n["inputs"][name] = list(source)
                    rewired.append(f"{other}.{name}")
        cls = wf.pop(nid)["class_type"]
        changes.append(f"  - {nid} ({cls}) bypassed -> {source}"
                       f"; rewired {', '.join(rewired) or 'nothing'}")

    # 6 · prune anything ComfyUI would not execute, repeatedly: removing one
    #     node can strand the node that only fed it.
    if PRUNE_INERT:
        protected = set(SEED_NODES) | {NODE_REF_IMAGE, NODE_DRIVE_VIDEO, NODE_FINAL_VIDEO}
        while True:
            inert = (set(wf) - reachable_from_outputs(wf)) - protected
            if not inert:
                break
            for nid in sorted(inert, key=lambda k: (len(k), k)):
                cls = wf.pop(nid)["class_type"]
                changes.append(f"  - {nid} ({cls}) pruned, never executed")

    return wf, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate the existing hardened file, do not rewrite")
    args = ap.parse_args()

    if args.check:
        wf = json.loads(DST.read_text())
        ok, dead = validate(wf, f"{DST.name} (as-is)")
        if dead:
            print(f"  inert nodes: {' '.join(dead)}")
        sys.exit(0 if ok else 1)

    src = json.loads(SRC.read_text())
    ok, _ = validate(src, f"{SRC.name} (before)")
    if not ok:
        sys.exit("source workflow does not validate; not touching it")

    out, changes = harden(src)
    print("\nchanges:")
    for c in changes:
        print(c)

    ok, dead = validate(out, f"{DST.name} (after)")
    if not ok:
        sys.exit("hardened workflow does not validate; nothing written")
    if dead:
        print(f"  inert nodes: {' '.join(dead)}")

    outputs = sorted((nid for nid, n in out.items()
                      if n["class_type"] in OUTPUT_CLASSES
                      and n["class_type"] not in ("PreviewImage", "PreviewAny",
                                                  "ShowText|pysssss")),
                     key=lambda k: (len(k), k))
    print("\nfile-writing outputs that remain:")
    for nid in outputs:
        n = out[nid]
        prefix = n["inputs"].get("filename_prefix", "?")
        print(f"  {nid:>6}  {n['class_type']:<18} prefix={prefix!r}")

    print("\nnode classes required:")
    for cls in sorted({n["class_type"] for n in out.values()}):
        print(f"  {cls}")

    DST.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {DST}  ({len(src)} -> {len(out)} nodes)")


if __name__ == "__main__":
    main()
