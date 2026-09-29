#!/usr/bin/env python3
"""
Run one job end to end against a ComfyUI instance, for either service.

Used to prove a backend container actually works: it exercises the same client,
adapter and workflow the UI will use, so a pass here means the only
untested layer left is the UI itself.

  python3 scripts/smoke.py --service replace --url http://127.0.0.1:8136 \
      --image ref.jpg --video drive.mp4 --frames 49

  python3 scripts/smoke.py --service animate --url http://127.0.0.1:8134 \
      --image ref.png --video drive.mp4 --frames 49
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.comfy_client import ComfyClient, ComfyError   # noqa: E402
from app.workflows import ADAPTERS                     # noqa: E402

DEFAULT_URL = {"replace": "http://127.0.0.1:8136", "animate": "http://127.0.0.1:8134"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=49,
                    help="frame_load_cap; 4n+1 keeps the latent aligned")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="verify the backend can resolve every node class, then stop")
    args = ap.parse_args()

    adapter = ADAPTERS[args.service]
    url = args.url or DEFAULT_URL[args.service]
    client = ComfyClient(url, name=args.service)

    if not client.is_up():
        sys.exit(f"no ComfyUI at {url}")
    running, pending = client.queue_depth()
    print(f"[{args.service}] {url}  (running={running} pending={pending})")

    # Pre-flight: a backend missing one custom node fails 20 minutes into a job
    # otherwise, so resolve every class up front.
    wf_probe = adapter.load()
    try:
        info = client._get("/object_info", timeout=180)
        needed = sorted({n["class_type"] for n in wf_probe.values()})
        missing = [c for c in needed if c not in info]
        print(f"node classes: {len(needed)} needed, {len(missing)} missing")
        if missing:
            for c in missing:
                print(f"   MISSING: {c}")
            sys.exit(f"{args.service}: backend cannot run this workflow")
    except ComfyError:
        raise
    except Exception as e:
        print(f"warning: could not check node classes ({e})")

    if args.check_only:
        print("check-only: backend resolves every node class")
        return 0

    image_name = client.upload(args.image)
    video_name = client.upload(args.video)
    print(f"uploaded  image -> {image_name}\n          video -> {video_name}")

    wf = adapter.build(image_name, video_name, frame_cap=args.frames, seed=args.seed)
    titles = client.node_titles(wf)

    stage_log: list[tuple[str, float]] = []
    state = {"node": None, "since": time.time(), "last": ""}

    def on_progress(p):
        if p.node_id != state["node"]:
            if state["node"] is not None:
                took = time.time() - state["since"]
                stage_log.append((f"{state['node']} {titles.get(state['node'],'')}", took))
                print(f"    ... {took:7.1f}s")
            state["node"], state["since"] = p.node_id, time.time()
            print(f"  [{time.strftime('%H:%M:%S')}] {p.node_id:>6}  "
                  f"{titles.get(p.node_id,'')}", end="", flush=True)
            state["last"] = ""
        if p.steps:
            line = f" {p.step}/{p.steps}"
            if line != state["last"]:
                print(f"\r  [{time.strftime('%H:%M:%S')}] {p.node_id:>6}  "
                      f"{titles.get(p.node_id,'')}{line}", end="", flush=True)
                state["last"] = line

    print(f"\nsubmitting ({args.frames} frames)...")
    started = time.time()
    try:
        prompt_id = client.submit(wf)
        print(f"prompt_id {prompt_id}\n")
        result = client.wait(prompt_id, timeout=args.timeout, on_progress=on_progress)
    except ComfyError as e:
        print(f"\n\nFAILED after {time.time()-started:.0f}s\n  {e}")
        return 1

    if state["node"] is not None:
        stage_log.append((f"{state['node']} {titles.get(state['node'],'')}",
                          time.time() - state["since"]))
        print(f"    ... {time.time()-state['since']:7.1f}s")

    print(f"\n{'='*66}\n{args.service}: completed in {result.elapsed/60:.1f} min "
          f"({args.frames} frames)\n{'='*66}")
    print("\nslowest stages:")
    for label, took in sorted(stage_log, key=lambda x: -x[1])[:8]:
        print(f"  {took:8.1f}s  {label}")

    print(f"\nvideos ({len(result.videos)}):")
    for v in result.videos:
        flag = "  <-- final" if v["node_id"] == adapter.output_node else ""
        print(f"  node {v['node_id']:>5}  {v['filename']}{flag}")
    for i in result.images:
        print(f"  image node {i['node_id']:>5}  {i['filename']}")

    finals = [v for v in result.videos if v["node_id"] == adapter.output_node]
    if not finals:
        print(f"\nFAILED: node {adapter.output_node} produced no video")
        return 1
    print(f"\nfinal: {client.view_url(finals[0])}")
    if args.out:
        print("saved:", client.download(finals[0], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
