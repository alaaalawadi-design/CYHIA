"""
cy_hia — one Gradio UI over two ComfyUI GPU backends.

  Animate tab  -> comfy-animate (GPU 1)  WAN 2.2 Animate
  Replace tab  -> comfy-replace (GPU 0)  SCAIL-2 character replacement

Each tab talks to its own backend, so the two GPUs work in parallel and a job
on one never blocks the other. Gradio's queue serialises *within* a tab
(concurrency_limit=1 per handler) because each backend is a single GPU that can
only run one ComfyUI job at a time anyway.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import gradio as gr

from .comfy_client import ComfyClient, ComfyError, ComfyTimeout
from .workflows import ANIMATE, REPLACE, Adapter

OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs"))

# Replacement costs ~3 min fixed plus ~3.3 s/frame (measured), so a full
# 1140-frame job runs 65-70 min. Three hours leaves room for a cold model load
# and a queued job ahead of it without stranding the user on a hang.
TIMEOUTS = {"replace": 3 * 3600.0, "animate": 2 * 3600.0}

BACKENDS = {
    "animate": ComfyClient(os.environ.get("ANIMATE_URL", "http://comfy-animate:8134"),
                           name="animate"),
    "replace": ComfyClient(os.environ.get("REPLACE_URL", "http://comfy-replace:8136"),
                           name="replace"),
}

LABELS = {"animate": "Animate (GPU 1)", "replace": "Replace (GPU 0)"}


# ── status ───────────────────────────────────────────────────────────────────
def status_line(name: str) -> str:
    client = BACKENDS[name]
    if not client.is_up():
        return f"🔴 **{LABELS[name]}** — offline ({client.base_url})"
    running, pending = client.queue_depth()
    if running:
        extra = f", {pending} queued" if pending else ""
        return f"🟡 **{LABELS[name]}** — busy{extra}"
    return f"🟢 **{LABELS[name]}** — ready"


def both_status() -> tuple[str, str]:
    return status_line("animate"), status_line("replace")


# ── running a job ────────────────────────────────────────────────────────────
def _run(adapter: Adapter, image: str | None, video: str | None,
         progress: gr.Progress) -> str:
    name = adapter.name
    client = BACKENDS[name]

    if not image or not video:
        raise gr.Error("Please provide both a reference image and a driving video.")
    if not client.is_up():
        raise gr.Error(f"The {LABELS[name]} backend is not reachable "
                       f"({client.base_url}). The other tab may still work.")

    progress(0.0, desc="Uploading…")
    try:
        image_name = client.upload(image)
        video_name = client.upload(video)
    except Exception as e:
        raise gr.Error(f"Upload failed: {e}") from e

    # Unset in production: the workflow's own cap (1140 frames for replace)
    # decides length. Set MAX_FRAMES on a staging deployment to keep jobs short.
    cap = os.environ.get("MAX_FRAMES")
    workflow = adapter.build(image_name, video_name,
                             frame_cap=int(cap) if cap else None)
    titles = client.node_titles(workflow)
    started = time.time()

    def on_progress(p):
        elapsed = time.time() - started
        stage = titles.get(p.node_id or "", "working")
        step = f" · step {p.step}/{p.steps}" if p.steps else ""
        # No honest overall fraction exists: the graph is ~100 nodes of very
        # uneven cost. Report elapsed time and the live stage instead of a
        # progress bar that would lie.
        progress(0.0, desc=f"{stage}{step} · {elapsed/60:.1f} min elapsed")

    try:
        result = client.run(workflow, timeout=TIMEOUTS[name], on_progress=on_progress)
    except ComfyTimeout as e:
        raise gr.Error(f"Timed out: {e}") from e
    except ComfyError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:                       # noqa: BLE001 - surfaced to the user
        traceback.print_exc()
        raise gr.Error(f"Unexpected failure: {e}") from e

    finals = [v for v in result.videos if v["node_id"] == adapter.output_node]
    if not finals:
        got = ", ".join(sorted({v["node_id"] for v in result.videos})) or "nothing"
        raise gr.Error(f"The job finished but produced no video on node "
                       f"{adapter.output_node} (got: {got}).")

    progress(1.0, desc="Downloading result…")
    return client.download(finals[0], OUT_DIR / name)


def run_animate(image, video, progress=gr.Progress()):
    return _run(ANIMATE, image, video, progress)


def run_replace(image, video, progress=gr.Progress()):
    return _run(REPLACE, image, video, progress)


# ── layout ───────────────────────────────────────────────────────────────────
def build_tab(adapter: Adapter, handler, blurb: str, video_label: str):
    """Both tabs take exactly the same two inputs, so they share a layout."""
    with gr.Row():
        status = gr.Markdown(status_line(adapter.name))
    gr.Markdown(blurb)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Reference character (image)", type="filepath",
                             sources=["upload"], height=280)
            video = gr.Video(label=video_label, sources=["upload"], height=280)
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            out = gr.Video(label="Result", height=580, autoplay=False)

    event = run_btn.click(
        fn=handler,
        inputs=[image, video],
        outputs=[out],
        # One GPU per backend: a second concurrent job would only fight for VRAM.
        concurrency_limit=1,
        concurrency_id=adapter.name,
    )
    # Disable the button for the duration so a double click cannot queue twice.
    event.then(lambda: gr.update(interactive=True), outputs=[run_btn])
    run_btn.click(lambda: gr.update(interactive=False), outputs=[run_btn],
                  queue=False)
    return status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="cy_hia — Animate & Replace", fill_height=True) as demo:
        gr.Markdown("# cy_hia\nTwo GPU services behind one interface.")

        with gr.Tabs():
            with gr.Tab("Replace character"):
                replace_status = build_tab(
                    REPLACE, run_replace,
                    "Puts the character from your image into the driving video, "
                    "matching pose, framing and lighting. The result is shown "
                    "side by side with the original, with the original audio.\n\n"
                    "A full-length clip takes roughly an hour.",
                    "Driving video (provides the motion)",
                )
            with gr.Tab("Animate"):
                animate_status = build_tab(
                    ANIMATE, run_animate,
                    "Animates the character in your image using the motion of "
                    "the driving video.",
                    "Driving video (provides the motion)",
                )

        # Poll both backends so a tab shows as offline instead of failing only
        # when someone presses Generate.
        timer = gr.Timer(15.0)
        timer.tick(both_status, outputs=[animate_status, replace_status],
                   show_progress="hidden")
        demo.load(both_status, outputs=[animate_status, replace_status])

    return demo


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, client in BACKENDS.items():
        print(f"{name:>8}: {client.base_url}  {'up' if client.is_up() else 'DOWN'}")
    build_app().queue(max_size=20).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
