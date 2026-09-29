"""
Workflow adapters: one per service.

Each adapter owns the node ids of its own graph, so the UI and the client never
have to know that "188" is the reference image. Adding a control later means
touching one adapter, not the UI.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"

# ComfyUI seeds are validated as signed 64-bit, but the UIs all stay well below
# that; 2**32 matches what the existing animate app uses.
SEED_MAX = 2 ** 32


def _seed() -> int:
    return secrets.randbelow(SEED_MAX)


@dataclass
class Adapter:
    """Turns (image, video) into a ready-to-submit prompt graph."""

    name: str
    workflow_file: str
    image_node: str
    video_node: str
    output_node: str
    seed_nodes: list[str] = field(default_factory=list)
    frame_cap_node: str | None = None
    fps_node: str | None = None

    def load(self) -> dict:
        path = WORKFLOW_DIR / self.workflow_file
        wf = json.loads(path.read_text())
        missing = [n for n in self.all_nodes if n not in wf]
        if missing:
            raise KeyError(f"{self.name}: {path.name} is missing node(s) {missing}")
        return wf

    @property
    def all_nodes(self) -> list[str]:
        base = [self.image_node, self.video_node, self.output_node, *self.seed_nodes]
        return [n for n in base if n]

    def build(self, image_name: str, video_name: str, *,
              frame_cap: int | None = None, seed: int | None = None) -> dict:
        """
        image_name / video_name are names already uploaded to that ComfyUI.

        seed is normally left None so every run differs; pass one to reproduce a
        result. Each sampler gets a distinct value derived from it, since giving
        every stage the same seed correlates their noise.
        """
        wf = self.load()
        wf[self.image_node]["inputs"]["image"] = image_name
        wf[self.video_node]["inputs"]["video"] = video_name

        base = _seed() if seed is None else int(seed)
        for offset, node in enumerate(self.seed_nodes):
            wf[node]["inputs"]["seed"] = (base + offset * 7919) % SEED_MAX

        if frame_cap is not None and self.frame_cap_node:
            wf[self.frame_cap_node]["inputs"]["frame_load_cap"] = int(frame_cap)

        return wf


# ── Service 2 · character replacement (SCAIL-2 + Flux2 Klein), GPU 1 ──────────
# Stages: clean first frame -> pose retarget -> relight -> windowed SCAIL-2.
# Node 335 emits the side-by-side (original | result) video with source audio.
REPLACE = Adapter(
    name="replace",
    workflow_file="replace_workflow.json",
    image_node="188",           # LoadImage      — reference character
    video_node="384",           # VHS_LoadVideo  — driving video
    output_node="335",          # VHS_VideoCombine — side-by-side + audio
    seed_nodes=["169", "212", "239", "373", "484", "408", "411"],
    frame_cap_node="384",
)

# ── Service 1 · WAN 2.2 Animate, GPU 0 ───────────────────────────────────────
# Node ids mirror patch_workflow() in comfyui-wan-docker/gradio_app_gpu1.py.
ANIMATE = Adapter(
    name="animate",
    workflow_file="animate_workflow.json",
    image_node="311",           # LoadImage      — reference image
    video_node="417",           # VHS_LoadVideo  — driving video
    output_node="462",          # VHS_VideoCombine — result
    seed_nodes=["324"],
    frame_cap_node="417",
)

ADAPTERS = {a.name: a for a in (ANIMATE, REPLACE)}
