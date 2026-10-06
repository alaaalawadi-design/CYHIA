"""
Workflow adapters: one per service.

An adapter is a small "settings card" for one ComfyUI workflow. It records which
node id is the reference image, which is the driving video, which is the final
output, and so on. The UI and the client never need to know that "188" is the
reference image; only this file knows. Adding a control later means touching one
adapter, not the UI.

Seeds are NOT touched here: whatever seed values are saved in the workflow JSON
are used as they are, so the same inputs reproduce the same result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# cy_hia/workflows/ holds the exported workflow JSON files.
WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"


@dataclass
class Adapter:
    """Turns (image, video) into a ready-to-submit ComfyUI prompt graph."""

    name: str                       # "replace" or "animate"
    workflow_file: str              # JSON file inside cy_hia/workflows/
    image_node: str                 # LoadImage node: the reference character
    video_node: str                 # VHS_LoadVideo node: the driving video
    output_node: str                # VHS_VideoCombine node: the final result
    frame_cap_node: str | None = None    # node that limits frames
    fps_node: str | None = None

    @property
    def output_prefix(self) -> str:
        """Start of the result's file name: Cy_Replace / Cy_Animate."""
        return f"Cy_{self.name.capitalize()}"

    @property
    def all_nodes(self) -> list[str]:
        base = [self.image_node, self.video_node, self.output_node]
        return [n for n in base if n]

    def load(self) -> dict:
        """Read the workflow JSON and check every node id we rely on exists."""
        path = WORKFLOW_DIR / self.workflow_file
        wf = json.loads(path.read_text())
        missing = [n for n in self.all_nodes if n not in wf]
        if missing:
            raise KeyError(f"{self.name}: {path.name} is missing node(s) {missing}")
        return wf

    def build(self, image_name: str, video_name: str, *,
              frame_cap: int | None = None) -> dict:
        """
        image_name / video_name are names already uploaded to that ComfyUI.
        Everything else in the workflow, seeds included, stays as saved.
        """
        wf = self.load()

        # 1. inputs: point the workflow at the files we just uploaded
        wf[self.image_node]["inputs"]["image"] = image_name
        wf[self.video_node]["inputs"]["video"] = video_name

        # 2. optional length cap (only used on a staging deployment)
        if frame_cap is not None and self.frame_cap_node:
            wf[self.frame_cap_node]["inputs"]["frame_load_cap"] = int(frame_cap)

        # 3. output name: ComfyUI will write Cy_Replace_00001-audio.mp4, ...
        wf[self.output_node]["inputs"]["filename_prefix"] = self.output_prefix

        return wf


# ── Character replacement (SCAIL-2 + Flux2 Klein) ────────────────────────────
# Stages: clean first frame -> pose retarget -> relight -> windowed SCAIL-2.
# Node 335 emits the side-by-side (original | result) video with source audio.
REPLACE = Adapter(
    name="replace",
    workflow_file="replace_workflow.json",
    image_node="188",           # LoadImage        - reference character
    video_node="384",           # VHS_LoadVideo    - driving video
    output_node="335",          # VHS_VideoCombine - side-by-side + audio
    frame_cap_node="384",
)

# ── WAN 2.2 Animate ──────────────────────────────────────────────────────────
# Node ids mirror patch_workflow() in comfyui-wan-docker/gradio_app_gpu1.py.
ANIMATE = Adapter(
    name="animate",
    workflow_file="animate_workflow.json",
    image_node="311",           # LoadImage        - reference image
    video_node="417",           # VHS_LoadVideo    - driving video
    output_node="462",          # VHS_VideoCombine - result
    frame_cap_node="417",
)

ADAPTERS = {a.name: a for a in (ANIMATE, REPLACE)}
