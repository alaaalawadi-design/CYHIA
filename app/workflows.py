"""
Workflow adapters: one per workflow listed in workflows/workflows.json.

An adapter is a small "settings card" for one ComfyUI workflow. It records which
node id is the reference image, which is the driving video, which is the final
output, and so on. The UI and the client never need to know that "188" is the
reference image; only workflows.json and this file know. Adding a workflow means
adding an entry to workflows.json, not touching the code.

Seeds are NOT randomised here: whatever seed values are saved in the workflow
JSON (or set by an admin on /admin) are used as they are, so the same inputs and
settings reproduce the same result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import workflow_settings

# cy_hia/workflows/ holds the exported workflow JSON files and workflows.json.
WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"
CATALOG_FILE = WORKFLOW_DIR / "workflows.json"


@dataclass
class Adapter:
    """Turns (image, video) into a ready-to-submit ComfyUI prompt graph."""

    name: str                       # key in workflows.json: "replace", "animate", ...
    workflow_file: str              # JSON file inside cy_hia/workflows/
    image_node: str                 # LoadImage node: the reference character
    video_node: str                 # VHS_LoadVideo node: the driving video
    output_node: str                # VHS_VideoCombine node: the final result
    frame_cap_node: str | None = None    # node that limits frames
    prefix: str = ""                # result file names start with this
    label: str = ""                 # tab title
    blurb: str = ""                 # one paragraph under the tab title
    gallery_blurb: str = ""
    timeout: float = 2 * 3600.0     # seconds a job may run before it is abandoned
    trim_frames: int = 0            # warm-up frames cut off the front of results
    free_after_run: bool = True     # unload the models from VRAM after every job
    port: int = 8188                # ComfyUI port inside its container
    default_gpu: int = 0            # only without generated/services.json

    @property
    def output_prefix(self) -> str:
        """Start of the result's file name: Cy_Replace / Cy_Animate."""
        return self.prefix or f"Cy_{self.name.capitalize()}"

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
        Everything else in the workflow, seeds included, stays as saved, except
        the controls an admin changed on /admin (app/workflow_settings.py).
        """
        wf = workflow_settings.apply(self.name, self.load())

        # 1. inputs: point the workflow at the files we just uploaded
        wf[self.image_node]["inputs"]["image"] = image_name
        wf[self.video_node]["inputs"]["video"] = video_name

        # 2. optional length cap (only used on a staging deployment)
        if frame_cap is not None and self.frame_cap_node:
            wf[self.frame_cap_node]["inputs"]["frame_load_cap"] = int(frame_cap)

        # 3. output name: ComfyUI will write Cy_Replace_00001-audio.mp4, ...
        wf[self.output_node]["inputs"]["filename_prefix"] = self.output_prefix

        return wf


def load_catalog(path: Path = CATALOG_FILE) -> dict:
    """The whole workflows.json, for callers that need more than the adapters."""
    return json.loads(path.read_text())


def _adapter(name: str, spec: dict) -> Adapter:
    nodes = spec["nodes"]
    return Adapter(
        name=name,
        workflow_file=spec["file"],
        image_node=str(nodes["image"]),
        video_node=str(nodes["video"]),
        output_node=str(nodes["output"]),
        frame_cap_node=str(nodes["frame_cap"]) if nodes.get("frame_cap") else None,
        prefix=spec.get("prefix", ""),
        label=spec.get("label") or name.capitalize(),
        blurb=spec.get("blurb", ""),
        gallery_blurb=spec.get("gallery_blurb", ""),
        timeout=float(spec.get("timeout_hours", 2)) * 3600.0,
        trim_frames=int(spec.get("trim_frames", 0)),
        free_after_run=bool(spec.get("free_after_run", True)),
        port=int(spec.get("port", 8188)),
        default_gpu=int(spec.get("default_gpu", 0)),
    )


# Replace: SCAIL-2 + Flux2 Klein. Node 335 emits the side-by-side
# (original | result) video with source audio.
# Animate: WAN 2.2 Animate. Node ids mirror patch_workflow() in
# comfyui-wan-docker/gradio_app_gpu1.py.
# Kept in workflows.json order, which is also the tab order.
ADAPTERS: dict[str, Adapter] = {
    name: _adapter(name, spec) for name, spec in load_catalog()["workflows"].items()
}
