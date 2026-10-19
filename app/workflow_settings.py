"""
Admin-editable workflow controls.

The workflow JSON stays the source of truth for every default: nothing here
repeats a value from it. CONTROLS only names which node inputs an admin may
change and how to validate them; the admin's changes are stored as overrides in
OUT_DIR/workflow_settings.json and laid over the graph by Adapter.build().
A control that is not overridden runs exactly as saved in the workflow.

    {"replace": {"overrides": {"408.steps": 6}, "updated": 1760000000.0,
                 "updated_by": "admin"}}

Overrides are read when a job *starts*, not when it is submitted, so a change
applies to every job that has not begun running yet.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs"))
SETTINGS_PATH = Path(
    os.environ.get("WORKFLOW_SETTINGS", OUT_DIR / "workflow_settings.json")
)

# ComfyUI takes 64-bit seeds, but the browser's numbers are exact only to 2**53.
SEED_MAX = 2**53 - 1
MAX_TEXT = 8000

# The names ComfyUI 0.27.0 accepts on KSampler / WanSCAILInfinity in comfy-replace
# (from /object_info). Kept to the ones that make sense with these models.
SAMPLERS = [
    "euler", "euler_ancestral", "heun", "dpm_2", "lms", "dpmpp_2s_ancestral",
    "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "lcm",
    "ipndm", "deis", "res_multistep", "uni_pc", "uni_pc_bh2",
]
SCHEDULERS = [
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "normal", "linear_quadratic", "kl_optimal",
]


@dataclass(frozen=True)
class Control:
    node: str
    input: str
    label: str
    group: str
    type: str  # int | float | text | bool | choice
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] = field(default_factory=list)
    help: str = ""

    @property
    def key(self) -> str:
        return f"{self.node}.{self.input}"


def _sampler(node: str, group: str, *, steps_max: int = 60, sampler: bool = True,
             what: str = "") -> list[Control]:
    out = [
        Control(node, "seed", "Seed", group, "int", 0, SEED_MAX, 1,
                help=f"Fixed seed{what}. Change it for a different take on the same inputs."),
        Control(node, "steps", "Steps", group, "int", 1, steps_max, 1),
        Control(node, "cfg", "CFG", group, "float", 0, 20, 0.1),
    ]
    if sampler:
        out += [
            Control(node, "sampler_name", "Sampler", group, "choice", choices=SAMPLERS),
            Control(node, "scheduler", "Scheduler", group, "choice", choices=SCHEDULERS),
        ]
    return out


G_INPUT = "Driving video"
G_CLEAN = "1 · Clean first frame"
G_POSE = "2 · Pose retarget"
G_PLACE = "3 · Character in pose"
G_LIGHT = "4 · Relight"
G_SEG = "5 · Segmentation (SAM3)"
G_VIDEO = "6 · Video (SCAIL-2)"
G_OUT = "Output"

REPLACE_CONTROLS: list[Control] = [
    # ── input ────────────────────────────────────────────────────────────────
    Control("384", "frame_load_cap", "Max frames", G_INPUT, "int", 1, 4000, 1,
            help="Frames read from the driving video at 24 fps. ~3.3 s of GPU per frame; "
                 "1040 ≈ 60 min. Keep it 4n+1 for an aligned Wan latent."),
    Control("384", "skip_first_frames", "Skip first frames", G_INPUT, "int", 0, 100000, 1),
    # ── 1. clean the driving video's first frame ────────────────────────────
    Control("368", "text", "Positive prompt", G_CLEAN, "text"),
    Control("405", "text", "Negative prompt", G_CLEAN, "text"),
    *_sampler("373", G_CLEAN),
    Control("114", "strength_model", "RefControl pose LoRA", G_CLEAN, "float", 0, 2, 0.05,
            help="Shared by the Klein samplers of stages 1–3."),
    Control("202", "strength_model", "RefControl depth LoRA", G_CLEAN, "float", 0, 2, 0.05,
            help="Shared by the Klein samplers of stages 1–3."),
    # ── 2. pose estimation / retarget ───────────────────────────────────────
    Control("135", "confidence_threshold", "Retarget keypoint confidence", G_POSE,
            "float", 0, 1, 0.01, help="DWPose keypoints below this are ignored by the bone-ratio retarget."),
    Control("135", "stick_width", "Pose stick width", G_POSE, "int", 1, 16, 1),
    Control("134", "confidence_threshold", "Coverage-prompt confidence", G_POSE,
            "float", 0, 1, 0.01, help="Keypoints that count as visible when writing the frontal-render prompt."),
    Control("400", "text", "Frontal reference — negative prompt", G_POSE, "text"),
    *_sampler("212", G_POSE, sampler=False, what=" of the frontal reference render"),
    Control("399", "text", "Frontal motion frame — negative prompt", G_POSE, "text"),
    *_sampler("239", G_POSE, sampler=False, what=" of the frontal motion-frame render"),
    # ── 3. reference character in the driving pose ──────────────────────────
    Control("168", "text", "Positive prompt", G_PLACE, "text"),
    Control("397", "text", "Negative prompt", G_PLACE, "text"),
    *_sampler("169", G_PLACE),
    # ── 4. relight ──────────────────────────────────────────────────────────
    Control("492", "value", "Relight instruction", G_LIGHT, "text",
            help="Joined with the QwenVL lighting description by \", \" to form the prompt."),
    Control("411", "custom_prompt", "QwenVL lighting-analysis prompt", G_LIGHT, "text",
            help="What QwenVL is asked about frame 0 of the driving video."),
    Control("411", "seed", "QwenVL seed", G_LIGHT, "int", 0, SEED_MAX, 1),
    Control("481", "strength_model", "Klein-Migration LoRA", G_LIGHT, "float", 0, 2, 0.05),
    *_sampler("484", G_LIGHT),
    # ── 5. segmentation ─────────────────────────────────────────────────────
    Control("309", "text", "Driving-video target", G_SEG, "text",
            help="What SAM3 tracks in the driving video."),
    Control("311", "detection_threshold", "Driving-video threshold", G_SEG, "float", 0, 1, 0.01),
    Control("311", "max_objects", "Driving-video max objects", G_SEG, "int", 1, 20, 1),
    Control("310", "text", "Reference target", G_SEG, "text",
            help="What SAM3 tracks in the relit reference image."),
    Control("317", "detection_threshold", "Reference threshold", G_SEG, "float", 0, 1, 0.01),
    Control("317", "max_objects", "Reference max objects", G_SEG, "int", 1, 20, 1),
    # ── 6. SCAIL-2 video ────────────────────────────────────────────────────
    Control("336", "text", "Positive prompt", G_VIDEO, "text"),
    Control("327", "text", "Negative prompt", G_VIDEO, "text"),
    *_sampler("408", G_VIDEO, steps_max=30),
    Control("408", "window_length", "Window length", G_VIDEO, "int", 5, 201, 4,
            help="Frames per chunk (4n+1). Bounds VRAM; 125 is what fits a 32 GB 5090. "
                 "SCAIL-2 was trained at 81."),
    Control("408", "previous_frame_count", "Window overlap frames", G_VIDEO, "int", 1, 33, 4,
            help="Frames carried over from the previous chunk. Trained at 5."),
    Control("408", "vary_seed_per_window", "Vary seed per window", G_VIDEO, "bool"),
    Control("408", "pose_strength", "Pose strength", G_VIDEO, "float", 0, 10, 0.01),
    Control("408", "pose_start", "Pose start", G_VIDEO, "float", 0, 1, 0.01),
    Control("408", "pose_end", "Pose end", G_VIDEO, "float", 0, 1, 0.01),
    Control("339", "shift", "Sampling shift", G_VIDEO, "float", 0, 20, 0.1),
    Control("326", "strength_model", "SCAIL-2 DPO LoRA", G_VIDEO, "float", 0, 2, 0.05),
    Control("314", "resize_type.megapixels", "Video resolution (MP)", G_VIDEO, "float",
            0.1, 1.5, 0.05, help="Output size before rounding to a multiple of 32. "
                                 "Time and VRAM grow with it."),
    # ── output ──────────────────────────────────────────────────────────────
    Control("335", "crf", "Video quality (CRF)", G_OUT, "int", 0, 51, 1,
            help="Lower is better quality and a larger file."),
]

CONTROLS: dict[str, list[Control]] = {"replace": REPLACE_CONTROLS}

_LOCK = threading.Lock()


class SettingsError(ValueError):
    pass


def _read() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        # A damaged file must never take a service down: run the saved workflow.
        print(f"WARNING: ignoring unreadable {SETTINGS_PATH}: {e}")
        return {}


def _write(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(SETTINGS_PATH)


def overrides(service: str) -> dict:
    """The admin's changes for one service, keyed "<node>.<input>"."""
    known = {c.key for c in CONTROLS.get(service, [])}
    saved = _read().get(service, {}).get("overrides", {})
    # Drop keys no control defines any more, so a removed control stops applying.
    return {k: v for k, v in saved.items() if k in known}


def apply(service: str, wf: dict) -> dict:
    for key, value in overrides(service).items():
        node, inp = key.split(".", 1)
        if node in wf:
            wf[node]["inputs"][inp] = value
    return wf


def _coerce(c: Control, v):
    if c.type == "bool":
        if not isinstance(v, bool):
            raise SettingsError(f"{c.label}: expected on/off.")
        return v
    if c.type == "text":
        if not isinstance(v, str):
            raise SettingsError(f"{c.label}: expected text.")
        if len(v) > MAX_TEXT:
            raise SettingsError(f"{c.label}: longer than {MAX_TEXT} characters.")
        return v
    if c.type == "choice":
        if v not in c.choices:
            raise SettingsError(f"{c.label}: '{v}' is not one of the allowed values.")
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SettingsError(f"{c.label}: expected a number.")
    if c.type == "int":
        if v != int(v):
            raise SettingsError(f"{c.label}: expected a whole number.")
        v = int(v)
    else:
        v = float(v)
    if c.min is not None and v < c.min or c.max is not None and v > c.max:
        raise SettingsError(f"{c.label}: must be between {c.min:g} and {c.max:g}.")
    if c.type == "int" and c.step and c.step > 1 and (v - (c.min or 0)) % c.step:
        raise SettingsError(
            f"{c.label}: must be {c.min:g} plus a multiple of {c.step:g}."
        )
    return v


def describe(service: str, workflow: dict) -> dict:
    """Every control with its workflow default and current value, for the admin page."""
    if service not in CONTROLS:
        raise KeyError(service)
    ov = overrides(service)
    meta = _read().get(service, {})
    rows = []
    for c in CONTROLS[service]:
        default = workflow.get(c.node, {}).get("inputs", {}).get(c.input)
        rows.append(
            {
                **asdict(c),
                "key": c.key,
                "default": default,
                "value": ov.get(c.key, default),
                "modified": c.key in ov,
                "title": workflow.get(c.node, {}).get("_meta", {}).get("title", ""),
            }
        )
    return {
        "service": service,
        "controls": rows,
        "updated": meta.get("updated"),
        "updated_by": meta.get("updated_by"),
    }


def save(service: str, workflow: dict, values: dict, username: str) -> None:
    """
    Replace the service's overrides with `values`. A value equal to the
    workflow's own default is not stored, so "modified" always means "differs
    from the JSON", and a later edit of the JSON still reaches untouched controls.
    """
    if service not in CONTROLS:
        raise KeyError(service)
    by_key = {c.key: c for c in CONTROLS[service]}
    unknown = sorted(set(values) - set(by_key))
    if unknown:
        raise SettingsError(f"Unknown control(s): {', '.join(unknown)}")
    new = {}
    for key, raw in values.items():
        c = by_key[key]
        v = _coerce(c, raw)
        if v != workflow.get(c.node, {}).get("inputs", {}).get(c.input):
            new[key] = v
    if service == "replace":
        scail = workflow["408"]["inputs"]
        start = new.get("408.pose_start", scail["pose_start"])
        end = new.get("408.pose_end", scail["pose_end"])
        if start > end:
            raise SettingsError("Pose start must not be after pose end.")
    with _LOCK:
        data = _read()
        data[service] = {"overrides": new, "updated": time.time(), "updated_by": username}
        _write(data)
