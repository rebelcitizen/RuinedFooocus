"""Register this RuinedFooocus instance as an image-generation worker with a JobHub server.

Configuration comes from environment variables, or from a `.env` file in the
RuinedFooocus folder (ignored by git):

    JOBHUB_URL=https://jobs.example.com
    JOBHUB_TOKEN=jhw_...
    JOBHUB_WORKER_NAME=RuinedFooocus on my PC   (optional)
    JOBHUB_ENABLED=1                            (optional, 0 disables)

Without a URL and token nothing happens. Jobs received from the hub go through
the same queue as the local UI (`modules.async_worker`), so both can be used at
the same time; the hub only ever sees image files and their PNG metadata.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

import modules.async_worker as worker
import modules.controlnet
import shared
from modules.jobworker import Job, JobCancelled, JobContext, JobWorker, load_env_file
from shared import path_manager, settings

log = logging.getLogger("remote_worker")

JOB_TYPE = "image.generate"
MAX_CUSTOM_PIXELS = 2_200_000
MAX_IMAGES = 8
MAX_INPUTS = 3
PROGRESS_STATUS_MAX = 120
JOB_TIMEOUT_SECONDS = 30 * 60


class ImageGenerationWorker(JobWorker):
    job_type = JOB_TYPE

    # ---- what the hub shows in its controls -------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        import modules.sdxl_styles as sdxl_styles

        perf = shared.performance_settings.load_performance()
        return {
            "models": list(shared.models.get_names("checkpoints")),
            "loras": list(shared.models.get_names("loras")),
            "performances": [p for p in perf.keys()],
            "resolutions": list(shared.resolution_settings.aspect_ratios.keys()),
            "cheat_codes": list(modules.controlnet.cn_options.keys()),
            "styles": list(sdxl_styles.allstyles),
            "defaults": {
                "model": settings.default_settings.get("base_model"),
                "performance": settings.default_settings.get("performance"),
                "resolution": settings.default_settings.get("resolution"),
            },
            "max_images": MAX_IMAGES,
            "max_inputs": MAX_INPUTS,
        }

    # ---- one job --------------------------------------------------------------------
    def run(self, job: Job, ctx: JobContext) -> None:
        spec = job.spec
        caps = self.capabilities()
        self._validate(spec, caps)

        ctx.progress(0, "downloading inputs")
        images: Dict[int, Image.Image] = {}
        for index, path in ctx.download_inputs().items():
            if index >= MAX_INPUTS:
                continue
            with Image.open(path) as im:
                images[index] = im.convert("RGB")
        if (1 in images or 2 in images) and 0 not in images:
            raise ValueError("reference images need an input image (image 1)")

        gen_data = self._build_gen_data(job, spec, images)
        # The web UI resets these before every generation; do the same for remote jobs.
        shared.state["interrupted"] = False
        worker.interrupt_ruined_processing = False
        task_id = worker.add_task(gen_data)
        log.info("job %s queued in RuinedFooocus as task %s", job.id, task_id)

        last_percent = 0.0
        results: List[Any] = []
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        while True:
            flag, product = worker.task_result(task_id)
            if flag == "preview":
                percent, status, preview = product
                if isinstance(percent, (int, float)) and percent >= 0:
                    last_percent = float(percent)
                preview_path = preview if isinstance(preview, str) and os.path.isfile(preview) else None
                ctx.progress(last_percent, _short(status), preview_path=preview_path)
                if ctx.cancelled:
                    worker.interrupt_ruined_processing = True
            elif flag == "results":
                results = list(product or [])
                break
            if time.monotonic() > deadline:
                worker.interrupt_ruined_processing = True
                raise RuntimeError(f"generation exceeded {JOB_TIMEOUT_SECONDS // 60} minutes")

        if ctx.cancelled:
            raise JobCancelled(job.id)

        preview_grid = Path(str(path_manager.model_paths["temp_preview_path"])).resolve()
        uploaded = 0
        for item in results:
            path = Path(str(item))
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved == preview_grid or path.suffix.lower() != ".png" or not path.is_file():
                continue
            ctx.add_output(path, kind="image", metadata=_png_metadata(path), mime="image/png")
            uploaded += 1
        if uploaded == 0:
            raise RuntimeError("generation finished without producing an image")

    def on_cancel(self, job_id: str) -> None:
        if job_id == self._current_job_id:
            worker.interrupt_ruined_processing = True

    # ---- helpers ------------------------------------------------------------------------
    def _validate(self, spec: Dict[str, Any], caps: Dict[str, Any]) -> None:
        def check(key: str, allowed: List[str], optional: bool = False) -> None:
            value = spec.get(key)
            if optional and value in (None, "", "None"):
                return
            if allowed and value not in allowed:
                raise ValueError(f"{key} '{value}' is not available on this worker")

        check("model", caps["models"])
        check("performance", caps["performances"])
        if spec.get("resolution") != "custom":
            check("resolution", caps["resolutions"])
        check("cheat_code", caps["cheat_codes"], optional=True)
        for style in spec.get("styles") or []:
            if style not in caps["styles"]:
                raise ValueError(f"style '{style}' is not available on this worker")
        if not str(spec.get("prompt", "")).strip():
            raise ValueError("prompt is empty")

    def _build_gen_data(self, job: Job, spec: Dict[str, Any], images: Dict[int, Image.Image]) -> Dict[str, Any]:
        prompt_text = str(spec["prompt"])
        prompts = [p.strip() for p in prompt_text.split("---")]
        prompts = [p for p in prompts if p] or [prompt_text.strip()]
        image_number = max(1, min(int(spec.get("image_number", 1) or 1), MAX_IMAGES))

        if spec.get("resolution") == "custom":
            width = int(spec.get("width") or 0)
            height = int(spec.get("height") or 0)
            if width < 256 or height < 256 or width * height > MAX_CUSTOM_PIXELS:
                raise ValueError("custom resolution must be at least 256 px per side and at most 2.2 megapixels")
            aspect = shared.resolution_settings.CUSTOM_RESOLUTION
        else:
            width, height = shared.resolution_settings.aspect_ratios[spec["resolution"]]
            aspect = spec["resolution"]

        try:
            seed = int(spec.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = -1

        cheat = spec.get("cheat_code")
        if cheat in (None, "", "None"):
            cheat = "None"

        defaults = settings.default_settings
        loras = [
            ("", f"{defaults.get(f'lora_{n}_weight', 1.0)} - {defaults.get(f'lora_{n}_model', 'None')}")
            for n in range(1, 6)
        ]

        return {
            "task_type": "process",
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "negative": str(spec.get("negative") or ""),
            "loras": loras,
            "style_selection": list(spec.get("styles") or []),
            "seed": seed,
            "base_model_name": spec["model"],
            "performance_selection": spec["performance"],
            "aspect_ratios_selection": aspect,
            "custom_width": width,
            "custom_height": height,
            "cn_selection": cheat,
            "cn_type": None,
            "image_number": image_number,
            "image_total": len(prompts) * image_number,
            "generate_forever": False,
            "input_image": images.get(0),
            "input_image2": images.get(1),
            "input_image3": images.get(2),
            "inpaint_toggle": False,
            "main_view": None,
            "remote_job_id": job.id,
        }


def _short(status: Any) -> Optional[str]:
    if status is None:
        return None
    text = str(status).strip()
    return text[:PROGRESS_STATUS_MAX] if text else None


def _png_metadata(path: Path) -> Dict[str, Any]:
    """Pick the useful bits of RuinedFooocus's embedded PNG metadata."""
    try:
        with Image.open(path) as im:
            raw = im.info.get("parameters")
            size = im.size
    except Exception:
        return {}
    meta: Dict[str, Any] = {"width": size[0], "height": size[1]}
    if not raw:
        return meta
    try:
        params = json.loads(raw)
    except Exception:
        return meta
    for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "base_model_name", "denoise"):
        if key in params:
            meta[key] = params[key]
    if "Prompt" in params:
        meta["prompt"] = str(params["Prompt"])[:2000]
    return meta


_instance: Optional[ImageGenerationWorker] = None


def start() -> Optional[ImageGenerationWorker]:
    """Start the worker thread if JOBHUB_URL and JOBHUB_TOKEN are configured. Never raises."""
    global _instance
    if _instance is not None:
        return _instance
    try:
        load_env_file(Path(".env"))
        url = os.environ.get("JOBHUB_URL", "").strip()
        token = os.environ.get("JOBHUB_TOKEN", "").strip()
        enabled = os.environ.get("JOBHUB_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
        if not url or not token:
            print("Remote worker: JOBHUB_URL / JOBHUB_TOKEN not set, not registering with a JobHub.")
            return None
        if not enabled:
            print("Remote worker: JOBHUB_ENABLED=0, staying disconnected.")
            return None

        _configure_logging()
        name = os.environ.get("JOBHUB_WORKER_NAME", "").strip() or f"RuinedFooocus on {socket.gethostname()}"
        try:
            from version import version as rf_version
        except Exception:
            rf_version = "unknown"
        work_dir = Path(path_manager.model_paths["cache_path"]) / "jobhub"
        _instance = ImageGenerationWorker(
            server_url=url,
            token=token,
            display_name=name,
            version=f"RuinedFooocus {rf_version}",
            work_dir=work_dir,
        )
        _instance.start()
        print(f"Remote worker: registering with {url} as '{name}'")
        return _instance
    except Exception as exc:  # never break RuinedFooocus startup
        print(f"Remote worker: failed to start ({exc})")
        return None


def _configure_logging() -> None:
    for name in ("jobworker", "remote_worker"):
        logger = logging.getLogger(name)
        if logger.handlers:
            continue
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
