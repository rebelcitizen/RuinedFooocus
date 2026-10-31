import traceback
from pathlib import Path

import numpy as np
import modules.async_worker as worker
import modules.controlnet
import modules.unblur as unblur
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class pipeline:
    """PowerUp type "unblur": deblur the input image, restore faces, optionally
    redraw details with the job's checkpoint, sharpen, upscale."""
    pipeline_type = ["unblur"]
    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # The checkpoint is only needed for the generative fix-up, which loads it itself.
        self.model_hash = name
        return

    def load_keywords(self, lora):
        return " "

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        task_id = gen_data["task_id"]

        def status(message):
            worker.add_result(task_id, "preview", (-1, message, None))

        def generative(image, denoise):
            """Run the job's checkpoint as Img2Img over `image` with the job's prompt.

            Edit models (Flux.2 Klein, Kontext, Qwen-Image-Edit) take the image as
            their reference and keep the denoise as the amount they redraw; SDXL
            models do a plain Img2Img pass. The intermediate image is saved to the
            outputs folder like any other generation.
            """
            data = dict(gen_data)
            prompt = data.get("prompt")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            data.update({
                "prompt": prompt or "",
                "input_image": image,
                "input_image2": None,
                "input_image3": None,
                "cn_selection": modules.controlnet.NEWCN,
                "cn_type": "img2img",
                "cn_strength": float(denoise),
                "cn_start": 0.0,
                "cn_stop": 1.0,
                "cn_edge_low": 0.2,
                "cn_edge_high": 0.8,
                "cn_upscale": "None",
                "image_number": 1,
                "image_total": 1,
                "index": (0, 1),
                "image_index": 0,
                "generate_forever": False,
                "show_preview": False,
            })
            results = worker._process(data)
            paths = [r for r in results if isinstance(r, (str, Path))]
            if not paths:
                raise RuntimeError("the generative fix-up produced no image")
            with Image.open(paths[-1]) as im:
                return im.convert("RGB")

        input_image = gen_data.get("input_image")
        if input_image is None:
            print("Unblur needs an input image.")
            status("Unblur needs an input image ...")
            return []

        settings = modules.controlnet.get_settings(gen_data)
        print(f"Unblur with: {settings}")
        try:
            result = unblur.run(input_image, settings, status, generative)
            status("Done ...")
            return [np.array(result.convert("RGB"))]
        except Exception:
            traceback.print_exc()
            worker.add_result(task_id, "preview", (-1, "Oops ...", "html/error.png"))
            return []
