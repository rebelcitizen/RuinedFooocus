import traceback
import numpy as np
import modules.async_worker as worker
import modules.controlnet
import modules.unblur as unblur
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class pipeline:
    """PowerUp type "unblur": deblur the input image, restore faces, sharpen, upscale."""
    pipeline_type = ["unblur"]
    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
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

        input_image = gen_data.get("input_image")
        if input_image is None:
            print("Unblur needs an input image.")
            status("Unblur needs an input image ...")
            return []

        settings = modules.controlnet.get_settings(gen_data)
        print(f"Unblur with: {settings}")
        try:
            result = unblur.run(input_image, settings, status)
            status("Done ...")
            return [np.array(result.convert("RGB"))]
        except Exception:
            traceback.print_exc()
            status("Oops ...")
            worker.add_result(task_id, "preview", (-1, "Oops ...", "html/error.png"))
            return []
