"""Unblur a photo: a deblur model, optional face restoration, sharpening and an
upscaler, chained in one PowerUp.

Every model loads through spandrel exactly like RuinedFooocus's upscalers do.
The deblur models are 1x restoration networks kept in the upscale_models folder
(NAFNet and FFTformer, both trained on the GoPro motion-blur set, MIT licensed).
Faces are found by OpenCV's YuNet detector and restored by GFPGAN v1.4 (Apache
2.0) on aligned 512 px crops that are blended back with a soft mask, the way
GFPGAN's own inference script does it. Both face files live in the faceswap
folder. Known files download on first use (see modules/pathdb/restoration.json).

Order of the steps: deblur -> find and restore faces -> unsharp mask ->
upscaler -> (resize back to the input size) -> paste the restored faces.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

import comfy.utils
import comfy.model_management as model_management
import comfy.model_patcher
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from spandrel import ModelLoader

FACE_SIZE = 512
# FFHQ alignment template for 512 px crops: left eye, right eye, nose, left and
# right mouth corner, all as seen in the image (left = smaller x).
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)
FACE_BORDER = (135, 133, 132)
DEFAULT_FACE_DETECTOR = "face_detection_yunet_2023mar.onnx"
MIN_FACE_PX = 24

NONE = (None, "", "None", "none", False)

DEFAULTS = {
    "deblur": "NAFNet-GoPro-width64.pth",
    "faces": "None",
    "face_strength": 1.0,
    "sharpen": 0,
    "upscaler": "None",
    "keep_size": False,
}


def model_file(name, folder_key):
    """Absolute path of a model file; known names download on first use."""
    from shared import path_manager

    folder = path_manager.paths[folder_key]
    if isinstance(folder, list):
        folder = folder[0]
    default = path_manager.get_abspath(folder) / name
    return Path(path_manager.get_file_path(name, default=default))


def load_model(name, folder_key):
    """A spandrel image model wrapped the way ComfyUI's upscale node expects."""
    path = model_file(name, folder_key)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    desc = ModelLoader().load_from_state_dict(sd).eval()
    desc.patcher = comfy.model_patcher.CoreModelPatcher(
        desc.model,
        load_device=model_management.get_torch_device(),
        offload_device=model_management.unet_offload_device(),
    )
    return desc


def to_tensor(image):
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def to_image(tensor):
    arr = np.clip(255.0 * tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_model(desc, image_t):
    """Run a 1x or Nx spandrel model over the whole image, tiled, with OOM fallback.

    ComfyUI's node halves the tile on a plain out-of-memory error. When the card is
    nearly full because a checkpoint is still resident, the CUDA driver can instead
    fail with "device not ready" / "Failed to create GPU mapping", which the node
    re-raises as is. Treat that like an OOM: unload everything else and try again.
    """
    try:
        return ImageUpscaleWithModel().upscale(desc, image_t)[0]
    except RuntimeError as e:
        if "cuda" not in str(e).lower():
            raise
        print(f"Unblur: {e}. Freeing VRAM and retrying once ...")
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        return ImageUpscaleWithModel().upscale(desc, image_t)[0]


def find_faces(rgb, detector_path):
    """5-point landmarks of every face YuNet finds, ordered like FACE_TEMPLATE."""
    h, w = rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (w, h), 0.7, 0.3, 50)
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    found = []
    for face in faces if faces is not None else []:
        if min(face[2], face[3]) < MIN_FACE_PX:
            continue
        pts = face[4:14].reshape(5, 2)
        eyes = sorted(pts[:2], key=lambda p: p[0])
        mouth = sorted(pts[3:5], key=lambda p: p[0])
        found.append(np.array([eyes[0], eyes[1], pts[2], mouth[0], mouth[1]], dtype=np.float32))
    return found


def restore_face(desc, crop, strength):
    """GFPGAN on one aligned 512 px RGB crop (uint8); returns float32 RGB."""
    device = desc.patcher.load_device
    model_management.load_models_gpu([desc.patcher], force_full_load=True)
    x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        y = desc.model(x * 2 - 1, return_rgb=False)[0]
    y = ((y.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    if strength < 1.0:
        y = strength * y + (1.0 - strength) * crop.astype(np.float32)
    return y.astype(np.float32)


def paste_face(canvas, face, affine, factor):
    """Blend a restored 512 px face back into canvas (float32 RGB at the final size).

    affine maps input-image coordinates to the crop; factor is final size / input size.
    """
    h, w = canvas.shape[:2]
    inverse = cv2.invertAffineTransform(affine) * factor
    warped = cv2.warpAffine(face, inverse, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), inverse, (w, h))
    k = max(2, int(2 * factor))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    edge = max(1, int(mask.sum() ** 0.5) // 20)
    center = cv2.erode(mask, np.ones((edge * 2, edge * 2), np.uint8))
    soft = cv2.GaussianBlur(center, (edge * 2 + 1, edge * 2 + 1), 0)[:, :, None]
    return soft * warped + (1.0 - soft) * canvas


def run(image, settings, status=lambda message: None):
    """Restore one PIL image with a PowerUp's settings and return the result."""
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in settings.items() if v is not None})
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    work = image

    deblur = opts.get("deblur")
    if deblur not in NONE:
        status(f"Deblurring with {deblur} ...")
        work = to_image(apply_model(load_model(deblur, "path_upscalers"), to_tensor(work)))

    faces = []
    face_model = opts.get("faces")
    if face_model not in NONE:
        status("Looking for faces ...")
        rgb = np.array(work)
        detector = model_file(opts.get("face_detector") or DEFAULT_FACE_DETECTOR, "path_faceswap")
        landmarks = find_faces(rgb, detector)
        if landmarks:
            desc = load_model(face_model, "path_faceswap")
            strength = max(0.0, min(1.0, float(opts.get("face_strength", 1.0))))
            for i, lm in enumerate(landmarks):
                affine, _ = cv2.estimateAffinePartial2D(lm, FACE_TEMPLATE, method=cv2.LMEDS)
                if affine is None:
                    continue
                status(f"Restoring face {i + 1} of {len(landmarks)} ...")
                crop = cv2.warpAffine(
                    rgb, affine, (FACE_SIZE, FACE_SIZE),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=FACE_BORDER,
                )
                faces.append((restore_face(desc, crop, strength), affine))
        else:
            status("No face found, skipping face restoration ...")

    sharpen = float(opts.get("sharpen") or 0)
    if sharpen > 0:
        status("Sharpening ...")
        work = work.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen), threshold=3))

    upscaler = opts.get("upscaler")
    if upscaler not in NONE:
        status(f"Upscaling with {upscaler} ...")
        work = to_image(apply_model(load_model(upscaler, "path_upscalers"), to_tensor(work)))
        if opts.get("keep_size"):
            work = work.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    if faces:
        status("Blending the restored faces ...")
        canvas = np.array(work).astype(np.float32)
        factor = work.width / orig_w
        for face, affine in faces:
            canvas = paste_face(canvas, face, affine, factor)
        work = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    return work
