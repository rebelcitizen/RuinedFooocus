"""Unblur a photo: a deblur model, optional face restoration, an optional
generative fix-up with the job's checkpoint, sharpening and an upscaler,
chained in one PowerUp.

Every model loads through spandrel exactly like RuinedFooocus's upscalers do.
The deblur models are 1x restoration networks kept in the upscale_models folder
(NAFNet and FFTformer, both trained on the GoPro motion-blur set, MIT licensed).
Faces are found by OpenCV's YuNet detector and restored by GFPGAN v1.4 (Apache
2.0) on aligned 512 px crops that are blended back with a soft mask, the way
GFPGAN's own inference script does it. Both face files live in the faceswap
folder. Known files download on first use (see modules/pathdb/restoration.json).

Order of the steps: deblur -> find and restore faces -> generative fix-up
(the pipeline passes a callable that runs the job's model as Img2Img on the
current image) -> faces again if asked -> unsharp mask -> upscaler -> (resize
back to the input size) -> paste the restored faces.
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
    "refine": "none",
    "refine_denoise": 0.4,
    "refine_faces": False,
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


GB = 1024 ** 3


def apply_model(desc, image_t, tile=None, overlap=32):
    """Run a 1x or Nx spandrel model over the whole image, tiled.

    Same tiling as ComfyUI's upscale node, but it asks model management for the
    memory an attention upscaler such as HAT really needs on a 512 px tile, so a
    resident checkpoint is offloaded before the run instead of the allocator
    failing mid-tile ("device not ready", "Failed to create GPU mapping" or a
    CUDACachingAllocator assert, none of which the node recognises as an OOM).
    Any CUDA error that still happens frees everything and halves the tile.
    """
    device = desc.patcher.load_device
    if tile is None:
        # 256 px tiles keep HAT-sized upscalers well inside the card even next to a
        # resident checkpoint; the 1x deblur nets are light enough for 512 px tiles.
        tile = 256 if desc.scale > 1 else 512
    memory_required = (6 * GB if desc.scale > 1 else 2 * GB) + image_t.nelement() * image_t.element_size() * 8
    model_management.load_models_gpu([desc.patcher], memory_required=memory_required, force_full_load=True)
    in_img = image_t.movedim(-1, -3).to(device)
    output_device = model_management.intermediate_device()
    while True:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            out = comfy.utils.tiled_scale(
                in_img, lambda a: desc(a.float()), tile_x=tile, tile_y=tile, overlap=overlap,
                upscale_amount=desc.scale, pbar=pbar, output_device=output_device,
            )
            break
        except Exception as e:
            text = str(e).lower()
            if "cuda" not in text and "out of memory" not in text:
                raise
            tile //= 2
            if tile < 128:
                raise
            print(f"Unblur: {str(e).splitlines()[0]}. Freeing VRAM and retrying with {tile} px tiles ...")
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
            model_management.load_models_gpu([desc.patcher], memory_required=memory_required, force_full_load=True)
    return torch.clamp(out.movedim(-3, -1), min=0, max=1.0)


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
    model_management.load_models_gpu([desc.patcher], memory_required=2 * GB, force_full_load=True)
    x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        y = desc.model(x * 2 - 1, return_rgb=False)[0]
    y = ((y.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    if strength < 1.0:
        y = strength * y + (1.0 - strength) * crop.astype(np.float32)
    return y.astype(np.float32)


def restore_faces(work, opts, status):
    """Find every face in a PIL image and restore it.

    Returns a list of (restored 512 px face, affine) where affine maps the
    coordinates of `work` to the crop; the faces are pasted later with paste_faces.
    """
    face_model = opts.get("faces")
    if face_model in NONE:
        return []
    status("Looking for faces ...")
    rgb = np.array(work.convert("RGB"))
    detector = model_file(opts.get("face_detector") or DEFAULT_FACE_DETECTOR, "path_faceswap")
    landmarks = find_faces(rgb, detector)
    if not landmarks:
        status("No face found, skipping face restoration ...")
        return []
    desc = load_model(face_model, "path_faceswap")
    strength = max(0.0, min(1.0, float(opts.get("face_strength", 1.0))))
    faces = []
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
    return faces


def paste_face(canvas, face, affine, factor):
    """Blend a restored 512 px face back into canvas (float32 RGB at the final size).

    affine maps the detection image's coordinates to the crop; factor is the
    final size divided by the detection image's size.
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


def paste_faces(work, faces, factor, status):
    if not faces:
        return work
    status("Blending the restored faces ...")
    canvas = np.array(work.convert("RGB")).astype(np.float32)
    for face, affine in faces:
        canvas = paste_face(canvas, face, affine, factor)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def run(image, settings, status=lambda message: None, generative=None):
    """Restore one PIL image with a PowerUp's settings and return the result.

    `generative(image, denoise)` is supplied by the pipeline and runs the job's
    checkpoint as Img2Img over the current image; it is only called when the
    PowerUp sets refine to "generative".
    """
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in settings.items() if v is not None})
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    work = image

    deblur = opts.get("deblur")
    if deblur not in NONE:
        status(f"Deblurring with {deblur} ...")
        work = to_image(apply_model(load_model(deblur, "path_upscalers"), to_tensor(work)))

    faces = restore_faces(work, opts, status)
    detected_w = work.width

    if str(opts.get("refine", "none")).lower() == "generative" and generative is not None:
        # The restored faces go in first so the checkpoint sees them as its reference.
        work = paste_faces(work, faces, 1.0, status)
        faces = []
        denoise = max(0.05, min(1.0, float(opts.get("refine_denoise", 0.4))))
        status(f"Generative fix-up with the job's model (denoise {denoise}) ...")
        work = generative(work, denoise).convert("RGB")
        if opts.get("refine_faces"):
            faces = restore_faces(work, opts, status)
        detected_w = work.width

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

    return paste_faces(work, faces, work.width / detected_w, status)
