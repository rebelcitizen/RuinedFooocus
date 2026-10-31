"""Unblur a photo: a deblur model, optional face restoration, a restoration
upscaler (SeedVR2 or a spandrel upscaler such as HAT), an optional generative
refine with the job's checkpoint (whole image or tiled at full resolution), an
optional second upscale and sharpening, chained in one PowerUp.

Every spandrel model loads exactly like RuinedFooocus's upscalers do. The deblur
models are 1x restoration networks kept in the upscale_models folder (NAFNet and
FFTformer, both trained on the GoPro motion-blur set, MIT licensed). SeedVR2
(ByteDance, MIT) is a one-step diffusion restorer that deblurs, denoises and
upscales in one pass and adds real texture; see modules/seedvr2.py. Faces are
found by OpenCV's YuNet detector and restored by GFPGAN v1.4 (Apache 2.0) on
aligned 512 px crops that are blended back with a soft mask, the way GFPGAN's
own inference script does it. Known files download on first use (see
modules/pathdb/restoration.json).

Order of the steps:
  deblur -> find and restore faces
  -> refine "generative": paste faces, run the checkpoint over the whole image
  -> upscaler (SeedVR2 at `upscale_factor`, or a fixed-scale spandrel model)
  -> refine "tiled": paste faces, run the checkpoint tile by tile at full size
  -> upscaler2 -> unsharp mask -> (resize back to the input size) -> paste faces.

PowerUp keys (all optional):
  deblur, faces, face_strength, face_detector, refine (none|generative|tiled),
  refine_denoise, refine_faces, refine_tile, refine_overlap, refine_controlnet,
  refine_strength, upscaler, upscale_factor, upscaler2, upscale_factor2,
  color_fix (lab|wavelet|adain|none, SeedVR2 only), sharpen, keep_size.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

import comfy.utils
import comfy.model_management as model_management
import comfy.model_patcher
from spandrel import ModelLoader

import modules.seedvr2 as seedvr2

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
    "refine_tile": 1024,
    "refine_overlap": 192,
    "refine_controlnet": "None",
    "refine_strength": 0.6,
    "upscaler": "None",
    "upscale_factor": 2,
    "upscaler2": "None",
    "upscale_factor2": 2,
    "color_fix": "lab",
    "seedvr2_tile": 2048,
    "sharpen": 0,
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


def upscale(work, name, factor, opts, seed, status):
    """One upscale pass: SeedVR2 at `factor`, or a spandrel model at its own scale."""
    if seedvr2.is_seedvr2(name):
        status(f"Loading SeedVR2 ({name}) ...")
        model, vae = seedvr2.load(
            model_file(name, "path_upscalers"),
            model_file(opts.get("seedvr2_vae") or seedvr2.DEFAULT_VAE, "path_vae"),
        )
        return seedvr2.restore(
            work, model, vae,
            scale=max(0.25, float(factor or 2)),
            seed=int(seed),
            color_fix=str(opts.get("color_fix") or "lab"),
            tile=int(opts.get("seedvr2_tile") or 2048),
            status=status,
        )
    status(f"Upscaling with {name} ...")
    return to_image(apply_model(load_model(name, "path_upscalers"), to_tensor(work)))


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


def _positions(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = max(1, tile - overlap)
    return list(range(0, size - tile, stride)) + [size - tile]


def _feather(h, w, overlap, top, left, bottom, right):
    """Weights that ramp over `overlap` px on the sides that meet another tile."""
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    n = min(overlap, h // 2, w // 2)
    ramp = (np.arange(n, dtype=np.float32) + 1) / max(1, n)
    if top:
        wy[:n] = np.minimum(wy[:n], ramp)
    if bottom:
        wy[h - n:] = np.minimum(wy[h - n:], ramp[::-1])
    if left:
        wx[:n] = np.minimum(wx[:n], ramp)
    if right:
        wx[w - n:] = np.minimum(wx[w - n:], ramp[::-1])
    return wy[:, None, None] * wx[None, :, None]


def refine_tiled(work, generative, opts, status):
    """Run the job's checkpoint over the image in overlapping tiles at full size.

    Each tile is a full generation at the job's resolution (the tile size), so
    the checkpoint adds texture everywhere instead of repainting a downscaled
    copy of the whole picture. Tiles are feathered together over the overlap.
    """
    tile = max(256, int(opts.get("refine_tile") or 1024))
    overlap = max(32, min(tile // 2, int(opts.get("refine_overlap") or 192)))
    denoise = max(0.05, min(1.0, float(opts.get("refine_denoise", 0.4))))
    controlnet = opts.get("refine_controlnet")
    controlnet = None if controlnet in NONE else str(controlnet)
    strength = float(opts.get("refine_strength", 0.6))
    W, H = work.size
    ys = _positions(H, tile, overlap)
    xs = _positions(W, tile, overlap)
    total = len(ys) * len(xs)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    done = 0
    for y in ys:
        for x in xs:
            done += 1
            h = min(tile, H - y)
            w = min(tile, W - x)
            status(f"Generative refine, tile {done} of {total} (denoise {denoise}) ...")
            crop = work.crop((x, y, x + w, y + h))
            out = generative(crop, denoise, controlnet, strength).convert("RGB")
            if out.size != (w, h):
                out = out.resize((w, h), Image.Resampling.LANCZOS)
            mask = _feather(h, w, overlap, y > 0, x > 0, y + h < H, x + w < W)
            canvas[y:y + h, x:x + w] += np.asarray(out, dtype=np.float32) * mask
            weight[y:y + h, x:x + w] += mask
    return Image.fromarray(np.clip(canvas / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8))


def run(image, settings, status=lambda message: None, generative=None, seed=0):
    """Restore one PIL image with a PowerUp's settings and return the result.

    `generative(image, denoise, controlnet, strength)` is supplied by the
    pipeline and runs the job's checkpoint as Img2Img over an image (with an
    optional ControlNet such as "tile" for SDXL checkpoints); it is only called
    when the PowerUp sets refine to "generative" or "tiled".
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
    refine = str(opts.get("refine", "none")).lower()
    if generative is None and refine in ("generative", "tiled"):
        print(f"Unblur: refine '{refine}' needs the pipeline's generative callback, skipping it.")
        refine = "none"

    if refine == "generative":
        # The restored faces go in first so the checkpoint sees them as its reference.
        work = paste_faces(work, faces, 1.0, status)
        faces = []
        denoise = max(0.05, min(1.0, float(opts.get("refine_denoise", 0.4))))
        status(f"Generative fix-up with the job's model (denoise {denoise}) ...")
        work = generative(work, denoise, None, 1.0).convert("RGB")
        if opts.get("refine_faces"):
            faces = restore_faces(work, opts, status)
        detected_w = work.width

    upscaler = opts.get("upscaler")
    if upscaler not in NONE:
        work = upscale(work, upscaler, opts.get("upscale_factor"), opts, seed, status)

    if refine == "tiled":
        work = paste_faces(work, faces, work.width / detected_w, status)
        faces = []
        work = refine_tiled(work, generative, opts, status)
        if opts.get("refine_faces"):
            faces = restore_faces(work, opts, status)
        detected_w = work.width

    upscaler2 = opts.get("upscaler2")
    if upscaler2 not in NONE:
        work = upscale(work, upscaler2, opts.get("upscale_factor2"), opts, seed, status)

    sharpen = float(opts.get("sharpen") or 0)
    if sharpen > 0:
        status("Sharpening ...")
        work = work.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen), threshold=3))

    if opts.get("keep_size") and work.size != (orig_w, orig_h):
        work = work.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    return paste_faces(work, faces, work.width / detected_w, status)
