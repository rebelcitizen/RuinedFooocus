"""SeedVR2 (ByteDance's one-step restoration transformer, MIT) through the
native ComfyUI implementation: restore, deblur and upscale a photo in one pass.

The transformer (seedvr2_3b_*.safetensors) lives in the upscale_models folder so
it shows up next to the other upscalers; its VAE lives in the vae folder. Both
download on first use (modules/pathdb/restoration.json).

The graph is the one from ComfyUI's own "SeedVR2 3B: Upscale Image" template:
resize the input to the output size (lanczos) -> pad to a multiple of 16 ->
tiled VAE encode -> SeedVR2 conditioning -> one euler step at cfg 1 from pure
noise -> tiled VAE decode -> colour transfer from the resized input.

Memory: ComfyUI measured about 0.55 GiB of activations per output megapixel for
the 3B model on top of the 3.4 GB fp8 weights, so a 12 GB card restores up to
about 2048 px in one piece. Larger outputs run as overlapping tiles that are
feathered together; any CUDA error frees the card and halves the tile.
"""
import numpy as np
import torch
from PIL import Image

import comfy.sd
import comfy.utils
import comfy.model_management as model_management
from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess
from nodes import VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, common_ksampler

DEFAULT_VAE = "seedvr2_ema_vae_fp16.safetensors"
VAE_TILE = (512, 128, 4096, 8)  # tile, overlap, temporal size, temporal overlap

_loaded = {}


def is_seedvr2(name):
    name = str(name).lower()
    return "seedvr2" in name and "vae" not in name


def load(model_path, vae_path):
    """The SeedVR2 model patcher and its VAE; the last pair stays cached.

    fp8 checkpoints keep their fp8 storage (3.2 GB for the 3B model) and are
    cast per layer at run time; left to itself ComfyUI would widen them to
    bf16 on a card without fp8 compute and double the resident size.
    """
    key = (str(model_path), str(vae_path))
    if key not in _loaded:
        _loaded.clear()
        sd = comfy.utils.load_torch_file(str(model_path), safe_load=True)
        model_options = {}
        if comfy.utils.weight_dtype(sd) in (torch.float8_e4m3fn, torch.float8_e5m2):
            model_options["dtype"] = comfy.utils.weight_dtype(sd)
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)
        if model is None:
            raise RuntimeError(f"{model_path} is not a SeedVR2 model ComfyUI can load")
        del sd
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_path), safe_load=True))
        _loaded[key] = (model, vae)
    return _loaded[key]


def unload():
    _loaded.clear()
    model_management.unload_all_models()
    model_management.soft_empty_cache(True)


def _is_view_error(e):
    text = str(e)
    return "view" in text and "inplace" in text


def _encode(vae, pixels):
    try:
        return VAEEncodeTiled().encode(vae, pixels, *VAE_TILE)[0]
    except RuntimeError as e:
        # ComfyUI 0.32's SeedVR2 tiled VAE multiplies a view in place, which
        # torch 2.14 refuses; the plain encoder works for tile-sized inputs.
        if not _is_view_error(e):
            raise
        return VAEEncode().encode(vae, pixels)[0]


def _decode(vae, samples):
    try:
        return VAEDecodeTiled().decode(vae, samples, *VAE_TILE)[0]
    except RuntimeError as e:
        if not _is_view_error(e):
            raise
        return VAEDecode().decode(vae, samples)[0]


@torch.inference_mode()
def _run_once(model, vae, tile_t, seed, color_fix):
    """One SeedVR2 pass over a (1, H, W, 3) tensor already at the output size.

    Inference mode matters: ComfyUI's executor normally provides it, and
    without it the VAE keeps every activation alive (a 512 px decode then
    takes 8 GB instead of about 1 GB).
    """
    padded = SeedVR2Preprocess.execute(tile_t).args[0]
    latent = _encode(vae, padded)
    positive, negative = SeedVR2Conditioning.execute(model, latent).args
    sampled = common_ksampler(model, seed, 1, 1.0, "euler", "simple", positive, negative, latent, denoise=1.0)[0]
    decoded = _decode(vae, sampled)
    return SeedVR2PostProcessing.execute(decoded, tile_t, color_fix).args[0]


def _positions(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = max(1, tile - overlap)
    return list(range(0, size - tile, stride)) + [size - tile]


def _feather(h, w, overlap, top, left, bottom, right):
    """Weights that ramp over `overlap` px on the sides that meet another tile."""
    wy = torch.ones(h)
    wx = torch.ones(w)
    ramp = (torch.arange(overlap, dtype=torch.float32) + 1) / overlap
    n = min(overlap, h // 2, w // 2)
    if top:
        wy[:n] = torch.minimum(wy[:n], ramp[:n])
    if bottom:
        wy[h - n:] = torch.minimum(wy[h - n:], ramp[:n].flip(0))
    if left:
        wx[:n] = torch.minimum(wx[:n], ramp[:n])
    if right:
        wx[w - n:] = torch.minimum(wx[w - n:], ramp[:n].flip(0))
    return (wy[:, None] * wx[None, :])[:, :, None]


def _restore_tensor(model, vae, image_t, seed, color_fix, tile, overlap, status):
    _, th, tw, _ = image_t.shape
    ys = _positions(th, tile, overlap)
    xs = _positions(tw, tile, overlap)
    if len(ys) * len(xs) == 1:
        status("SeedVR2 restoring ...")
        return _run_once(model, vae, image_t, seed, color_fix)
    canvas = torch.zeros(th, tw, 3)
    weight = torch.zeros(th, tw, 1)
    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            done += 1
            status(f"SeedVR2 restoring tile {done} of {total} ...")
            h = min(tile, th - y)
            w = min(tile, tw - x)
            out = _run_once(model, vae, image_t[:, y:y + h, x:x + w].contiguous(), seed, color_fix)[0].cpu()
            oh, ow = out.shape[:2]
            mask = _feather(oh, ow, overlap, y > 0, x > 0, y + h < th, x + w < tw)
            canvas[y:y + oh, x:x + ow] += out.float() * mask
            weight[y:y + oh, x:x + ow] += mask
    return (canvas / weight.clamp(min=1e-6))[None]


def restore(image, model, vae, scale=2.0, seed=0, color_fix="lab", tile=2048, overlap=192, status=lambda m: None):
    """Restore a PIL image at `scale` times its size (rounded to multiples of 16)."""
    w, h = image.size
    tw = max(16, int(round(w * scale / 16)) * 16)
    th = max(16, int(round(h * scale / 16)) * 16)
    resized = image.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
    image_t = torch.from_numpy(np.array(resized).astype(np.float32) / 255.0)[None]
    tile = max(512, int(tile) // 16 * 16)
    overlap = max(32, int(overlap) // 16 * 16)
    while True:
        try:
            out = _restore_tensor(model, vae, image_t, seed, color_fix, tile, overlap, status)
            break
        except Exception as e:
            text = str(e).lower()
            if "cuda" not in text and "out of memory" not in text:
                raise
            if tile <= 512:
                raise
            tile = max(512, tile // 2)
            print(f"SeedVR2: {str(e).splitlines()[0]}. Freeing VRAM and retrying with {tile} px tiles ...")
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
    arr = np.clip(255.0 * out[0].float().cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
