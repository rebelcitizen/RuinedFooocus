"""Which base model a checkpoint or LoRA belongs to.

Nothing here touches the network. The answer comes from the safetensors header
(training metadata and tensor names), from the CivitAI record RuinedFooocus has
already cached, or from the folder the inbox filed the model into.

A family is ``{"arch": key, "base": label}``: ``arch`` says which checkpoints a
LoRA loads on (``sdxl``, ``flux2-klein-4b``, ...) and ``base`` is the finer
label shown to the user ("Illustrious", "SDXL 1.0", "Flux.2 Klein 4B").
"""
import json
import re
import struct
from pathlib import Path

UNKNOWN = "unknown"

ARCH_LABELS = {
    "sd15": "SD 1.5",
    "sd2": "SD 2.x",
    "sdxl": "SDXL",
    "sd3": "SD 3",
    "flux1": "Flux.1",
    "flux2": "Flux.2",
    "flux2-klein-4b": "Flux.2 Klein 4B",
    "flux2-klein-9b": "Flux.2 Klein 9B",
    "krea2": "Krea 2",
    "zimage": "Z-Image",
    "qwen": "Qwen Image",
    "hunyuan-video": "Hunyuan Video",
    "wan-video": "Wan Video",
    UNKNOWN: "?",
}

# Lower-case prefixes of a CivitAI baseModel, a kohya ss_base_model_version or a
# modelspec.architecture, in the order they must be tried.
_LABEL_RULES = [
    ("flux.2 klein 4b", "flux2-klein-4b"),
    ("flux2_klein_4b", "flux2-klein-4b"),
    ("flux-2-klein-4b", "flux2-klein-4b"),
    ("flux.2 klein 9b", "flux2-klein-9b"),
    ("flux2_klein_9b", "flux2-klein-9b"),
    ("flux-2-klein-9b", "flux2-klein-9b"),
    ("flux.2", "flux2"),
    ("flux2", "flux2"),
    ("flux-2", "flux2"),
    ("flux.1", "flux1"),
    ("flux1", "flux1"),
    ("flux-1", "flux1"),
    ("krea 2", "krea2"),
    ("krea2", "krea2"),
    ("sdxl", "sdxl"),
    ("stable-diffusion-xl", "sdxl"),
    ("pony", "sdxl"),
    ("illustrious", "sdxl"),
    ("noobai", "sdxl"),
    ("sd 1.", "sd15"),
    ("sd_v1", "sd15"),
    ("sd1", "sd15"),
    ("stable-diffusion-v1", "sd15"),
    ("sd 2.", "sd2"),
    ("sd_v2", "sd2"),
    ("sd2", "sd2"),
    ("stable-diffusion-v2", "sd2"),
    ("sd 3", "sd3"),
    ("sd3", "sd3"),
    ("stable-diffusion-v3", "sd3"),
    ("zimage", "zimage"),
    ("z-image", "zimage"),
    ("z_image", "zimage"),
    ("qwen", "qwen"),
    ("hunyuan video", "hunyuan-video"),
    ("hunyuan_video", "hunyuan-video"),
    ("wan video", "wan-video"),
    ("wan2", "wan-video"),
]

_BLOCK = re.compile(
    r"(?:^|[._])(double_blocks|transformer_blocks|single_blocks|single_transformer_blocks)[._](\d+)[._]",
    re.MULTILINE,
)
_MAX_HEADER = 256 * 1024 * 1024


def arch_from_label(label):
    """Map a base-model label (CivitAI, kohya, modelspec, folder name) to an arch key."""
    if not label:
        return UNKNOWN
    text = str(label).strip().lower()
    for prefix, arch in _LABEL_RULES:
        if text.startswith(prefix):
            return arch
    return UNKNOWN


def arch_from_filename(name):
    """Last resort: hints in the file name."""
    text = str(name).lower()
    if "klein" in text:
        if "9b" in text:
            return "flux2-klein-9b"
        if "4b" in text:
            return "flux2-klein-4b"
        return "flux2"
    for needle, arch in (
        ("flux2", "flux2"),
        ("flux.2", "flux2"),
        ("flux-2", "flux2"),
        ("flux", "flux1"),
        ("krea2", "krea2"),
        ("zimage", "zimage"),
        ("z-image", "zimage"),
        ("qwen", "qwen"),
        ("pony", "sdxl"),
        ("illustrious", "sdxl"),
        ("noob", "sdxl"),
        ("sdxl", "sdxl"),
        ("sd15", "sd15"),
        ("sd1.5", "sd15"),
        ("v1-5", "sd15"),
    ):
        if needle in text:
            return arch
    if re.search(r"(^|[^a-z])xl([^a-z]|$)", text):
        return "sdxl"
    return UNKNOWN


def read_header(path):
    """(metadata dict, tensor names) of a safetensors file, reading only its header."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        return {}, []
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if size <= 0 or size > _MAX_HEADER:
                return {}, []
            header = json.loads(f.read(size))
    except Exception:
        return {}, []
    meta = header.pop("__metadata__", None) or {}
    return (meta if isinstance(meta, dict) else {}), list(header.keys())


def arch_from_metadata(meta):
    """Training metadata written by kohya-ss and friends."""
    for key in ("ss_base_model_version", "modelspec.architecture"):
        arch = arch_from_label(meta.get(key))
        if arch != UNKNOWN:
            return arch
    trained_on = str(meta.get("ss_sd_model_name", "")).lower()
    if "sd_xl" in trained_on or "sdxl" in trained_on:
        return "sdxl"
    return UNKNOWN


def arch_from_keys(keys):
    """Tensor names: works for checkpoints and for LoRAs in kohya, diffusers or ComfyUI naming."""
    joined = "\n".join(keys)

    def has(s):
        return s in joined

    if has("conditioner.embedders.1."):
        return "sdxl"
    if has("cond_stage_model."):
        return "sd2" if has("cond_stage_model.model.") else "sd15"
    if has("lora_te2_") or has("lora_unet_input_blocks_4_1_transformer_blocks_1_"):
        return "sdxl"
    if has("lora_unet_input_blocks") or has("lora_te_text_model"):
        return "sd15"
    if has("joint_blocks"):
        return "sd3"
    if has("txtfusion") or has(".attn.gate."):
        return "krea2"
    if has("context_refiner") or has("noise_refiner"):
        return "zimage"
    if has("individual_token_refiner"):
        return "hunyuan-video"
    if has("img_mod") and has("transformer_blocks") and not has("double_blocks"):
        return "qwen"

    double, single = -1, -1
    for m in _BLOCK.finditer(joined):
        kind, idx = m.group(1), int(m.group(2))
        if kind in ("double_blocks", "transformer_blocks"):
            double = max(double, idx)
        else:
            single = max(single, idx)
    has_single = has("single_blocks") or has("single_transformer_blocks")
    if has("double_stream_modulation") or (0 <= double <= 9 and has_single):
        # Flux.2: Klein 4B has 5 double-stream blocks, Klein 9B has 8
        if double <= 4:
            return "flux2-klein-4b"
        if double <= 7 and single <= 23:
            return "flux2-klein-9b"
        return "flux2"
    if double >= 10 and (has_single or has("img_attn")):
        return "flux1"
    return UNKNOWN


def detect(path=None, civit_base=None, folder_hint=None, name=None):
    """Family of one model file. Every argument is optional; unknown stays unknown."""
    meta, keys = read_header(path) if path else ({}, [])
    arch = arch_from_metadata(meta) if meta else UNKNOWN
    if arch == UNKNOWN and keys:
        arch = arch_from_keys(keys)
    elif arch == "flux2" and keys:
        # the metadata only says Flux.2: the tensor names tell which Klein it is
        refined = arch_from_keys(keys)
        if refined.startswith("flux2-"):
            arch = refined

    label_arch = arch_from_label(civit_base)
    if arch == UNKNOWN:
        arch = label_arch
    elif arch == "flux2" and label_arch.startswith("flux2-"):
        arch = label_arch
    if arch == UNKNOWN and folder_hint:
        arch = arch_from_label(folder_hint)
    if arch == UNKNOWN and name:
        arch = arch_from_filename(name)

    base = ARCH_LABELS.get(arch, arch)
    if civit_base and civit_base != "Unknown" and (label_arch == arch or label_arch == UNKNOWN and arch == UNKNOWN):
        base = str(civit_base)
    elif folder_hint and arch != UNKNOWN and arch_from_label(folder_hint) == arch:
        base = str(folder_hint)
    return {"arch": arch, "base": base}


def compatible(lora_arch, checkpoint_arch):
    """True/False when both sides are known, None when either is unknown."""
    if UNKNOWN in (lora_arch, checkpoint_arch) or not lora_arch or not checkpoint_arch:
        return None
    if lora_arch == checkpoint_arch:
        return True
    lora_family, checkpoint_family = lora_arch.split("-")[0], checkpoint_arch.split("-")[0]
    if lora_family == checkpoint_family and (lora_arch == lora_family or checkpoint_arch == checkpoint_family):
        # one side only knows the family (plain "flux2"): give it the benefit of the doubt
        return True
    return False
