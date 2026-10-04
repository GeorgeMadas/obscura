import os
import hashlib
import io
import json
import queue
import random
import re
import threading
import time
import traceback
import uuid
import copy
import math
from pathlib import Path

# Configure memory allocator before heavy operations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import requests
import torch
import spandrel
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from accelerate import init_empty_weights
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift
from diffusers.models.model_loading_utils import load_gguf_checkpoint
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
from huggingface_hub import constants as hf_constants, hf_hub_download
from PIL import Image, ImageOps
from safetensors.torch import load_file as safetensors_load_file

import comfy_bridge
import prompt_rewriter

# Model Configuration
MODEL_ID = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"


def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_settings(**changes):
    SETTINGS_FILE.write_text(json.dumps({**_read_settings(), **changes}, indent=2))


# The model picked in the UI (settings.json) wins over the environment default
CHECKPOINT = _read_settings().get("checkpoint") or os.environ.get("QWEN_GGUF_CHECKPOINT", "qwen-image-2.1-UC-BF16.gguf")
COMPANION_ID = "Qwen/Qwen-Image-2.1"
COMPANION_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

SHA256_CHECKSUMS = {
    # Uncensored (UC) GGUFs
    "qwen-image-2.1-UC-Q4_K_M.gguf": "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41",
    "qwen-image-2.1-UC-Q4_0.gguf": "13f59f20656efc0aa385d03c1fcac1a9dc2ad6e5ccc0ea9bfe1d6ac636f2c5b9",
    "qwen-image-2.1-UC-Q5_K_M.gguf": "af0bf278cf16d204fb31c384dc82fd41dca82d976b15fe9305a60c726fd6f821",
    "qwen-image-2.1-UC-Q6_K.gguf": "e14bb312109333b3d73b92ad9b9ac8b29b51f2edca1e7b86d1af1981bf1c4ee3",
    "qwen-image-2.1-UC-Q8_0.gguf": "cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8",
    "qwen-image-2.1-UC-BF16.gguf": "f151c683a8aed4b310777017ebbbe3f2180f1180f7867115171adb7d50b0762a",
}

MODES = ["Text to Image", "Edit Image (1 Ref)", "Transform & Swap (2 Refs)", "Transparent PNG"]
# 4MP sizes are the official Qwen Image 2.1 presets; 1MP renders roughly 4x faster
RATIO_SIZES = {
    "1:1": {"1MP": (1024, 1024), "4MP": (2048, 2048)},
    "16:9": {"1MP": (1344, 768), "4MP": (2752, 1536)},
    "9:16": {"1MP": (768, 1344), "4MP": (1536, 2752)},
    "4:3": {"1MP": (1152, 864), "4MP": (2400, 1792)},
    "3:4": {"1MP": (864, 1152), "4MP": (1792, 2400)},
}
# Side length the pipeline also uses to size reference images in Edit / Blend
OUTPUT_RESOLUTION = {"1MP": 1024, "4MP": 2048}

# Checkpoints the UI can switch between
CHECKPOINT_INFO = {
    "qwen-image-2.1-UC-BF16.gguf": {"id": "full", "label": "Full", "detail": "BF16"},
    "qwen-image-2.1-UC-Q4_K_M.gguf": {"id": "small", "label": "Small", "detail": "Q4_K_M"},
}
MAX_SEED = 2**31 - 1

# ============================================================
# Curated All-In-One LoRA Specifications
# ============================================================
NONE_LORA = "None (Base Uncensored)"

# Only LoRAs trained for Qwen Image 2.1 (32 blocks, width 4096). LoRAs for the original
# Qwen-Image / Qwen-Image-Edit (60 blocks, width 3072) cannot load into this transformer.
ADAPTER_SPECS = {
    "Anime Consistency": {
        "repo": "WarmBloodAban/Qwen-Image-2.1-LoRAs",
        "weights": "Qwen2.1_Anime_consistency.safetensors",
        "adapter_name": "anime_consistency",
        "default_strength": 0.85,
        "default_steps": 28,
        "preset_prompt": "masterpiece, highly detailed anime illustration, vibrant colors, expressive eyes",
        "category": "Style",
    },
    "Natural Exposure (Photorealism)": {
        "repo": "prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA",
        "weights": "Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors",
        "adapter_name": "natural_exposure",
        "default_strength": 0.8,
        "default_steps": 30,
        "preset_prompt": "natural daylight exposure, authentic colors, unedited 35mm photograph, soft organic textures",
        "category": "Style",
    },
    "Photo Aesthetics": {
        "repo": "SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v2",
        "weights": "pytorch_lora_weights.safetensors",
        "adapter_name": "photo_aesthetics",
        "default_strength": 1.0,
        "default_steps": 30,
        "preset_prompt": "",
        "category": "Style",
    },
    "Expired Film": {
        "repo": "AIImageStudio/RadianceChromeVoluptuous_QwenImage2.1_v1.0",
        "weights": "RadianceChromeVoluptuous_QwenImage2.1_v1.0.safetensors",
        "adapter_name": "expired_film",
        "default_strength": 0.9,
        "default_steps": 25,
        "preset_prompt": "lighting and shadow effects, color cast, wrong white balance, expired film",
        "category": "Color",
    },
    "Quality Fix": {
        "repo": "e-n-v-y/Qwen-Image-2.1-Fix",
        "weights": "qwen-image-2.1-fix-1.0-comfy.safetensors",
        "adapter_name": "quality_fix",
        "default_strength": 1.0,
        "default_steps": 30,
        "preset_prompt": "",
        "category": "Utility",
    },
    "Viggle Turbo (4-Step Acceleration)": {
        "repo": "Viggle/Qwen-Image-2.1-viggle-turbo",
        "weights": "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
        "adapter_name": "viggle_turbo",
        "default_strength": 1.0,
        "default_steps": 4,
        "preset_prompt": "",
        "category": "Turbo Speed",
    },
}

LOCAL_LORA = "Local LoRA file..."
LORA_DIR = Path(os.environ.get("QWEN_LORA_DIR", Path(__file__).resolve().parent / "loras"))
LORA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Generation parameters are logged here, never embedded in the images themselves
HISTORY_LOG = Path(os.environ.get("QWEN_HISTORY_LOG", Path(__file__).resolve().parent / "history.jsonl"))

LORA_CHOICES = [NONE_LORA] + list(ADAPTER_SPECS.keys()) + ["Custom HuggingFace LoRA...", LOCAL_LORA]


def list_local_loras() -> list[str]:
    """All .safetensors files under LORA_DIR (recursive), relative to it."""
    return sorted(str(p.relative_to(LORA_DIR)) for p in LORA_DIR.rglob("*.safetensors"))


def resolve_local_lora(value: str) -> Path:
    """Accept a path relative to LORA_DIR or an absolute path."""
    value = (value or "").strip()
    if not value:
        raise UserError(f"Pick a LoRA from {LORA_DIR} or enter an absolute path to a .safetensors file.")
    path = Path(value)
    if not path.is_absolute():
        path = LORA_DIR / path
    if not path.is_file():
        raise UserError(f"LoRA file not found: {path}")
    return path

class UserError(Exception):
    """A problem the person can fix; shown verbatim in the UI."""


# Track dynamically loaded adapters
LOADED_ADAPTERS = set()

# ============================================================
# GGUF Checkpoint Initialization on CUDA
# ============================================================
VERIFIED_CHECKPOINTS: set[str] = set()


def _checkpoint_blob(name: str) -> Path | None:
    """Hub cache blobs are named by their SHA256, so a file is on disk if its blob exists."""
    sha = SHA256_CHECKSUMS.get(name)
    if not sha:
        return None
    blob = Path(hf_constants.HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}" / "blobs" / sha
    return blob if blob.is_file() else None


def load_checkpoint_weights(name: str, progress=lambda phase: None) -> dict:
    progress("Locating file")
    path = hf_hub_download(MODEL_ID, name)
    if name in SHA256_CHECKSUMS and name not in VERIFIED_CHECKPOINTS:
        progress("Verifying checksum")
        with open(path, "rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        if checksum != SHA256_CHECKSUMS[name]:
            raise RuntimeError(f"Checksum verification failed for {name}! Got {checksum}")
        VERIFIED_CHECKPOINTS.add(name)
        print(f"Checksum verified: {checksum}", flush=True)
    progress("Reading weights")
    # Expand the GGUF weights to bfloat16 so inference runs on plain tensors
    weights = load_gguf_checkpoint(path)
    progress("Expanding weights")
    for key in list(weights.keys()):
        weights[key] = dequantize_gguf_tensor(weights[key]).to(torch.bfloat16)
    return weights


print(f"Loading checkpoint {CHECKPOINT} from {MODEL_ID}...", flush=True)
weights = load_checkpoint_weights(CHECKPOINT)

config = QwenImage21Transformer2DModel.load_config(
    COMPANION_ID, subfolder="transformer", revision=COMPANION_REVISION
)
with init_empty_weights():
    transformer = QwenImage21Transformer2DModel.from_config(config)

transformer.load_state_dict(weights, strict=True, assign=True)
transformer.eval().requires_grad_(False)
# Plain (LoRA-free) layout, used to validate checkpoints before a live swap
BASE_SHAPES = {k: v.shape for k, v in transformer.state_dict().items()}
print(f"Loaded {len(weights)} tensors into QwenImage21Transformer2DModel.", flush=True)
del weights

# Load full pipeline with companion VAE and text encoder, eagerly placed on 'cuda'
pipe = QwenImage21Pipeline.from_pretrained(
    COMPANION_ID,
    revision=COMPANION_REVISION,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")
print(f"{MODEL_ID} successfully initialized on CUDA for ZeroGPU.", flush=True)


# ============================================================
# Sharing the GPU with ComfyUI: Qwen waits in CPU RAM while a ComfyUI job runs
# ============================================================
QWEN = {"on_gpu": True, "moving": False}
QWEN_BYTES = sum(t.numel() * t.element_size() for m in (pipe.transformer, pipe.text_encoder, pipe.vae)
                 for t in [*m.parameters(), *m.buffers()])
UNPARK_MARGIN = 6e9


def park_qwen(job: dict | None = None):
    """Move Qwen (and the prompt rewriter) to CPU RAM so ComfyUI gets the VRAM. Call with GPU_LOCK held."""
    if QWEN["on_gpu"]:
        if job:
            job["phase"] = "Moving Qwen off the GPU"
        pipe.to("cpu")
        QWEN["on_gpu"] = False
    prompt_rewriter.park()
    torch.cuda.empty_cache()


def unpark_qwen(job: dict | None = None):
    """Bring Qwen back to the GPU before anything uses the pipe. Call with GPU_LOCK held.

    ComfyUI is asked to drop its models first; it does that between prompts, so this waits for the VRAM
    (longer while ComfyUI is still busy with a run started from its editor)."""
    if QWEN["on_gpu"]:
        return
    comfy_bridge.free()
    deadline = time.time() + 90
    while torch.cuda.mem_get_info()[0] < QWEN_BYTES + UNPARK_MARGIN:
        if job and job["cancel"]:
            raise Cancelled()
        if comfy_bridge.busy():
            if job:
                job["phase"] = "Waiting for ComfyUI to finish"
            deadline = time.time() + 90
        elif time.time() > deadline:
            break  # try anyway; an OOM below is reported clearly
        elif job:
            job["phase"] = "Waiting for ComfyUI to free the GPU"
        time.sleep(0.5)
    if job:
        job["phase"] = "Loading Qwen back to GPU"
    try:
        pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        pipe.to("cpu")
        torch.cuda.empty_cache()
        raise UserError("Not enough free GPU memory to bring Qwen back. Something else (ComfyUI?) is still using the GPU.")
    QWEN["on_gpu"] = True


def clear_gpu_for_render(job: dict):
    """Free VRAM that a render can use: move the idle prompt rewriter (~18 GB) to CPU RAM and ask ComfyUI to
    drop its cached memory. Call with GPU_LOCK held."""
    job["phase"] = "Freeing GPU memory"
    prompt_rewriter.park()  # waits for a rewrite in progress
    comfy_bridge.free()
    torch.cuda.empty_cache()


# ============================================================
# LoRA Adapter Loading Helpers
# ============================================================
def _inject_missing_alpha_keys(state_dict: dict) -> dict:
    bases = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_down.weight") and v.ndim >= 1:
            base = k[:-len(".lora_down.weight")]
            rank = int(v.shape[0])
            bases[base] = rank

    for base, rank in bases.items():
        alpha_tensor = torch.tensor(float(rank), dtype=torch.float32)
        full_alpha = f"{base}.alpha"
        if full_alpha not in state_dict:
            state_dict[full_alpha] = alpha_tensor
        if base.startswith("diffusion_model."):
            stripped_base = base[len("diffusion_model."):]
            stripped_alpha = f"{stripped_base}.alpha"
            if stripped_alpha not in state_dict:
                state_dict[stripped_alpha] = alpha_tensor
    return state_dict


def _filter_to_diffusers_lora_keys(state_dict: dict) -> dict:
    keep_suffixes = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_mid.weight",
        ".alpha",
        ".lora_alpha",
    )
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".diff") or k.endswith(".diff_b"):
            continue
        if not k.endswith(keep_suffixes):
            continue
        if k.endswith(".lora_alpha"):
            base = k[:-len(".lora_alpha")]
            k2 = f"{base}.alpha"
            out[k2] = v.float() if v.dtype != torch.float32 else v
            continue
        out[k] = v
    return out


def _duplicate_stripped_prefix_keys(state_dict: dict, prefix: str = "diffusion_model.") -> dict:
    out = dict(state_dict)
    for k, v in list(state_dict.items()):
        if not k.startswith(prefix):
            continue
        stripped = k[len(prefix):]
        if stripped not in out:
            out[stripped] = v
    return out


QWEN21_WIDTH = transformer.config.num_attention_heads * transformer.config.attention_head_dim
QWEN21_BLOCKS = transformer.config.num_layers


def _check_compatible(state_dict: dict, label: str):
    """Reject LoRAs trained for other Qwen models before they fail deep inside diffusers."""
    width, last_block = None, -1
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        m = re.search(r"blocks[._](\d+)[._]", k)
        if m:
            last_block = max(last_block, int(m.group(1)))
        if v.ndim == 2 and re.search(r"attn[._]to_q[._].*(lora_A|lora_down)", k):
            width = v.shape[1]
    if (width and width != QWEN21_WIDTH) or last_block >= QWEN21_BLOCKS:
        raise UserError(
            f"{label} was trained for a different Qwen model (such as Qwen-Image or Qwen-Image-Edit), "
            "not Qwen Image 2.1, so it can't be used here."
        )


def _adapter_registered(adapter_name: str) -> bool:
    return adapter_name in pipe.get_list_adapters().get("transformer", [])


def _convert_bare_lora_keys(state_dict: dict) -> dict:
    """Formats like alibaba-pai's "prefused" export use bare `X.lora_down` / `X.lora_up` keys with no
    alpha (scale 1) and ship unchanged base weights alongside; keep only the LoRA pairs, in PEFT naming."""
    if not any(k.endswith((".lora_down", ".lora_up")) for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.endswith(".lora_down"):
            out[f"transformer.{k[:-len('.lora_down')]}.lora_A.weight"] = v
        elif k.endswith(".lora_up"):
            out[f"transformer.{k[:-len('.lora_up')]}.lora_B.weight"] = v
    return out


def _load_state_dict(path: str, adapter_name: str, label: str, needs_alpha_fix: bool = False):
    sd = _convert_bare_lora_keys(safetensors_load_file(path))
    _check_compatible(sd, label)
    try:
        try:
            pipe.load_lora_weights(dict(sd), adapter_name=adapter_name)
        except Exception as e:
            print(f"Direct LoRA load failed ({str(e)[:300]}), attempting key-conversion fallback...", flush=True)
            # A failed attempt can leave a partially registered adapter behind
            if _adapter_registered(adapter_name):
                pipe.delete_adapters(adapter_name)
            sd = _convert_bare_lora_keys(safetensors_load_file(path))
            if needs_alpha_fix:
                sd = _inject_missing_alpha_keys(sd)
            sd = _filter_to_diffusers_lora_keys(sd)
            sd = _duplicate_stripped_prefix_keys(sd)
            pipe.load_lora_weights(sd, adapter_name=adapter_name)
    except UserError:
        raise
    except Exception as e:
        if _adapter_registered(adapter_name):
            pipe.delete_adapters(adapter_name)
        raise UserError(f"{label} could not be loaded: {str(e)[:200]}")
    # diffusers only warns when no keys match, so confirm the adapter really exists
    if not _adapter_registered(adapter_name):
        raise UserError(f"{label} loaded, but none of its weights matched Qwen Image 2.1. It may use a format this app can't read.")


def _load_lora_with_fallback(repo: str, weight_name: str, adapter_name: str, needs_alpha_fix: bool = False, label: str = ""):
    try:
        local_path = hf_hub_download(repo_id=repo, filename=weight_name)
    except Exception as e:
        raise UserError(f"Could not download {weight_name} from {repo}: {str(e)[:200]}")
    _load_state_dict(local_path, adapter_name, label or f"{repo}/{weight_name}", needs_alpha_fix)


def _load_local_lora(path: Path, adapter_name: str):
    _load_state_dict(str(path), adapter_name, path.name, needs_alpha_fix=True)


def ensure_adapter_ready(selected_lora: str, custom_repo: str = "", custom_file: str = "", local_lora: str = "") -> tuple[str, float]:
    if selected_lora == NONE_LORA:
        return "", 1.0

    if selected_lora == LOCAL_LORA:
        path = resolve_local_lora(local_lora)
        # Key on path + mtime so a re-trained/overwritten file is reloaded
        stat = path.stat()
        adapter_name = f"local_{hashlib.md5(f'{path}:{stat.st_mtime_ns}'.encode()).hexdigest()[:8]}"
        if adapter_name not in LOADED_ADAPTERS:
            print(f"Loading local LoRA {path}...", flush=True)
            _load_local_lora(path, adapter_name)
            LOADED_ADAPTERS.add(adapter_name)
        return adapter_name, 1.0

    if selected_lora == "Custom HuggingFace LoRA...":
        custom_repo = (custom_repo or "").strip()
        custom_file = (custom_file or "").strip()
        if not custom_repo or not custom_file:
            raise UserError("Please enter both a Hugging Face Repo ID and LoRA filename for Custom LoRA.")
        adapter_name = f"custom_{hashlib.md5((custom_repo + custom_file).encode()).hexdigest()[:8]}"
        if adapter_name not in LOADED_ADAPTERS:
            print(f"Loading custom LoRA from {custom_repo} / {custom_file}...", flush=True)
            _load_lora_with_fallback(custom_repo, custom_file, adapter_name, needs_alpha_fix=True)
            LOADED_ADAPTERS.add(adapter_name)
        return adapter_name, 1.0

    spec = ADAPTER_SPECS.get(selected_lora)
    if not spec:
        return "", 1.0

    adapter_name = spec["adapter_name"]
    if adapter_name not in LOADED_ADAPTERS:
        print(f"Loading LoRA {selected_lora} ({spec['repo']} / {spec['weights']})...", flush=True)
        _load_lora_with_fallback(
            spec["repo"],
            spec["weights"],
            adapter_name,
            needs_alpha_fix=spec.get("needs_alpha_fix", False),
            label=LORA_SHORT_NAMES.get(selected_lora, selected_lora),
        )
        LOADED_ADAPTERS.add(adapter_name)

    return adapter_name, float(spec.get("default_strength", 1.0))


# ============================================================
# Studio metadata (surfaced to the web UI)
# ============================================================
MODE_INFO = {
    "Text to Image": {"id": "create", "label": "Create", "icon": "sparkle", "refs": 0,
                      "blurb": "Generate from a prompt"},
    "Edit Image (1 Ref)": {"id": "edit", "label": "Edit", "icon": "paint-brush-broad", "refs": 1,
                           "blurb": "Change one image with words"},
    # Internal name kept so older history entries still match; Qwen Image 2.1 accepts up to 10 references
    "Transform & Swap (2 Refs)": {"id": "blend", "label": "Blend", "icon": "intersect", "refs": 4, "min_refs": 2,
                                  "blurb": "Combine two to four references"},
    "Transparent PNG": {"id": "cutout", "label": "Cutout", "icon": "selection-background", "refs": 0,
                        "blurb": "Subject on a transparent background"},
}

LORA_BLURBS = {
    "Anime Consistency": "Anime look with stable characters. Strongest in Edit.",
    "Natural Exposure (Photorealism)": "Balanced, neutral exposure with honest color. Strongest in Edit.",
    "Photo Aesthetics": "Pushes renders toward well-composed photography.",
    "Expired Film": "Color casts, off white balance and expired-film grain.",
    "Quality Fix": "Corrects common Qwen Image 2.1 artifacts.",
    "Viggle Turbo (4-Step Acceleration)": "Drafts in four steps. Ideal for exploring ideas.",
}

LORA_SHORT_NAMES = {
    "Natural Exposure (Photorealism)": "Natural Exposure",
    "Viggle Turbo (4-Step Acceleration)": "Viggle Turbo",
}

SAFE_NAME = re.compile(r"^[\w\-.]+\.(png|mp4|webm)$")
VIDEO_SUFFIXES = (".mp4", ".webm")
MEDIA_TYPES = {".png": "image/png", ".mp4": "video/mp4", ".webm": "video/webm"}


THUMB_DIR = OUTPUT_DIR / ".thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"
HISTORY_LOCK = threading.Lock()


def _is_video(name: str) -> bool:
    return name.endswith(VIDEO_SUFFIXES)

# ============================================================
# Upscalers (4x ESRGAN family, run through spandrel)
# ============================================================
UPSCALER_DIR = Path(__file__).resolve().parent / "upscalers"
UPSCALERS = {
    "realesrgan": {"label": "Real-ESRGAN", "file": "RealESRGAN_x4plus.pth",
                   "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"},
    "ultrasharp": {"label": "UltraSharp", "file": "4x-UltraSharp.safetensors",
                   "sha256": "36a340b5509b699d2c06cb445ddc1d3d39199ac734d889ed6d7915f60e05bcbc"},
}
UPSCALE_TILE = 512
UPSCALE_PAD = 32
UPSCALE_MAX_INPUT = 8_500_000  # pixels; a 4MP render becomes ~67MP, so larger inputs are refused
_UPSCALER_MODELS: dict = {}


def _load_upscaler(key: str):
    if key not in _UPSCALER_MODELS:
        spec = UPSCALERS[key]
        path = UPSCALER_DIR / spec["file"]
        with open(path, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != spec["sha256"]:
                raise UserError(f"{spec['label']} weights failed their checksum. Re-download {spec['file']}.")
        if path.suffix == ".safetensors":
            state = safetensors_load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        model = spandrel.ModelLoader().load_from_state_dict(state)
        _UPSCALER_MODELS[key] = model.cuda().eval().half()
    return _UPSCALER_MODELS[key]


def _tile_grid(width: int, height: int) -> list[tuple[int, int, int, int]]:
    return [(x, y, min(x + UPSCALE_TILE, width), min(y + UPSCALE_TILE, height))
            for y in range(0, height, UPSCALE_TILE) for x in range(0, width, UPSCALE_TILE)]


def _esrgan(rgb: Image.Image, key: str, on_tile=None, cancelled=lambda: False) -> Image.Image:
    """Tiled 4x ESRGAN pass over an RGB image."""
    model = _load_upscaler(key)
    scale = model.scale
    w, h = rgb.size
    x = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).unsqueeze(0).cuda().half().div_(255)
    out = torch.empty((3, h * scale, w * scale), dtype=torch.uint8)
    tiles = _tile_grid(w, h)
    with torch.inference_mode():
        for i, (x0, y0, x1, y1) in enumerate(tiles):
            if cancelled():
                raise Cancelled()
            # Pad each tile with real neighbouring pixels so seams get context, then crop it away
            px0, py0 = max(x0 - UPSCALE_PAD, 0), max(y0 - UPSCALE_PAD, 0)
            px1, py1 = min(x1 + UPSCALE_PAD, w), min(y1 + UPSCALE_PAD, h)
            result = model(x[:, :, py0:py1, px0:px1])
            cx, cy = (x0 - px0) * scale, (y0 - py0) * scale
            crop = result[0, :, cy:cy + (y1 - y0) * scale, cx:cx + (x1 - x0) * scale]
            out[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = crop.clamp_(0, 1).mul_(255).round_().byte().cpu()
            if on_tile:
                on_tile(i, len(tiles))
    del x
    torch.cuda.empty_cache()
    return Image.fromarray(out.permute(1, 2, 0).numpy(), "RGB")


def run_upscale(job: dict) -> dict:
    req = job["request"]
    spec = UPSCALERS[req["upscaler"]]
    job["phase"] = f"Loading {spec['label']}"
    scale = _load_upscaler(req["upscaler"]).scale
    src = Image.open(OUTPUT_DIR / req["file"])
    src.load()
    alpha = src.getchannel("A") if src.mode in ("RGBA", "LA") else None
    job["phase"] = f"Upscaling with {spec['label']}"
    start_time = time.perf_counter()

    def on_tile(i, n):
        if i == 0:
            job["first_step_at"] = time.time()
        job["step"] = i + 1

    img = _esrgan(src.convert("RGB"), req["upscaler"], on_tile, lambda: job["cancel"])
    job["phase"] = "Saving"
    if alpha is not None:
        img.putalpha(alpha.resize(img.size, Image.LANCZOS))
    elapsed = time.perf_counter() - start_time

    src_record = _read_records().get(req["file"], {"file": req["file"]})
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000:03d}_{src_record.get('seed', 'img')}_{req['upscaler']}x{scale}.png"
    # Pixels only, like renders; fast compression because these files are large
    img.save(OUTPUT_DIR / name, "PNG", compress_level=3)
    record = {
        **{k: v for k, v in src_record.items() if k not in ("file", "dimensions", "elapsed_seconds", "created")},
        "file": name,
        "dimensions": f"{img.width}x{img.height}",
        "elapsed_seconds": round(elapsed, 2),
        "upscaler": f"{spec['label']} {scale}x",
        "upscaled_from": req["file"],
        "created": time.time(),
    }
    with HISTORY_LOCK, open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _public_record(record)


# ============================================================
# Enhance detail: low-strength re-render through Qwen ("hires fix")
# ============================================================
ENHANCE_TILE = 2048          # Qwen's native 4MP side; larger images are enhanced in overlapping tiles
ENHANCE_OVERLAP = 256
ENHANCE_SINGLE_MAX = 4_400_000   # up to ~4MP runs as one pass at the official 4MP size
ENHANCE_MAX_INPUT = 72_000_000   # the largest upscales (~67MP) still fit
SIGMA_END = 0.02             # matches the scheduler's shift_terminal
# Sent instead of the image's own prompt: without skin cues Qwen repaints skin smooth and plastic
ENHANCE_DEFAULT_PROMPT = "Skin texture, Visible natural pores, Fine skin detail"


def _enhance_positions(length: int) -> list[int]:
    """Tile start offsets along one axis: evenly spread, multiples of 16, last tile flush with the edge."""
    if length <= ENHANCE_TILE:
        return [0]
    n = math.ceil((length - ENHANCE_OVERLAP) / (ENHANCE_TILE - ENHANCE_OVERLAP))
    return [min(length - ENHANCE_TILE, round(i * (length - ENHANCE_TILE) / (n - 1) / 16) * 16) for i in range(n)]


def enhance_plan(width: int, height: int, ratio: str | None) -> dict:
    """How an image of this size is enhanced: one 4MP pass, or tiles at its native size."""
    if width * height <= ENHANCE_SINGLE_MAX:
        tw, th = (RATIO_SIZES[ratio]["4MP"] if ratio in RATIO_SIZES else (0, 0))
        # Use the official 4MP preset only when it matches the image's shape
        if not tw or abs(tw / th - width / height) > 0.01:
            tw, th, _ = calculate_dimensions(2048 * 2048, width / height)
        return {"tiled": False, "width": tw, "height": th, "tiles": 1}
    xs, ys = _enhance_positions(width), _enhance_positions(height)
    return {"tiled": True, "width": width, "height": height, "tiles": len(xs) * len(ys), "xs": xs, "ys": ys}


def _enhance_sigmas(strength: float, n: int, seq_len: int) -> tuple[list[float], float]:
    """Raw sigmas whose *shifted* values run evenly from `strength` down to the terminal sigma.

    At 4MP the scheduler shifts sigmas heavily toward noise (a raw 0.3 lands near 0.95), so strength is
    defined on the effective noise level: we invert the exponential time shift, then confirm with a copy of
    the scheduler and return the real starting sigma used to noise the latents."""
    sch = copy.deepcopy(pipe.scheduler)
    cfg = sch.config
    mu = calculate_shift(seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    target = np.linspace(strength, SIGMA_END, n)
    raw = 1.0 / (1.0 + math.exp(mu) * (1.0 / target - 1.0))
    sch.set_timesteps(sigmas=list(raw), mu=mu, device="cpu")
    return list(raw), float(sch.sigmas[0])


def _enhance_tile(tile: Image.Image, prompt: str, strength: float, n_steps: int, seed: int, on_step) -> Image.Image:
    """Noise the tile's latents to `strength` and let Qwen finish the last part of the schedule."""
    w, h = tile.size
    dtype = pipe.transformer.dtype
    with torch.inference_mode():
        pixels = pipe.image_processor.preprocess(tile.convert("RGBA"), width=w, height=h).unsqueeze(2)
        gen = torch.Generator("cuda").manual_seed(seed)
        x0 = pipe._encode_vae_image(pixels.to("cuda", pipe.vae.dtype), gen)
        c, lh, lw = x0.shape[1], x0.shape[3], x0.shape[4]
        x0 = pipe._pack_latents(x0, 1, c, lh, lw)
        raw, sigma0 = _enhance_sigmas(strength, n_steps, x0.shape[1])
        noise = torch.randn(x0.shape, generator=gen, device="cuda", dtype=torch.float32)
        latents = ((1.0 - sigma0) * x0.float() + sigma0 * noise).to(dtype)
        return pipe(
            prompt=prompt, width=w, height=h,
            num_inference_steps=len(raw), sigmas=raw, latents=latents,
            generator=gen, callback_on_step_end=on_step, output_resolution=2048,
        ).images[0]


def _ramp(length: int, lead: bool, trail: bool) -> np.ndarray:
    """Blend weights along one tile axis: fade in/out across the overlap where a neighbour exists."""
    wgt = np.ones(length, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, ENHANCE_OVERLAP + 2, dtype=np.float32)[1:-1]
    if lead:
        wgt[:ENHANCE_OVERLAP] = ramp
    if trail:
        wgt[-ENHANCE_OVERLAP:] = np.minimum(wgt[-ENHANCE_OVERLAP:], ramp[::-1])
    return wgt


def run_enhance(job: dict) -> dict:
    req = job["request"]
    strength = req["strength"]
    src_record = _read_records().get(req["file"], {"file": req["file"]})
    src = Image.open(OUTPUT_DIR / req["file"])
    src.load()
    alpha = src.getchannel("A") if src.mode in ("RGBA", "LA") else None
    rgb = src.convert("RGB")
    plan = enhance_plan(rgb.width, rgb.height, src_record.get("ratio"))
    n_steps = req["n_steps"]
    prompt = req.get("enhance_prompt") or ENHANCE_DEFAULT_PROMPT
    seed = int(src_record.get("seed") or random.randint(0, MAX_SEED))

    # Same style the image was made with, when it's one of the built-in LoRAs
    lora = src_record.get("lora")
    if lora in ADAPTER_SPECS:
        job["phase"] = "Loading style"
        adapter, _ = ensure_adapter_ready(lora)
        pipe.enable_lora()
        pipe.set_adapters([adapter], adapter_weights=[float(src_record.get("lora_strength") or ADAPTER_SPECS[lora]["default_strength"])])
    else:
        pipe.disable_lora()

    start_time = time.perf_counter()
    done = {"tiles": 0}

    def on_step(p, i, t, kwargs):
        if job["step"] == 0:
            job["first_step_at"] = time.time()
        job["step"] = done["tiles"] * n_steps + i + 1
        if job["cancel"]:
            p._interrupt = True
        return {}

    if not plan["tiled"]:
        base = rgb
        if rgb.width < plan["width"] or rgb.height < plan["height"]:
            # Enlarge with Real-ESRGAN first so Qwen refines real edges instead of blur
            job["phase"] = "Enlarging with Real-ESRGAN"
            base = _esrgan(rgb, "realesrgan", cancelled=lambda: job["cancel"])
        base = base.resize((plan["width"], plan["height"]), Image.LANCZOS)
        job["phase"] = "Enhancing detail"
        out_img = _enhance_tile(base, prompt, strength, n_steps, seed, on_step)
        if job["cancel"]:
            raise Cancelled()
        out_img = out_img.convert("RGB")
    else:
        W, H = plan["width"], plan["height"]
        acc = np.zeros((H, W, 3), dtype=np.float32)
        wsum = np.zeros((H, W, 1), dtype=np.float32)
        tw, th = min(ENHANCE_TILE, W) // 16 * 16, min(ENHANCE_TILE, H) // 16 * 16
        k = 0
        for yi, y in enumerate(plan["ys"]):
            for xi, x in enumerate(plan["xs"]):
                if job["cancel"]:
                    raise Cancelled()
                k += 1
                job["phase"] = f"Enhancing tile {k} of {plan['tiles']}"
                tile = rgb.crop((x, y, x + tw, y + th))
                res = _enhance_tile(tile, prompt, strength, n_steps, seed + k, on_step)
                if job["cancel"]:
                    raise Cancelled()
                wy = _ramp(th, yi > 0, yi < len(plan["ys"]) - 1)
                wx = _ramp(tw, xi > 0, xi < len(plan["xs"]) - 1)
                wgt = (wy[:, None] * wx[None, :])[..., None]
                acc[y:y + th, x:x + tw] += np.asarray(res.convert("RGB"), dtype=np.float32) * wgt
                wsum[y:y + th, x:x + tw] += wgt
                done["tiles"] = k
        # Any sliver no tile reached keeps the original pixels
        original = np.asarray(rgb, dtype=np.float32)
        covered = wsum > 0
        out = np.where(covered, acc / np.maximum(wsum, 1e-6), original)
        out_img = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")
        del acc, wsum, original, out
    torch.cuda.empty_cache()

    job["phase"] = "Saving"
    if alpha is not None:
        out_img.putalpha(alpha.resize(out_img.size, Image.LANCZOS))
    elapsed = time.perf_counter() - start_time
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000:03d}_{src_record.get('seed', 'img')}_enhanced.png"
    out_img.save(OUTPUT_DIR / name, "PNG", compress_level=3)
    record = {
        **{k: v for k, v in src_record.items() if k not in ("file", "dimensions", "elapsed_seconds", "created")},
        "file": name,
        "dimensions": f"{out_img.width}x{out_img.height}",
        "elapsed_seconds": round(elapsed, 2),
        "upscaler": f"Enhanced {strength:.2f}",
        "upscaled_from": req["file"],
        "enhance_strength": strength,
        "enhance_prompt": prompt,
        "created": time.time(),
    }
    with HISTORY_LOCK, open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _public_record(record)


# ============================================================
# Inference
# ============================================================
def _prepare_ref(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    img.thumbnail((3072, 3072))
    return img


MAX_BLEND_REFS = 4
# Rewritten prompts run to 400-500 words, well past what people type by hand
MAX_PROMPT_CHARS = 8000


def validate_request(p: dict, refs: list[Image.Image]) -> dict:
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise UserError("Describe what you want to see first.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise UserError(f"Prompt is too long. Keep it under {MAX_PROMPT_CHARS} characters.")
    mode = p.get("mode", "Text to Image")
    if mode not in MODES:
        raise UserError(f"Unknown mode: {mode}")
    ratio = p.get("ratio", "9:16")
    if ratio not in RATIO_SIZES:
        raise UserError("Unknown aspect ratio.")
    resolution = p.get("resolution", "4MP")
    if resolution not in OUTPUT_RESOLUTION:
        raise UserError("Resolution must be 1MP or 4MP.")
    width, height = RATIO_SIZES[ratio][resolution]
    try:
        steps = int(p.get("steps", 40))
    except (TypeError, ValueError):
        raise UserError("Steps must be a number.")
    if not 4 <= steps <= 40:
        raise UserError("Steps must be between 4 and 40.")
    lora = p.get("lora") or NONE_LORA
    if lora not in LORA_CHOICES:
        raise UserError(f"Unknown style: {lora}")
    if mode == "Edit Image (1 Ref)" and not refs:
        raise UserError("Add a reference image to edit.")
    if mode == "Transform & Swap (2 Refs)" and not 2 <= len(refs) <= MAX_BLEND_REFS:
        raise UserError(f"Blend needs 2 to {MAX_BLEND_REFS} reference images.")
    randomize = bool(p.get("randomize_seed", True))
    try:
        seed = random.randint(0, MAX_SEED) if randomize else int(p.get("seed", 42)) % (MAX_SEED + 1)
    except (TypeError, ValueError):
        raise UserError("Seed must be a whole number.")
    original = str(p.get("original_prompt") or "").strip()[:MAX_PROMPT_CHARS]
    return {
        "prompt": prompt,
        "original_prompt": original if original and original != prompt else "",
        "mode": mode,
        "ratio": ratio,
        "resolution": resolution,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "lora": lora,
        "lora_strength": max(0.0, min(1.5, float(p.get("lora_strength", 1.0)))),
        "custom_repo": p.get("custom_repo", ""),
        "custom_file": p.get("custom_file", ""),
        "local_lora": p.get("local_lora", ""),
        "refs": [_prepare_ref(r) for r in (refs[:1] if mode == "Edit Image (1 Ref)" else refs if mode == "Transform & Swap (2 Refs)" else [])],
    }


class Cancelled(Exception):
    pass


def run_generation(job: dict) -> dict:
    req = job["request"]
    width, height = req["width"], req["height"]

    job["phase"] = "Loading style" if req["lora"] != NONE_LORA else "Preparing"
    try:
        active_adapter, _ = ensure_adapter_ready(req["lora"], req["custom_repo"], req["custom_file"], req["local_lora"])
    except UserError:
        raise
    except Exception as e:
        raise UserError(f"Could not load the style: {e}")
    if active_adapter:
        pipe.enable_lora()
        pipe.set_adapters([active_adapter], adapter_weights=[req["lora_strength"]])
        lora_label = f"Local: {req['local_lora']}" if req["lora"] == LOCAL_LORA else req["lora"]
        active_lora_desc = f"{lora_label} (scale={round(req['lora_strength'], 2)})"
    else:
        pipe.disable_lora()
        active_lora_desc = "None"

    effective_prompt = req["prompt"]
    if req["mode"] == "Transparent PNG":
        effective_prompt = (
            "This is an RGBA image with transparency. " + req["prompt"]
            + " The image has alpha channel and the background is transparent."
        )

    call_kwargs = {}
    if req["mode"] == "Edit Image (1 Ref)":
        call_kwargs["image"] = req["refs"][0]
    elif req["mode"] == "Transform & Swap (2 Refs)":
        call_kwargs["image"] = list(req["refs"])

    def on_step(p, i, t, kwargs):
        if job["step"] == 0:
            job["first_step_at"] = time.time()
        job["step"] = i + 1
        job["phase"] = "Finishing" if i + 1 >= req["steps"] else "Rendering"
        if job["cancel"]:
            p._interrupt = True
        return {}

    if job["cancel"]:
        raise Cancelled()
    job["phase"] = "Reading prompt"
    start_time = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=effective_prompt,
            width=width,
            height=height,
            num_inference_steps=req["steps"],
            generator=torch.Generator("cuda").manual_seed(req["seed"]),
            callback_on_step_end=on_step,
            output_resolution=OUTPUT_RESOLUTION[req["resolution"]],
            **call_kwargs,
        ).images[0]
    elapsed = time.perf_counter() - start_time
    if job["cancel"]:
        raise Cancelled()

    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000:03d}_{req['seed']}.png"
    # Save pixels only: no text/EXIF/ICC chunks, so the file carries no generation metadata
    Image.frombytes(result.mode, result.size, result.tobytes()).save(OUTPUT_DIR / name, "PNG", optimize=True)

    record = {
        "file": name,
        "model": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "lora_adapter": active_lora_desc,
        "prompt": req["prompt"],
        **({"original_prompt": req["original_prompt"]} if req.get("original_prompt") else {}),
        "mode": req["mode"],
        "seed": req["seed"],
        "steps": req["steps"],
        "dimensions": f"{result.width}x{result.height}",
        "elapsed_seconds": round(elapsed, 2),
        # Structured fields so the UI can restore settings exactly
        "lora": req["lora"],
        "lora_strength": round(req["lora_strength"], 2),
        "ratio": req["ratio"],
        "resolution": req["resolution"],
        "created": time.time(),
    }
    with HISTORY_LOCK, open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _public_record(record)


# ============================================================
# ComfyUI workflows
# ============================================================
def _save_comfy_output(item: dict, req: dict, elapsed: float, index: int) -> dict:
    """Copy one ComfyUI output into OUTPUT_DIR (browser-playable, pixels only) and log it in the history."""
    ext = Path(item["filename"]).suffix.lower()
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000:03d}"
    base = f"{stamp}_{req.get('seed') if req.get('seed') is not None else 'wf'}_{req['slug']}" + (f"-{index + 1}" if index else "")
    tmp = OUTPUT_DIR / f".incoming-{uuid.uuid4().hex}.part"
    comfy_bridge.download(item, tmp)
    info = {}
    try:
        if ext in comfy_bridge.VIDEO_EXTS:
            info = comfy_bridge.probe(tmp)
            if ext in VIDEO_SUFFIXES and info.get("codec") in comfy_bridge.BROWSER_CODECS:
                name = base + ext
                tmp.replace(OUTPUT_DIR / name)
            else:
                name = base + ".mp4"
                comfy_bridge.to_browser_mp4(tmp, OUTPUT_DIR / name)
                info = comfy_bridge.probe(OUTPUT_DIR / name)
            media, width, height = "video", info.get("width"), info.get("height")
        else:
            with Image.open(tmp) as im:
                im.load()
                img = im if im.mode in ("RGB", "RGBA", "L", "LA") else im.convert("RGBA")
                name = base + ".png"
                # Pixels only, like renders
                Image.frombytes(img.mode, img.size, img.tobytes()).save(OUTPUT_DIR / name, "PNG", optimize=True)
                media, (width, height) = "image", img.size
    finally:
        tmp.unlink(missing_ok=True)
    record = {
        "file": name,
        "kind": "comfy",
        "media": media,
        "workflow": req["slug"],
        "workflow_name": req["name"],
        "prompt": req.get("prompt", ""),
        "mode": "Workflow",
        "seed": req.get("seed"),
        "steps": req.get("steps_value"),
        "dimensions": f"{width}x{height}",
        "elapsed_seconds": round(elapsed, 2),
        **({k: info[k] for k in ("fps", "frames", "duration") if info.get(k)}),
        "values": req.get("display", []),
        "form": req.get("values", {}),
        **({"source_image": req["source"]} if req.get("source") else {}),
        "created": time.time(),
    }
    with HISTORY_LOCK, open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _public_record(record)


def run_comfy(job: dict) -> dict:
    req = job["request"]
    if not comfy_bridge.status().get("up"):
        job["phase"] = "Starting ComfyUI"
        comfy_bridge.ensure_running()
    park_qwen(job)
    if job["cancel"]:
        raise Cancelled()
    job["phase"] = "Sending inputs to ComfyUI"
    media = {key: comfy_bridge.upload(data, filename) for key, (filename, data) in req["media"].items()}
    graph = comfy_bridge.patch(req["graph"], req["meta"], req["values"], media)
    start = time.perf_counter()
    try:
        hist = comfy_bridge.execute(graph, job, lambda: job["cancel"])
    except comfy_bridge.Interrupted:
        raise Cancelled()
    finally:
        # Hand the VRAM back unless the person wants ComfyUI's models kept loaded between runs
        if not _read_settings().get("comfy_keep_loaded"):
            comfy_bridge.free()
    elapsed = time.perf_counter() - start
    files = comfy_bridge.output_files(hist)
    if not files:
        raise UserError("The workflow finished but saved nothing. Add a Save Image or Save Video node.")
    job["phase"] = "Saving"
    records = [_save_comfy_output(item, req, elapsed, i) for i, item in enumerate(files)]
    job["results"] = records
    return records[0]


# ============================================================
# Job queue: one GPU, one worker, any number of queued requests
# ============================================================
JOBS: dict[str, dict] = {}
JOB_ORDER: "queue.Queue[str]" = queue.Queue()


def _job_view(job: dict) -> dict:
    ahead = 0
    if job["status"] == "queued":
        ahead = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running") and j["created"] < job["created"])
    return {
        "id": job["id"],
        "status": job["status"],
        "phase": job["phase"],
        "step": job["step"],
        "steps": job["request"]["steps"],
        "ahead": ahead,
        "created": job["created"],
        "started": job.get("started"),
        "first_step_at": job.get("first_step_at"),
        "server_time": time.time(),
        "prompt": job["request"]["prompt"],
        "ratio": job["request"]["ratio"],
        "resolution": job["request"]["resolution"],
        "mode": job["request"]["mode"],
        "kind": job["request"].get("kind", "render"),
        "aspect": job["request"].get("aspect"),
        "upscaler": job["request"].get("upscaler"),
        "strength": job["request"].get("strength"),
        "workflow": job["request"].get("name"),
        "media": job["request"].get("output"),
        "result": job.get("result"),
        "results": job.get("results"),
        "error": job.get("error"),
    }


def _worker():
    while True:
        job = JOBS.get(JOB_ORDER.get())
        if job is None or job["status"] == "cancelled":
            continue
        job.update(status="running", started=time.time())
        try:
            # Renders queued behind a model switch should use the new model
            while SWITCH["state"] == "loading" and not job["cancel"]:
                job["phase"] = "Waiting for model switch"
                time.sleep(0.5)
            with GPU_LOCK:
                kind = job["request"].get("kind")
                if kind == "comfy":
                    job["result"] = run_comfy(job)
                else:
                    clear_gpu_for_render(job)
                    if kind != "upscale":
                        unpark_qwen(job)
                    job["result"] = (run_upscale(job) if kind == "upscale"
                                     else run_enhance(job) if kind == "enhance" else run_generation(job))
            job["status"] = "done"
        except Cancelled:
            job["status"] = "cancelled"
        except (UserError, comfy_bridge.BridgeError) as e:
            job.update(status="error", error=str(e))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            job.update(status="error", error="The GPU ran out of memory. Try a smaller size or fewer references.")
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=f"Generation failed: {e}")
        finally:
            # Reference images and uploads are large; drop them once the job is over
            job["request"]["refs"] = []
            job["request"]["media"] = {}
            job["finished"] = time.time()
            _prune_jobs()


# ============================================================
# Model switching: swap transformer weights in place, no restart
# ============================================================
GPU_LOCK = threading.Lock()
SWITCH = {"state": "idle", "target": None, "phase": "", "error": None, "started": None}


def _switch_worker(name: str):
    global CHECKPOINT
    try:
        weights = load_checkpoint_weights(name, progress=lambda phase: SWITCH.update(phase=phase))
        mismatched = [k for k, shape in BASE_SHAPES.items() if k not in weights or weights[k].shape != shape]
        if mismatched or len(weights) != len(BASE_SHAPES):
            raise RuntimeError(f"{name} does not match the loaded architecture ({len(mismatched)} tensors differ).")
        SWITCH["phase"] = "Waiting for the current render"
        with GPU_LOCK:
            SWITCH["phase"] = "Swapping weights"
            # LoRA layers wrap the base weights; remove them so the swap sees plain modules
            if pipe.get_list_adapters():
                pipe.unload_lora_weights()
            LOADED_ADAPTERS.clear()
            transformer.load_state_dict(weights, strict=True)
            del weights
            torch.cuda.empty_cache()
            CHECKPOINT = name
        _write_settings(checkpoint=name)
        print(f"Switched transformer to {name}.", flush=True)
        SWITCH.update(state="idle", target=None, phase="", error=None)
    except Exception as e:
        traceback.print_exc()
        SWITCH.update(state="error", target=None, phase="", error=f"Could not switch model: {e}")


def _models_view() -> list[dict]:
    items = []
    for name, info in CHECKPOINT_INFO.items():
        blob = _checkpoint_blob(name)
        items.append({
            "file": name, **info,
            "available": blob is not None,
            "size_gb": round(blob.stat().st_size / 1e9, 1) if blob else None,
            "active": name == CHECKPOINT,
        })
    return items


def _status_view() -> dict:
    return {"checkpoint": CHECKPOINT, "switch": {**SWITCH, "server_time": time.time()}, "models": _models_view()}


def _prune_jobs():
    cutoff = time.time() - 3600
    for jid in [k for k, j in JOBS.items() if j.get("finished", time.time()) < cutoff]:
        JOBS.pop(jid, None)


threading.Thread(target=_worker, daemon=True, name="generation-worker").start()


# ============================================================
# History
# ============================================================
def _read_records() -> dict[str, dict]:
    records = {}
    if HISTORY_LOG.exists():
        for line in HISTORY_LOG.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                records[record["file"]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def _public_record(r: dict) -> dict:
    name = r["file"]
    dims = r.get("dimensions", "")
    w, _, h = dims.partition("x")
    lora = r.get("lora")
    if lora is None:
        # Older entries only stored a description like "Name (scale=0.8)"
        desc = r.get("lora_adapter") or "None"
        lora = NONE_LORA if desc == "None" else desc.rsplit(" (scale=", 1)[0]
    video = _is_video(name)
    return {
        "file": name,
        "url": f"/outputs/{name}",
        "thumb": f"/thumbs/{name}",
        "media": "video" if video else "image",
        "kind": r.get("kind", "render"),
        "workflow": r.get("workflow"),
        "workflow_name": r.get("workflow_name"),
        "fps": r.get("fps"),
        "frames": r.get("frames"),
        "duration": r.get("duration"),
        "values": r.get("values"),
        "form": r.get("form"),
        "source_image": r.get("source_image"),
        "prompt": r.get("prompt", ""),
        "mode": r.get("mode", "Text to Image"),
        "seed": r.get("seed"),
        "steps": r.get("steps"),
        "width": int(w) if w.isdigit() else None,
        "height": int(h) if h.isdigit() else None,
        "lora": lora,
        "lora_strength": r.get("lora_strength"),
        "ratio": r.get("ratio"),
        "resolution": r.get("resolution"),
        "checkpoint": r.get("checkpoint"),
        "elapsed": r.get("elapsed_seconds"),
        "upscaler": r.get("upscaler"),
        "upscaled_from": r.get("upscaled_from"),
        "enhance_prompt": r.get("enhance_prompt"),
        "original_prompt": r.get("original_prompt"),
        "preview": f"/outputs/{name}" if video else f"/previews/{name}",
        "created": r.get("created") or _mtime(name),
    }


def _mtime(name: str) -> float | None:
    try:
        return (OUTPUT_DIR / name).stat().st_mtime
    except OSError:
        return None


def _checked_output(name: str) -> Path:
    if not SAFE_NAME.match(name):
        raise HTTPException(404)
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(404)
    return path


# ============================================================
# Web API + custom frontend
# ============================================================
api = FastAPI(title="Obscura", docs_url=None, redoc_url=None)
api.add_middleware(GZipMiddleware, minimum_size=1024)


async def _read_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read()
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        raise UserError(f"{upload.filename} is not an image we can read.")


@api.exception_handler(UserError)
async def _user_error(_, exc: UserError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@api.get("/api/config")
def get_config():
    loras = []
    for name, spec in ADAPTER_SPECS.items():
        loras.append({
            "id": name,
            "name": LORA_SHORT_NAMES.get(name, name),
            "category": spec.get("category", "Style"),
            "blurb": LORA_BLURBS.get(name, ""),
            "steps": spec.get("default_steps", 40),
            "strength": spec.get("default_strength", 1.0),
            "preset": spec.get("preset_prompt", ""),
            "two_images": bool(spec.get("requires_two_images")),
            "image2_label": spec.get("image2_label", ""),
            "loaded": spec["adapter_name"] in LOADED_ADAPTERS,
        })
    return {
        "app": {"name": "Obscura", "model": "Qwen Image 2.1", "checkpoint": CHECKPOINT, "repo": MODEL_ID},
        "modes": [{"value": k, **v} for k, v in MODE_INFO.items()],
        "ratios": [{"ratio": r, "sizes": sizes} for r, sizes in RATIO_SIZES.items()],
        "resolutions": list(OUTPUT_RESOLUTION),
        "upscalers": [{"id": k, "label": v["label"], "available": (UPSCALER_DIR / v["file"]).is_file()} for k, v in UPSCALERS.items()],
        "upscale_max_input": UPSCALE_MAX_INPUT,
        "enhance": {"max_input": ENHANCE_MAX_INPUT, "single_max": ENHANCE_SINGLE_MAX, "default_prompt": ENHANCE_DEFAULT_PROMPT,
                    "tile": ENHANCE_TILE, "overlap": ENHANCE_OVERLAP},
        "status": _status_view(),
        "loras": loras,
        "none_lora": NONE_LORA,
        "custom_lora": "Custom HuggingFace LoRA...",
        "local_lora": LOCAL_LORA,
        "local_loras": list_local_loras(),
        "lora_dir": str(LORA_DIR),
        "max_seed": MAX_SEED,
    }


@api.get("/api/status")
def get_status():
    return _status_view()


@api.post("/api/model")
def post_model(body: dict):
    name = body.get("checkpoint")
    if name not in CHECKPOINT_INFO:
        raise UserError("Unknown model.")
    if SWITCH["state"] == "loading":
        raise UserError("A model switch is already in progress.")
    if name == CHECKPOINT:
        return _status_view()
    if _checkpoint_blob(name) is None:
        raise UserError(f"{CHECKPOINT_INFO[name]['label']} model is not downloaded on this machine.")
    SWITCH.update(state="loading", target=name, phase="Starting", error=None, started=time.time())
    threading.Thread(target=_switch_worker, args=(name,), daemon=True, name="model-switch").start()
    return _status_view()


@api.get("/api/local-loras")
def get_local_loras():
    return {"items": list_local_loras(), "dir": str(LORA_DIR)}


@api.post("/api/generate")
async def post_generate(
    params: str = Form(...),
    ref1: UploadFile | None = File(None),
    ref2: UploadFile | None = File(None),
    ref3: UploadFile | None = File(None),
    ref4: UploadFile | None = File(None),
):
    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        raise UserError("Malformed request.")
    refs = [img for img in [await _read_image(u) for u in (ref1, ref2, ref3, ref4)] if img is not None]
    req = validate_request(p, refs)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "phase": "Queued", "step": 0,
        "cancel": False, "created": time.time(), "request": req,
    }
    JOB_ORDER.put(job_id)
    return _job_view(JOBS[job_id])


def _rewrite_job(view: dict | None) -> dict:
    if view is None:
        raise HTTPException(404, "That rewrite is gone. Try again.")
    return view


@api.post("/api/rewrite")
async def post_rewrite(
    params: str = Form(...),
    ref1: UploadFile | None = File(None),
    ref2: UploadFile | None = File(None),
    ref3: UploadFile | None = File(None),
    ref4: UploadFile | None = File(None),
):
    """Expand a prompt with the official Qwen rewriter (PE-I2I when the mode uses references, PE-T2I otherwise)."""
    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        raise UserError("Malformed request.")
    mode = p.get("mode", "Text to Image")
    if mode not in MODES:
        raise UserError(f"Unknown mode: {mode}")
    refs = [img for img in [await _read_image(u) for u in (ref1, ref2, ref3, ref4)] if img is not None]
    refs = refs[:1] if mode == "Edit Image (1 Ref)" else refs if mode == "Transform & Swap (2 Refs)" else []
    try:
        return prompt_rewriter.start(p.get("prompt", ""), refs)
    except prompt_rewriter.RewriteError as e:
        raise UserError(str(e))


@api.get("/api/rewrite/{job_id}")
def get_rewrite(job_id: str):
    return _rewrite_job(prompt_rewriter.get(job_id))


@api.post("/api/rewrite/{job_id}/cancel")
def cancel_rewrite(job_id: str):
    return _rewrite_job(prompt_rewriter.cancel(job_id))


@api.get("/api/jobs")
def get_jobs(ids: str = ""):
    wanted = [i for i in ids.split(",") if i]
    return {"items": [_job_view(JOBS[i]) for i in wanted if i in JOBS]}


@api.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404)
    job["cancel"] = True
    if job["status"] == "queued":
        job.update(status="cancelled", finished=time.time())
    return _job_view(job)


@api.get("/api/history")
def get_history(offset: int = 0, limit: int = 60, q: str = ""):
    records = _read_records()
    files = sorted((p.name for p in OUTPUT_DIR.iterdir() if SAFE_NAME.match(p.name) and not p.name.startswith(".")), reverse=True)
    items = [_public_record(records.get(f, {"file": f})) for f in files]
    if q.strip():
        needle = q.strip().lower()
        items = [i for i in items if needle in i["prompt"].lower() or needle in (i["original_prompt"] or "").lower()
                 or needle in (i["lora"] or "").lower() or needle in (i["workflow_name"] or "").lower()
                 or (needle == "video" and i["media"] == "video")]
    limit = max(1, min(limit, 200))
    return {"items": items[offset:offset + limit], "total": len(items)}


def _delete_outputs(names: list[str]) -> list[str]:
    """Remove images with their thumbnails, previews and history entries. Returns what was deleted."""
    deleted = []
    for name in dict.fromkeys(names):
        try:
            path = _checked_output(name)
        except HTTPException:
            continue
        path.unlink()
        (THUMB_DIR / (path.stem + ".webp")).unlink(missing_ok=True)
        (THUMB_DIR / (path.stem + ".preview.webp")).unlink(missing_ok=True)
        deleted.append(name)
    if deleted:
        gone = set(deleted)
        with HISTORY_LOCK:
            if HISTORY_LOG.exists():
                kept = []
                for line in HISTORY_LOG.read_text(encoding="utf-8").splitlines():
                    try:
                        if json.loads(line).get("file") in gone:
                            continue
                    except json.JSONDecodeError:
                        pass
                    kept.append(line)
                HISTORY_LOG.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    return deleted


@api.delete("/api/history/{name}")
def delete_history(name: str):
    _checked_output(name)
    _delete_outputs([name])
    return {"ok": True}


@api.post("/api/history/delete")
def delete_history_many(body: dict):
    names = body.get("files") or []
    if not isinstance(names, list) or not names:
        raise UserError("Nothing selected.")
    return {"deleted": _delete_outputs([str(n) for n in names][:1000])}


PROMPTS_FILE = Path(os.environ.get("QWEN_PROMPTS_FILE", Path(__file__).resolve().parent / "prompts.json"))
PROMPTS_LOCK = threading.Lock()


def _read_prompts() -> list[dict]:
    try:
        return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _write_prompts(items: list[dict]):
    tmp = PROMPTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PROMPTS_FILE)


def _prompt_fields(body: dict) -> tuple[str, str]:
    title = str(body.get("title") or "").strip()[:120]
    text = str(body.get("prompt") or "").strip()[:MAX_PROMPT_CHARS]
    if not title or not text:
        raise UserError("A saved prompt needs a title and the prompt text.")
    return title, text


@api.get("/api/prompts")
def get_prompts():
    return {"items": _read_prompts()}


@api.post("/api/prompts")
def post_prompt(body: dict):
    title, text = _prompt_fields(body)
    item = {"id": uuid.uuid4().hex[:12], "title": title, "prompt": text, "created": time.time()}
    with PROMPTS_LOCK:
        _write_prompts([item, *_read_prompts()])
    return item


@api.put("/api/prompts/{prompt_id}")
def put_prompt(prompt_id: str, body: dict):
    title, text = _prompt_fields(body)
    with PROMPTS_LOCK:
        items = _read_prompts()
        item = next((x for x in items if x["id"] == prompt_id), None)
        if item is None:
            raise UserError("That saved prompt no longer exists.")
        item.update(title=title, prompt=text)
        _write_prompts(items)
    return item


@api.delete("/api/prompts/{prompt_id}")
def delete_prompt(prompt_id: str):
    with PROMPTS_LOCK:
        _write_prompts([x for x in _read_prompts() if x["id"] != prompt_id])
    return {"ok": True}


@api.get("/outputs/{name}")
def get_output(name: str, download: int = 0):
    path = _checked_output(name)
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    media_type = MEDIA_TYPES[path.suffix]
    if download:
        return FileResponse(path, media_type=media_type, filename=f"obscura-{name}", headers=headers)
    return FileResponse(path, media_type=media_type, headers=headers)


@api.post("/api/enhance")
def post_enhance(body: dict):
    path = _checked_output(body.get("file", ""))
    if _is_video(path.name):
        raise UserError("Enhance works on images, not videos.")
    try:
        strength = float(body.get("strength", 0.3))
        steps = int(body.get("steps", 40))
    except (TypeError, ValueError):
        raise UserError("Strength and steps must be numbers.")
    if not 0.1 <= strength <= 0.7:
        raise UserError("Strength must be between 0.10 and 0.70.")
    steps = max(4, min(40, steps))
    enhance_prompt = str(body.get("prompt") or "").strip() or ENHANCE_DEFAULT_PROMPT
    if len(enhance_prompt) > 4000:
        raise UserError("Enhance prompt is too long. Keep it under 4000 characters.")
    with Image.open(path) as im:
        w, h = im.size
    if w * h > ENHANCE_MAX_INPUT:
        raise UserError(f"This image is {w}x{h}, too large to enhance.")
    record = _read_records().get(path.name, {})
    plan = enhance_plan(w, h, record.get("ratio"))
    # Only the tail of the schedule runs: strength x steps, at least 4
    n_steps = max(4, round(steps * strength))
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "phase": "Queued", "step": 0,
        "cancel": False, "created": time.time(),
        "request": {
            "kind": "enhance", "file": path.name, "strength": round(strength, 2), "n_steps": n_steps,
            "enhance_prompt": enhance_prompt,
            "steps": n_steps * plan["tiles"], "aspect": plan["width"] / plan["height"],
            "prompt": record.get("prompt", ""), "ratio": record.get("ratio") or "1:1",
            "resolution": record.get("resolution") or "4MP", "mode": record.get("mode", "Text to Image"),
            "refs": [],
        },
    }
    JOB_ORDER.put(job_id)
    return _job_view(JOBS[job_id])


@api.post("/api/upscale")
def post_upscale(body: dict):
    key = body.get("upscaler")
    if key not in UPSCALERS:
        raise UserError("Unknown upscaler.")
    if not (UPSCALER_DIR / UPSCALERS[key]["file"]).is_file():
        raise UserError(f"{UPSCALERS[key]['label']} weights are missing from {UPSCALER_DIR}.")
    path = _checked_output(body.get("file", ""))
    if _is_video(path.name):
        raise UserError("Upscaling works on images, not videos.")
    with Image.open(path) as im:
        w, h = im.size
    if w * h > UPSCALE_MAX_INPUT:
        raise UserError(f"This image is already {w}x{h}. Upscale renders of 4 MP or less.")
    record = _read_records().get(path.name, {})
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "phase": "Queued", "step": 0,
        "cancel": False, "created": time.time(),
        "request": {
            "kind": "upscale", "file": path.name, "upscaler": key,
            "steps": len(_tile_grid(w, h)), "aspect": w / h,
            "prompt": record.get("prompt", ""), "ratio": record.get("ratio") or "1:1",
            "resolution": record.get("resolution") or "1MP", "mode": record.get("mode", "Text to Image"),
            "refs": [],
        },
    }
    JOB_ORDER.put(job_id)
    return _job_view(JOBS[job_id])


@api.get("/api/related/{name}")
def get_related(name: str):
    """The original behind an image plus every upscale made from that original, for comparisons."""
    _checked_output(name)
    records = _read_records()
    source = records.get(name, {}).get("upscaled_from") or name
    upscales = [r for r in records.values() if r.get("upscaled_from") == source and (OUTPUT_DIR / r["file"]).is_file()]
    src = records.get(source, {"file": source}) if (OUTPUT_DIR / source).is_file() else None
    return {
        "source": _public_record(src) if src else None,
        "upscales": [_public_record(r) for r in sorted(upscales, key=lambda r: r.get("created", 0))],
    }


@api.get("/previews/{name}")
def get_preview(name: str):
    """Large images (upscales) get a screen-sized WebP; everything else is served as is."""
    src = _checked_output(name)
    if _is_video(name):
        return get_output(name)
    with Image.open(src) as img:
        if img.width * img.height <= 9_000_000:
            return FileResponse(src, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})
    preview = THUMB_DIR / (src.stem + ".preview.webp")
    if not preview.exists():
        with Image.open(src) as img:
            img.thumbnail((3072, 3072), Image.LANCZOS)
            img.save(preview, "WEBP", quality=92, method=4)
    return FileResponse(preview, media_type="image/webp", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@api.get("/thumbs/{name}")
def get_thumb(name: str):
    src = _checked_output(name)
    thumb = THUMB_DIR / (src.stem + ".webp")
    if not thumb.exists():
        # Videos are represented by their first frame
        source = io.BytesIO(comfy_bridge.first_frame_png(src)) if _is_video(name) else src
        with Image.open(source) as img:
            img.thumbnail((640, 640), Image.LANCZOS)
            img.save(thumb, "WEBP", quality=86, method=5)
    return FileResponse(thumb, media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


# ============================================================
# Workflows (ComfyUI)
# ============================================================
def _bridge(fn, *a):
    try:
        return fn(*a)
    except comfy_bridge.BridgeError as e:
        raise UserError(str(e))


def _comfy_status() -> dict:
    return {**comfy_bridge.status(), "qwen_on_gpu": QWEN["on_gpu"], "qwen_moving": QWEN["moving"],
            "keep_loaded": bool(_read_settings().get("comfy_keep_loaded"))}


@api.get("/api/workflows")
def get_workflows():
    return {"items": comfy_bridge.list_workflows(), "comfy": _comfy_status()}


@api.post("/api/workflows")
def post_workflow(body: dict):
    return _bridge(comfy_bridge.import_workflow, body.get("name", ""), body.get("graph"), body.get("description", ""))


@api.put("/api/workflows/{slug}")
def put_workflow(slug: str, body: dict):
    return _bridge(comfy_bridge.update_workflow, slug, body)


@api.delete("/api/workflows/{slug}")
def delete_workflow(slug: str):
    _bridge(comfy_bridge.delete_workflow, slug)
    return {"ok": True}


@api.get("/api/workflows/{slug}/graph")
def get_workflow_graph(slug: str):
    """The stored API graph, so the Edit fields dialog can offer every node input."""
    graph, _ = _bridge(comfy_bridge.load, slug)
    return graph


async def _media_upload(field: dict, upload) -> tuple[str, bytes]:
    data = await upload.read()
    if not data:
        raise UserError(f"{field['label']}: the file is empty.")
    if field["type"] == "image":
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            img.load()
        except Exception:
            raise UserError(f"{upload.filename} is not an image we can read.")
        buf = io.BytesIO()
        img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").save(buf, "PNG")
        return "input.png", buf.getvalue()
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in comfy_bridge.VIDEO_EXTS:
        raise UserError(f"{upload.filename} is not a video file (use MP4, WebM, MOV, MKV or GIF).")
    return f"input{ext}", data


@api.post("/api/workflows/{slug}/run")
async def run_workflow(slug: str, request: Request):
    form = await request.form()
    try:
        params = json.loads(form.get("params") or "{}")
    except json.JSONDecodeError:
        raise UserError("Malformed request.")
    graph, meta = _bridge(comfy_bridge.load, slug)
    values = _bridge(comfy_bridge.resolve_values, meta, params.get("values") or {}, params.get("randomize") or {})
    library = params.get("library") or {}
    media, display, source = {}, [], None
    for f in meta.get("fields", []):
        key = f["key"]
        if f["type"] in ("image", "video"):
            upload = form.get(f"file_{key}")
            if upload is not None and getattr(upload, "filename", None):
                media[key] = await _media_upload(f, upload)
                display.append([f["label"], upload.filename])
            elif library.get(key):
                path = _checked_output(str(library[key]))
                if _is_video(path.name) != (f["type"] == "video"):
                    raise UserError(f"{f['label']} needs {'a video' if f['type'] == 'video' else 'an image'}.")
                media[key] = (path.name, path.read_bytes())
                display.append([f["label"], path.name])
                if f["type"] == "image" and source is None:
                    source = path.name
            elif not f.get("hidden") and not f.get("default"):
                raise UserError(f"Add {f['label'].lower()} first.")
        elif not f.get("hidden") and f["type"] not in ("text", "seed") and key in values:
            display.append([f["label"], values[key]])
    texts = [f for f in meta.get("fields", []) if f["type"] == "text" and not f.get("hidden")]
    prompt_field = next((f for f in texts if "negative" not in f["label"].lower()), None)
    prompt = str(values.get(prompt_field["key"], "")) if prompt_field else ""
    seed_field = next((f for f in meta.get("fields", []) if f["type"] == "seed"), None)
    steps_field = next((f for f in meta.get("fields", []) if f["key"] == "steps"), None)
    size = meta.get("size") or {}
    w, h = values.get(size.get("width")), values.get(size.get("height"))
    estimate = sum(comfy_bridge.sampler_steps(comfy_bridge.patch(graph, meta, values, {})).values())
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "phase": "Queued", "step": 0,
        "cancel": False, "created": time.time(),
        "request": {
            "kind": "comfy", "slug": slug, "name": meta.get("name", slug), "output": meta.get("output", "image"),
            "graph": graph, "meta": meta, "values": values, "media": media, "display": display, "source": source,
            "prompt": prompt, "seed": values.get(seed_field["key"]) if seed_field else None,
            "steps_value": values.get("steps") if steps_field else None,
            "steps": estimate or 1, "aspect": (w / h) if w and h else 1.0,
            "ratio": "", "resolution": "", "mode": "Workflow", "refs": [],
        },
    }
    JOB_ORDER.put(job_id)
    return _job_view(JOBS[job_id])


PORTAL_API = "http://localhost:11111"
COMFY_PORTAL_TARGET = "http://localhost:10200"


def _comfy_tunnel(start: bool) -> str:
    """The Cloudflare quick tunnel to the ComfyUI portal port, started on demand. Empty if unavailable."""
    try:
        for t in requests.get(f"{PORTAL_API}/get-all-quick-tunnels", timeout=3).json():
            if t.get("targetUrl") == COMFY_PORTAL_TARGET and t.get("tunnelUrl"):
                return t["tunnelUrl"]
        if start:
            r = requests.post(f"{PORTAL_API}/start-quick-tunnel/{requests.utils.quote(COMFY_PORTAL_TARGET, safe='')}", timeout=30)
            return r.json().get("tunnel_url") or ""
    except (requests.RequestException, ValueError):
        pass
    return ""


@api.get("/api/comfy/editor")
def get_comfy_editor(tunnel: int = 0):
    """Where the ComfyUI editor is reachable from the browser. The page passes tunnel=1 when it was itself opened
    through a Cloudflare tunnel, since the direct port isn't on that hostname. The token is the one this page was
    opened with (Caddy already checked it), so the new tab doesn't ask again."""
    token = os.environ.get("OPEN_BUTTON_TOKEN") or os.environ.get("WEB_PASSWORD") or ""
    return {
        "url": os.environ.get("COMFY_EDITOR_URL", ""),
        "port": os.environ.get("VAST_TCP_PORT_10200", ""),
        "tunnel": _comfy_tunnel(start=True) if tunnel else "",
        "token": token,
    }


@api.post("/api/comfy/start")
def post_comfy_start():
    """Start ComfyUI for the editor button; returns once it answers."""
    _bridge(comfy_bridge.ensure_running)
    return _comfy_status()


@api.get("/api/comfy/status")
def get_comfy_status():
    return _comfy_status()


def _move_qwen(to_gpu: bool):
    try:
        with GPU_LOCK:
            unpark_qwen() if to_gpu else park_qwen()
    except Exception:
        traceback.print_exc()
    finally:
        QWEN["moving"] = False


@api.post("/api/comfy/park")
def post_comfy_park(body: dict):
    """Free the GPU for the ComfyUI editor (park), or bring Qwen back early (unpark)."""
    to_gpu = body.get("action") == "unpark"
    if not QWEN["moving"] and QWEN["on_gpu"] != to_gpu:
        QWEN["moving"] = True
        threading.Thread(target=_move_qwen, args=(to_gpu,), daemon=True, name="qwen-move").start()
    return _comfy_status()


@api.put("/api/comfy/settings")
def put_comfy_settings(body: dict):
    if "keep_loaded" in body:
        _write_settings(comfy_keep_loaded=bool(body["keep_loaded"]))
    return _comfy_status()


@api.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(
        api,
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "17860")),
        log_level="warning",
    )
