"""Download the ComfyUI video models (Wan 2.2 I2V 14B, Wan 2.2 TI2V 5B, Wan Animate 2) into ComfyUI/models.

Files are fetched through the Hub and moved into place under the names the workflows expect (~68 GB).
Already-present files with the right size are skipped, so this is safe to rerun.
"""
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODELS = Path(os.environ.get("COMFY_MODELS", "/workspace/ComfyUI/models"))
STAGING = MODELS / ".download"

WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
ANIMATE2 = "Comfy-Org/Wan-Animate-2"

# (repo, path in repo, ComfyUI models subfolder)
FILES = [
    # Shared
    (WAN21, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    (WAN22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    # Wan 2.2 Image -> Video 14B (two experts) + 4-step Lightx2v LoRAs
    (WAN22, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    (WAN22, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    (WAN22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "loras"),
    (WAN22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "loras"),
    # Wan 2.2 TI2V 5B drafts (uses the 2.2 VAE)
    (WAN22, "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
    (WAN22, "split_files/vae/wan2.2_vae.safetensors", "vae"),
    # Wan Animate 2 motion transfer
    (ANIMATE2, "diffusion_models/wan_animate_2_int8_convrot.safetensors", "diffusion_models"),
    (ANIMATE2, "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
    (ANIMATE2, "clip_vision/clip_vision_h.safetensors", "clip_vision"),
    (ANIMATE2, "vae/Wan2_1_VAE_bf16.safetensors", "vae"),
]

api = HfApi()
sizes = {}
for repo in {r for r, _, _ in FILES}:
    sizes[repo] = {s.rfilename: s.size for s in api.model_info(repo, files_metadata=True).siblings}

for repo, path, folder in FILES:
    dest = MODELS / folder / Path(path).name
    expected = sizes[repo].get(path)
    if dest.is_file() and dest.stat().st_size == expected:
        print(f"OK      {folder}/{dest.name}", flush=True)
        continue
    print(f"Fetching {repo}/{path} ({expected / 1e9:.1f} GB)", flush=True)
    local = Path(hf_hub_download(repo, path, local_dir=STAGING))
    if local.stat().st_size != expected:
        raise SystemExit(f"Size mismatch for {path}: {local.stat().st_size} != {expected}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(local, dest)
shutil.rmtree(STAGING, ignore_errors=True)
print("ComfyUI model prefetch complete", flush=True)
