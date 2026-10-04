"""Write the built-in ComfyUI presets into workflows/ (API graphs + form metadata).

Graphs follow ComfyUI's official Wan 2.2 and Wan Animate 2 templates, flattened from their subgraphs into
API format. Rerun after editing: python deploy/build_presets.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "workflows"
OUT.mkdir(exist_ok=True)

# Wan's own default negative prompt (the models were trained with it)
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
                "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
MAX_SEED = 2**31 - 1


def node(class_type, title=None, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title or class_type}}


def write(slug, graph, meta):
    (OUT / f"{slug}.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", slug)


def text_fields(prompt_default, neg_node="3", pos_node="2"):
    return [
        {"key": "prompt", "label": "Prompt", "type": "text", "default": prompt_default, "targets": [[pos_node, "text"]],
         "help": "Describe the motion and the camera. The image already sets the look."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "default": WAN_NEGATIVE,
         "targets": [[neg_node, "text"]], "hidden": True},
    ]


def size_fields(w, h, targets_w, targets_h, step):
    return [
        {"key": "width", "label": "Width", "type": "int", "default": w, "min": 256, "max": 1920, "step": step, "targets": targets_w},
        {"key": "height", "label": "Height", "type": "int", "default": h, "min": 256, "max": 1920, "step": step, "targets": targets_h},
    ]


# ------------------------------------------------------------------ Wan 2.2 Image -> Video 14B (two experts)
def wan22_i2v(fast: bool):
    g = {
        "1": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "2": node("CLIPTextEncode", "Positive Prompt", clip=["1", 0], text=""),
        "3": node("CLIPTextEncode", "Negative Prompt", clip=["1", 0], text=WAN_NEGATIVE),
        "4": node("VAELoader", vae_name="wan_2.1_vae.safetensors"),
        "5": node("LoadImage", "Start image", image=""),
        "6": node("UNETLoader", "High noise model", unet_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", weight_dtype="default"),
        "7": node("UNETLoader", "Low noise model", unet_name="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", weight_dtype="default"),
    }
    high, low = ["6", 0], ["7", 0]
    if fast:
        g["16"] = node("LoraLoaderModelOnly", "Lightx2v 4-step (high)", model=high,
                       lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", strength_model=1.0)
        g["17"] = node("LoraLoaderModelOnly", "Lightx2v 4-step (low)", model=low,
                       lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", strength_model=1.0)
        high, low = ["16", 0], ["17", 0]
    steps, cfg = (4, 1.0) if fast else (20, 3.5)
    g.update({
        "8": node("ModelSamplingSD3", model=high, shift=5.0),
        "9": node("ModelSamplingSD3", model=low, shift=5.0),
        "10": node("WanImageToVideo", positive=["2", 0], negative=["3", 0], vae=["4", 0], start_image=["5", 0],
                   width=832, height=480, length=81, batch_size=1),
        "11": node("KSamplerAdvanced", "High noise pass", model=["8", 0], add_noise="enable", noise_seed=0, steps=steps,
                   cfg=cfg, sampler_name="euler", scheduler="simple", positive=["10", 0], negative=["10", 1],
                   latent_image=["10", 2], start_at_step=0, end_at_step=steps // 2, return_with_leftover_noise="enable"),
        "12": node("KSamplerAdvanced", "Low noise pass", model=["9", 0], add_noise="disable", noise_seed=0, steps=steps,
                   cfg=cfg, sampler_name="euler", scheduler="simple", positive=["10", 0], negative=["10", 1],
                   latent_image=["11", 0], start_at_step=steps // 2, end_at_step=10000, return_with_leftover_noise="disable"),
        "13": node("VAEDecode", samples=["12", 0], vae=["4", 0]),
        "14": node("CreateVideo", images=["13", 0], fps=16.0),
        "15": node("SaveVideo", video=["14", 0], filename_prefix="obscura/wan22-i2v", format="auto", **{"format.codec": "auto"}),
    })
    fields = text_fields("The camera slowly pushes in while the subject turns their head and smiles, hair moving in a light breeze.")
    fields += [
        {"key": "image", "label": "Start image", "type": "image", "default": "", "targets": [["5", "image"]]},
        *size_fields(832, 480, [["10", "width"]], [["10", "height"]], 16),
        {"key": "length", "label": "Frames", "type": "int", "default": 81, "min": 5, "max": 241, "step": 4,
         "targets": [["10", "length"]], "help": "81 frames is about 5 seconds at 16 fps. Use 4n+1."},
        {"key": "fps", "label": "FPS", "type": "float", "default": 16, "min": 8, "max": 60, "step": 1, "targets": [["14", "fps"]]},
        {"key": "steps", "label": "Steps", "type": "int", "default": steps, "min": 2 if fast else 6, "max": 12 if fast else 60, "step": 2,
         "targets": [["11", "steps"], ["12", "steps"], ["11", "end_at_step", "half"], ["12", "start_at_step", "half"]],
         "help": "Split evenly between the high-noise and low-noise models."},
        {"key": "cfg", "label": "CFG", "type": "float", "default": cfg, "min": 1, "max": 10, "step": 0.5,
         "targets": [["11", "cfg"], ["12", "cfg"]], "hidden": fast},
        {"key": "seed", "label": "Seed", "type": "seed", "default": 0, "min": 0, "max": MAX_SEED, "targets": [["11", "noise_seed"]]},
    ]
    meta = {
        "name": "Image → Video" + (" (fast)" if fast else ""),
        "description": ("Wan 2.2 14B with the Lightx2v 4-step LoRAs. Quick, a little less detail." if fast
                        else "Wan 2.2 14B, full 20-step quality. Animates a still image from a motion prompt."),
        "output": "video", "preset": True, "order": 2 if fast else 1,
        "size": {"width": "width", "height": "height", "from": "image", "multiple": 16},
        "models": ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"]
                  + (["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"] if fast else []),
        "fields": fields,
    }
    write("wan22-i2v-fast" if fast else "wan22-i2v", g, meta)


# ------------------------------------------------------------------ Wan 2.2 TI2V 5B (drafts)
def wan22_5b():
    g = {
        "1": node("UNETLoader", unet_name="wan2.2_ti2v_5B_fp16.safetensors", weight_dtype="default"),
        "2": node("CLIPTextEncode", "Positive Prompt", clip=["4", 0], text=""),
        "3": node("CLIPTextEncode", "Negative Prompt", clip=["4", 0], text=WAN_NEGATIVE),
        "4": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "5": node("VAELoader", vae_name="wan2.2_vae.safetensors"),
        "6": node("LoadImage", "Start image", image=""),
        "7": node("ModelSamplingSD3", model=["1", 0], shift=8.0),
        "8": node("Wan22ImageToVideoLatent", vae=["5", 0], start_image=["6", 0], width=1280, height=704, length=121, batch_size=1),
        "9": node("KSampler", model=["7", 0], seed=0, steps=20, cfg=5.0, sampler_name="uni_pc", scheduler="simple",
                  positive=["2", 0], negative=["3", 0], latent_image=["8", 0], denoise=1.0),
        "10": node("VAEDecode", samples=["9", 0], vae=["5", 0]),
        "11": node("CreateVideo", images=["10", 0], fps=24.0),
        "12": node("SaveVideo", video=["11", 0], filename_prefix="obscura/wan22-5b", format="auto", **{"format.codec": "auto"}),
    }
    fields = text_fields("The camera slowly pushes in while the subject turns their head and smiles, hair moving in a light breeze.")
    fields += [
        {"key": "image", "label": "Start image", "type": "image", "default": "", "targets": [["6", "image"]]},
        *size_fields(1280, 704, [["8", "width"]], [["8", "height"]], 32),
        {"key": "length", "label": "Frames", "type": "int", "default": 121, "min": 5, "max": 241, "step": 4,
         "targets": [["8", "length"]], "help": "121 frames is about 5 seconds at 24 fps. Use 4n+1."},
        {"key": "fps", "label": "FPS", "type": "float", "default": 24, "min": 8, "max": 60, "step": 1, "targets": [["11", "fps"]]},
        {"key": "steps", "label": "Steps", "type": "int", "default": 20, "min": 4, "max": 50, "step": 1, "targets": [["9", "steps"]]},
        {"key": "cfg", "label": "CFG", "type": "float", "default": 5.0, "min": 1, "max": 10, "step": 0.5, "targets": [["9", "cfg"]]},
        {"key": "seed", "label": "Seed", "type": "seed", "default": 0, "min": 0, "max": MAX_SEED, "targets": [["9", "seed"]]},
    ]
    write("wan22-5b-draft", g, {
        "name": "Image → Video draft (5B)",
        "description": "Wan 2.2 TI2V 5B at 720p, 24 fps. Much faster than 14B; good for trying ideas.",
        "output": "video", "preset": True, "order": 4,
        "size": {"width": "width", "height": "height", "from": "image", "multiple": 32},
        "models": ["wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_vae.safetensors"],
        "fields": fields,
    })


# ------------------------------------------------------------------ Wan Animate 2 motion transfer (one segment)
def wan_animate2():
    g = {
        "1": node("UNETLoader", unet_name="wan_animate_2_int8_convrot.safetensors", weight_dtype="default"),
        "2": node("LoraLoaderModelOnly", "Lightx2v distill", model=["1", 0],
                  lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength_model=1.0),
        "3": node("WanAnimate2Cache", model=["2", 0], device="gpu", dtype="int8"),
        "4": node("ModelSamplingSD3", model=["3", 0], shift=5.0),
        "5": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "6": node("CLIPTextEncode", "Positive Prompt", clip=["5", 0], text=""),
        "7": node("CLIPTextEncode", "Negative Prompt", clip=["5", 0], text=WAN_NEGATIVE),
        "8": node("CLIPTextEncode", "Motion Prompt", clip=["5", 0], text=""),
        "9": node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors"),
        "10": node("VAELoader", vae_name="Wan2_1_VAE_bf16.safetensors"),
        "11": node("LoadImage", "Character image", image=""),
        "12": node("LoadVideo", "Driving video", file=""),
        "13": node("GetVideoComponents", video=["12", 0]),
        "14": node("ImageFromBatch", "Driving frames", image=["13", 0], batch_index=0, length=81),
        "15": node("ResizeImageMaskNode", "Resize driving video", input=["14", 0], scale_method="area",
                   resize_type="scale dimensions", **{"resize_type.width": 480, "resize_type.height": 832, "resize_type.crop": "center"}),
        "16": node("ResizeImageMaskNode", "Resize character", input=["11", 0], scale_method="area",
                   resize_type="scale dimensions", **{"resize_type.width": 480, "resize_type.height": 832, "resize_type.crop": "center"}),
        "17": node("CLIPVisionEncode", clip_vision=["9", 0], image=["16", 0], crop="none"),
        "18": node("ImageFromBatch", "First driving frame", image=["15", 0], batch_index=0, length=1),
        "19": node("CLIPVisionEncode", clip_vision=["9", 0], image=["18", 0], crop="none"),
        "20": node("WanAnimate2ToVideo", positive=["6", 0], negative=["7", 0], vae=["10", 0], reference_image=["16", 0],
                   pose_video=["15", 0], clip_vision_output=["17", 0], positive_pose=["8", 0], clip_vision_output_pose=["19", 0],
                   width=480, height=832, length=81, batch_size=1, video_frame_offset=0, pose_strength=1.0,
                   pose_start_percent=0.0, pose_end_percent=1.0, reference_image_strength=1.0),
        "21": node("KSamplerSelect", sampler_name="lcm"),
        "22": node("BasicScheduler", model=["4", 0], scheduler="simple", steps=6, denoise=1.0),
        "23": node("SamplerCustom", model=["4", 0], add_noise=True, noise_seed=0, cfg=1.0, positive=["20", 0],
                   negative=["20", 1], sampler=["21", 0], sigmas=["22", 0], latent_image=["20", 2]),
        "24": node("TrimVideoLatent", samples=["23", 0], trim_amount=["20", 3]),
        "25": node("VAEDecode", samples=["24", 0], vae=["10", 0]),
        "26": node("CreateVideo", images=["25", 0], fps=["13", 2]),
        "27": node("SaveVideo", video=["26", 0], filename_prefix="obscura/wan-animate2", format="auto", **{"format.codec": "auto"}),
    }
    fields = [
        {"key": "prompt", "label": "Prompt", "type": "text",
         "default": "Character description: a young woman with long dark hair in a red jacket.\nBackground description: a sunlit city street.",
         "targets": [["6", "text"]], "help": "Describe how the character and background look. No motion words."},
        {"key": "motion", "label": "Motion prompt", "type": "text", "default": "A person dancing energetically, background stationary.",
         "targets": [["8", "text"]], "help": "Describe the motion in the driving video."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "default": WAN_NEGATIVE, "targets": [["7", "text"]], "hidden": True},
        {"key": "image", "label": "Character image", "type": "image", "default": "", "targets": [["11", "image"]]},
        {"key": "video", "label": "Driving video", "type": "video", "default": "", "targets": [["12", "file"]],
         "help": "Its motion is copied onto the character. Keep the framing similar (full body to full body)."},
        *size_fields(480, 832, [["15", "resize_type.width"], ["16", "resize_type.width"], ["20", "width"]],
                     [["15", "resize_type.height"], ["16", "resize_type.height"], ["20", "height"]], 16),
        {"key": "length", "label": "Frames", "type": "int", "default": 81, "min": 5, "max": 241, "step": 4,
         "targets": [["14", "length"], ["20", "length"]],
         "help": "Frames taken from the start of the driving video, played at its frame rate. Use 4n+1."},
        {"key": "steps", "label": "Steps", "type": "int", "default": 6, "min": 4, "max": 20, "step": 1, "targets": [["22", "steps"]]},
        {"key": "pose_strength", "label": "Motion strength", "type": "float", "default": 1.0, "min": 0, "max": 2, "step": 0.05,
         "targets": [["20", "pose_strength"]]},
        {"key": "reference_strength", "label": "Identity strength", "type": "float", "default": 1.0, "min": 0, "max": 2, "step": 0.05,
         "targets": [["20", "reference_image_strength"]]},
        {"key": "seed", "label": "Seed", "type": "seed", "default": 0, "min": 0, "max": MAX_SEED, "targets": [["23", "noise_seed"]]},
    ]
    write("wan-animate2-motion", g, {
        "name": "Motion transfer",
        "description": "Wan Animate 2: your character performs the motion from a driving video. Background follows the prompt.",
        "output": "video", "preset": True, "order": 3,
        "size": {"width": "width", "height": "height", "from": "image", "multiple": 16},
        "models": ["wan_animate_2_int8_convrot.safetensors", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                   "clip_vision_h.safetensors", "Wan2_1_VAE_bf16.safetensors"],
        "fields": fields,
    })


wan22_i2v(fast=False)
wan22_i2v(fast=True)
wan22_5b()
wan_animate2()
