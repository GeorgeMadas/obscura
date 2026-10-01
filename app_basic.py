import os
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

# Configure memory allocator before heavy operations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import torch
from accelerate import init_empty_weights
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.models.model_loading_utils import load_gguf_checkpoint
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, PngImagePlugin

# Model Configuration
MODEL_ID = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
CHECKPOINT = os.environ.get("QWEN_GGUF_CHECKPOINT", "qwen-image-2.1-UC-BF16.gguf")
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
    # Legacy / Base GGUFs
    "qwen-image-2.1-Q4_K_M.gguf": "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8",
    "qwen-image-2.1-Q4_0.gguf": "8efd261419f4d60bf0eda8ae86a656e229ed440a248a22e20bf1f1f9606f6124",
    "qwen-image-2.1-Q5_K_M.gguf": "88ce8e90e5b959cce5e248f697d7f6c9c7ca5696c1eac64a10dadb041dd7fd07",
    "qwen-image-2.1-Q6_K.gguf": "a3a0d39bb03cda26302fc048b49d019baaea1c381cbda3994f6f2a7826344fb9",
    "qwen-image-2.1-Q8_0.gguf": "9a7ec02f4c9d5cf5b78e8efa6ccd81ec0c35a898defbf8e77ac7c02a32fe0d7e",
}

MODES = ["Create an image", "Edit an image", "Transparent PNG"]
SIZES = {
    "Square · 1:1 (1024x1024)": (1024, 1024),
    "Landscape · 16:9 (1344x768)": (1344, 768),
    "Portrait · 9:16 (768x1344)": (768, 1344),
    "Landscape · 4:3 (1152x864)": (1152, 864),
    "Portrait · 3:4 (864x1152)": (864, 1152),
}
MAX_SEED = 2**31 - 1

print(f"Loading checkpoint {CHECKPOINT} from {MODEL_ID}...", flush=True)
checkpoint_path = hf_hub_download(MODEL_ID, CHECKPOINT)

if CHECKPOINT in SHA256_CHECKSUMS:
    with open(checkpoint_path, "rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    if checksum != SHA256_CHECKSUMS[CHECKPOINT]:
        raise RuntimeError(f"Checksum verification failed for {CHECKPOINT}! Got {checksum}")
    print(f"Checksum verified: {checksum}", flush=True)

# Expand the GGUF weights to bfloat16 once at startup for ZeroGPU eager packing
weights = load_gguf_checkpoint(checkpoint_path)
for name in list(weights.keys()):
    weights[name] = dequantize_gguf_tensor(weights[name]).to(torch.bfloat16)

config = QwenImage21Transformer2DModel.load_config(
    COMPANION_ID, subfolder="transformer", revision=COMPANION_REVISION
)
with init_empty_weights():
    transformer = QwenImage21Transformer2DModel.from_config(config)

transformer.load_state_dict(weights, strict=True, assign=True)
transformer.eval().requires_grad_(False)
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


def generate(
    prompt: str,
    mode: str = "Create an image",
    reference: Image.Image | None = None,
    aspect_ratio: str = "Square · 1:1 (1024x1024)",
    steps: int = 40,
    seed: int = 42,
    randomize_seed: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, int, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt describing the image you want to generate or modify.")
    if len(prompt) > 4000:
        raise gr.Error("Prompt is too long. Please keep under 4000 characters.")
    if mode not in MODES:
        raise gr.Error(f"Invalid mode: {mode}")
    if aspect_ratio not in SIZES:
        raise gr.Error(f"Invalid aspect ratio: {aspect_ratio}")
    if steps is None or not (4 <= int(steps) <= 40):
        raise gr.Error("Inference steps must be between 4 and 40.")
    if mode == "Edit an image" and reference is None:
        raise gr.Error("Please upload a reference image for 'Edit an image' mode.")

    actual_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    width, height = SIZES[aspect_ratio]

    effective_prompt = prompt
    if mode == "Transparent PNG":
        effective_prompt = (
            "This is an RGBA image with transparency. " + prompt
            + " The image has alpha channel and the background is transparent."
        )

    call_kwargs = {}
    if mode == "Edit an image":
        ref_img = ImageOps.exif_transpose(reference).convert("RGBA")
        ref_img.thumbnail((2048, 2048))
        call_kwargs["image"] = ref_img

    start_time = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=effective_prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            generator=torch.Generator("cuda").manual_seed(actual_seed),
            **call_kwargs,
        ).images[0]
    elapsed = time.perf_counter() - start_time

    # Embed generation parameters inside PNG metadata
    metadata = {
        "model": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "prompt": prompt,
        "effective_prompt": effective_prompt,
        "mode": mode,
        "seed": actual_seed,
        "steps": int(steps),
        "width": result.width,
        "height": result.height,
        "elapsed_seconds": round(elapsed, 2),
    }
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", json.dumps(metadata, ensure_ascii=False))

    upload_folder = gr.utils.get_upload_folder()
    out_dir = Path(tempfile.mkdtemp(prefix="qwen21-gguf-", dir=upload_folder))
    saved_path = out_dir / f"qwen-image-2.1-{actual_seed}.png"
    result.save(saved_path, pnginfo=png_info)

    details = (
        f"Resolution: {result.width}x{result.height} | Steps: {int(steps)} | "
        f"Time: {elapsed:.2f}s | Seed: {actual_seed} | Format: {result.mode} PNG"
    )
    if mode == "Transparent PNG" and (
        "A" not in result.getbands() or result.getchannel("A").getextrema()[0] == 255
    ):
        details += " (Note: background may not be fully transparent; try re-generating with higher steps or adjusted prompt)"

    print(f"Generated successfully: {details}", flush=True)
    return str(saved_path), str(saved_path), actual_seed, details


def update_ui_mode(selected_mode: str):
    """Toggle visibility of the reference image input based on mode."""
    return gr.update(visible=(selected_mode == "Edit an image"))


CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
#app-title { text-align: center; margin-bottom: 0.5rem; }
#app-header-desc { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1.5rem; }
#generate-btn { font-weight: 600; font-size: 1.05rem; }
.promo-banner {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.12), rgba(219, 39, 119, 0.12));
    border: 1px solid rgba(147, 51, 234, 0.35);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 1rem 0 1.5rem 0;
    text-align: center;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.08);
}
.promo-btn {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white !important;
    font-weight: 700;
    padding: 5px 15px;
    border-radius: 8px;
    margin-left: 8px;
    text-decoration: none !important;
    transition: opacity 0.2s ease;
}
.promo-btn:hover {
    opacity: 0.9;
}
"""

with gr.Blocks(title="Qwen Image 2.1 Uncensored GGUF", delete_cache=(3600, 86400)) as demo:
    gr.HTML(
        """
        <div class="promo-banner">
            🚀 <b>NEW RELEASE:</b> Want <b>4-Step Turbo Speed</b>, <b>Anime</b>, <b>Photorealism</b>, or <b>Face & Pose Swap</b>?
            <a href="https://huggingface.co/spaces/arudradey/qwen-image-2.1-uncensored-aio-loras" target="_blank" class="promo-btn">
                Try Qwen 2.1 All-In-One LoRA Studio →
            </a>
        </div>
        """
    )
    gr.Markdown(
        "# 🎨 Qwen Image 2.1 Uncensored GGUF Studio\n"
        "### High-performance local text-to-image and image editing powered by Hugging Face ZeroGPU\n"
        f"Model: **[{MODEL_ID}](https://huggingface.co/{MODEL_ID})** ({CHECKPOINT}) · "
        "License: [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)",
        elem_id="app-title",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            mode_radio = gr.Radio(
                MODES,
                value=MODES[0],
                label="Generation Mode",
                info="Choose between generating from scratch, editing an existing image, or creating isolated transparent graphics.",
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to create (e.g. 'A futuristic cyberpunk street in Tokyo with neon reflections on wet asphalt')...",
                lines=4,
                max_lines=8,
            )
            ref_image_input = gr.Image(
                label="Reference Image (for Edit Mode)",
                type="pil",
                image_mode="RGBA",
                sources=["upload", "clipboard"],
                visible=False,
            )
            aspect_ratio_dropdown = gr.Dropdown(
                list(SIZES.keys()),
                value="Square · 1:1 (1024x1024)",
                label="Aspect Ratio & Resolution",
            )
            with gr.Accordion("Advanced Parameters", open=False):
                steps_slider = gr.Slider(
                    minimum=4,
                    maximum=40,
                    value=40,
                    step=1,
                    label="Inference Steps",
                    info="40 is recommended for highest quality. Use 10-20 for fast preview.",
                )
                randomize_seed_cb = gr.Checkbox(
                    value=True,
                    label="Randomize Seed",
                    info="Uncheck to lock in a specific seed for reproducible results.",
                )
                seed_number = gr.Number(
                    value=42,
                    label="Seed",
                    precision=0,
                    minimum=0,
                    maximum=MAX_SEED,
                )

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary",
                elem_id="generate-btn",
            )
            gr.Markdown(
                "> **ZeroGPU Hardware**: Requests dynamically claim compute from an NVIDIA RTX PRO 6000 GPU pool. "
                "No private API keys or personal GPU required."
            )

        with gr.Column(scale=6):
            output_image = gr.Image(
                label="Generated Output",
                type="filepath",
                image_mode="RGBA",
                format="png",
                interactive=False,
            )
            download_file = gr.File(
                label="Download Full Resolution PNG",
                interactive=False,
            )
            details_box = gr.Textbox(
                label="Generation Details",
                interactive=False,
            )

    # Event handlers
    mode_radio.change(
        fn=update_ui_mode,
        inputs=mode_radio,
        outputs=ref_image_input,
    )

    generate_btn.click(
        fn=generate,
        inputs=[
            prompt_input,
            mode_radio,
            ref_image_input,
            aspect_ratio_dropdown,
            steps_slider,
            seed_number,
            randomize_seed_cb,
        ],
        outputs=[output_image, download_file, seed_number, details_box],
        api_name="generate",
        concurrency_limit=1,
        concurrency_id="qwen-pipeline",
    )

    gr.Examples(
        examples=[
            ["A futuristic cybernetic tiger with glowing azure circuitry in a dark misty jungle, 8k resolution, Unreal Engine 5 render", MODES[0]],
            ["A cozy cabin in a snowy pine forest at twilight, warm yellow light from the windows, starry sky above", MODES[0]],
            ["A cute baby panda wearing astronaut helmet, sticker art style, vector illustration with clean borders", MODES[2]],
        ],
        inputs=[prompt_input, mode_radio],
        label="Prompt Examples",
    )

if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        css=CUSTOM_CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "17860")),
    )
