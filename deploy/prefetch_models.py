"""Download the weights app.py loads at startup into $HF_HOME, so the first launch doesn't stall."""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

APP_DIR = Path(__file__).resolve().parent.parent
GGUF_REPO = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
COMPANION_ID = "Qwen/Qwen-Image-2.1"
COMPANION_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

settings = APP_DIR / "settings.json"
checkpoint = json.loads(settings.read_text()).get("checkpoint") if settings.exists() else None
checkpoint = checkpoint or "qwen-image-2.1-UC-BF16.gguf"

print(f"Fetching {GGUF_REPO}/{checkpoint}", flush=True)
hf_hub_download(GGUF_REPO, checkpoint)

# The transformer comes from the GGUF, so skip the companion repo's transformer weights
print(f"Fetching {COMPANION_ID}@{COMPANION_REVISION[:8]}", flush=True)
snapshot_download(COMPANION_ID, revision=COMPANION_REVISION,
                  ignore_patterns=["transformer/*.safetensors"])
print("Model prefetch complete", flush=True)
