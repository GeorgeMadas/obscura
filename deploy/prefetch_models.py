"""Download the weights app.py loads at startup into $HF_HOME, so the first launch doesn't stall."""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

APP_DIR = Path(__file__).resolve().parent.parent
GGUF_REPO = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
COMPANION_ID = "Qwen/Qwen-Image-2.1"
COMPANION_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

# "Full" and "Small" in the UI's model picker; settings.json may name another one
checkpoints = ["qwen-image-2.1-UC-BF16.gguf", "qwen-image-2.1-UC-Q4_K_M.gguf"]
settings = APP_DIR / "settings.json"
selected = json.loads(settings.read_text()).get("checkpoint") if settings.exists() else None
if selected and selected not in checkpoints:
    checkpoints.append(selected)

for checkpoint in checkpoints:
    print(f"Fetching {GGUF_REPO}/{checkpoint}", flush=True)
    hf_hub_download(GGUF_REPO, checkpoint)

# The transformer comes from the GGUF, so skip the companion repo's transformer weights
print(f"Fetching {COMPANION_ID}@{COMPANION_REVISION[:8]}", flush=True)
snapshot_download(COMPANION_ID, revision=COMPANION_REVISION,
                  ignore_patterns=["transformer/*.safetensors"])
# Official prompt rewriters behind the Create popup (prompt_rewriter.py), ~19 GB each
for repo, rev in [("Qwen/Qwen-Image-2.1-PE-T2I", "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"),
                  ("Qwen/Qwen-Image-2.1-PE-I2I", "72927bc08afc99b7888ceb7d7d51a12db3700bbd")]:
    print(f"Fetching {repo}@{rev[:8]}", flush=True)
    snapshot_download(repo, revision=rev)
print("Model prefetch complete", flush=True)
