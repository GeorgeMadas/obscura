#!/bin/bash
# Install ComfyUI next to Obscura (SETUP.md step 9). Safe to rerun: every step skips work that is already done.
#   - /workspace/ComfyUI at a pinned commit, plus pinned custom node packs (list below)
#   - its own venv /venv/comfy with the same torch build as /venv/main (cu128, needed for Blackwell)
#   - supervisor service "comfyui" on 127.0.0.1:18188 and the portal entry "ComfyUI" (10200 -> 18188)
#   - the Wan 2.2 / Wan Animate models (~97 GB, deploy/prefetch_comfy.py)
# Never runs a workflow.
set -eo pipefail
log(){ echo "[comfyui $(date +%T)] $*"; }
APP=/workspace/qwen-image
COMFY=/workspace/ComfyUI
VENV=/venv/comfy
EXT_PORT=${COMFY_EXTERNAL_PORT:-10200}

COMFY_COMMIT=f1072eb0350638a3390ddb6afbcaa8c6b237c6fd
NODES=(
  "https://github.com/Comfy-Org/ComfyUI-Manager 855a0f50ecc842adacab08aa24d8655a742fae40"
  "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite 4d907bee61e92c2e65af3bd6383a4e4d356126d1"
  "https://github.com/kijai/ComfyUI-KJNodes d3cfe21625e5170126ce06fbfcfe1d88108688c3"
  # For the "ultimate motion" workflow (WanVideoWrapper + Wan 2.2 Animate with pose detection)
  "https://github.com/kijai/ComfyUI-WanVideoWrapper 088128b224242e110d3906c6750e9a3a348a659b"
  "https://github.com/kijai/ComfyUI-WanAnimatePreprocess 0e0b6a2a555625acf4d4aefb780e27d06937132f"
  "https://github.com/teskor-hub/comfyui-teskors-utils b5be3ffb73d80019a7605c626407a0e686cd1ec0"
  "https://github.com/rgthree/rgthree-comfy 449c58fcdd612f7733e54c51f6758ead63fa180b"
  # Installed by the user through ComfyUI-Manager
  "https://github.com/wuwukaka/ComfyUI-WanAnimatePlus 95d909a"
  "https://github.com/LeonQ8/ComfyUI-Dynamic-Lora-Scheduler b5a796b"
)

clone_at() {  # url dir commit
  # A pack ComfyUI-Manager installed as a plain download (no .git) is left as it is
  if [ -d "$2" ] && [ ! -d "$2/.git" ]; then return; fi
  if [ ! -d "$2/.git" ]; then git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q origin && git -C "$2" checkout -q "$3"
}

log "ComfyUI @ ${COMFY_COMMIT:0:8}"
clone_at https://github.com/comfyanonymous/ComfyUI "$COMFY" "$COMFY_COMMIT"
for entry in "${NODES[@]}"; do
  read -r url commit <<<"$entry"
  clone_at "$url" "$COMFY/custom_nodes/$(basename "$url")" "$commit"
done

# Own venv: ComfyUI-Manager installs whatever a node asks for, which must not touch Obscura's pinned packages
if [ ! -x "$VENV/bin/python" ]; then
  log "Creating $VENV"
  uv venv -q --python 3.12 "$VENV"
fi
TORCH=$(/venv/main/bin/python -c 'import torch; print(torch.__version__.split("+")[0])')
TV=$(/venv/main/bin/python -c 'import torchvision; print(torchvision.__version__.split("+")[0])')
TA=$(/venv/main/bin/python -c 'import torchaudio; print(torchaudio.__version__.split("+")[0])' 2>/dev/null || echo "$TORCH")
log "torch $TORCH / torchvision $TV / torchaudio $TA (cu128)"
uv pip install -q --python "$VENV/bin/python" "torch==$TORCH" "torchvision==$TV" "torchaudio==$TA" \
  --index-url https://download.pytorch.org/whl/cu128
reqs=(-r "$COMFY/requirements.txt")
for f in "$COMFY"/custom_nodes/*/requirements.txt; do reqs+=(-r "$f"); done
# torch is already satisfied by the cu128 build above, so these requirements leave it alone
uv pip install -q --python "$VENV/bin/python" "${reqs[@]}" websocket-client
"$VENV/bin/python" -c 'import torch; print("comfy venv torch", torch.__version__, torch.cuda.is_available())'

log "Service files"
cp "$APP/deploy/service/comfyui.sh" /opt/supervisor-scripts/comfyui.sh
chmod +x /opt/supervisor-scripts/comfyui.sh
python3 - "$EXT_PORT" <<'EOF'
import sys, yaml
p = "/etc/portal.yaml"
d = yaml.safe_load(open(p)) or {}
apps = d.setdefault("applications", {})
if "ComfyUI" not in apps:
    apps["ComfyUI"] = {"hostname": "localhost", "external_port": int(sys.argv[1]), "internal_port": 18188,
                       "open_path": "/", "name": "ComfyUI"}
    yaml.safe_dump(d, open(p, "w"), sort_keys=False)
    print("portal entry added")
EOF

log "Models (~97 GB)"
HF_HOME=/workspace/.hf_home /venv/main/bin/python "$APP/deploy/prefetch_comfy.py"

until supervisorctl pid >/dev/null 2>&1; do sleep 5; done
cp "$APP/deploy/service/comfyui.conf" /etc/supervisor/conf.d/comfyui.conf
supervisorctl restart caddy >/dev/null
supervisorctl reread >/dev/null && supervisorctl update
log "ComfyUI installed (editor on portal port $EXT_PORT)"
