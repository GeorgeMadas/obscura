#!/bin/bash
# Obscura / Qwen Image auto-provisioning for Vast.ai (implements SETUP.md steps 0-9 + Claude Code)
# The Vast template's on-start downloads this file from main and runs it in the background, so
# edits here apply to every NEW instance. Needs GITHUB_TOKEN in the Vast account env vars.
# Log: /var/log/obscura-provision.log   Done marker: /workspace/.obscura-provisioned
set -o pipefail
log(){ echo "[obscura $(date +%T)] $*"; }
APP=/workspace/qwen-image
PY=/venv/main/bin/python
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="/root/.local/bin:$PATH" HOME=/root

# On-start runs before the Vast entrypoint creates /workspace and starts supervisor; wait for both
until [ -d /workspace ] && supervisorctl pid >/dev/null 2>&1; do sleep 5; done
log "Vast entrypoint is up, provisioning"

# 0. Repo
if [ ! -f "$APP/app.py" ]; then
  log "Cloning repo"
  [ -n "$GITHUB_TOKEN" ] || log "GITHUB_TOKEN not set - private repo clone will fail"
  git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/GeorgeMadas/obscura.git" "$APP" || { log "clone FAILED"; exit 1; }
  git -C "$APP" remote set-url origin https://github.com/GeorgeMadas/obscura.git
fi

# 2. Git identity (+ push credentials only if GITHUB_TOKEN is set in Vast account env)
git config --global user.email georgemadas5@gmail.com
git config --global user.name GeorgeMadas
git config --global --add safe.directory "$APP"
if [ -n "$GITHUB_TOKEN" ]; then
  git config --global credential.helper store
  printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > ~/.git-credentials
  chmod 600 ~/.git-credentials
fi

# Claude Code
if ! command -v claude >/dev/null; then
  log "Installing Claude Code"
  curl -fsSL https://claude.ai/install.sh | bash
fi
# Vast's .bashrc reloads /etc/environment (resetting PATH), so expose claude via /usr/local/bin
ln -sf /root/.local/bin/claude /usr/local/bin/claude

# 7. Claude extras
mkdir -p ~/.claude/skills ~/.claude/projects/-workspace/memory
cp -n "$APP/deploy/claude/settings.json" ~/.claude/settings.json
cp -rn "$APP"/deploy/claude/skills/* ~/.claude/skills/
cp -n "$APP"/deploy/claude/memory/*.md ~/.claude/projects/-workspace/memory/

# 3. Python packages
log "Installing Python packages"
uv pip install --python $PY -r "$APP/deploy/requirements.txt" || log "pip install FAILED"
# Fast kernels for the prompt rewriter; causal-conv1d compiles against the installed torch (~6 min)
uv pip install --python $PY flash-linear-attention==0.5.2 || log "flash-linear-attention install FAILED"
MAX_JOBS=16 uv pip install --python $PY --no-build-isolation causal-conv1d==1.7.0 ninja==1.13.2 || log "causal-conv1d install FAILED"
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. Service files (portal entry comes from PORTAL_CONFIG in the template)
cp "$APP/deploy/service/qwen-image.sh" /opt/supervisor-scripts/qwen-image.sh
chmod +x /opt/supervisor-scripts/qwen-image.sh

# 5. Models
cd "$APP/upscalers"
[ -s 4x-UltraSharp.safetensors ] || wget -q -O 4x-UltraSharp.safetensors https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors
[ -s RealESRGAN_x4plus.pth ] || wget -q -O RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
sha256sum 4x-UltraSharp.safetensors RealESRGAN_x4plus.pth
log "Prefetching models (~75 GB)"
cd "$APP" && $PY deploy/prefetch_models.py || log "prefetch FAILED"

# 8. Register + start the service once supervisor is up
until supervisorctl pid >/dev/null 2>&1; do sleep 5; done
cp "$APP/deploy/service/qwen-image.conf" /etc/supervisor/conf.d/qwen-image.conf
supervisorctl reread && supervisorctl update

# 9. ComfyUI (video workflows; editor on portal port 10200)
bash "$APP/deploy/install_comfyui.sh" || log "ComfyUI install FAILED"

touch /workspace/.obscura-provisioned
log "Provisioning complete - Qwen Image is loading (portal port 10100), ComfyUI editor on 10200"
