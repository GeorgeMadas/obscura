#!/bin/bash
# Obscura / Qwen Image auto-provisioning for Vast.ai (implements SETUP.md steps 0-8 + Claude Code)
# The Vast template's on-start downloads this file from main and runs it in the background, so
# edits here apply to every NEW instance. Needs GITHUB_TOKEN in the Vast account env vars.
# Log: /workspace/obscura-provision.log   Done marker: /workspace/.obscura-provisioned
set -o pipefail
log(){ echo "[obscura $(date +%T)] $*"; }
APP=/workspace/qwen-image
PY=/venv/main/bin/python
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="/root/.local/bin:$PATH"

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
  grep -q '.local/bin' /root/.bashrc || echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
fi

# 7. Claude extras
mkdir -p ~/.claude/skills ~/.claude/projects/-workspace/memory
cp -n "$APP/deploy/claude/settings.json" ~/.claude/settings.json
cp -rn "$APP"/deploy/claude/skills/* ~/.claude/skills/
cp -n "$APP"/deploy/claude/memory/*.md ~/.claude/projects/-workspace/memory/

# 3. Python packages
log "Installing Python packages"
uv pip install --python $PY -r "$APP/deploy/requirements.txt" || log "pip install FAILED"
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
touch /workspace/.obscura-provisioned
log "Provisioning complete - Qwen Image is loading (portal port 10100)"
