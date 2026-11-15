#!/bin/bash
# Obscura / Qwen Image auto-provisioning for Vast.ai (implements SETUP.md steps 0-9 + Claude Code)
# The Vast template's on-start downloads this file from main and runs it in the background, so
# edits here apply to every NEW instance. Needs GITHUB_TOKEN in the Vast account env vars.
# Works on both the plain vastai/pytorch image and the prebuilt ghcr.io/georgemadas/obscura image (Dockerfile):
# on the prebuilt one the installs below are already satisfied and finish in seconds.
# Log: /var/log/obscura-provision.log   Done marker: /workspace/.obscura-provisioned
set -o pipefail
log(){ echo "[obscura $(date +%T)] $*"; }
APP=/workspace/qwen-image
PY=/venv/main/bin/python
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
export PATH="/root/.local/bin:$PATH" HOME=/root

# On-start runs before the Vast entrypoint creates /workspace and starts supervisor; wait for both
until [ -d /workspace ] && supervisorctl pid >/dev/null 2>&1; do sleep 5; done
log "Vast entrypoint is up, provisioning"

# Setup dashboard (deploy/dashboard): live phases, downloads, network, disk, GPU and services on portal port 6006.
# Starts first so it sees every download. Its files come straight from main, because the repo may not be cloned yet.
# No port is free in the template, so it takes Tensorboard's (unused here): Tensorboard leaves /etc/portal.yaml
# and self-skips from then on.
# fetch_deploy <dir> <deploy/ paths...>: download each file from main into <dir> (keeping any subfolder after the
# first path segment, e.g. terminal/vendor/x.js -> <dir>/vendor/x.js), falling back to the local clone
fetch_deploy(){
  local dir=$1 f rel; shift
  for f in "$@"; do
    rel=${f#*/}; mkdir -p "$dir/$(dirname "$rel")"
    curl -fsSL -H "Authorization: token $GITHUB_TOKEN" -o "$dir/$rel.new" \
      "https://raw.githubusercontent.com/GeorgeMadas/obscura/main/deploy/$f" \
      && mv "$dir/$rel.new" "$dir/$rel" \
      || { rm -f "$dir/$rel.new"; [ -f "$APP/deploy/$f" ] && cp "$APP/deploy/$f" "$dir/$rel"; }
  done
}
DASH=/opt/obscura-dashboard
fetch_deploy "$DASH" dashboard/server.py dashboard/index.html service/obscura-dashboard.sh service/obscura-dashboard.conf
if [ -s "$DASH/server.py" ] && [ -s "$DASH/obscura-dashboard.conf" ]; then
  install -m 755 "$DASH/obscura-dashboard.sh" /opt/supervisor-scripts/obscura-dashboard.sh
  cp "$DASH/obscura-dashboard.conf" /etc/supervisor/conf.d/obscura-dashboard.conf
  python3 - <<'EOF'
import os, yaml
p = "/etc/portal.yaml"
d = yaml.safe_load(open(p)) or {}
apps = d.setdefault("applications", {})
changed = apps.pop("Tensorboard", None) is not None
if "Setup Dashboard" not in apps:
    apps["Setup Dashboard"] = {"hostname": "localhost", "external_port": 6006, "internal_port": 16060,
                               "open_path": "/", "name": "Setup Dashboard"}
    changed = True
if changed:
    tmp = p + ".tmp"
    yaml.safe_dump(d, open(tmp, "w"), sort_keys=False)
    os.replace(tmp, p)
    print("portal: Setup Dashboard on 6006")
EOF
  supervisorctl stop tensorboard >/dev/null 2>&1
  supervisorctl reread >/dev/null && supervisorctl update
  supervisorctl restart obscura-dashboard >/dev/null  # pick up new files when it was already running
  supervisorctl restart caddy >/dev/null
  log "Setup dashboard running (portal port 6006)"
else
  log "Setup dashboard download FAILED - continuing without it"
fi

# Web terminal (deploy/terminal): ttyd + tmux session "main" with a phone key bar, on portal port 8384.
# Independent of the models and the app, so it is usable while the rest is still installing.
# No port is free in the template, so it takes Syncthing's (no shared folders or devices here):
# Syncthing leaves /etc/portal.yaml and self-skips from then on.
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux >/dev/null; } || log "tmux install FAILED"
if ! command -v ttyd >/dev/null; then
  # Static binary, same pin and checksum as the Dockerfile
  curl -fsSL -o /tmp/ttyd https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    && echo "8a217c968aba172e0dbf3f34447218dc015bc4d5e59bf51db2f2cd12b7be4f55  /tmp/ttyd" | sha256sum -c --quiet \
    && install -m 755 /tmp/ttyd /usr/local/bin/ttyd || log "ttyd install FAILED"
  rm -f /tmp/ttyd
fi
TERMDIR=/opt/obscura-terminal
fetch_deploy "$TERMDIR" terminal/index.html terminal/build.py terminal/attach.sh terminal/tmux.conf \
  terminal/vendor/xterm.js terminal/vendor/xterm.css terminal/vendor/addon-fit.js terminal/vendor/addon-webgl.js \
  terminal/vendor/addon-unicode11.js terminal/vendor/addon-web-links.js \
  service/obscura-terminal.sh service/obscura-terminal.conf
if command -v ttyd >/dev/null && command -v tmux >/dev/null && [ -s "$TERMDIR/vendor/xterm.js" ] && [ -s "$TERMDIR/obscura-terminal.conf" ]; then
  chmod +x "$TERMDIR/attach.sh"
  install -m 755 "$TERMDIR/obscura-terminal.sh" /opt/supervisor-scripts/obscura-terminal.sh
  cp "$TERMDIR/obscura-terminal.conf" /etc/supervisor/conf.d/obscura-terminal.conf
  python3 - <<'EOF'
import os, yaml
p = "/etc/portal.yaml"
d = yaml.safe_load(open(p)) or {}
apps = d.setdefault("applications", {})
changed = apps.pop("Syncthing", None) is not None
if "Web Terminal" not in apps:
    apps["Web Terminal"] = {"hostname": "localhost", "external_port": 8384, "internal_port": 17681,
                            "open_path": "/", "name": "Web Terminal"}
    changed = True
if changed:
    tmp = p + ".tmp"
    yaml.safe_dump(d, open(tmp, "w"), sort_keys=False)
    os.replace(tmp, p)
    print("portal: Web Terminal on 8384")
EOF
  supervisorctl stop syncthing >/dev/null 2>&1
  supervisorctl reread >/dev/null && supervisorctl update
  supervisorctl restart obscura-terminal >/dev/null  # pick up new files when it was already running
  supervisorctl restart caddy >/dev/null
  log "Web terminal running (portal port 8384)"
else
  log "Web terminal setup FAILED - continuing without it"
fi

# 0. Repo
if [ ! -f "$APP/app.py" ]; then
  log "Cloning repo"
  [ -n "$GITHUB_TOKEN" ] || log "GITHUB_TOKEN not set - private repo clone will fail"
  git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/GeorgeMadas/obscura.git" "$APP" || { log "clone FAILED"; exit 1; }
  git -C "$APP" remote set-url origin https://github.com/GeorgeMadas/obscura.git
elif [ -z "$(git -C "$APP" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  # Repo kept on a persistent volume: catch up with main, but only when nothing local would be touched
  log "Updating repo (fast-forward only)"
  git -C "$APP" -c safe.directory="$APP" -c credential.helper= \
    pull -q --ff-only "https://x-access-token:${GITHUB_TOKEN}@github.com/GeorgeMadas/obscura.git" main \
    || log "repo update skipped (not a fast-forward or no network)"
else
  log "Repo has uncommitted changes - leaving it as it is (commit and push them!)"
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
# The prebuilt image has ComfyUI in /opt/ComfyUI with models/input/output/user linked onto /workspace
if [ -d /opt/ComfyUI ]; then
  mkdir -p /workspace/models/comfyui /workspace/comfyui-data/input /workspace/comfyui-data/output /workspace/comfyui-data/user
  [ -e /workspace/ComfyUI ] || ln -s /opt/ComfyUI /workspace/ComfyUI
fi
bash "$APP/deploy/install_comfyui.sh" || log "ComfyUI install FAILED"

touch /workspace/.obscura-provisioned
log "Provisioning complete - Qwen Image is loading (portal port 10100), ComfyUI editor on 10200"
