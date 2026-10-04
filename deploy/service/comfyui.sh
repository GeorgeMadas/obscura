#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "ComfyUI"

# Own venv so nodes installed through ComfyUI-Manager can't change Obscura's packages (/venv/main)
source /venv/comfy/bin/activate
cd "${WORKSPACE:-/workspace}/ComfyUI"
# Behind Caddy (and Cloudflare tunnels) the browser's Origin never matches the proxied Host, so ComfyUI's
# same-origin guard would 403 the editor. Caddy's token check protects it instead; --enable-cors-header swaps
# the guard for CORS headers that only name ComfyUI's own local address.
pty python main.py --listen 127.0.0.1 --port 18188 --disable-auto-launch \
    --enable-cors-header http://127.0.0.1:18188 \
    --output-directory "${WORKSPACE:-/workspace}/ComfyUI/output" 2>&1
