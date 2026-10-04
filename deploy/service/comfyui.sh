#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "ComfyUI"

# Own venv so nodes installed through ComfyUI-Manager can't change Obscura's packages (/venv/main)
source /venv/comfy/bin/activate
cd "${WORKSPACE:-/workspace}/ComfyUI"
pty python main.py --listen 127.0.0.1 --port 18188 --disable-auto-launch \
    --output-directory "${WORKSPACE:-/workspace}/ComfyUI/output" 2>&1
