#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Qwen Image"

source /venv/main/bin/activate
export HF_HOME="${HF_HOME:-${WORKSPACE:-/workspace}/.hf_home}"
cd "${WORKSPACE:-/workspace}/qwen-image"
pty python app.py 2>&1
