#!/bin/bash
# Obscura setup dashboard (deploy/dashboard). Stdlib-only Python, so it runs before any package or model exists.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Setup Dashboard"

export HF_HOME="${HF_HOME:-${WORKSPACE:-/workspace}/.hf_home}"
cd /opt/obscura-dashboard
pty /usr/bin/python3 -u server.py 2>&1
