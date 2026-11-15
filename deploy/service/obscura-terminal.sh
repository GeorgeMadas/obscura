#!/bin/bash
# Obscura web terminal (deploy/terminal): ttyd serving a phone-friendly xterm.js page, attached to the
# tmux session "main". Needs only ttyd, tmux and python3, so it runs before any package or model exists.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Web Terminal"

T=/opt/obscura-terminal
export HOME=/root PATH="/root/.local/bin:/usr/local/bin:$PATH" LANG="${LANG:-C.UTF-8}"
/usr/bin/python3 "$T/build.py" "$T" "$T/page.html" || exit 1
cd "${WORKSPACE:-/workspace}"
# 127.0.0.1 only: Caddy fronts it on portal port 8384 with the instance token
pty ttyd --interface 127.0.0.1 --port 17681 --writable --index "$T/page.html" \
  --ping-interval 20 "$T/attach.sh" 2>&1
