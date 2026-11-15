#!/bin/bash
# Web Terminal: what ttyd runs for each browser connection. Every phone, tab and reconnect lands in the
# same tmux session "main", so closing the page or losing the network never kills what is running.
CONF=/opt/obscura-terminal/tmux.conf
cd "${WORKSPACE:-/workspace}" 2>/dev/null || cd /root
tmux -f "$CONF" new-session -d -s main 2>/dev/null   # first connect: start the server and the session
tmux source-file "$CONF"                               # server already running: apply the current settings
tmux set-option -t main status off                     # the page has its own header; phones need the rows
exec tmux new -A -s main
