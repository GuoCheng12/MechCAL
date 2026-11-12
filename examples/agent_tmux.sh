#!/bin/bash
# Open a tmux session with live views of MechCAL agent runs (read-only).
#   bash examples/agent_tmux.sh outputs/agent/dev2   then: tmux attach -t mechcal
# Left: one row per case. Right top: event stream. Right bottom: compute processes.
set -euo pipefail
ROOT=${1:-outputs/agent/dev2}
SESSION=${2:-mechcal}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/bin/python"
DASH="$REPO/examples/agent_dashboard.py"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 260 -y 70 -c "$REPO" "$PY $DASH board $ROOT"
tmux split-window -h -t "$SESSION:0" -p 42 -c "$REPO" "$PY $DASH feed $ROOT"
tmux split-window -v -t "$SESSION:0.1" -p 35 -c "$REPO" "$PY $DASH procs $ROOT"
tmux select-pane -t "$SESSION:0.0"
tmux set -t "$SESSION" status-style "bg=colour24,fg=white"
tmux set -t "$SESSION" status-left " MechCAL · $ROOT "
tmux set -t "$SESSION" status-left-length 60
tmux set -t "$SESSION" status-right " %H:%M  ·  Ctrl-b d to detach "
tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" mouse on
tmux select-pane -t "$SESSION:0.0" -T "cases"
tmux select-pane -t "$SESSION:0.1" -T "events"
tmux select-pane -t "$SESSION:0.2" -T "compute"
echo "tmux session '$SESSION' is running. Attach with:  tmux attach -t $SESSION"
