#!/bin/bash
# Resume ChatGPT phase2 session experiment.
# Assumes Chrome windows for session_00..02 are already open on ports 9250..9252.
# Runs mode:sync so each experiment gets its own subprocess with experiment_index.

set -e

YAML="yamls/layer2_resume_session_chatgpt_final.yaml"
for arg in "$@"; do
  case "$arg" in
    --yaml=*) YAML="${arg#--yaml=}" ;;
  esac
done

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2_session"
PORT_BASE=9250

echo "Attaching to existing Chrome windows on ports ${PORT_BASE}...$((PORT_BASE + 2))."
echo "YAML: $YAML"
echo ""
read -p "Press Enter to fire resume (3 sessions, sync mode)..."

exec python audit/audit_chatgpt_layer2.py \
  --configs "$YAML" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --resume
