#!/bin/bash
# Layer 2 phase 2 — ChatGPT session variance, sequential (s1 → s2 → s3).
# Runs one session at a time. Chrome windows must already be open and logged in.
# Use login_layer2_chatgpt_session.sh first if starting fresh.

set -e

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2_session"
PORT_BASE=9250

SESSIONS=("s1" "s2" "s3")
YAMLS=(
  "yamls/layer2_resume_session_chatgpt_s1.yaml"
  "yamls/layer2_resume_session_chatgpt_s2.yaml"
  "yamls/layer2_resume_session_chatgpt_s3.yaml"
)

for i in 0 1 2; do
  SESSION="${SESSIONS[$i]}"
  YAML="${YAMLS[$i]}"
  PORT=$((PORT_BASE + i))
  PROFILE="$BASE/session_0$i"

  echo ""
  echo "=================================================="
  echo "  Running ${SESSION} (port ${PORT}, profile session_0${i})"
  echo "  YAML: ${YAML}"
  echo "=================================================="

  python audit_chatgpt_layer2.py \
    --configs "$YAML" \
    --profile-base "$BASE" \
    --attach-port-base "$PORT_BASE" \
    --sessions 1 \
    --session-index-offset "$i" \
    --output-tag "layer2-session-chatgpt-instant-bbq-${SESSION}" \
    --resume

  echo "  ${SESSION} complete."
done

echo ""
echo "=================================================="
echo "  All 3 sessions complete."
echo "=================================================="
