#!/bin/bash
# Layer 2 phase 3 — request-level variance, ChatGPT instant.
# 1 Chrome window, 1 account, 3 runs per question.

set -e

RESUME_FLAG=""
YAML="yamls/layer2_request_chatgpt_instant_bbq.yaml"
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME_FLAG="--resume" ;;
    --yaml=*) YAML="${arg#--yaml=}" ;;
  esac
done

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2_request"
PORT=9260  # clear of all other ranges

kill_chrome() {
  pkill -f "user-data-dir=$BASE" 2>/dev/null || true
  sleep 3
}

echo "Wipe profile before starting? [y/N]"
echo "  (Say 'y' to start fresh; 'n' to reuse existing login.)"
read -r WIPE
kill_chrome
if [[ "$WIPE" =~ ^[Yy]$ ]]; then rm -rf "$BASE/session_00"; fi
mkdir -p "$BASE/session_00"

echo ""
echo "=================================================="
echo "  Opening ChatGPT — log in as your test account."
echo "=================================================="
open -na "Google Chrome" --args \
  --user-data-dir="$BASE/session_00" \
  --remote-debugging-port=$PORT \
  --no-first-run \
  --no-default-browser-check \
  https://chatgpt.com
read -p "Press Enter once logged in..."

echo ""
read -p "Press Enter to start (3 runs x 200 questions)..."

exec python audit/audit_chatgpt_layer2.py \
  --configs "$YAML" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT" \
  --sessions 1 \
  --output-tag layer2-request-chatgpt-instant-bbq \
  $RESUME_FLAG
