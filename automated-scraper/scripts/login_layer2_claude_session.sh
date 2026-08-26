#!/bin/bash
# Layer 2 phase 2 — within-account session variance, Claude haiku.
# 3 Chrome windows (1 account x 3 sessions), sync barrier.
# All 3 sessions must be logged into the SAME account.

set -e

RESUME_FLAG=""
YAML="yamls/layer2_session_claude_haiku_bbq.yaml"
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME_FLAG="--resume" ;;
    --yaml=*) YAML="${arg#--yaml=}" ;;
  esac
done

cd "$(dirname "$0")/.."
BASE="$PWD/claude_data/chrome_profiles_layer2_session"
PORT_BASE=9231  # 9231..9233 (clear of ChatGPT 9222-9224/9250-9252, Gemini 9240-9242)

kill_chrome() {
  pkill -f "user-data-dir=$BASE" 2>/dev/null || true
  sleep 3
}

echo "Wipe session_00..02 before starting? [y/N]"
read -r WIPE
kill_chrome
if [[ "$WIPE" =~ ^[Yy]$ ]]; then
  for i in 00 01 02; do rm -rf "$BASE/session_$i"; done
fi
mkdir -p "$BASE"
for i in 00 01 02; do mkdir -p "$BASE/session_$i"; done

for i in 00 01 02; do
  PORT=$((PORT_BASE + 10#$i))
  echo ""
  echo "=================================================="
  echo "  session_$i  ->  session $((10#$i + 1))/3  (port $PORT)"
  echo "  (All 3 sessions must use the SAME account login.)"
  echo "=================================================="
  open -na "Google Chrome" --args \
    --user-data-dir="$BASE/session_$i" \
    --remote-debugging-port=$PORT \
    --no-first-run \
    --no-default-browser-check \
    https://claude.ai
  read -p "Press Enter once logged in..."
done

echo ""
echo "=================================================="
echo "  All 3 windows logged in and STAYING OPEN."
echo "  Verify all 3 are the same account."
echo "=================================================="
read -p "Press Enter to fire the parallel run (3 sessions, sync barrier)..."

exec python audit/audit_claude_layer2.py \
  --configs "$YAML" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --output-tag layer2-session-claude-haiku-bbq \
  $RESUME_FLAG
