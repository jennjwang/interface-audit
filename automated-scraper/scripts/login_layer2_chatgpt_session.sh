#!/bin/bash
# Layer 2 phase 2 — within-account session variance, ChatGPT instant.
# 3 Chrome windows (1 account x 3 sessions), staggered start (no sync barrier).
# All 3 sessions must be logged into the SAME account.

set -e

RESUME_FLAG=""
YAML_S1="yamls/layer2_session_chatgpt_instant_bbq.yaml"
YAML_S2="yamls/layer2_session_chatgpt_instant_bbq.yaml"
YAML_S3="yamls/layer2_session_chatgpt_instant_bbq.yaml"
STAGGER=120  # seconds between session starts
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME_FLAG="--resume" ;;
    --yaml=*) YAML_S1="${arg#--yaml=}"; YAML_S2="$YAML_S1"; YAML_S3="$YAML_S1" ;;
    --yaml-s1=*) YAML_S1="${arg#--yaml-s1=}" ;;
    --yaml-s2=*) YAML_S2="${arg#--yaml-s2=}" ;;
    --yaml-s3=*) YAML_S3="${arg#--yaml-s3=}" ;;
    --stagger=*) STAGGER="${arg#--stagger=}" ;;
  esac
done

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2_session"
PORT_BASE=9250  # 9250..9252 (clear of phase1: ChatGPT 9222-9224, Gemini 9225-9227, Claude 9228-9230)

kill_chrome() {
  pkill -f "user-data-dir=$BASE" 2>/dev/null || true
  sleep 3
}

echo "Wipe session_00..02 before starting? [y/N]"
echo "  (Say 'y' to start fresh; 'n' to reuse existing logins.)"
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
    https://chatgpt.com
  read -p "Press Enter once logged in..."
done

echo ""
echo "=================================================="
echo "  All 3 windows logged in and STAYING OPEN."
echo "  Verify all 3 are the same account."
echo "  Sessions will start staggered: s1 now, s2 +${STAGGER}s, s3 +$((STAGGER*2))s"
echo "=================================================="
read -p "Press Enter to fire staggered runs..."

python audit_chatgpt_layer2.py \
  --configs "$YAML_S1" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --sessions 1 \
  --session-index-offset 0 \
  --output-tag layer2-session-chatgpt-instant-bbq-s1 \
  $RESUME_FLAG &

sleep "$STAGGER"
python audit_chatgpt_layer2.py \
  --configs "$YAML_S2" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --sessions 1 \
  --session-index-offset 1 \
  --output-tag layer2-session-chatgpt-instant-bbq-s2 \
  $RESUME_FLAG &

sleep "$STAGGER"
python audit_chatgpt_layer2.py \
  --configs "$YAML_S3" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --sessions 1 \
  --session-index-offset 2 \
  --output-tag layer2-session-chatgpt-instant-bbq-s3 \
  $RESUME_FLAG &

wait
echo "All 3 sessions complete."
