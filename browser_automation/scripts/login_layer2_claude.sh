#!/bin/bash
# Layer 2 phase 1 — Claude Sonnet, login + parallel smoke, 3 sessions (one per account).
#
# Phase 1: sequential per-session logins. Chrome windows stay open after login.
# Phase 2: parallel smoke via audit/audit_claude_layer2.py --attach-port-base.
#
# Account mapping (3 accounts, 1 session each):
#   session_00 -> account 1
#   session_01 -> account 2
#   session_02 -> account 3

set -e

# Optional --resume flag: appended to audit invocation; the audit script
# skips queries already completed in claude_data (any prior run).
RESUME_FLAG=""
YAML="yamls/layer2_smoke_claude_sonnet_bbq.yaml"
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME_FLAG="--resume" ;;
    --yaml=*) YAML="${arg#--yaml=}" ;;
  esac
done

cd "$(dirname "$0")/.."
BASE="$PWD/claude_data/chrome_profiles_layer2_phase1"
PORT_BASE=9228  # Claude: 9228, 9229, 9230 (offset from ChatGPT/Gemini)

# Only kill Chrome instances rooted at $BASE so a concurrent ChatGPT/Gemini
# smoke isn't disrupted.
kill_chrome() {
  pkill -f "user-data-dir=$BASE" 2>/dev/null || true
  sleep 3
}

echo "Wipe session_00..02 before starting? [y/N]"
echo "  (Tip: 'N' leaves existing profiles intact, so Chrome opens only"
echo "   one window per session instead of an extra welcome window.)"
read -r WIPE
kill_chrome
if [[ "$WIPE" =~ ^[Yy]$ ]]; then
  for i in 00 01 02; do rm -rf "$BASE/session_$i"; done
fi
mkdir -p "$BASE"
for i in 00 01 02; do mkdir -p "$BASE/session_$i"; done

# Phase 1: launch each Chrome on a unique debug port so we can attach later.
for i in 00 01 02; do
  case "$i" in
    00) ACCT=1 ;;
    01) ACCT=2 ;;
    02) ACCT=3 ;;
  esac
  PORT=$((PORT_BASE + 10#$i))
  echo ""
  echo "=================================================="
  echo "  session_$i  ->  log in as ACCOUNT $ACCT  (port $PORT)"
  echo "  (One Claude session per account; account dimension only.)"
  echo "=================================================="
  open -na "Google Chrome" --args \
    --user-data-dir="$BASE/session_$i" \
    --remote-debugging-port=$PORT \
    --no-first-run \
    --no-default-browser-check \
    https://claude.ai
  read -p "Press Enter once session_$i is logged in as ACCOUNT $ACCT..."
done

echo ""
echo "=================================================="
echo "  All 3 windows logged in and STAYING OPEN."
echo "  Verify each window's email matches its account."
echo "=================================================="
read -p "Press Enter to fire the parallel smoke (attaches to these 3 Chromes)..."

# Phase 2: parallel smoke, attaching to the 3 existing Chromes via debug ports.
exec python audit/audit_claude_layer2.py \
  --configs "$YAML" \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --output-tag layer2-resume-claude-haiku-bbq \
  $RESUME_FLAG
