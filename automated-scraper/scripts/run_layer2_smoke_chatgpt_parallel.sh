#!/bin/bash
# Phase 2: parallel smoke. Launches all 6 sessions simultaneously with
# sync barrier — every session sends query #i at the same wall-clock moment.
# Use this after login_layer2_chatgpt.sh has populated session_00..05.

set -e

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2"

# Sanity: verify all 6 session dirs have Cookies
echo "Verifying Cookies files in all 6 session dirs:"
ALL_OK=1
for i in 00 01 02 03 04 05; do
  cookies="$BASE/session_$i/Default/Cookies"
  if [ -f "$cookies" ]; then
    size=$(stat -f%z "$cookies" 2>/dev/null || echo "?")
    echo "  session_$i: ok (${size}B)"
  else
    echo "  session_$i: MISSING Cookies — run login_layer2_chatgpt.sh first"
    ALL_OK=0
  fi
done
if [ "$ALL_OK" != "1" ]; then
  exit 1
fi

echo ""
echo "=================================================="
echo "  Launching parallel smoke: 6 sessions, sync barrier"
echo "  (you should see 6 Chrome windows pop up at once)"
echo "=================================================="
echo ""

exec python audit_chatgpt.py \
  --configs yamls/layer2_smoke_chatgpt_instant_bbq.yaml \
  --profile-base "$BASE" \
  --output-tag layer2-smoke-chatgpt-instant-bbq
