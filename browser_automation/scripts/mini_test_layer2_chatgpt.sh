#!/bin/bash
# Minimum-viable test of the Layer 2 attach-mode wiring.
# - 2 Chrome instances on ports 9222 / 9223
# - Same account is fine (we're testing the mechanism, not variance)
# - 2 BBQ queries via audit/audit_chatgpt_layer2.py --attach-port-base

set -e

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2"
PORT_BASE=9222

kill_chrome() {
  pkill -f "Google Chrome.app/Contents/MacOS/Google Chrome" 2>/dev/null || true
  sleep 3
}

echo "Wipe session_00 and session_01 first? [y/N]"
read -r WIPE
kill_chrome
if [[ "$WIPE" =~ ^[Yy]$ ]]; then
  for i in 00 01; do rm -rf "$BASE/session_$i"; done
fi
mkdir -p "$BASE/session_00" "$BASE/session_01"

for i in 00 01; do
  PORT=$((PORT_BASE + 10#$i))
  echo ""
  echo "=================================================="
  echo "  session_$i  ->  any ChatGPT account  (port $PORT)"
  echo "  (Same account both times is fine; this only tests wiring.)"
  echo "=================================================="
  open -na "Google Chrome" --args \
    --user-data-dir="$BASE/session_$i" \
    --remote-debugging-port=$PORT \
    --no-first-run \
    --no-default-browser-check \
    https://chatgpt.com
  read -p "Press Enter once session_$i is logged in..."
done

echo ""
echo "=================================================="
echo "  Both windows logged in, STAYING OPEN."
echo "  Pressing Enter fires the mini smoke (2 queries each)"
echo "  attached to these existing Chromes via debug ports."
echo "=================================================="
read -p "Press Enter to launch..."

exec python audit/audit_chatgpt_layer2.py \
  --configs yamls/layer2_mini_smoke_chatgpt.yaml \
  --profile-base "$BASE" \
  --attach-port-base "$PORT_BASE" \
  --output-tag layer2-mini-smoke
