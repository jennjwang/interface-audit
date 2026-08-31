#!/bin/bash
# Combined: 2 logins → replicate to 6 profile dirs → 6 sequential per-session
# smoke tests so you can visually verify each session in isolation.
#
# Workflow:
#   1. Optional wipe of session_00..05
#   2. Log in to ACCOUNT 1 once (in a temp dir); replicate to session_00, _01, _02
#   3. Log in to ACCOUNT 2 once (in a temp dir); replicate to session_03, _04, _05
#   4. Verify Cookies files in all 6 session dirs
#   5. For each session_NN, fire a 1-session smoke (10 queries) you can watch end-to-end

set -e

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2"
SETUP_DIR="$BASE/_setup_tmp"
SMOKE_QUERY_FILE="../benchmark_creation/results/bbq-subset-10.csv"

kill_chrome() {
  pkill -f "Google Chrome.app/Contents/MacOS/Google Chrome" 2>/dev/null || true
  sleep 3
}

replicate() {
  local src="$1"
  shift
  for dst in "$@"; do
    rm -rf "$dst"
    rsync -a \
      --exclude='Singleton*' \
      --exclude='Default/Cache' \
      --exclude='Default/Code Cache' \
      --exclude='Default/GPUCache' \
      "$src"/ "$dst"/
  done
}

# 0) Wipe prompt
echo "Wipe profile dirs session_00..05 and start from scratch? [y/N]"
read -r WIPE
kill_chrome
if [[ "$WIPE" =~ ^[Yy]$ ]]; then
  for i in 00 01 02 03 04 05; do rm -rf "$BASE/session_$i"; done
  rm -rf "$SETUP_DIR"
fi
mkdir -p "$BASE"
for i in 00 01 02 03 04 05; do mkdir -p "$BASE/session_$i"; done

# 1) Account 1 login → replicate
echo ""
echo "=================================================="
echo "  STEP 1/2: Log in to ACCOUNT 1"
echo "  This account populates session_00, _01, _02"
echo "=================================================="
mkdir -p "$SETUP_DIR/acct1"
open -na "Google Chrome" --args \
  --user-data-dir="$SETUP_DIR/acct1" \
  --no-first-run \
  --no-default-browser-check \
  https://chatgpt.com
read -p "Press Enter once ACCOUNT 1 is logged in..."
kill_chrome
echo "Replicating acct1 → session_00, _01, _02..."
replicate "$SETUP_DIR/acct1" "$BASE/session_00" "$BASE/session_01" "$BASE/session_02"

# 2) Account 2 login → replicate
echo ""
echo "=================================================="
echo "  STEP 2/2: Log in to ACCOUNT 2"
echo "  This account populates session_03, _04, _05"
echo "=================================================="
mkdir -p "$SETUP_DIR/acct2"
open -na "Google Chrome" --args \
  --user-data-dir="$SETUP_DIR/acct2" \
  --no-first-run \
  --no-default-browser-check \
  https://chatgpt.com
read -p "Press Enter once ACCOUNT 2 is logged in..."
kill_chrome
echo "Replicating acct2 → session_03, _04, _05..."
replicate "$SETUP_DIR/acct2" "$BASE/session_03" "$BASE/session_04" "$BASE/session_05"

# Cleanup setup dirs
rm -rf "$SETUP_DIR"

# 3) Verify cookies landed
echo ""
echo "Verifying Cookies files in all 6 session dirs:"
ALL_OK=1
for i in 00 01 02 03 04 05; do
  cookies="$BASE/session_$i/Default/Cookies"
  if [ -f "$cookies" ]; then
    size=$(stat -f%z "$cookies" 2>/dev/null || echo "?")
    echo "  session_$i: ok (${size}B)"
  else
    echo "  session_$i: MISSING Cookies"
    ALL_OK=0
  fi
done
if [ "$ALL_OK" != "1" ]; then
  echo "Cookie verification failed — aborting smoke."
  exit 1
fi

# 4) Sequential per-session smoke
# We use a temp profile-base whose session_00 is a symlink to the real
# session_NN, so audit/audit_chatgpt.py operates on one specific session per run.
TMP_ROOT="$(mktemp -d -t layer2_smoke)"
TMP_YAML_DIR="$(mktemp -d -t layer2_smoke_yamls)"
trap 'rm -rf "$TMP_ROOT" "$TMP_YAML_DIR"' EXIT

for i in 00 01 02 03 04 05; do
  case "$i" in
    00) ACCT=1; SUFFIX="acct1_s1" ;;
    01) ACCT=1; SUFFIX="acct1_s2" ;;
    02) ACCT=1; SUFFIX="acct1_s3" ;;
    03) ACCT=2; SUFFIX="acct2_s1" ;;
    04) ACCT=2; SUFFIX="acct2_s2" ;;
    05) ACCT=2; SUFFIX="acct2_s3" ;;
  esac

  echo ""
  echo "=================================================="
  echo "  Smoke $((10#$i + 1))/6: session_$i  (acct $ACCT, $SUFFIX)"
  echo "=================================================="
  echo "  Watch the Chrome window — it should:"
  echo "    1. Open already logged in as ACCOUNT $ACCT"
  echo "    2. Ask 10 BBQ questions and save responses"
  echo "    3. Close cleanly"
  echo ""
  read -p "Press Enter to fire this session's smoke..."

  # Build a temp profile-base whose session_00 -> the real session_NN
  TMP_BASE="$TMP_ROOT/base_$i"
  rm -rf "$TMP_BASE"
  mkdir -p "$TMP_BASE"
  ln -s "$BASE/session_$i" "$TMP_BASE/session_00"

  # Generate a 1-experiment yaml
  YAML="$TMP_YAML_DIR/smoke_$i.yaml"
  cat > "$YAML" <<EOF
defaults:
  query_file: $SMOKE_QUERY_FILE
  runs: 1
  shuffle: false
  fast: true
  wait_after_prompts: 0
  wait_after_saves: 0

mode: sync

experiments:
  - name: $SUFFIX
    type: non-temporary
    interface_model: instant
EOF

  python audit/audit_chatgpt.py \
    --configs "$YAML" \
    --profile-base "$TMP_BASE" \
    --output-tag layer2-smoke-chatgpt-instant-bbq-session-$i

  echo ""
  echo "  session_$i smoke finished."
done

echo ""
echo "=================================================="
echo "  All 6 per-session smokes complete."
echo "=================================================="
