#!/bin/bash
# Run 6 sequential per-session smoke tests on already-set-up profile dirs.
# Use this after setup_layer2_logins_chatgpt.sh has populated session_00..05.

set -e

cd "$(dirname "$0")/.."
BASE="$PWD/chatgpt_data/chrome_profiles_layer2"
SMOKE_QUERY_FILE="../benchmark_creation/results/bbq-subset-10.csv"
MAPPING_FILE="$BASE/account_mapping.json"

# Sanity: verify all 6 session dirs exist
ALL_OK=1
for i in 00 01 02 03 04 05; do
  if [ ! -d "$BASE/session_$i" ]; then
    echo "session_$i: MISSING dir"
    ALL_OK=0
  fi
done
if [ "$ALL_OK" != "1" ]; then
  echo "Run setup_layer2_logins_chatgpt.sh first."
  exit 1
fi

# Account-email mapping: load existing or prompt user
if [ -f "$MAPPING_FILE" ]; then
  ACCT1_EMAIL=$(python3 -c "import json; print(json.load(open('$MAPPING_FILE'))['account_1'])")
  ACCT2_EMAIL=$(python3 -c "import json; print(json.load(open('$MAPPING_FILE'))['account_2'])")
  echo "Loaded existing account mapping from $MAPPING_FILE:"
  echo "  account_1 = $ACCT1_EMAIL  (sessions 00/01/02)"
  echo "  account_2 = $ACCT2_EMAIL  (sessions 03/04/05)"
else
  echo "Set up the account → email mapping (one-time):"
  read -p "  Email for ACCOUNT 1 (sessions 00/01/02): " ACCT1_EMAIL
  read -p "  Email for ACCOUNT 2 (sessions 03/04/05): " ACCT2_EMAIL
  cat > "$MAPPING_FILE" <<EOF
{
  "account_1": "$ACCT1_EMAIL",
  "account_2": "$ACCT2_EMAIL",
  "session_to_account": {
    "session_00": "account_1",
    "session_01": "account_1",
    "session_02": "account_1",
    "session_03": "account_2",
    "session_04": "account_2",
    "session_05": "account_2"
  }
}
EOF
  echo "Saved to $MAPPING_FILE"
fi

TMP_ROOT="$(mktemp -d -t layer2_smoke)"
TMP_YAML_DIR="$(mktemp -d -t layer2_smoke_yamls)"
trap 'rm -rf "$TMP_ROOT" "$TMP_YAML_DIR"' EXIT

for i in 00 01 02 03 04 05; do
  case "$i" in
    00) ACCT=1; EMAIL="$ACCT1_EMAIL"; SUFFIX="acct1_s1" ;;
    01) ACCT=1; EMAIL="$ACCT1_EMAIL"; SUFFIX="acct1_s2" ;;
    02) ACCT=1; EMAIL="$ACCT1_EMAIL"; SUFFIX="acct1_s3" ;;
    03) ACCT=2; EMAIL="$ACCT2_EMAIL"; SUFFIX="acct2_s1" ;;
    04) ACCT=2; EMAIL="$ACCT2_EMAIL"; SUFFIX="acct2_s2" ;;
    05) ACCT=2; EMAIL="$ACCT2_EMAIL"; SUFFIX="acct2_s3" ;;
  esac

  echo ""
  echo "=================================================="
  echo "  Smoke $((10#$i + 1))/6: session_$i  ($SUFFIX)"
  echo "  ACCOUNT $ACCT  →  $EMAIL"
  echo "=================================================="
  echo "  Watch the Chrome window:"
  echo "    - If logged in: smoke runs 10 BBQ queries and closes"
  echo "    - If login screen: log in as $EMAIL, then smoke proceeds"
  echo ""
  read -p "Press Enter to fire this session's smoke..."

  TMP_BASE="$TMP_ROOT/base_$i"
  rm -rf "$TMP_BASE"
  mkdir -p "$TMP_BASE"
  ln -s "$BASE/session_$i" "$TMP_BASE/session_00"

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

  python audit_chatgpt.py \
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
