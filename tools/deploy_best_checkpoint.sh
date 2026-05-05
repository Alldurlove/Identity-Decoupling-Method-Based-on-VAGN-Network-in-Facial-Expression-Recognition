#!/usr/bin/env bash
set -euo pipefail

SERVICE_FILE="/etc/systemd/system/vgan-uvicorn.service"
SERVICE_NAME="vgan-uvicorn"
DEFAULT_HEALTH_URL="http://127.0.0.1:8000/api/health"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_path> [health_url]"
  echo "Example: $0 /home/ubuntu/VGAN-Project/checkpoints/finetune/netG_finetuned_best.pth"
  exit 1
fi

CHECKPOINT_PATH="$1"
HEALTH_URL="${2:-$DEFAULT_HEALTH_URL}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [[ ! -f "$SERVICE_FILE" ]]; then
  echo "Service file not found: $SERVICE_FILE"
  exit 1
fi

TMP_FILE="$(mktemp)"
cp "$SERVICE_FILE" "$TMP_FILE"

if rg -q "^Environment=CHECKPOINT_PATH=" "$TMP_FILE"; then
  sed -i "s|^Environment=CHECKPOINT_PATH=.*|Environment=CHECKPOINT_PATH=$CHECKPOINT_PATH|g" "$TMP_FILE"
else
  awk -v line="Environment=CHECKPOINT_PATH=$CHECKPOINT_PATH" '
    /^\[Service\]$/ { print; inserted=1; next }
    inserted==1 && !done { print line; done=1; inserted=0 }
    { print }
  ' "$TMP_FILE" > "${TMP_FILE}.new"
  mv "${TMP_FILE}.new" "$TMP_FILE"
fi

sudo cp "$TMP_FILE" "$SERVICE_FILE"
rm -f "$TMP_FILE"

sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true

echo "Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager -n 20

echo ""
echo "Health check: $HEALTH_URL"
curl -s "$HEALTH_URL"
echo ""

echo "Deployment complete. Active checkpoint:"
sudo systemctl cat "$SERVICE_NAME" | rg "CHECKPOINT_PATH|ExecStart"
