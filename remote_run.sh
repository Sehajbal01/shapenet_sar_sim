#!/bin/bash

REMOTE="berian@engr-mahala01s.engr.arizona.edu"
REMOTE_DIR="/workspace/berian/remote-run"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="run_log.txt"

echo "=== Uploading code ==="
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='figures' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "=== Running on remote ==="
ssh "$REMOTE" "cd $REMOTE_DIR && bash test.sh 2>&1 | tee $LOG_FILE"

echo "=== Downloading figures ==="
rsync -av "$REMOTE:$REMOTE_DIR/figures/" "$LOCAL_DIR/figures/"

echo "=== Downloading log ==="
rsync -av "$REMOTE:$REMOTE_DIR/$LOG_FILE" "$LOCAL_DIR/$LOG_FILE"

echo "=== Done ==="
