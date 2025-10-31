#!/bin/bash
# upload_figures.sh
# Upload all files in ./figures to a remote server.

# === CONFIGURATION ===
REMOTE_USER="berian"   # <-- change this to your remote username
REMOTE_HOST="150.135.222.51"
REMOTE_DIR="~/Documents/figures"  # destination directory on the remote server

# === SCRIPT ===
set -e  # stop if any command fails

if [ ! -d "./figures" ]; then
        echo "Error: ./figures directory not found."
            exit 1
fi

echo "Uploading all files from ./figures to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR ..."
scp ./figures/* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "✅ Upload complete!"

