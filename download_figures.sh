#!/bin/bash
REMOTE_HOST="engr-mahala01s.engr.arizona.edu"
REMOTE_USER="berian"
REMOTE_BASE="/workspace/berian/shapenet_sar_sim"

# Each entry is "remote_subdir:local_path"
FOLDERS=(
    "figures:/home/berian/Documents/remote-figures"
    "latex:/home/berian/Documents/paper"
)

RSYNC_ARGS=(-avz --progress)
if [ -n "$1" ]; then
    RSYNC_ARGS+=(--include="*$1*" --exclude="*")
fi

for entry in "${FOLDERS[@]}"; do
    REMOTE_SUBDIR="${entry%%:*}"
    LOCAL_PATH="${entry#*:}"
    rm -rf "$LOCAL_PATH"/*
    mkdir -p "$LOCAL_PATH"
    rsync "${RSYNC_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/$REMOTE_SUBDIR/" "$LOCAL_PATH/"
done
