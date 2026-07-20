#!/bin/bash
REMOTE_HOST="engr-mahala01s.engr.arizona.edu"
REMOTE_USER="berian"
REMOTE_BASE="/workspace/berian/shapenet_sar_sim"

UPLOAD=0
while getopts "u" opt; do
    case "$opt" in
        u) UPLOAD=1 ;;
        *) echo "Usage: $0 [-u] [filter]" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Each entry is "remote_subdir:local_path"
FOLDERS=(
    "figures:/home/berian/Documents/remote-figures"
    "latex:/home/berian/Documents/remote-paper"
)

RSYNC_ARGS=(-avz --progress)
if [ -n "$1" ]; then
    RSYNC_ARGS+=(--include="*$1*" --exclude="*")
fi

for entry in "${FOLDERS[@]}"; do
    REMOTE_SUBDIR="${entry%%:*}"
    LOCAL_PATH="${entry#*:}"
    REMOTE="$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/$REMOTE_SUBDIR/"
    if [ "$UPLOAD" -eq 1 ]; then
        rsync "${RSYNC_ARGS[@]}" "$LOCAL_PATH/" "$REMOTE"
    else
        rm -rf "$LOCAL_PATH"/*
        mkdir -p "$LOCAL_PATH"
        rsync "${RSYNC_ARGS[@]}" "$REMOTE" "$LOCAL_PATH/"
    fi
done
