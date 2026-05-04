#!/bin/bash
REMOTE_HOST="engr-mahala01s.engr.arizona.edu"
REMOTE_USER="berian"
REMOTE_PATH="/workspace/berian/shapenet_sar_sim/figures/"
LOCAL_PATH="/home/berian/Documents/remote-figures/"

rm -rf "$LOCAL_PATH"
mkdir -p "$LOCAL_PATH"
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
