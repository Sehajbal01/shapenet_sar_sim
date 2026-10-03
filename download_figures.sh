#!/bin/bash
set -euo pipefail

REMOTE_HOST="engr-mahala01s.engr.arizona.edu"
REMOTE_USER="berian"

# "<remote path>|<local path>"
REMOTE_LOCAL_PAIRS=(
    # "/workspace/berian/sony-proposal/OurGeNVS/figures|$HOME/Documents/remote-figures/sony-proposal"
    # "/workspace/berian/realsimir/figures|$HOME/Documents/remote-figures/realsimir"
    "/workspace/berian/sar-sim/shapenet_sar_sim/figures|$HOME/Documents/remote-figures/sarsim"
)

# optional filename substring, e.g. "./download_figures.sh stitched" gets only *stitched* files
substr="${1:-}"

for pair in "${REMOTE_LOCAL_PAIRS[@]}"; do
    remote_dir="${pair%%|*}"
    local_dir="${pair#*|}"

    # delete each local folder, even for a filtered download
    if [[ -d "$local_dir" ]]; then
        echo "Removing $local_dir"
        rm -rf "$local_dir"
    fi
    mkdir -p "$local_dir"

    # rsync remote to local; recurse into dirs, keep files matching *substr*, prune empty dirs
    echo "Downloading $remote_dir/*$substr* -> $local_dir"
    rsync -rvm --include='*/' --include="*$substr*" --exclude='*' "$REMOTE_USER@$REMOTE_HOST:$remote_dir/" "$local_dir/"
done

echo "Done."
