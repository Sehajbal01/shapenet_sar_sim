#!/bin/bash

# Check if a message was provided
if [ -z "$1" ]; then
  echo "Usage: $0 \"commit message\""
  exit 1
fi

# Add all changes
git add .

# Commit with message
git commit -m "$1"

# Push to current branch
git push
