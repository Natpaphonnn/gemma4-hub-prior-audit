#!/bin/bash
# Download individual competition files into data/, preserving paths.
# Usage: analysis/fetch.sh <file> [<file>...]   (retries on transient 5xx)
SLUG=gemma-4-developer-agent
for f in "$@"; do
  dir="data/$(dirname "$f")"; mkdir -p "$dir"
  [ -f "data/$f" ] && continue
  for i in 1 2 3; do
    .venv/bin/kaggle competitions download "$SLUG" -f "$f" -p "$dir" -q >/dev/null 2>&1 && break
    sleep 2
  done
  z="$dir/$(basename "$f").zip"
  [ -f "$z" ] && unzip -oq "$z" -d "$dir" && rm "$z"
  [ -f "data/$f" ] || echo "FAILED: $f"
done
