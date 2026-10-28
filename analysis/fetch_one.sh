#!/bin/bash
# Download a single competition file into data/, skip if present, retry on 5xx.
f="$1"; SLUG=gemma-4-developer-agent
[ -f "data/$f" ] && exit 0
dir="data/$(dirname "$f")"; mkdir -p "$dir"
for i in 1 2 3 4; do
  .venv/bin/kaggle competitions download "$SLUG" -f "$f" -p "$dir" -q >/dev/null 2>&1
  z="$dir/$(basename "$f").zip"
  [ -f "$z" ] && unzip -oq "$z" -d "$dir" && rm -f "$z"
  [ -f "data/$f" ] && exit 0
  sleep $((i*2))
done
echo "FAILED: $f"
