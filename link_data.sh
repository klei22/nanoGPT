#!/usr/bin/env bash
# link_data.sh - Create a symbolic link named 'data' to a shared dataset directory.

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/shared_data" >&2
    exit 1
fi

TARGET="$1"

if [ -e data ] && [ ! -L data ]; then
    echo "Existing 'data' directory found. Move or remove it before linking." >&2
    exit 1
fi

rm -f data
ln -s "$TARGET" data

echo "Linked data -> $TARGET"
