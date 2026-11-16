#!/bin/sh
# Keep the named topics of maps before they are retrained: runs/_naming_backup/<map>/{main,fine}_topics.json and
# {main,fine}_topic_pages.json (see scripts/naming/GUIDE.md). Run from the repository root.
#   scripts/naming/backup.sh law communication ...
set -e
for m in "$@"; do
    mkdir -p "runs/_naming_backup/$m"
    for res in main fine; do
        b="runs/$m/bundle"
        [ "$res" = fine ] && b="runs/$m/fine/bundle"
        cp -p "$b/topics.json" "runs/_naming_backup/$m/${res}_topics.json"
        cp -p "$b/topic_pages.json" "runs/_naming_backup/$m/${res}_topic_pages.json"
    done
    echo "kept $m"
done
