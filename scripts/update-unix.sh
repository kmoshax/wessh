#!/bin/sh
# Internal updater helper. Paths and application arguments are separate argv entries.
set -eu
parent_pid="$1"
candidate="$2"
target="$3"
backup="$4"
kind="$5"
restart="$6"
shift 6

while kill -0 "$parent_pid" 2>/dev/null; do sleep 0.2; done

start_app() {
    if [ "$restart" = yes ]; then
        if [ "$kind" = bundle ]; then /usr/bin/open -n "$target" --args "$@"
        else "$target" "$@" &
        fi
    fi
}

if ! mv "$target" "$backup"; then
    echo "Update failed: could not back up the installed application." >&2
    start_app "$@"
    exit 1
fi
if ! mv "$candidate" "$target"; then
    echo "Update failed: restoring the previous application." >&2
    mv "$backup" "$target"
    start_app "$@"
    exit 1
fi
echo "Update installed. Previous application: $backup"
start_app "$@"
