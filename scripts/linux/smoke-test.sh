#!/usr/bin/env bash
# Run the extracted AppImage on a clean distribution, without FUSE or GPU access.
set -euo pipefail
image="${1:?usage: smoke-test.sh <AppImage>}"
work="$(mktemp -d)"
display_pid=""
trap 'if [[ -n "$display_pid" ]]; then kill "$display_pid" 2>/dev/null || true; fi; rm -rf "$work"' EXIT
image="$(realpath "$image")"
(cd "$work" && "$image" --appimage-extract >/dev/null)
app="$work/squashfs-root/AppRun"
"$app" --version
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp >"$work/xvfb.log" 2>&1 &
display_pid="$!"
export DISPLAY=:99
for attempt in {1..50}; do
    [[ -S /tmp/.X11-unix/X99 ]] && break
    sleep 0.1
done
WESSH_HOME="$work/data" WESSH_UI_ONBOARDING=welcome WESSH_UI_CAPTURE="$work/welcome.png" \
    LIBGL_ALWAYS_SOFTWARE=1 timeout 45s "$app" --device cpu
[[ -s "$work/welcome.png" ]] || { echo 'No onboarding screenshot produced' >&2; exit 1; }
echo 'AppImage CLI and graphical onboarding passed'
