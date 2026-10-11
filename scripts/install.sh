#!/bin/sh
# Install the latest weshy release and prepare the machine (Linux and macOS).
#
#   curl -fsSL https://raw.githubusercontent.com/kmoshax/weshy/main/scripts/install.sh | sh
#
# Environment: WESHY_VERSION (tag, default latest), WESHY_INSTALL_DIR (default ~/.local/bin),
# WESHY_SKIP_SETUP=1 to only install the binary. Extra arguments are passed to `weshy setup`.
set -eu

repo=kmoshax/weshy

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
    Linux-aarch64 | Linux-arm64) target=aarch64-unknown-linux-gnu ;;
    Darwin-arm64) target=aarch64-apple-darwin ;;
    Darwin-x86_64) target=x86_64-apple-darwin ;;
    *) echo "unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

tag="${WESHY_VERSION:-$(curl -fsSL "https://api.github.com/repos/$repo/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)}"
[ -n "$tag" ] || { echo "could not determine the latest release; set WESHY_VERSION" >&2; exit 1; }
dir="${WESHY_INSTALL_DIR:-$HOME/.local/bin}"
archive="weshy-$tag-$target.tar.gz"
base="https://github.com/$repo/releases/download/$tag"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading weshy $tag for $target"
curl -fL --progress-bar -o "$tmp/$archive" "$base/$archive"
curl -fsSL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS"
(
    cd "$tmp"
    grep " $archive\$" SHA256SUMS > expected
    if command -v sha256sum >/dev/null; then sha256sum -c expected; else shasum -a 256 -c expected; fi
)
tar -xzf "$tmp/$archive" -C "$tmp"
mkdir -p "$dir"
install -m 755 "$tmp/weshy-$tag-$target/weshy" "$dir/weshy"
echo "installed $dir/weshy"
case ":$PATH:" in *":$dir:"*) ;; *) echo "note: add $dir to your PATH" ;; esac

if [ "${WESHY_SKIP_SETUP:-0}" != 1 ]; then
    "$dir/weshy" setup "$@"
fi
