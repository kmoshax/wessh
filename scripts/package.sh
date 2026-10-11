#!/usr/bin/env bash
# Build a portable release archive from an already built binary.
#
#   scripts/package.sh <rust-target> [version]
#   scripts/package.sh x86_64-unknown-linux-gnu v0.2.0   ->  dist/weshy-v0.2.0-x86_64-unknown-linux-gnu.tar.gz
#
# Build first with `cargo build --release --target <rust-target> -p weshy`. The archive holds the
# binary, README and licenses; `weshy setup` installs ONNX Runtime and the models on first run.
set -euo pipefail

target="${1:?usage: scripts/package.sh <rust-target> [version]}"
version="${2:-v$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

exe=weshy
[[ "$target" == *windows* ]] && exe=weshy.exe
binary="target/$target/release/$exe"
[[ -f "$binary" ]] || { echo "missing $binary; run: cargo build --release --target $target -p weshy" >&2; exit 1; }

name="weshy-$version-$target"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage"
cp "$binary" README.md LICENSE "$stage/"

cd dist
if [[ "$target" == *windows* ]]; then
    rm -f "$name.zip"
    if command -v 7z >/dev/null; then 7z a -tzip -bso0 "$name.zip" "$name"; else zip -qr "$name.zip" "$name"; fi
    echo "dist/$name.zip"
else
    tar -czf "$name.tar.gz" "$name"
    echo "dist/$name.tar.gz"
fi
rm -rf "$name"
