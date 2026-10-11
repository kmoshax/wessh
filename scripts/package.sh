#!/usr/bin/env bash
# Build the release downloads for one target from already built binaries.
#
#   scripts/package.sh <rust-target> [version]
#
# Every target gets a portable archive (what the one-line installers fetch):
#   dist/wessh-<target>.tar.gz             Linux, macOS
#   dist/wessh-<target>.zip                Windows (wessh.exe and wessh-cli.exe)
# and the download people double-click:
#   dist/wessh-linux-<arch>.AppImage       with APPIMAGETOOL set to an appimagetool executable
#   dist/wessh-windows-x64-setup.exe       with ISCC set to Inno Setup's compiler
#   dist/wessh-macos-<arch>.dmg            on macOS (iconutil, sips, hdiutil, codesign)
#
# Names carry no version, so https://github.com/kmoshax/wessh/releases/latest/download/<name>
# always points at the newest release. Build first with
#   cargo build --release --target <rust-target> -p wessh
# (on Windows also `--features console`, copied to wessh-cli.exe).
set -euo pipefail

target="${1:?usage: scripts/package.sh <rust-target> [version]}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
version="${2:-v$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)}"
version="${version#v}"
release="target/$target/release"
icon=crates/wessh-app/assets/icon
mkdir -p dist

stage() { # <dir> <binaries...>: binaries, README and license in <dir>
    local dir="$1"
    shift
    rm -rf "$dir"
    mkdir -p "$dir"
    for binary in "$@"; do
        [[ -f "$release/$binary" ]] || { echo "missing $release/$binary; build it first" >&2; exit 1; }
        cp "$release/$binary" "$dir/"
    done
    cp README.md LICENSE "$dir/"
}

case "$target" in
*windows*)
    stage dist/stage wessh.exe wessh-cli.exe
    rm -f "dist/wessh-$target.zip"
    (cd dist/stage && if command -v 7z >/dev/null; then 7z a -tzip -bso0 "../wessh-$target.zip" .; else zip -qr "../wessh-$target.zip" .; fi)
    echo "dist/wessh-$target.zip"
    if [[ -n "${ISCC:-}" ]]; then
        "$ISCC" /Q "/DAppVersion=$version" "/DSourceDir=$root/dist/stage" "/DOutputDir=$root/dist" scripts/windows/wessh.iss
        echo "dist/wessh-windows-x64-setup.exe"
    fi
    rm -rf dist/stage
    ;;

*apple-darwin*)
    arch="${target%%-*}"
    [[ "$arch" == aarch64 ]] && arch=arm64
    stage "dist/wessh-$target" wessh
    tar -C dist -czf "dist/wessh-$target.tar.gz" "wessh-$target"
    rm -rf "dist/wessh-$target"
    echo "dist/wessh-$target.tar.gz"

    app=dist/dmg/wessh.app
    rm -rf dist/dmg
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$release/wessh" "$app/Contents/MacOS/"
    sed "s/@VERSION@/$version/g" scripts/macos/Info.plist >"$app/Contents/Info.plist"
    iconset=dist/wessh.iconset
    mkdir -p "$iconset"
    for size in 16 32 128 256 512; do
        sips -z "$size" "$size" "$icon/wessh-1024.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
        sips -z $((size * 2)) $((size * 2)) "$icon/wessh-1024.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$iconset" -o "$app/Contents/Resources/wessh.icns"
    rm -rf "$iconset"
    # Ad-hoc signature: Apple Silicon refuses unsigned code, and the camera prompt needs a sealed bundle.
    codesign --force --deep --sign - "$app"
    ln -s /Applications dist/dmg/Applications
    rm -f "dist/wessh-macos-$arch.dmg"
    hdiutil create -quiet -volname wessh -srcfolder dist/dmg -ov -format UDZO "dist/wessh-macos-$arch.dmg"
    rm -rf dist/dmg
    echo "dist/wessh-macos-$arch.dmg"
    ;;

*linux*)
    arch="${target%%-*}"
    stage "dist/wessh-$target" wessh
    tar -C dist -czf "dist/wessh-$target.tar.gz" "wessh-$target"
    rm -rf "dist/wessh-$target"
    echo "dist/wessh-$target.tar.gz"
    if [[ -n "${APPIMAGETOOL:-}" ]]; then
        appdir=dist/wessh.AppDir
        rm -rf "$appdir"
        mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" "$appdir/usr/share/icons/hicolor/256x256/apps"
        cp "$release/wessh" "$appdir/usr/bin/"
        cp scripts/linux/wessh.desktop "$appdir/wessh.desktop"
        cp scripts/linux/wessh.desktop "$appdir/usr/share/applications/wessh.desktop"
        cp "$icon/wessh-256.png" "$appdir/wessh.png"
        cp "$icon/wessh-256.png" "$appdir/usr/share/icons/hicolor/256x256/apps/wessh.png"
        ln -s usr/bin/wessh "$appdir/AppRun"
        ARCH="$arch" "$APPIMAGETOOL" --no-appstream "$appdir" "dist/wessh-linux-$arch.AppImage"
        rm -rf "$appdir"
        echo "dist/wessh-linux-$arch.AppImage"
    fi
    ;;

*)
    echo "unsupported target $target" >&2
    exit 1
    ;;
esac
