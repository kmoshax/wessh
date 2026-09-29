#!/usr/bin/env python3
"""Rebuild the embedded Material Symbols Rounded subsets used by the UI.

Usage: subset_icons.py <MaterialSymbolsRounded[FILL,GRAD,opsz,wght].ttf>

Pins the variable axes to static outlined (FILL 0) and filled (FILL 1) instances,
keeps only the icons listed in ICONS, and prints the matching Rust constants for
`src/ui/icons.rs`. Requires `fonttools`.
"""

import pathlib
import sys

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ICONS = """
add_photo_alternate arrow_drop_down arrow_drop_up auto_awesome blur_on check check_circle close
dark_mode error face flip folder_open graphic_eq groups hourglass_top image info
light_mode memory movie photo_camera play_arrow speed stop swap_horiz timer tune
upload videocam videocam_off crop_free center_focus_strong cast sensors
face_retouching_natural refresh expand_more palette settings warning frame_person
back_hand mood sort hd
save save_as content_copy compare compare_arrows cancel task_alt replay schedule
aspect_ratio burst_mode folder video_file terminal edit
download pause arrow_forward verified_user bolt cloud_off
""".split()

HERE = pathlib.Path(__file__).parent


def codepoints(font):
    by_name = {}
    for code, name in sorted(font.getBestCmap().items()):
        by_name.setdefault(name, code)
    missing = [n for n in ICONS if n not in by_name]
    if missing:
        sys.exit(f"unknown icons: {missing}")
    return {n: by_name[n] for n in ICONS}


def build(source, codes, fill, out):
    font = TTFont(source)
    options = subset.Options()
    options.layout_features = []
    options.name_IDs = ["*"]
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=list(codes.values()))
    subsetter.subset(font)
    font = instancer.instantiateVariableFont(font, {"FILL": fill, "wght": 400, "GRAD": 0, "opsz": 24})
    font.save(out)


def main():
    source = sys.argv[1]
    codes = codepoints(TTFont(source))
    build(source, codes, 0, HERE / "MaterialSymbolsRounded-Outlined.ttf")
    build(source, codes, 1, HERE / "MaterialSymbolsRounded-Filled.ttf")
    for name, code in sorted(codes.items()):
        print(f"pub const {name.upper()}: char = '\\u{{{code:x}}}';")


if __name__ == "__main__":
    main()
