#!/usr/bin/env python3
"""Generate the unified OpenINTJ icon set from the avatar master artwork.

Source : resources/avatar-master.jpg (1024x1024 anime avatar, purple hair)
Outputs:
  - resources/icon.png            1024x1024, window / taskbar / builder (mac+linux)
  - resources/icon.ico            multi-size (16..256), Windows app + NSIS installer
  - src/renderer/assets/app-icon.png  256x256, renderer (title bar logo; the
    1024 master lives in resources/, this copy keeps the bundle lean)

The crop matches the title-bar close-up (face-centred, ~1.65x zoom) so the
mark reads identically at 16px and at 1024px. Re-run after replacing the
master artwork:  python scripts/generate-avatar-icons.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "resources" / "avatar-master.jpg"

# Face-centred crop of the 1024x1024 master (left, upper, right, lower).
CROP = (202, 129, 822, 749)
ICO_SIZES = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]


def main() -> None:
    master = Image.open(MASTER).convert("RGB")
    if master.size != (1024, 1024):
        raise SystemExit(f"master must be 1024x1024, got {master.size}")

    icon = master.crop(CROP).resize((1024, 1024), Image.LANCZOS)

    png_path = ROOT / "resources" / "icon.png"
    icon.save(png_path, format="PNG")

    ico_path = ROOT / "resources" / "icon.ico"
    icon.save(ico_path, format="ICO", sizes=ICO_SIZES)

    renderer_path = ROOT / "src" / "renderer" / "assets" / "app-icon.png"
    icon.resize((256, 256), Image.LANCZOS).save(renderer_path, format="PNG")

    for path in (png_path, ico_path, renderer_path):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
