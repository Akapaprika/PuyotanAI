# GUI package
import sys
from pathlib import Path

# Ensure native library is loaded before any submodules
# The `gui` folder is now at the root, so the parent is the project root
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
DEBUG_DIR = BASE_DIR / "native" / "build_Debug" / "Debug"
for d in [DIST_DIR, RELEASE_DIR, DEBUG_DIR]:
    if d.exists():
        sys.path.insert(0, str(d))
        break
