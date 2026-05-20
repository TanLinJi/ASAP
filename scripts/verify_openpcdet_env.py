"""Check whether the active Python environment can run OpenPCDet.

This script only checks importability. It does not install dependencies.

Usage:
    python scripts/verify_openpcdet_env.py
"""

from __future__ import annotations

import importlib.util
import sys

REQUIRED_MODULES = ["torch", "numpy", "yaml", "skimage", "SharedArray"]
OPTIONAL_OR_BACKEND_MODULES = ["spconv", "pcdet"]


def main() -> int:
    print(f"[ASAP] Python executable: {sys.executable}")
    print(f"[ASAP] Python version: {sys.version.split()[0]}")
    failed = 0
    for name in REQUIRED_MODULES:
        found = importlib.util.find_spec(name) is not None
        print(f"[ASAP][{'OK' if found else 'FAIL'}] {name}: {'found' if found else 'missing'}")
        if not found:
            failed += 1
    for name in OPTIONAL_OR_BACKEND_MODULES:
        found = importlib.util.find_spec(name) is not None
        print(f"[ASAP][INFO] {name}: {'found' if found else 'missing'}")
    if failed:
        print(f"[ASAP] Environment check FAILED with {failed} missing required module(s).")
        return 1
    print("[ASAP] Basic environment check PASSED. OpenPCDet may still require compiled CUDA operators for full detection evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
