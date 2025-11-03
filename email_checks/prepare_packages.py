from __future__ import annotations

"""
Prepare offline wheel packages into packages/ using the current machine's internet.

Usage (from this folder):
    python prepare_packages.py
This is equivalent to:
    pip download -r requirements.txt -d packages/
"""

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    root = Path(__file__).parent
    req = root / "requirements.txt"
    out = root / "packages"
    out.mkdir(exist_ok=True)
    if not req.exists():
        print(f"requirements.txt not found at {req}")
        return 1
    return run(["pip", "download", "-r", str(req), "-d", str(out)])


if __name__ == "__main__":
    raise SystemExit(main())


