from __future__ import annotations

"""
Offline bootstrap: create a venv and install dependencies from local packages/.

Usage (from this folder):
    python bootstrap_offline.py
Then run:
    .venv/ Scripts/activate.bat (CMD) or Activate.ps1 (PowerShell) if you want an interactive shell
Or just execute checkers directly via:
    .venv/Script s/python.exe check_outlook_com.py
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def main() -> int:
    root = Path(__file__).parent
    req = root / "requirements.txt"
    pkg_dir = root / "packages"
    venv_dir = root / ".venv"

    if not req.exists():
        print(f"requirements.txt not found at {req}")
        return 1
    if not pkg_dir.exists():
        print(f"packages/ not found at {pkg_dir}. Prepare packages on an online machine using prepare_packages.py.")
        return 1

    # Create venv if missing
    if not venv_dir.exists():
        print(f"Creating virtualenv at {venv_dir} ...")
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))

    # Compute venv python
    if os.name == "nt":
        vpy = venv_dir / "Scripts" / "python.exe"
    else:
        vpy = venv_dir / "bin" / "python"
    if not vpy.exists():
        print("Virtualenv python not found.")
        return 1

    # Install from local packages folder
    code = run([str(vpy), "-m", "pip", "install", "--no-index", "--find-links", str(pkg_dir), "-r", str(req)])
    if code != 0:
        return code

    print("\n[OK] Offline installation complete.")
    print("Use the venv python to run checkers, e.g.:")
    if os.name == "nt":
        print(" ", str(vpy), "check_outlook_com.py")
    else:
        print(" ", str(vpy), "check_outlook_com.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


