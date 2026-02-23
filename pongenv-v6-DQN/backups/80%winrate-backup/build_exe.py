#!/usr/bin/env python3
import sys
import platform
import subprocess
from pathlib import Path

"""
PyInstaller build script for Pong DQN project.
Usage:
  1) Activate your conda env:  conda activate myenv
  2) Install build deps:       python -m pip install -U pyinstaller pyinstaller-hooks-contrib
  3) Run this script:          python build_exe.py

This builds an onedir distribution under ./dist/pong_dqn/
Resources (bg.png, dqn.pth) are included.
"""


def build():
    project_root = Path(__file__).parent
    main_script = project_root / "pong_bot_DQN.py"
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")

    name = "pong_dqn"
    sep = ";" if platform.system() == "Windows" else ":"

    # Resources to bundle next to the executable
    add_data = [
        f"bg.png{sep}.",
        f"dqn.pth{sep}.",
    ]

    # Hidden imports commonly missed by analysis
    hidden_imports = [
        "pygame",
        "numpy",
        "torch",
        "torch.nn",
        "matplotlib",
        "matplotlib.pyplot",
    ]

    # Collect everything for complex libs (code, data, binaries)
    collect_all = [
        "pygame",
        "torch",
        "numpy",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(main_script),
        "--name", name,
        "--onedir",            # prefer onedir for complex libs like torch
        "--clean",
        "--noconfirm",
        "--console",           # keep console for logs
    ]

    for d in add_data:
        cmd += ["--add-data", d]

    # add hidden imports
    for mod in hidden_imports:
        cmd += ["--hidden-import", mod]

    # collect-all for libs
    for lib in collect_all:
        cmd += ["--collect-all", lib]

    # also collect submodules just in case
    cmd += [
        "--collect-submodules", "torch",
        "--collect-submodules", "pygame",
    ]

    print("Running PyInstaller command:\n ", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(project_root))
    print(f"\nBuild complete. See dist/{name}/ for the generated files.")


if __name__ == "__main__":
    build()
