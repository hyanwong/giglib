"""
Tests for the experimental giglib.tools subpackage
"""

import fnmatch
import pathlib
import subprocess
import sys

import tomllib


def test_not_imported_by_giglib():
    # Run in a fresh interpreter, as other tests may already have imported giglib.tools
    code = "import sys, giglib; print('giglib.tools' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_importable():
    import giglib.tools

    assert "experimental" in giglib.tools.__doc__.lower()


def test_packaged():
    # setuptools finds the packages to put in wheels by matching these patterns, so every
    # package (each directory of giglib with an __init__.py) must match one
    root = pathlib.Path(__file__).parent.parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)
    patterns = config["tool"]["setuptools"]["packages"]["find"]["include"]
    packages = [".".join(path.parent.relative_to(root).parts) for path in (root / "giglib").rglob("__init__.py")]
    assert "giglib.tools.simulation" in packages
    for package in packages:
        assert any(fnmatch.fnmatch(package, pattern) for pattern in patterns), package
