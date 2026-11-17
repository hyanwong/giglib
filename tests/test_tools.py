"""
Tests for the experimental giglib.tools subpackage
"""

import pathlib
import subprocess
import sys

import pytest


def test_not_imported_by_giglib():
    # Run in a fresh interpreter, as other tests may already have imported giglib.tools
    code = "import sys, giglib; print('giglib.tools' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_importable():
    import giglib.tools

    assert "experimental" in giglib.tools.__doc__.lower()


def test_packaged():
    # The packages are listed explicitly, so a subpackage is left out of wheels unless listed
    tomllib = pytest.importorskip("tomllib")
    root = pathlib.Path(__file__).parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)
    assert "giglib.tools" in config["tool"]["setuptools"]["packages"]
