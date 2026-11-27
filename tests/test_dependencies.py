"""
Check that giglib only needs its declared dependencies
"""

import ast
import pathlib
import subprocess
import sys

import pytest
import tomllib

ROOT = pathlib.Path(__file__).parent.parent

# Modules that tests and optional features use, but giglib itself must not need
OPTIONAL = ["matplotlib", "pandas", "networkx", "msprime"]


@pytest.mark.parametrize("optional", OPTIONAL)
def test_import_without_optional_module(optional):
    # Block the module in a fresh interpreter, then import and use giglib
    code = f"""
import sys
sys.modules[{optional!r}] = None
import giglib, giglib.tools, giglib.tools.indels, giglib.tools.homology
import giglib.tools.simulation.no_recombination, giglib.tools.simulation.one_break_inversions
import giglib.tools.simulation.microsatellites_forward, giglib.tools.simulation.microsatellites_backward
tables = giglib.Tables()
tables.nodes.add_row(1)
tables.nodes.add_row(0, flags=giglib.NODE_IS_SAMPLE)
tables.iedges.add_row(1, 0, child_right=10)
tables.mutations.add_row(0, 3, "A")
gig = tables.graph()
assert gig.allele_at(1, 3) == "A"
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_imports_are_declared():
    # Every third-party module imported (at the top level) by giglib is a declared
    # dependency
    with open(ROOT / "pyproject.toml", "rb") as f:
        declared = {
            d.split(">")[0].split("=")[0].split("<")[0].strip() for d in tomllib.load(f)["project"]["dependencies"]
        }
    imported = set()
    for path in (ROOT / "giglib").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:  # top-level imports only
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = {m for m in imported if m not in sys.stdlib_module_names and m != "giglib"}
    assert third_party <= declared, f"Undeclared dependencies: {third_party - declared}"
