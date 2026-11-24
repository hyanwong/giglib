(sec_tools_homology)=

# Homology

The `giglib.tools.homology` module finds the homology between two genomes (positions that
trace back to the same position in a common ancestor), building on
{meth}`~giglib.Tables.find_mrca_regions`, as described in {ref}`sec_design_homology`. It
can join homology into *chains*, which distinguish allelic homology (long chains, even across
many small differences in length, e.g. at microsatellites) from non-allelic homology (e.g.
between copies of a transposable element), and can find *all* the homology between two
genomes, not just the most recent.

```{eval-rst}
.. automodule:: giglib.tools.homology
  :members:
```
