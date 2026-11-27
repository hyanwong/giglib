(sec_tools_homology)=

# Homology

The `giglib.tools.homology` module builds on the homology between two genomes (positions that
trace back to the same position in a common ancestor) found by
{meth}`~giglib.Tables.find_homology` (see {ref}`sec_usage_homology`), as described in
{ref}`sec_design_homology`. It can join homology into *chains*, which distinguish allelic
homology (long chains, even across many small differences in length, e.g. at microsatellites)
from non-allelic homology (e.g. between copies of a transposable element), pick a random pair
of homologous positions (e.g. for a crossover), and plot the homology between two genomes.

```{eval-rst}
.. automodule:: giglib.tools.homology
  :members:
```
