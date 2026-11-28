(sec_tools_recombination)=

# Recombination

The `giglib.tools.recombination` module has rules for placing crossovers between two genomes,
using their homology (found with {mod}`giglib.tools.homology`), shared by the simulators in
{ref}`giglib.tools.simulation <sec_tools_simulation>`. They are *policy*, i.e. choices that depend
on the biology being modelled (see the {ref}`recombination design note <sec_design_recombination>`),
so a simulator can use them, or choose its own. At present, crossovers are allowed in long chains of
homology (a stand-in for allelic homology), where both genomes are in the same orientation, and
outside regions that the simulator excludes, such as microsatellites. One of the allowed places is
then chosen uniformly.

```{eval-rst}
.. automodule:: giglib.tools.recombination
  :members:
```
