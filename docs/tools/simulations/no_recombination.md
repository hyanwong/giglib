---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_tools_simulation_no_recombination)=

# No recombination

`DTWF_no_recombination_sim` is the simplest forward-time simulator (see
{ref}`sec_tools_simulation`). Each child genome inherits each chromosome whole, from one of
its two parent genomes, chosen at random. With no recombination and no structural variation,
the result is a set of trees, one for each chromosome.

```{code-cell}
from giglib.tools.simulation.no_recombination import DTWF_no_recombination_sim

simulator = DTWF_no_recombination_sim()
gig = simulator.run(num_diploids=10, seq_lens=[100], gens=5, random_seed=1)
print(f"{len(gig.nodes)} nodes and {len(gig.iedges)} iedges")
```

```{eval-rst}
.. automodule:: giglib.tools.simulation.no_recombination
  :members:
```
