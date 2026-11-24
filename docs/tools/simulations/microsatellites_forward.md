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

(sec_tools_simulation_microsatellites_forward)=

# Microsatellites, forward in time

`DTWF_microsatellite_sim` is a forward-time simulator (see {ref}`sec_tools_simulation`) with
microsatellites whose numbers of repeats change ({issue}`202`). It is mainly a test harness
for the {ref}`homology tools <sec_tools_homology>`, rather than a realistic model: the changes
in length mean that the homology between two parent genomes is broken into many blocks, so a
crossover can only be placed in the right (allelic) place by joining the blocks into *chains*
(see {ref}`sec_design_homology`).

Its rules are:

- An expansion copies the last repeat of a microsatellite from the parent, so the copy's
  homology is recorded, as for slippage; a contraction deletes the last repeat.
- Each meiosis has one crossover, placed in a chain of homology between the parent genomes
  that is at least `min_chain_length` long, with gaps of at most `max_gap` between its blocks,
  and never inside a microsatellite.

```{code-cell}
from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim

# Microsatellites at positions 20, 70 and 120, with repeats 2, 3 and 2 positions long,
# repeated 5, 4 and 6 times in the founders
loci = [(20, 2, 5), (70, 3, 4), (120, 2, 6)]
simulator = DTWF_microsatellite_sim(loci, mutation_rate=0.2, min_chain_length=100, max_gap=20)
gig = simulator.run(num_diploids=5, seq_lens=[200], gens=8, random_seed=1)
print("Genome lengths:", sorted({gig.sequence_length(u) for u in gig.sample_nodes().ids}))
```

```{eval-rst}
.. automodule:: giglib.tools.simulation.microsatellites_forward
  :members:
```
