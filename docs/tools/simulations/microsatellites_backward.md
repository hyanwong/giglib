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

(sec_tools_simulation_microsatellites_backward)=

# Microsatellites, backward in time

`sim_microsatellites_backward` simulates microsatellites backward in time ({issue}`202`),
which is much faster than {ref}`forward in time <sec_tools_simulation_microsatellites_forward>`.
Changes in the length of a microsatellite don't affect where crossovers can happen, so the
genealogy doesn't depend on them: it is simulated with msprime, and the microsatellites are
then laid on top of it, using msprime's stepwise mutation model. The
{ref}`indels tool <sec_tools_indels>` then turns the changes in the number of repeats
into changes in the coordinates of the genomes. Gained repeats are inserted material of
unknown origin, rather than copies of a neighbouring repeat as in the forward-time simulator.

This needs msprime, which is not a requirement of _giglib_ itself.

```{code-cell}
from giglib.tools.simulation.microsatellites_backward import sim_microsatellites_backward

# Microsatellites at positions 200, 500 and 800 of a 1000-position sequence
loci = [(200, "AT"), (500, "CAG"), (800, "GATA")]
gig = sim_microsatellites_backward(
    10, 1000, loci,
    max_units=10, population_size=1e4, recombination_rate=1e-8, mutation_rate=1e-4,
    random_seed=1,
)
print("Genome lengths:", sorted({gig.sequence_length(u) for u in gig.sample_nodes().ids}))
# The known bases of a genome are its microsatellites (other positions are unknown, N)
print("Known bases of genome 0:", gig.haplotype(0).replace("N", ""))
```

```{eval-rst}
.. automodule:: giglib.tools.simulation.microsatellites_backward
  :members:
```
