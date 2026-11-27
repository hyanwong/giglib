(sec_tools_simulation)=

# Simulations

The `giglib.tools.simulation` package has some simple simulators, which produce GIGs. They are used in the tests, and are a starting point for simulating
structural variants (see {ref}`sec_usage_simulations`). Each simulator has a page of its own:

- {ref}`sec_tools_simulation_no_recombination`: forward in time, with no recombination.
- {ref}`sec_tools_simulation_one_break`: forward in time, with one crossover per meiosis,
  avoiding inverted regions.
- {ref}`sec_tools_simulation_single_origin` (planned): forward in time, growing genomes from a
  single base pair by tandem duplication.

The rules a simulator uses (e.g. where crossovers can happen, between which copies of a
duplicated region) are its own *policy*, built on the *mechanism* provided by _giglib_ (see
the {ref}`homology design note <sec_design_homology>`). Each simulator's page says which rules it uses.

Code shared by several simulators is described on this page, below.

## Forward-time simulators

The forward-time simulators are discrete-time Wright-Fisher models of a diploid population,
based on the _tskit_
[forward-time simulation tutorial](https://tskit.dev/tutorials/forward_sims.html). Each is a
subclass of `DTWF_simulator`, which builds the population one generation at a time, and adds
iedges from each child genome to its parent genomes as it goes. Subclasses decide how a child
inherits from its parents, by overriding `add_inheritance_paths()`.

```{eval-rst}
.. automodule:: giglib.tools.simulation

.. automodule:: giglib.tools.simulation.dtwf
  :members:
```

## Backward-time simulators

Changes in the length of genomes, such as indels, don't affect where crossovers can happen
(as long as homology is chained across small gaps, see the {ref}`homology design note <sec_design_homology>`), so the
genealogy doesn't depend on them ({issue}`93`). So a GIG with indels can be simulated much
faster backward in time: simulate the genealogy, e.g. with msprime, add the indels as
mutations, and convert the resulting tree sequence with the
{ref}`indels tool <sec_tools_indels>`. This only gives changes that fit in a multiple
sequence alignment (substitutions, insertions and deletions, including changes in the
number of repeats at microsatellites), not other structural variants, which need the
forward-time simulators. A backward-time simulator of SNP-like indels is planned
({issue}`216`).
