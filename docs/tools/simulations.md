(sec_tools_simulation)=

# Simulations

The `giglib.tools.simulation` module has some simple simulators, which record their
genealogies as GIGs. They are used in the tests, and are a starting point for simulating
structural variants (see {ref}`sec_usage_simulations`). Each simulator has a page of its own:

- {ref}`sec_tools_simulation_no_recombination`: forward in time, with no recombination.
- {ref}`sec_tools_simulation_one_break`: forward in time, with one crossover per meiosis,
  avoiding inverted regions.

The rules a simulator uses (e.g. where crossovers can happen, between which copies of a
duplicated region) are its own *policy*, built on the *mechanism* provided by _giglib_ (see
{ref}`sec_design_homology`). Each simulator's page says which rules it uses.

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
