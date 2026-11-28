(sec_tools)=

# Experimental tools

The `giglib.tools` subpackage contains extra functionality that is built on top of
_giglib_, but is **not part of the core API**:

- **Not imported by default.** `import giglib` does not load it, so you need to import it
  explicitly, e.g. `import giglib.tools`.
- **Experimental.** Tools may change, or be removed, without warning, and may only work
  in restricted cases (raising an error otherwise).
- **Built on the public API.** Tools only use the documented _giglib_ API, so they do not
  depend on how _giglib_ stores its data internally (see {ref}`sec_design_storage`).
- **Tested.** Tools are tested along with the rest of _giglib_.

The tools are:

- {ref}`sec_tools_indels` (`giglib.tools.indels`): converting between GIGs and tree
  sequences with insertions and deletions, whose coordinates are those of a multiple
  sequence alignment.
- {ref}`sec_tools_homology` (`giglib.tools.homology`): finding the homology between two
  genomes, joining it into chains, and plotting it.
- {ref}`sec_tools_recombination` (`giglib.tools.recombination`): rules for placing crossovers
  between two genomes using their homology, shared by the simulators.
- {ref}`sec_tools_simulation` (`giglib.tools.simulation`): simple simulators that record
  their genealogies as GIGs, each described on a page of its own.
